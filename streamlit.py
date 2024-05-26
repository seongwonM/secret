import streamlit as st
import sqlite3
import requests
import json
import datetime
import time
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from io import StringIO
from datetime import timedelta
import pytz
import yfinance as yf

# model py파일 import
from stock import Stock, Mymodel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터베이스 연결
conn = sqlite3.connect('stock_prices.db')
cursor = conn.cursor()

# 테이블에 'close'와 'volume' 열이 존재하는지 확인하고, 없으면 추가
cursor.execute("PRAGMA table_info(price_info)")
columns = [info[1] for info in cursor.fetchall()]
if 'close' not in columns:
    cursor.execute('ALTER TABLE price_info ADD COLUMN close INTEGER')
if 'volume' not in columns:
    cursor.execute('ALTER TABLE price_info ADD COLUMN volume INTEGER')
conn.commit()

# 테이블 생성 (필요할 경우)
cursor.execute('''
CREATE TABLE IF NOT EXISTS price_info (
    time_key TEXT,
    stock_code TEXT,
    high INTEGER,
    low INTEGER,
    open INTEGER,
    close INTEGER,
    volume INTEGER,
    PRIMARY KEY (time_key, stock_code)
)
''')
conn.commit()

# 모델1 부분

def get_model_prediction(stock_code, current_hour_key):
    # current_hour_key 이전 10개 데이터 가져오기
    cursor.execute('SELECT * FROM price_info WHERE stock_code = ? AND time_key < ? ORDER BY time_key DESC LIMIT ?', (stock_code, current_hour_key, 5))
    rows = cursor.fetchall()

    if len(rows) < 5:
        return None  # 데이터가 충분하지 않으면 None 반환

    # 데이터를 DataFrame으로 변환
    df = pd.DataFrame(rows, columns=['time_key', 'stock_code', 'high', 'low', 'open', 'close', 'Volume'])

    stock=Stock(df)
    stock.preprocessing()
    stock.add_change(stock.df.columns)
    stock.df.loc[stock.df['Volume_chg']==np.inf,'Volume_chg']=0
    # stock.scale_col(stock.df.columns[[3,0,1,2,4]]) # 종가
    stock.scale_col(stock.df.columns[[8,5,6,7,9]]) # 종가(변화율)
    train_loader=stock.data_loader(stock.seq_len, 'train')
    valid_loader=stock.data_loader(stock.seq_len, 'valid')
    test_loader=stock.data_loader(stock.seq_len, 't')
    stock.create_model(1, 0.2)
    stock.model.load_state_dict(torch.load('chg_close_loss.pth'))
    loss=stock.train(train_loader, valid_loader, test_loader, 10, 0.1, 20, 'test')
    predicted_class, act=stock.pred_value('t')

    return predicted_class


# 전역 변수 설정
ACCESS_TOKEN = None
token_issue_time = None
TOKEN_VALIDITY_DURATION = 3600 * 6  # 토큰 유효 기간 (6시간)

# 함수 정의
def send_message(msg, DISCORD_WEBHOOK_URL):
    """디스코드 메세지 전송"""
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(msg)}"}
    requests.post(DISCORD_WEBHOOK_URL, data=message)
    print(message)

def get_access_token(APP_KEY, APP_SECRET, URL_BASE):
    """토큰 발급"""
    global ACCESS_TOKEN, token_issue_time
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    if res.status_code == 200:
        ACCESS_TOKEN = res.json()["access_token"]
        token_issue_time = datetime.datetime.now()
        return ACCESS_TOKEN
    else:
        raise Exception("Failed to get access token")

def ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE):
    """토큰의 유효성을 보장하고 필요시 갱신"""
    global ACCESS_TOKEN, token_issue_time
    if ACCESS_TOKEN is None or (datetime.datetime.now() - token_issue_time).total_seconds() >= TOKEN_VALIDITY_DURATION:
        ACCESS_TOKEN = get_access_token(APP_KEY, APP_SECRET, URL_BASE)

def hashkey(datas, APP_KEY, APP_SECRET, URL_BASE):
    """암호화"""
    PATH = "uapi/hashkey"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        'content-Type': 'application/json',
        'appKey': APP_KEY,
        'appSecret': APP_SECRET,
    }
    res = requests.post(URL, headers=headers, data=json.dumps(datas))
    hashkey = res.json()["HASH"]
    return hashkey

def reset_database():
    cursor.execute('DROP TABLE IF EXISTS price_info')
    cursor.execute('''
    CREATE TABLE price_info (
        time_key TEXT,
        stock_code TEXT,
        high INTEGER,
        low INTEGER,
        open INTEGER,
        close INTEGER,
        volume INTEGER,
        PRIMARY KEY (time_key, stock_code)
    )
    ''')
    conn.commit()



def get_current_price_and_volume(code, APP_KEY, APP_SECRET, URL_BASE):
    """현재가와 누적 거래량 조회"""
    ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
    PATH = "uapi/domestic-stock/v1/quotations/inquire-price"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST01010100"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": code,
    }
    res = requests.get(URL, headers=headers, params=params)
    data = res.json()['output']
    current_price = int(data['stck_prpr'])
    acml_vol = int(data['acml_vol'])
    return current_price, acml_vol

def get_previous_row(stock_code, current_hour_key):
    cursor.execute('''
        SELECT high, low 
        FROM price_info 
        WHERE stock_code = ? AND time_key < ? 
        ORDER BY time_key DESC 
        LIMIT 1
    ''', (stock_code, current_hour_key))
    return cursor.fetchone()

def get_target_price_change(stock_code):
    now = datetime.datetime.now()
    current_hour_key = now.strftime('%Y-%m-%d %H')

    if now.hour == 9:
        previous_day = now - datetime.timedelta(days=1)
        previous_hour_key = previous_day.replace(hour=14).strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key)  # 수정된 부분
    else:
        previous_hour = now - datetime.timedelta(hours=1)
        previous_hour_key = previous_hour.strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key)  # 수정된 부분

    cursor.execute('SELECT open FROM price_info WHERE time_key = ? AND stock_code = ?', (current_hour_key, stock_code))
    stck_oprc = cursor.fetchone()

    if stck_oprc and prev_data:
        stck_oprc = stck_oprc[0]
        stck_hgpr, stck_lwpr = prev_data
        target_price = stck_oprc + (stck_hgpr - stck_lwpr) * 0.5
        return target_price
    else:
        return None
    

def sell_target_price_change(stock_code):
    now = datetime.datetime.now()
    current_hour_key = now.strftime('%Y-%m-%d %H')

    if now.hour == 9:
        previous_day = now - datetime.timedelta(days=1)
        previous_hour_key = previous_day.replace(hour=14).strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key)  # 수정된 부분
    else:
        previous_hour = now - datetime.timedelta(hours=1)
        previous_hour_key = previous_hour.strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key)  # 수정된 부분

    cursor.execute('SELECT open FROM price_info WHERE time_key = ? AND stock_code = ?', (current_hour_key, stock_code))
    stck_oprc = cursor.fetchone()

    if stck_oprc and prev_data:
        stck_oprc = stck_oprc[0]
        stck_hgpr, stck_lwpr = prev_data
        target_price = stck_oprc - (stck_hgpr - stck_lwpr) * 0.5
        return target_price
    else:
        return None


def get_target_price_ma(stock_code):
    cursor.execute('SELECT time_key, open FROM price_info WHERE stock_code = ?', (stock_code,))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['time_key', 'open'])
    df['time_key'] = pd.to_datetime(df['time_key'])
    df.set_index('time_key', inplace=True)
    df['SMA2'] = df['open'].rolling(window=2, min_periods=1).mean()
    df['SMA4'] = df['open'].rolling(window=4, min_periods=1).mean()
    return df

def get_accumulated_volume(stock_code, current_hour_key):
    current_date = current_hour_key.split(' ')[0]  # YYYY-MM-DD 부분만 추출
    cursor.execute('''
        SELECT SUM(volume) 
        FROM price_info 
        WHERE stock_code = ? AND time_key < ? AND time_key LIKE ?
    ''', (stock_code, current_hour_key, f'{current_date}%'))
    result = cursor.fetchone()
    return result[0] if result[0] is not None else 0


def update_price_info(current_price, current_volume, current_time, stock_code):
    time_key = current_time.strftime('%Y-%m-%d %H')
    
    cursor.execute('SELECT * FROM price_info WHERE time_key = ? AND stock_code = ?', (time_key, stock_code))
    data = cursor.fetchone()

    # 누적 거래량 계산
    accumulated_volume_before = get_accumulated_volume(stock_code, time_key)
    volume = current_volume - accumulated_volume_before

    if data is None:
        cursor.execute('''
        INSERT INTO price_info (time_key, stock_code, high, low, open, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (time_key, stock_code, current_price, current_price, current_price, current_price, volume))
        conn.commit()
    else:
        cursor.execute('''
        UPDATE price_info
        SET high = MAX(high, ?),
            low = MIN(low, ?),
            close = ?,
            volume = ?
        WHERE time_key = ? AND stock_code = ?
        ''', (current_price, current_price, current_price, volume, time_key, stock_code))
        conn.commit()

def fetch_recent_5_hours_data(stock_code):
    try:
        stock = yf.Ticker(stock_code+'.KS')
        data = stock.history(period="1d", interval="1h")
        if data.empty:
            stock = yf.Ticker(stock_code+'.KQ')
            data = stock.history(period="1d", interval="1h")
            if data.empty:
                st.error(f"No data fetched for {stock_code}")
    except Exception as e:
        st.error(f"Error fetching data for {stock_code}: {e}")

    data=data.tail(5)
    for idx, row in data.iterrows():
        time_key = idx.strftime('%Y-%m-%d %H')
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        volume = row['Volume']
        
        cursor.execute('''
        INSERT OR REPLACE INTO price_info (time_key, stock_code, high, low, open, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (time_key, stock_code, high_price, low_price, open_price, close_price, volume))
    
    conn.commit()

def get_stock_balance(APP_KEY, APP_SECRET, URL_BASE):
    """주식 잔고조회"""
    ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
    PATH = "uapi/domestic-stock/v1/trading/inquire-balance"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "TTTC8434R",
        "custtype": "P"
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    res = requests.get(URL, headers=headers, params=params)
    stock_list = res.json()['output1']
    evaluation = res.json()['output2']
    stock_dict = {}
    send_message(f"====주식 보유잔고====", DISCORD_WEBHOOK_URL)
    for stock in stock_list:
        if int(stock['hldg_qty']) > 0:
            stock_dict[stock['pdno']] = stock['hldg_qty']
            send_message(f"{stock['prdt_name']}({stock['pdno']}): {stock['hldg_qty']}주", DISCORD_WEBHOOK_URL)
            time.sleep(0.1)
    send_message(f"주식 평가 금액: {evaluation[0]['scts_evlu_amt']}원", DISCORD_WEBHOOK_URL)
    time.sleep(0.1)
    send_message(f"평가 손익 합계: {evaluation[0]['evlu_pfls_smtl_amt']}원", DISCORD_WEBHOOK_URL)
    time.sleep(0.1)
    send_message(f"총 평가 금액: {evaluation[0]['tot_evlu_amt']}원", DISCORD_WEBHOOK_URL)
    time.sleep(0.1)
    send_message(f"=================", DISCORD_WEBHOOK_URL)
    return stock_dict

def get_balance(APP_KEY, APP_SECRET, URL_BASE):
    """현금 잔고조회"""
    ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
    PATH = "uapi/domestic-stock/v1/trading/inquire-psbl-order"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "TTTC8908R",
        "custtype": "P"
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": "005930",
        "ORD_UNPR": "65500",
        "ORD_DVSN": "01",
        "CMA_EVLU_AMT_ICLD_YN": "Y",
        "OVRS_ICLD_YN": "Y"
    }
    res = requests.get(URL, headers=headers, params=params)
    cash = res.json()['output']['ord_psbl_cash']
    send_message(f"주문 가능 현금 잔고: {cash}원", DISCORD_WEBHOOK_URL)
    return int(cash)

def buy(code, qty, APP_KEY, APP_SECRET, URL_BASE):
    """주식 시장가 매수"""
    ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
    PATH = "uapi/domestic-stock/v1/trading/order-cash"
    URL = f"{URL_BASE}/{PATH}"
    data = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": code,
        "ORD_DVSN": "01",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": "0",
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "TTTC0802U",
        "custtype": "P",
        "hashkey": hashkey(data, APP_KEY, APP_SECRET, URL_BASE)
    }
    res = requests.post(URL, headers=headers, data=json.dumps(data))
    if res.json()['rt_cd'] == '0':
        send_message(f"[매수 성공]{str(res.json())}", DISCORD_WEBHOOK_URL)
        return True
    else:
        send_message(f"[매수 실패]{str(res.json())}", DISCORD_WEBHOOK_URL)
        return False

def sell(code, qty, APP_KEY, APP_SECRET, URL_BASE):
    """주식 시장가 매도"""
    ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
    PATH = "uapi/domestic-stock/v1/trading/order-cash"
    URL = f"{URL_BASE}/{PATH}"
    data = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": code,
        "ORD_DVSN": "01",
        "ORD_QTY": qty,
        "ORD_UNPR": "0",
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "TTTC0801U",
        "custtype": "P",
        "hashkey": hashkey(data, APP_KEY, APP_SECRET, URL_BASE)
    }
    res = requests.post(URL, headers=headers, data=json.dumps(data))
    if res.json()['rt_cd'] == '0':
        send_message(f"[매도 성공]{str(res.json())}", DISCORD_WEBHOOK_URL)
        return True
    else:
        send_message(f"[매도 실패]{str(res.json())}", DISCORD_WEBHOOK_URL)
        return False

if 'stop' not in st.session_state:
    st.session_state.stop = False

def stop_button_callback():
    st.session_state.stop = True

# Streamlit UI
st.title('자동 주식 매매 프로그램')

# 사용자 정보 입력
APP_KEY = st.text_input('APP_KEY', value='')
APP_SECRET = st.text_input('APP_SECRET', value='', type='password')
CANO = st.text_input('CANO', value='')
ACNT_PRDT_CD = st.text_input('ACNT_PRDT_CD', value='')
DISCORD_WEBHOOK_URL = st.text_input('DISCORD_WEBHOOK_URL', value='')
URL_BASE = st.text_input('URL_BASE', value='https://openapi.koreainvestment.com:9443')

# 종목 코드 입력
stock_code = st.text_input('종목 코드', value='005930')

# 현재 주가 조회 버튼
if st.button('현재 주가 조회'):
    try:
        ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
        current_price, acml_vol = get_current_price_and_volume(stock_code, APP_KEY, APP_SECRET, URL_BASE)
        st.write(f'{stock_code}의 현재 주가는 {current_price}원입니다.')
    except Exception as e:
        st.error(f'주가 정보를 가져오는 데 오류가 발생했습니다: {e}')

# 현금 잔고 조회 버튼
if st.button('현금 잔고 조회'):
    try:
        ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
        cash_balance = get_balance(APP_KEY, APP_SECRET, URL_BASE)
        st.write(f'현재 현금 잔고는 {cash_balance}원입니다.')
    except Exception as e:
        st.error(f'현금 잔고를 가져오는 데 오류가 발생했습니다: {e}')

# 특정 주식 종목 데이터 조회 버튼
if st.button('종목 데이터 조회'):
    try:
        cursor.execute('SELECT * FROM price_info WHERE stock_code = ?', (stock_code,))
        rows = cursor.fetchall()
        if rows:
            st.write(f'{stock_code}의 데이터:')
            st.write(pd.DataFrame(rows, columns=['time_key', 'stock_code', 'high', 'low', 'open','close','volume']))
        else:
            st.write(f'{stock_code}에 대한 데이터가 없습니다.')
    except Exception as e:
        st.error(f'종목 데이터를 가져오는 데 오류가 발생했습니다: {e}')

if st.button('최근 5시간 데이터 미리 가져오기'):
    try:
        fetch_recent_5_hours_data(stock_code)
        st.write('최근 5시간의 데이터가 성공적으로 DB에 저장되었습니다.')
    except Exception as e:
        st.error(f'최근 5시간의 데이터를 가져오는 데 오류가 발생했습니다: {e}')

# DB 초기화 버튼
if st.button('DB 초기화'):
    try:
        reset_database()
        st.write('데이터베이스가 초기화되었습니다.')
    except Exception as e:
        st.error(f'데이터베이스 초기화 중 오류가 발생했습니다: {e}')


if st.button('자동매매 시작'):
    try:
        ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
        total_cash = get_balance(APP_KEY, APP_SECRET, URL_BASE)
        bought = False
        buy_price = 0
        sell_price = 0
        total_profit = 0

        st.write('===국내 주식 자동매매 프로그램을 시작합니다===')
        send_message('===국내 주식 자동매매 프로그램을 시작합니다===', DISCORD_WEBHOOK_URL)
        
        profit_display = st.sidebar.empty()
        stop_button_placeholder = st.empty()
        stop_button_placeholder.button('종료', key='stop_button', on_click=stop_button_callback)

        while True:
            if st.session_state.stop:
                break

            loop_start_time = datetime.datetime.now()

            t_now = datetime.datetime.now()
            t_start = t_now.replace(hour=9, minute=0, second=0, microsecond=0)
            t_sell = t_now.replace(hour=15, minute=00, second=0, microsecond=0)
            t_end = t_now.replace(hour=15, minute=20, second=0, microsecond=0)
            today = datetime.datetime.today().weekday()

            if today in [5]:  # 토요일이나 일요일이면 자동 종료
                send_message("토요일이므로 프로그램을 종료합니다.", DISCORD_WEBHOOK_URL)
                break

            if t_now >= t_end:
                send_message(f"현재 시각: {t_now} \n 오후 3시가 지났으므로 프로그램을 종료합니다.", DISCORD_WEBHOOK_URL)
            if t_start <= t_now <= t_sell:
                current_price, current_volume = get_current_price_and_volume(stock_code, APP_KEY, APP_SECRET, URL_BASE)
                update_price_info(current_price, current_volume, t_now, stock_code)

            current_hour_key = t_now.strftime('%Y-%m-%d %H')  # current_hour_key 할당

            # 매수
            if t_start < t_now < t_sell and not bought:
                target_price = get_target_price_change(stock_code)
                model_prediction = get_model_prediction(stock_code, current_hour_key)

                if target_price and target_price < current_price and current_price < model_prediction[0][0]:
                    buy_qty = int(total_cash // current_price)
                    if buy_qty > 0:
                        result = buy(stock_code, buy_qty, APP_KEY, APP_SECRET, URL_BASE)
                        if result:
                            bought = True
                            buy_price = current_price
                            send_message(f"{stock_code} 매수 완료", DISCORD_WEBHOOK_URL)
                            st.write(f"{stock_code} 매수 완료")

            sell_price = get_target_price_change(stock_code)

            # 매도
            if bought and (target_price <= sell_price or current_price > model_prediction[0][0]):
                stock_dict = get_stock_balance(APP_KEY, APP_SECRET, URL_BASE)
                qty = stock_dict.get(stock_code, 0)
                if qty:
                    qty = int(qty)
                if qty > 0:
                    result = sell(stock_code, qty, APP_KEY, APP_SECRET, URL_BASE)
                    if result:
                        bought = False
                        sell_price = current_price
                        profit = ((sell_price - buy_price) / buy_price) * 100 - 0.2
                        total_profit += profit
                        send_message(f"{stock_code} 매도 완료", DISCORD_WEBHOOK_URL)
                        st.write(f"{stock_code} 매도 완료")

            if t_now >= t_sell and bought:
                stock_dict = get_stock_balance(APP_KEY, APP_SECRET, URL_BASE)
                qty = stock_dict.get(stock_code, 0)
                if qty > 0:
                    sell(stock_code, qty, APP_KEY, APP_SECRET, URL_BASE)
                    bought = False
                    sell_price = current_price
                    profit = ((sell_price - buy_price) / buy_price) * 100 - 0.2
                    total_profit += profit
                    send_message(f"장 마감 강제 매도: {stock_code}", DISCORD_WEBHOOK_URL)
                    st.write(f"장 마감 강제 매도: {stock_code}")

            # 수익률 표시
            profit_display.write(f"오늘의 수익률: {total_profit:.2f}%")

            loop_end_time = datetime.datetime.now()
            elapsed_time = (loop_end_time - loop_start_time).total_seconds()
            sleep_time = max(5 - elapsed_time, 0)

            time.sleep(sleep_time)

    except Exception as e:
        send_message(f"[오류 발생]{e}", DISCORD_WEBHOOK_URL)
        st.error(f"오류 발생: {e}")

    finally:
        send_message("프로그램이 종료되었습니다.", DISCORD_WEBHOOK_URL)
        st.write("프로그램이 종료되었습니다.")
