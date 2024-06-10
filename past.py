# past.py
import streamlit as st
import datetime
import pytz
import pandas as pd
import time
import sqlite3
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import torch
from stock2 import Stock
from database import ACCESS_TOKEN, reset_database, fetch_recent_data, get_current_price_and_volume, update_price_info, get_target_price_change, sell_target_price_change, get_model_prediction
from trading import get_balance, get_stock_balance, buy, sell, send_message, ensure_token_valid

conn = sqlite3.connect('stock_prices.db')
cursor = conn.cursor()

st.set_page_config(
    page_title="자동 주식 매매 프로그램",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title('📈 자동 주식 매매 프로그램')

# 사용자 정보 입력
with st.expander("🔑 사용자 정보 입력"):
    col1, col2 = st.columns(2)

    with col1:
        APP_KEY = st.text_input('APP_KEY', value='', placeholder="한국투자증권 API의 앱 키를 입력하세요", help='한국투자증권에서 자동매매를 하기위해 발급해주는 키입니다.')
        CANO = st.text_input('계좌번호', value='', placeholder="계좌번호를 입력하세요")

    with col2:
        APP_SECRET = st.text_input('APP_SECRET', value='', type='password', placeholder="한국투자증권 API의 앱 시크릿을 입력하세요", help='한국투자증권에서 자동매매를 하기위해 발급해주는 시크릿 번호입니다.')
        ACNT_PRDT_CD = st.text_input('계좌 구분 번호', value='', placeholder="계좌 구분 번호를 입력하세요")

    DISCORD_WEBHOOK_URL = st.text_input('디스코드 URL', value='', placeholder="디스코드 웹훅 URL을 입력하세요", help='디스코드 알림을 받기 위한 웹훅 URL입니다.')
    URL_BASE = st.text_input('API 기본 URL', value='https://openapi.koreainvestment.com:9443', placeholder="한국 투자 증권 API 기본 URL을 입력하세요", help=f'입력된 URL은 기본적인 실제 매매 URL입니다')
    tendency = st.selectbox("주식 투자를 해보신 경험이 있으십니까?", ["예", "아니요"], help='예를 선택하시면 공격적 투자를, 아니요의 경우 보수적 투자를 진행합니다.')


st.write("---")

st.markdown(
    f"""
    <div style="font-size:20px; font-weight:bold;">
        투자 종목 코드
    </div>
    <div style="height: 10px;"></div>
    """,
    unsafe_allow_html=True
)

stock_code = st.text_input('', value='', placeholder="종목 코드를 입력하세요", help='005390과 같이 6자리 숫자로 입력하세요.')

if stock_code:
    try:
        fetch_recent_data(stock_code, conn, cursor)
        st.write('최근 데이터가 성공적으로 DB에 저장되었습니다.')
    except Exception as e:
        st.error(f'최근 데이터를 가져오는 데 오류가 발생했습니다: {e}')

st.write("---")

st.markdown(
    f"""
    <div style="font-size:20px; font-weight:bold;">
        부가기능
    </div>
    """,
    unsafe_allow_html=True
)

# CSS 스타일 정의
button_style = """
    <style>
    div.row-widget.stButton > button {
        background-color: #434654 !important;
        width: 700px;
        height: 30px;
        font-size: 16px;
    }
    </style>

    <style>
    div.row-widget.stButton > div > div > div > div > button {
        background-color: #434654 !important;
        width: 340px;
        height: 30px;
        font-size: 16px;
    }
    </style>

"""
st.markdown(button_style, unsafe_allow_html=True)

# 추가: 시작 날짜, 종료 날짜 및 차트 유형 선택
st.sidebar.write('차트 조회 기간 선택')
start_date = st.sidebar.date_input("시작 날짜: ", value=pd.to_datetime("2024-06-01"))
end_date = st.sidebar.date_input("종료 날짜: ", value=pd.to_datetime("2024-06-07"))
interval = st.sidebar.selectbox("간격을 선택하세요.", ["1m", "5m", "15m", "30m", "1h", "1d"])
chart_type = st.sidebar.radio("차트 타입을 선택하세요.", ("봉 차트", "선 차트"))

# 2개의 열 생성
col1, col2 = st.columns(2)

# 현재 주가 조회 버튼
with col1:
    button_current_price = st.button('현재 주가 조회', key='current_price', help='현재 주가를 조회할 수 있습니다.')
    if button_current_price:
        try:
            current_price, acml_vol = get_current_price_and_volume(stock_code, "APP_KEY", "APP_SECRET", "URL_BASE")
            st.write(f'{stock_code}의 현재 주가는 {current_price}원입니다.')
        except Exception as e:
            st.error(f'주가 정보를 가져오는 데 오류가 발생했습니다: {e}')
    
    button_chart_view = st.button('차트 조회', key='chart_view', help='주식 차트를 조회할 수 있습니다.')
    if button_chart_view:
        try:
            stock = yf.Ticker(stock_code + '.KS')
            data = stock.history(start=start_date, end=end_date, interval=interval)
            if data.empty:
                stock = yf.Ticker(stock_code + '.KQ')
                data = stock.history(start=start_date, end=end_date, interval=interval)
            st.dataframe(data)

            if chart_type == "봉 차트":
                fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
            elif chart_type == "선 차트":
                fig = go.Figure(data=[go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close')])
            else:
                st.error("차트 유형 선택에 오류가 발생했습니다.")

            fig.update_layout(title=f"{stock_code} {chart_type} 차트", xaxis_title="날짜", yaxis_title="가격")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f'주가 정보를 가져오는 데 오류가 발생했습니다: {e}')

# 현금 잔고 조회 버튼
with col2:
    button_cash_balance = st.button('현금 잔고 조회', key='cash_balance', help='현재 계좌 잔고를 조회할 수 있습니다.')
    if button_cash_balance:
        try:
            ensure_token_valid("APP_KEY", "APP_SECRET", "URL_BASE")
            cash_balance = get_balance("APP_KEY", "APP_SECRET", "URL_BASE", "CANO", "ACNT_PRDT_CD", "DISCORD_WEBHOOK_URL")
            st.write(f'현재 현금 잔고는 {cash_balance}원입니다.')
        except Exception as e:
            st.error(f'현금 잔고를 가져오는 데 오류가 발생했습니다: {e}')

    button_stock_data = st.button('종목 데이터 조회', key='stock_data', help='DB에 저장된 데이터를 조회할 수 있습니다.')
    if button_stock_data:
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



# # DB 초기화 버튼
# if st.button('DB 초기화'):
#     try:
#         reset_database(conn, cursor)
#         st.write('데이터베이스가 초기화되었습니다.')
#     except Exception as e:
#         st.error(f'데이터베이스 초기화 중 오류가 발생했습니다: {e}')


st.write("---")

def stop_button_callback():
    st.session_state.stop = True

st.markdown(
    f"""
    <div style="font-size:20px; font-weight:bold;">
        자동매매
    </div>
    <div style="height: 10px;"></div>
    """,
    unsafe_allow_html=True
)

total_cash=st.text_input('보유 금액', value='', placeholder="보유 금액을 입력하세요.")

cash_ratio = st.number_input('예수금 비율 (%)', min_value=0, max_value=100, value=100, help='투자할 금액의 비율을 설정해주세요.')

# 자동 매매 시작 버튼
if st.button('🚀 자동매매 시작'):
    # 주식 데이터 가져오기 및 차트 표시

    st.write('===국내 주식 자동매매 프로그램을 시작합니다===')
    send_message('===국내 주식 자동매매 프로그램을 시작합니다===', DISCORD_WEBHOOK_URL)
    
    profit_display = st.sidebar.empty()
    stop_button_placeholder = st.empty()
    stop_button_placeholder.button('⏹️ 종료', key='stop_button', on_click=stop_button_callback)

    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    tick = yf.Ticker(stock_code+'.KS')
    df_get = tick.history(start=now - datetime.timedelta(days=7), end=now, interval='1m')
    df_pred = tick.history(start=now - datetime.timedelta(days=10), end=now, interval='1h')
    if df_get.empty:
        tick = yf.Ticker(stock_code+'.KQ')
        df_get = tick.history(start=now - datetime.timedelta(days=7), end=now, interval='1m')
        df_pred = tick.history(start=now - datetime.timedelta(days=10), end=now, interval='1h')
    stock=Stock(df_pred)
    df_pred=stock.preprocessing()
    stock.add_change(['High', 'Low', 'Open', 'Close', 'Volume'])
    stock.df.loc[stock.df['Volume_chg']==np.inf,'Volume_chg']=0
    stock.scale_col(['Close_chg', 'High_chg', 'Low_chg', 'Open_chg', 'Volume_chg']) # 종가(변화율)
    train_loader=stock.data_loader(5, 't')
    valid_loader=stock.data_loader(5, 't')
    test_loader=stock.data_loader(5, 't')
    stock.create_model()
    stock.model.load_state_dict(torch.load('chg_close_loss.pth'))
    stock.train(train_loader, valid_loader, test_loader, 7, 0.001, 80, 'test')
    pred=stock.pred_value('t')
    # stock.diff()
    # stock.show('chg')

    # 데이터셋 예측값 합치기
    stock.df=df_get.copy()
    df_get=stock.preprocessing()
    df_pred['pred']=0
    df_pred.iloc[len(df_pred)-len(pred):,-1]=pred
    df_get['key']=pd.to_datetime(df_get.index).strftime('%d-%H')
    df_pred['key']=pd.to_datetime(df_pred.index).strftime('%d-%H')
    df_get.loc[:,'pred']=pd.merge(df_get[['key']], df_pred[['key', 'pred']], how='left', on='Datetime')['pred']
    df_get.fillna(method='ffill', inplace=True)

    # short=60
    # long=2

    # # 이평선
    # df_get['4H_MA'] = df_get['Close'].rolling(window=short).mean()
    # df_get['8H_MA'] = df_get['Close'].rolling(window=short*long).mean()
    # # 이전 행의 4H_MA와 8H_MA 비교를 위해 shift() 사용
    # df_get['Previous_4H_MA'] = df_get['4H_MA'].shift(1)
    # df_get['Previous_8H_MA'] = df_get['8H_MA'].shift(1)

    # 매수 조건: 이전 4H_MA <= 이전 8H_MA 이고 현재 4H_MA > 현재 8H_MA
    # df_get['Buy_Signal'] = (df_get['Previous_4H_MA'] <= df_get['Previous_8H_MA']) & (df_get['4H_MA'] > df_get['8H_MA'])

    # 매도 조건: 각 시간대의 마지막 분(59분)에 매도
    # df_get['Sell_Signal'] = pd.to_datetime(df_get.index).minute == 59
    # df_get['Sell_Signal'] = (df_get['Previous_4H_MA'] >= df_get['Previous_8H_MA']) & (df_get['4H_MA'] < df_get['8H_MA'])

    # results=[]
    # moneys=[]
    # money=[]
    # first=[]
    # key=False
    # for i, row in df_get.iterrows():
    #     if row['Buy_Signal'] and row['Open']<=row['pred']:
    #         money.append(row['Open'])
    #         key=True
    #     if (row['Sell_Signal'] or row['Open']>row['pred']) and key:
    #         results.append((row['Close']-money[0])/money[0]*100)
    #         moneys.append(row['Close']-money[0])
    #         first.append(money[0])
    #         money=[]
    #         key=False
    # try:
    #     print('-'*100)
    #     print(f'이동평균선 / {name}')
    #     print(f'모델 수익률: {round(sum(moneys)/first[0]*100, 3)}%')
    #     print(f'모델 수익: {int(sum(moneys))}')
    #     print(f'수수료 포함 수익률: {round(sum(moneys)/first[0]*100-len(moneys)*0.2,3)}%')
    #     print(f'buy & hold 수익률: {round((df_get.iloc[-1,3]-df_get.iloc[0,0])/df_get.iloc[0,0]*100, 3)}%')
    #     print(f'buy & hold 수익: {int(df_get.iloc[-1,3]-df_get.iloc[0,0])}')
    #     print(f'매매횟수: {len(moneys)}')
    # except:
    #     print('매매하지 않음')


    k=0.5
    # 변동성 돌파 전략
    df_pred['Point']=(df_pred['High'].shift(1)-df_pred['Low'].shift(1))*k+df_pred['Open']
    df_pred['Point2']=-(df_pred['High'].shift(1)-df_pred['Low'].shift(1))*k+df_pred['Open']
    df_get['Point']=pd.merge(df_get[['key']], df_pred[['key', 'Point']], how='left', on='Datetime')['Point']
    df_get['Point2']=pd.merge(df_get[['key']], df_pred[['key', 'Point2']], how='left', on='Datetime')['Point2']
    df_get['Point'].fillna(method='ffill', inplace=True)
    df_get['Point2'].fillna(method='ffill', inplace=True)

    df_get['Buy_Signal'] = (df_get['Point'] <= df_get['Open'])
    df_get['Sell_Signal'] = (df_get['Point2'] > df_get['Open'])

    moneys=0
    money=0
    first=[]
    key=False
    profit=0
    total_profit=0
    total_cash= int(int(total_cash) * (cash_ratio / 100))
    first_cash=total_cash
    j=0
    now=datetime.datetime.now()
    for i, row in df_get.iterrows():
        loop_start_time=datetime.datetime.now(pytz.timezone('Asia/Seoul'))
        if row['Buy_Signal'] and row['Open']<=row['pred'] and key == False:
            send_message("매수 신호 발생", DISCORD_WEBHOOK_URL)
            money=row['Open']
            buy_qty = int(int(total_cash)*0.9 // int(money))
            key=True
            first.append(money)
            send_message(f"{i.replace(tzinfo=None)}: {stock_code} 종목 {money}에 {buy_qty}주 매수 완료", DISCORD_WEBHOOK_URL)
            st.write(f"{i.replace(tzinfo=None)}: {stock_code} 종목 {money}에 {buy_qty}주 매수 완료")
        if key and (row['Sell_Signal'] or row['Open']>row['pred']):
            send_message("매도 신호 발생", DISCORD_WEBHOOK_URL)
            moneys+=row['Close']-money
            key=False
            total_cash+=int((row['Close']-money)*buy_qty)
            total_cash=int(total_cash*0.998)
            total_profit=(total_cash-first_cash)/first_cash*100
            send_message(f"{i.replace(tzinfo=None)}: {stock_code} 종목 {row['Close']}에 {buy_qty}주 매도 완료", DISCORD_WEBHOOK_URL)
            st.write(f"{i.replace(tzinfo=None)}: {stock_code} 종목 {row['Close']}에 {buy_qty}주 매도 완료, 잔액: {total_cash}")
            profit_display.write(f"매도 후 수익률: {total_profit:.2f}%")
        j+=1
        
        loop_end_time = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
        elapsed_time = (loop_end_time - loop_start_time).total_seconds()
        p=30
        sleep_time = max(1/p - elapsed_time, 0)
        time.sleep(sleep_time)

    st.write(f"배속: 약 {60*p}배")
    st.write(f"총 소요 시간: {datetime.datetime.now()-now}")
    st.write(f"기존 소요됐어야 하는 시간: {j//60}시간 {j%60//60}분")



    if key:
        first.pop()
        
    if len(first)>0:
        send_message(f"총 보유금: {total_cash}, 총 수익: {total_cash-first_cash}, 총 수익률: {total_profit:.2f}%, 매매횟수: {len(first)}", DISCORD_WEBHOOK_URL)
        st.write(f"총 보유금: {total_cash}, 총 수익: {total_cash-first_cash}, 총 수익률: {total_profit:.2f}%, 매매횟수: {len(first)}")

    else:
        send_message(f"매매하지 않음", DISCORD_WEBHOOK_URL)
        st.write(f"매매하지 않음")
