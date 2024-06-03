# app.py
import streamlit as st
import datetime
import pytz
import pandas as pd
import time
import sqlite3
import plotly.graph_objects as go
import yfinance as yf
from database import reset_database, fetch_recent_data, get_current_price_and_volume, update_price_info, get_target_price_change, sell_target_price_change, get_model_prediction
from trading import get_balance, get_stock_balance, buy, sell, send_message, ensure_token_valid

conn = sqlite3.connect('stock_prices.db')
cursor = conn.cursor()

st.set_page_config(
    page_title="자동 주식 매매 프로그램",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get help': 'https://github.com/lucaseo/streamlit_app_stockprice_downloader',
        'Report a bug': 'https://github.com/lucaseo/streamlit_app_stockprice_downloader/issues',
        'About': '''
        ## Stock price downloader
        Download stock price data in US, South Korean stock market
        '''
    }
)

st.title('📈 자동 주식 매매 프로그램')

# 사용자 정보 입력
with st.expander("🔑 사용자 정보 입력"):
    col1, col2 = st.columns(2)

    with col1:
        APP_KEY = st.text_input('APP_KEY', value='', placeholder="앱 키를 입력하세요")
        CANO = st.text_input('CANO', value='', placeholder="계좌번호를 입력하세요")

    with col2:
        APP_SECRET = st.text_input('APP_SECRET', value='', type='password', placeholder="앱 시크릿을 입력하세요")
        ACNT_PRDT_CD = st.text_input('ACNT_PRDT_CD', value='', placeholder="계좌상품코드를 입력하세요")

    DISCORD_WEBHOOK_URL = st.text_input('DISCORD_WEBHOOK_URL', value='', placeholder="디스코드 웹훅 URL을 입력하세요")
    URL_BASE = st.text_input('URL_BASE', value='https://openapi.koreainvestment.com:9443', placeholder="API 기본 URL을 입력하세요")


st.write("---")

stock_code = st.text_input('종목 코드', value='005930', placeholder="종목 코드를 입력하세요")

st.write("---")

# 현재 주가 조회 버튼
if st.button('현재 주가 조회'):
    try:
        ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
        current_price, acml_vol = get_current_price_and_volume(stock_code, APP_KEY, APP_SECRET, URL_BASE, conn, cursor)
        st.write(f'{stock_code}의 현재 주가는 {current_price}원입니다.')
    except Exception as e:
        st.error(f'주가 정보를 가져오는 데 오류가 발생했습니다: {e}')

# 현금 잔고 조회 버튼
if st.button('현금 잔고 조회'):
    try:
        ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
        cash_balance = get_balance(APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
        st.write(f'현재 현금 잔고는 {cash_balance}원입니다.')
    except Exception as e:
        st.error(f'현금 잔고를 가져오는 데 오류가 발생했습니다: {e}')

# 추가: 시작 날짜, 종료 날짜 및 차트 유형 선택
start_date = st.sidebar.date_input("시작 날짜: ", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("종료 날짜: ", value=pd.to_datetime("2023-07-28"))
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
chart_type = st.sidebar.radio("Select Chart Type", ("Candle Stick", "Line"))

# 현재 주가 조회 버튼
if st.button('차트 조회', key='candle_chart'):
    try:
        # 주식 데이터 가져오기 및 차트 표시
        stock = yf.Ticker(stock_code+'.KS')
        data = stock.history(start=start_date, end=end_date, interval=interval)
        if data.empty:
            stock = yf.Ticker(stock_code+'.KQ')
            data = stock.history(start=start_date, end=end_date, interval=interval)
        st.dataframe(data)

        if chart_type == "Candle Stick":
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        elif chart_type == "Line":
            fig = go.Figure(data=[go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close')])
        else:
            st.error("차트 유형 선택에 오류가 발생했습니다.")

        fig.update_layout(title=f"{stock_code} Stock {chart_type} Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f'주가 정보를 가져오는 데 오류가 발생했습니다: {e}')

# 종목 데이터 조회 버튼
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

# DB 초기화 버튼
if st.button('DB 초기화'):
    try:
        reset_database(conn, cursor)
        st.write('데이터베이스가 초기화되었습니다.')
    except Exception as e:
        st.error(f'데이터베이스 초기화 중 오류가 발생했습니다: {e}')

if 'stop' not in st.session_state:
    st.session_state.stop = False

st.write("---")

def stop_button_callback():
    st.session_state.stop = True

# 자동 매매 시작 버튼
if st.button('🚀 자동매매 시작'):
    bought = False
    try:
        ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
        total_cash = get_balance(APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
        buy_price = 0
        sell_price = 0
        total_profit = 0
        st.session_state.stop = False

        st.write('===국내 주식 자동매매 프로그램을 시작합니다===')
        send_message('===국내 주식 자동매매 프로그램을 시작합니다===', DISCORD_WEBHOOK_URL)
        
        profit_display = st.sidebar.empty()
        stop_button_placeholder = st.empty()
        stop_button_placeholder.button('⏹️ 종료', key='stop_button', on_click=stop_button_callback)
        
        while True:
            if st.session_state.stop:
                send_message(f"현재 시각: {datetime.datetime.now(pytz.timezone('Asia/Seoul'))} \n 프로그램을 종료합니다.", DISCORD_WEBHOOK_URL)
                break

            loop_start_time = datetime.datetime.now(pytz.timezone('Asia/Seoul'))

            t_now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
            t_start = t_now.replace(hour=9, minute=0, second=0, microsecond=0)
            t_sell = t_now.replace(hour=15, minute=00, second=0, microsecond=0)
            t_end = t_now.replace(hour=15, minute=20, second=0, microsecond=0)
            today = t_now.weekday()

            if today in [5]:  # 토요일이면 자동 종료
                send_message("토요일이므로 프로그램을 종료합니다.", DISCORD_WEBHOOK_URL)
                break

            if (t_now >= t_end + datetime.timedelta(minutes=30)) or (t_now<=t_start-datetime.timedelta(hours=1)):
                send_message(f"현재 시각: {t_now} \n 장이 마감되었으므로 프로그램을 종료합니다.", DISCORD_WEBHOOK_URL)
                break

            if t_start <= t_now <= t_sell:
                current_price, current_volume = get_current_price_and_volume(stock_code, APP_KEY, APP_SECRET, URL_BASE, conn, cursor)
                update_price_info(current_price, current_volume, t_now, stock_code, conn, cursor)

            current_hour_key = t_now.strftime('%Y-%m-%d %H')  # current_hour_key 할당

            # 매수
            if t_start < t_now < t_sell and not bought:
                target_price = get_target_price_change(stock_code, conn, cursor)
                model_prediction = get_model_prediction(stock_code, current_hour_key, conn, cursor)

                if target_price and target_price < current_price and current_price < int(model_prediction[0][0]):
                    send_message("매수 신호 발생", DISCORD_WEBHOOK_URL)
                    st.write(f"모델 예측 가격: {model_prediction[0][0]}")
                    buy_qty = int(total_cash*0.9 // int(current_price))
                    if buy_qty > 0:
                        result = buy(stock_code, buy_qty, APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
                        if result:
                            bought = True
                            buy_price = int(current_price)
                            send_message(f"{stock_code} 종목 {buy_price}에 {buy_qty}만큼 매수 완료", DISCORD_WEBHOOK_URL)
                            st.write(f"{stock_code} 종목 {buy_price}에 {buy_qty}만큼 매수 완료")

            sell_price = sell_target_price_change(stock_code, conn, cursor)

            # 매도
            if bought and (target_price <= sell_price or current_price > int(model_prediction[0][0])):
                stock_dict = get_stock_balance(APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
                qty = stock_dict.get(stock_code, 0)
                send_message("매도 신호 발생", DISCORD_WEBHOOK_URL)
                if qty:
                    qty = int(qty)
                if qty > 0:
                    result = sell(stock_code, qty, APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
                    if result:
                        bought = False
                        sell_price = int(current_price)
                        profit = ((sell_price - buy_price) / buy_price) * 100 - 0.2
                        total_profit += profit
                        send_message(f"{stock_code} 종목 {sell_price}에 {qty}만큼 매도 완료", DISCORD_WEBHOOK_URL)
                        st.write(f"{stock_code} 종목 {sell_price}에 {qty}만큼 매도 완료")
                        profit_display.write(f"매도 후 수익률: {total_profit:.2f}%")

            if t_now >= t_sell and bought:
                stock_dict = get_stock_balance(APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
                qty = stock_dict.get(stock_code, 0)
                if qty > 0:
                    sell(stock_code, qty, APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
                    bought = False
                    sell_price = current_price
                    profit = ((sell_price - buy_price) / buy_price) * 100 - 0.2
                    total_profit += profit
                    send_message(f"장 마감 강제 매도: {stock_code}", DISCORD_WEBHOOK_URL)
                    st.write(f"장 마감 강제 매도: {stock_code}")
                    profit_display.write(f"매도 후 수익률: {total_profit:.2f}%")

            # 수익률 표시
            profit_display.write(f"오늘의 수익률: {total_profit:.2f}%")

            loop_end_time = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
            elapsed_time = (loop_end_time - loop_start_time).total_seconds()
            sleep_time = max(5 - elapsed_time, 0)

            time.sleep(sleep_time)

    except Exception as e:
            send_message(f"[오류 발생]{e}", DISCORD_WEBHOOK_URL)
            st.error(f"오류 발생: {e}")

    finally:
        if bought:
                stock_dict = get_stock_balance(APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
                qty = stock_dict.get(stock_code, 0)
                if qty > 0:
                    sell(stock_code, qty, APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL)
                    bought = False
                    sell_price = current_price
                    profit = ((sell_price - buy_price) / buy_price) * 100 - 0.2
                    total_profit += profit
                    send_message(f"강제 매도: {stock_code}", DISCORD_WEBHOOK_URL)
                    st.write(f"강제 매도: {stock_code}")
        send_message("프로그램이 종료되었습니다.", DISCORD_WEBHOOK_URL)
        send_message(f"오늘의 수익률: {total_profit:.2f}%", DISCORD_WEBHOOK_URL)
        st.write("프로그램이 종료되었습니다.")
        st.write(f"오늘의 수익률: {total_profit:.2f}%")
        st.session_state.stop = False  # Reset stop state
        stop_button_placeholder.empty()  # 종료 버튼 제거
