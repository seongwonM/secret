# database.py
import sqlite3
import pandas as pd
import yfinance as yf
import pytz
import datetime
import requests
from trading import ensure_token_valid, ACCESS_TOKEN

def reset_database(conn, cursor):
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

def fetch_recent_data(stock_code, conn, cursor):
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    stock = yf.Ticker(stock_code + '.KS')
    data = stock.history(start=now - datetime.timedelta(days=4), end=now, interval="1h")
    if data.empty:
        stock = yf.Ticker(stock_code + '.KQ')
        data = stock.history(start=now - datetime.timedelta(days=4), end=now, interval="1h")
        if data.empty:
            return None
    data = data.tail(7)[:-1]
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
    return data

def get_current_price_and_volume(code, APP_KEY, APP_SECRET, URL_BASE, conn, cursor):
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

def get_accumulated_volume(stock_code, current_hour_key, conn, cursor):
    current_date = current_hour_key.split(' ')[0]  # YYYY-MM-DD 부분만 추출
    cursor.execute('''
        SELECT SUM(volume) 
        FROM price_info 
        WHERE stock_code = ? AND time_key < ? AND time_key LIKE ?
    ''', (stock_code, current_hour_key, f'{current_date}%'))
    result = cursor.fetchone()
    return result[0] if result[0] is not None else 0

def update_price_info(current_price, current_volume, current_time, stock_code, conn, cursor):
    time_key = current_time.strftime('%Y-%m-%d %H')
    
    cursor.execute('SELECT * FROM price_info WHERE time_key = ? AND stock_code = ?', (time_key, stock_code))
    data = cursor.fetchone()

    # 누적 거래량 계산
    accumulated_volume_before = get_accumulated_volume(stock_code, time_key, conn, cursor)
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

def get_previous_row(stock_code, current_hour_key, conn, cursor):
    cursor.execute('''
        SELECT high, low 
        FROM price_info 
        WHERE stock_code = ? AND time_key < ? 
        ORDER BY time_key DESC 
        LIMIT 1
    ''', (stock_code, current_hour_key))
    return cursor.fetchone()

def get_target_price_change(stock_code, conn, cursor):
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    current_hour_key = now.strftime('%Y-%m-%d %H')

    if now.hour == 9:
        previous_day = now - datetime.timedelta(days=1)
        previous_hour_key = previous_day.replace(hour=14).strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key, conn, cursor)  # 수정된 부분
    else:
        previous_hour = now - datetime.timedelta(hours=1)
        previous_hour_key = previous_hour.strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key, conn, cursor)  # 수정된 부분

    cursor.execute('SELECT open FROM price_info WHERE time_key = ? AND stock_code = ?', (current_hour_key, stock_code))
    stck_oprc = cursor.fetchone()

    if stck_oprc and prev_data:
        stck_oprc = stck_oprc[0]
        stck_hgpr, stck_lwpr = prev_data
        target_price = stck_oprc + (stck_hgpr - stck_lwpr) * 0.5
        return target_price
    else:
        return None
    
def sell_target_price_change(stock_code, conn, cursor):
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    current_hour_key = now.strftime('%Y-%m-%d %H')

    if now.hour == 9:
        previous_day = now - datetime.timedelta(days=1)
        previous_hour_key = previous_day.replace(hour=14).strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key, conn, cursor)  # 수정된 부분
    else:
        previous_hour = now - datetime.timedelta(hours=1)
        previous_hour_key = previous_hour.strftime('%Y-%m-%d %H')
        cursor.execute('SELECT high, low FROM price_info WHERE time_key = ? AND stock_code = ?', (previous_hour_key, stock_code))
        prev_data = cursor.fetchone()
        if not prev_data:
            prev_data = get_previous_row(stock_code, current_hour_key, conn, cursor)  # 수정된 부분

    cursor.execute('SELECT open FROM price_info WHERE time_key = ? AND stock_code = ?', (current_hour_key, stock_code))
    stck_oprc = cursor.fetchone()

    if stck_oprc and prev_data:
        stck_oprc = stck_oprc[0]
        stck_hgpr, stck_lwpr = prev_data
        target_price = stck_oprc - (stck_hgpr - stck_lwpr) * 0.5
        return target_price
    else:
        return None
    
def get_target_price_ma(stock_code, conn, cursor):
    cursor.execute('SELECT time_key, open FROM price_info WHERE stock_code = ?', (stock_code,))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['time_key', 'open'])
    df['time_key'] = pd.to_datetime(df['time_key'])
    df.set_index('time_key', inplace=True)
    df['SMA2'] = df['open'].rolling(window=2, min_periods=1).mean()
    df['SMA4'] = df['open'].rolling(window=4, min_periods=1).mean()
    return df

def get_model_prediction(stock_code, current_hour_key, conn, cursor):
    cursor.execute('SELECT * FROM price_info WHERE stock_code = ? AND time_key < ? ORDER BY time_key DESC LIMIT ?', (stock_code, current_hour_key, 6))
    rows = cursor.fetchall()
    if len(rows) < 6:
        return None
    df = pd.DataFrame(rows, columns=['Datetime', 'stock_code', 'High', 'Low', 'Open', 'Close', 'Volume'])
    stock = Stock(df)
    stock.preprocessing()
    stock.add_change(['High', 'Low', 'Open', 'Close', 'Volume'])
    stock.df.loc[stock.df['Volume_chg'] == np.inf, 'Volume_chg'] = 0
    stock.seq_len = 5
    stock.scale_col(['Close_chg', 'High_chg', 'Low_chg', 'Open_chg', 'Volume_chg'])
    train_loader = stock.data_loader(stock.seq_len, 'train')
    valid_loader = stock.data_loader(stock.seq_len, 'valid')
    test_loader = stock.data_loader(stock.seq_len, 't')
    stock.create_model(1, 0.2)
    stock.model.load_state_dict(torch.load('chg_close_loss.pth'))
    loss = stock.train(train_loader, valid_loader, test_loader, 10, 0.1, 20, 't')
    predicted = stock.pred_value('t')
    return predicted