# trading.py
import requests
import json
import datetime
import pytz
import time
import pandas as pd
import torch
import numpy as np
from stock import Stock

ACCESS_TOKEN = None
token_issue_time = None
TOKEN_VALIDITY_DURATION = 3600 * 6

def send_message(msg, DISCORD_WEBHOOK_URL):
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(msg)}"}
    requests.post(DISCORD_WEBHOOK_URL, data=message)
    print(message)

def get_access_token(APP_KEY, APP_SECRET, URL_BASE):
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
        token_issue_time = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
        return ACCESS_TOKEN
    else:
        raise Exception("Failed to get access token")

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

def ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE):
    global ACCESS_TOKEN, token_issue_time
    if ACCESS_TOKEN is None or (datetime.datetime.now(pytz.timezone('Asia/Seoul')) - token_issue_time).total_seconds() >= TOKEN_VALIDITY_DURATION:
        ACCESS_TOKEN = get_access_token(APP_KEY, APP_SECRET, URL_BASE)
    return ACCESS_TOKEN

def get_balance(APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL):
    """현금 잔고조회"""
    ACCESS_TOKEN=ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
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

def get_stock_balance(APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL):
    """주식 잔고조회"""
    ACCESS_TOKEN=ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
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

def buy(code, qty, APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL):
    """주식 시장가 매수"""
    ACCESS_TOKEN=ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
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

def sell(code, qty, APP_KEY, APP_SECRET, URL_BASE, CANO, ACNT_PRDT_CD, DISCORD_WEBHOOK_URL):
    """주식 시장가 매도"""
    ACCESS_TOKEN=ensure_token_valid(APP_KEY, APP_SECRET, URL_BASE)
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

