import yfinance as yf


def y_loader(stock_code):
    try:
        stock = yf.Ticker(str(stock_code)+'.KS')
        data = stock.history(period="6h", interval="1h")

    except:
        stock = yf.Ticker(str(stock_code)+'.KQ')
        data = stock.history(period="6h", interval="1h")

    data.to_csv(f'{stock_code}.csv', index=False)