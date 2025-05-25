import yfinance as yf
import pandas as pd

"""
yfinance kullanarak hisse senedi verisini indirir ve teknik göstergeleri ekler.
"""
def load_stock_data(symbol="AAPL", start_date="2020-01-01", end_date="2025-02-14"):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data