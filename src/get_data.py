# get_data.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# CONFIG
TICKER = "SPY"
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Download underlying prices
def fetch_prices(ticker=TICKER, period="1y", interval="1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    out_path = os.path.join(RAW_DIR, f"{ticker}_prices.csv")
    hist.to_csv(out_path, index=False)
    print(f"[OK] Saved {ticker} prices → {out_path}")
    return hist

# Download option chains
def fetch_options(ticker=TICKER):
    stock = yf.Ticker(ticker)
    expiries = stock.options  # list of expiry dates (YYYY-MM-DD)

    all_options = []
    for expiry in expiries[:5]:  # only first 5 expiries for demo
        calls = stock.option_chain(expiry).calls
        puts = stock.option_chain(expiry).puts
        calls["type"] = "C"
        puts["type"] = "P"
        df = pd.concat([calls, puts], ignore_index=True)
        df["expiry"] = expiry
        df["download_date"] = datetime.today().strftime("%Y-%m-%d")
        all_options.append(df)

    options_df = pd.concat(all_options, ignore_index=True)
    out_path = os.path.join(RAW_DIR, f"{ticker}_options.csv")
    options_df.to_csv(out_path, index=False)
    print(f"[OK] Saved {ticker} options → {out_path}")
    return options_df

# MAIN
if __name__ == "__main__":
    prices = fetch_prices()
    options = fetch_options()
    print(prices.head())
    print(options.head())
