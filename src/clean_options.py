# clean_options.py
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# CONFIG
RAW_DIR = "data/raw"
ARTIFACTS_DIR = "data/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Risk-free rate (approx, annualized)
R = 0.05  

# Helper: compute tau in years
def compute_tau(expiry, trade_date):
    expiry_date = pd.to_datetime(expiry)
    trade_date = pd.to_datetime(trade_date)
    return max((expiry_date - trade_date).days / 365, 0)

# Cleaning Script
def clean_options():
    # Load ALL option snapshots
    option_files = glob.glob(os.path.join(RAW_DIR, "SPY_options_*.csv"))
    if not option_files:
        raise FileNotFoundError("No SPY_options_*.csv files found in data/raw/")
    
    df = pd.concat([pd.read_csv(f) for f in option_files], ignore_index=True)

    # Basic preprocessing
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)

    # Date & tau
    df["download_date"] = pd.to_datetime(df["download_date"])
    df["tau"] = df.apply(lambda row: compute_tau(row["expiry"], row["download_date"]), axis=1)

    # Moneyness: (strike / underlying)
    # For Yahoo Finance, underlying price is lastPrice of ATM option’s underlying
    # but better to pull from SPY_prices.csv
    try:
        prices = pd.read_csv("data/raw/SPY_prices.csv")
        latest_close = prices["Close"].iloc[-1]
    except Exception:
        latest_close = df["strike"].median()  # fallback estimate
    df["moneyness"] = df["strike"] / latest_close

    # Filtering bad rows
    clean_df = df.copy()
    clean_df = clean_df[(clean_df["volume"].fillna(0) > 0) | (clean_df["openInterest"].fillna(0) > 0)]
    clean_df = clean_df[clean_df["mid"] > 0]
    clean_df = clean_df[clean_df["spread"] < 0.2]  # drop wide spreads
    clean_df = clean_df[clean_df["tau"] > 0.01]   # drop contracts expiring too soon

    # Save cleaned data
    out_path = os.path.join(ARTIFACTS_DIR, "options_clean.parquet")
    clean_df.to_parquet(out_path, index=False)
    print(f"[OK] Saved cleaned options → {out_path}")
    print(f"[INFO] Remaining rows: {len(clean_df)}")

    return clean_df

# Main
if __name__ == "__main__":
    df_clean = clean_options()
    print(df_clean.head())
