import os
import yfinance as yf
import pandas as pd

RAW_PATH = os.getenv("RAW_PATH", "data/raw/btc_raw.csv")


def fetch_data():
    df = yf.download("BTC-USD", start="2018-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH)
    print(f"Saved {len(df)} rows to {RAW_PATH}")
    return RAW_PATH


if __name__ == "__main__":
    fetch_data()
