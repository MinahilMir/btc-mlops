import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from xgboost import XGBRegressor

RAW_PATH = os.getenv("RAW_PATH", "data/raw/btc_raw.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")


def make_features(closes):
    s = np.array(closes, dtype=float)
    lags = [s[-i] for i in range(1, 8)]
    return lags + [s[-7:].mean(), s[-30:].mean(), s[-7:].std()]


def build_xy(closes):
    X, y = [], []
    for i in range(30, len(closes)):
        X.append(make_features(closes[:i]))
        y.append(closes[i])
    return np.array(X), np.array(y)


def forecast(model, closes, days=7):
    history = list(closes)
    preds = []
    for _ in range(days):
        p = float(model.predict(np.array([make_features(history)]))[0])
        preds.append(p)
        history.append(p)
    return preds


def train():
    df = pd.read_csv(RAW_PATH, index_col=0)
    closes = df["Close"].values

    # validation: hold out last 7 days
    X, y = build_xy(closes[:-7])
    val_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
    val_model.fit(X, y)
    pred = np.array(forecast(val_model, closes[:-7], 7))
    actual = closes[-7:]
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

    # final model on all data
    X, y = build_xy(closes)
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    with open(os.path.join(MODEL_DIR, "last_closes.json"), "w") as f:
        json.dump([float(x) for x in closes[-60:]], f)
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "last_data_date": str(df.index[-1]),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)
    return metrics


if __name__ == "__main__":
    train()