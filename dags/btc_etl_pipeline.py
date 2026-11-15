import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

BASE = "/opt/airflow"
RAW = f"{BASE}/data/raw/btc_raw.csv"
PROCESSED = f"{BASE}/data/processed/btc_features.csv"
STAGING = f"{BASE}/models/staging"
LIVE = f"{BASE}/models"
MAPE_THRESHOLD = 15.0
API_URL = os.getenv("API_URL", "http://api:5000")


def run_script(name, extra_env=None):
    env = {**os.environ, "RAW_PATH": RAW, **(extra_env or {})}
    subprocess.run([sys.executable, f"{BASE}/src/{name}"], env=env, check=True)


def fetch_data():
    run_script("etl_fetch.py")


def preprocess():
    df = pd.read_csv(RAW, index_col=0).dropna()
    df = df[~df.index.duplicated()]
    df["ma7"] = df["Close"].rolling(7).mean()
    df["ma30"] = df["Close"].rolling(30).mean()
    df["volatility7"] = df["Close"].pct_change().rolling(7).std()
    df["target_close_7d"] = df["Close"].shift(-7)
    df = df.dropna(subset=["ma30", "volatility7"])
    os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
    df.to_csv(PROCESSED)


def train_model():
    run_script("train_forecast.py", {"MODEL_DIR": STAGING})


def passes_threshold():
    with open(f"{STAGING}/metrics.json") as f:
        mape = json.load(f)["MAPE"]
    print(f"MAPE={mape}, threshold={MAPE_THRESHOLD}")
    return mape < MAPE_THRESHOLD


def register_model():
    for name in ("model.pkl", "metrics.json", "last_closes.json"):
        shutil.copy(f"{STAGING}/{name}", f"{LIVE}/{name}")


def reload_api():
    r = requests.post(f"{API_URL}/reload", timeout=30)
    r.raise_for_status()
    print(r.json())


with DAG(
    dag_id="btc_etl_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
) as dag:
    t1 = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t2 = PythonOperator(task_id="preprocess_features", python_callable=preprocess)
    t3 = PythonOperator(task_id="train_and_evaluate", python_callable=train_model)
    t4 = ShortCircuitOperator(
        task_id="check_threshold", python_callable=passes_threshold
    )
    t5 = PythonOperator(task_id="register_model", python_callable=register_model)
    t6 = PythonOperator(task_id="reload_api", python_callable=reload_api)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
