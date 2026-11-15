from kfp import compiler, dsl

BASE_IMAGE = "python:3.12-slim"

INGEST_SCRIPT = """
import os, sys
import yfinance as yf
out = sys.argv[1]
os.makedirs(os.path.dirname(out), exist_ok=True)
df = yf.download("BTC-USD", start="2018-01-01", auto_adjust=True, progress=False)
df.columns = df.columns.get_level_values(0)
df[["Open", "High", "Low", "Close", "Volume"]].dropna().to_csv(out)
print("rows:", len(df))
"""

FEATURE_SCRIPT = """
import os, sys
import pandas as pd
src, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out), exist_ok=True)
df = pd.read_csv(src, index_col=0).dropna()
df["ma7"] = df["Close"].rolling(7).mean()
df["ma30"] = df["Close"].rolling(30).mean()
df["volatility7"] = df["Close"].pct_change().rolling(7).std()
df["target_close_7d"] = df["Close"].shift(-7)
df = df.dropna(subset=["ma30", "volatility7"])
df.to_csv(out)
print("feature rows:", len(df))
"""

TRAIN_SCRIPT = """
import json, os, sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
df = pd.read_csv(src, index_col=0).dropna()
cols = ["Close", "ma7", "ma30", "volatility7"]
train, test = df.iloc[:-7], df.iloc[-7:]
model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
model.fit(train[cols], train["target_close_7d"])
pred = model.predict(test[cols])
actual = test["target_close_7d"].values
mae = float(np.mean(np.abs(actual - pred)))
rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
joblib.dump(model, os.path.join(out, "model.pkl"))
with open(os.path.join(out, "metrics.json"), "w") as f:
    json.dump(metrics, f)
print(metrics)
"""


def shell_cmd(packages):
    return ["sh", "-c", f'pip install -q {packages} && python -c "$0" "$@"']


@dsl.container_component
def ingest_data(raw_data: dsl.Output[dsl.Dataset]):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=shell_cmd("yfinance pandas"),
        args=[INGEST_SCRIPT, raw_data.path],
    )


@dsl.container_component
def generate_features(
    raw_data: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]
):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=shell_cmd("pandas"),
        args=[FEATURE_SCRIPT, raw_data.path, features.path],
    )


@dsl.container_component
def train_model(features: dsl.Input[dsl.Dataset], model: dsl.Output[dsl.Model]):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=shell_cmd("pandas numpy xgboost scikit-learn joblib"),
        args=[TRAIN_SCRIPT, features.path, model.path],
    )


@dsl.pipeline(
    name="btc-forecast-pipeline",
    description="Ingest BTC-USD data, build features, train 7-day forecaster",
)
def btc_pipeline():
    ingest = ingest_data()
    feats = generate_features(raw_data=ingest.outputs["raw_data"])
    train_model(features=feats.outputs["features"])


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=btc_pipeline,
        package_path="kubeflow/btc_kfp_pipeline.yaml",
    )
    print("Compiled to kubeflow/btc_kfp_pipeline.yaml")
