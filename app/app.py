import os
import json
import time
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_DIR = os.getenv("MODEL_DIR", "models")
START_TIME = time.time()

app = Flask(__name__)
state = {"model": None, "metrics": {}, "closes": []}


def make_features(closes):
    s = np.array(closes, dtype=float)
    lags = [s[-i] for i in range(1, 8)]
    return lags + [s[-7:].mean(), s[-30:].mean(), s[-7:].std()]


def load_model():
    state["model"] = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        state["metrics"] = json.load(f)
    with open(os.path.join(MODEL_DIR, "last_closes.json")) as f:
        state["closes"] = json.load(f)


def forecast(closes, days):
    history = list(closes)
    preds = []
    for _ in range(days):
        x = np.array([make_features(history)])
        p = float(state["model"].predict(x)[0])
        preds.append(p)
        history.append(p)
    return preds


try:
    load_model()
except Exception as e:
    print(f"Model not loaded: {e}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_version": state["metrics"].get("model_version", "not_loaded"),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if state["model"] is None:
            return jsonify({"error": "Model not loaded"}), 500

        body = request.get_json(silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify({"error": "Payload must be a JSON object"}), 400

        days = body.get("forecast_days", 7)
        if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 30:
            return jsonify({"error": "forecast_days must be an integer 1 to 30"}), 400

        history = body.get("history")
        if history is None:
            history = state["closes"]
        else:
            ok = isinstance(history, list) and len(history) >= 30
            if not ok or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in history
            ):
                return (
                    jsonify({"error": "history must be a list of 30+ numbers"}),
                    400,
                )

        preds = forecast(history, days)
        today = datetime.now(timezone.utc).date()
        result = [
            {
                "date": str(today + timedelta(days=i + 1)),
                "predicted_close": round(p, 2),
            }
            for i, p in enumerate(preds)
        ]
        return jsonify({"forecast_days": days, "predictions": result})
    except Exception as e:
        return jsonify({"error": f"Internal error: {e}"}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    if not state["metrics"]:
        return jsonify({"error": "Metrics not available"}), 500
    return jsonify(state["metrics"])


@app.route("/reload", methods=["POST"])
def reload_model():
    try:
        load_model()
        return jsonify(
            {"status": "reloaded", "model_version": state["metrics"]["model_version"]}
        )
    except Exception as e:
        return jsonify({"error": f"Reload failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
