import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import app as appmod  # noqa: E402


class DummyModel:
    def predict(self, x):
        return np.array([50000.0])


def setup_module():
    appmod.state["model"] = DummyModel()
    appmod.state["metrics"] = {"MAE": 1.0, "RMSE": 2.0, "MAPE": 3.0, "model_version": "test"}
    appmod.state["closes"] = [40000.0 + i for i in range(60)]


client = appmod.app.test_client()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "MAE" in r.get_json()


def test_predict_default_7_days():
    r = client.post("/predict", json={})
    assert r.status_code == 200
    assert len(r.get_json()["predictions"]) == 7


def test_predict_bad_days():
    r = client.post("/predict", json={"forecast_days": "abc"})
    assert r.status_code == 400


def test_predict_bad_history():
    r = client.post("/predict", json={"history": [1, 2, 3]})
    assert r.status_code == 400