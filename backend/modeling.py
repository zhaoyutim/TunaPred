from __future__ import annotations

import json
import importlib
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict) -> None:
    try:
        log_path = Path("/Users/zhaoyu/code_repos/TunaPred/.cursor/debug.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": f"log_{int(time.time() * 1000)}_{hypothesis_id}",
            "timestamp": int(time.time() * 1000),
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
        }
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


# region agent log
_debug_log(
    "H2",
    "backend/modeling.py:32",
    "modeling module import start",
    {"python": platform.python_version(), "platform": platform.platform()},
)
# endregion
def _get_lightgbm():
    try:
        module = importlib.import_module("lightgbm")
        # region agent log
        _debug_log("H1", "backend/modeling.py:50", "deferred lightgbm import succeeded", {})
        # endregion
        return module
    except Exception as exc:
        # region agent log
        _debug_log(
            "H1",
            "backend/modeling.py:56",
            "deferred lightgbm import failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH", ""),
                "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", ""),
            },
        )
        # endregion
        raise

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tuna_lgbm_ensemble.joblib"
FEATURES = ["species_id", "grid_lat", "grid_lon", "month", "day", "sst", "wind_speed", "pressure", "precipitation", "activity_count", "cmems_thetao", "cmems_so", "cmems_uo", "cmems_vo", "cmems_zos", "cmems_current_speed"]
TARGET = "catch_index"


def train_model(df: pd.DataFrame) -> Dict:
    lgb = _get_lightgbm()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models: List[Any] = []
    preds = []
    for seed in [11, 22, 33, 44, 55]:
        try:
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                device_type="gpu",
            )
            model.fit(X_train, y_train)
        except Exception:
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                device_type="cpu",
            )
            model.fit(X_train, y_train)
        models.append(model)
        preds.append(model.predict(X_test))

    pred_stack = np.vstack(preds)
    y_pred = pred_stack.mean(axis=0)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    artifact = {
        "models": models,
        "features": FEATURES,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)
    return metrics


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet")
    return joblib.load(MODEL_PATH)


def predict_with_confidence(df: pd.DataFrame, artifact: Dict) -> Tuple[np.ndarray, np.ndarray]:
    X = df[artifact["features"]]
    all_preds = np.vstack([model.predict(X) for model in artifact["models"]])
    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)
    confidence = 1.0 / (1.0 + std_pred)
    return mean_pred, confidence
