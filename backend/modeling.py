from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tuna_lgbm_ensemble.joblib"
FEATURES = ["species_id", "grid_lat", "grid_lon", "month", "day", "sst", "wind_speed", "pressure", "precipitation", "activity_count"]
TARGET = "catch_index"


def train_model(df: pd.DataFrame) -> Dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models: List[lgb.LGBMRegressor] = []
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
