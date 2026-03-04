from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DISPLAY_NAME = "高精度modelling"


@dataclass(frozen=True)
class StageConfig:
    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)
    test_size: float = 0.2
    random_state: int = 42


def model_name() -> str:
    return MODEL_DISPLAY_NAME


def _get_lightgbm():
    import importlib

    return importlib.import_module("lightgbm")


def build_hotspot_labels(
    df: pd.DataFrame,
    cpue_col: str = "cpue",
    species_col: str = "species_name",
    region_col: str = "region",
    month_col: str = "month",
    quantile: float = 0.7,
) -> pd.DataFrame:
    out = df.copy()
    if cpue_col not in out.columns:
        if {"catch", "effort"}.issubset(out.columns):
            out[cpue_col] = out["catch"] / out["effort"].replace(0, np.nan)
        else:
            raise ValueError("cpue column missing and cannot be derived from catch/effort")

    group_cols = [c for c in [species_col, region_col, month_col] if c in out.columns]
    if not group_cols:
        group_cols = [species_col] if species_col in out.columns else []

    if group_cols:
        thresholds = out.groupby(group_cols, dropna=False)[cpue_col].transform(lambda s: s.quantile(quantile))
    else:
        thresholds = pd.Series(out[cpue_col].quantile(quantile), index=out.index)

    out["hotspot"] = (out[cpue_col] >= thresholds).astype(int)
    return out


def _fit_binary_ensemble(df: pd.DataFrame, features: Sequence[str], label_col: str, cfg: StageConfig) -> Dict[str, Any]:
    lgb = _get_lightgbm()
    x = df[list(features)].copy()
    y = df[label_col].astype(int).copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    models: List[Any] = []
    test_probs: List[np.ndarray] = []
    for seed in cfg.seeds:
        try:
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=127,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=80,
                random_state=seed,
                device_type="gpu",
            )
            model.fit(x_train, y_train)
        except Exception:
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=127,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=80,
                random_state=seed,
                device_type="cpu",
            )
            model.fit(x_train, y_train)
        models.append(model)
        test_probs.append(model.predict_proba(x_test)[:, 1])

    mean_prob = np.vstack(test_probs).mean(axis=0)
    metrics = {
        "auc": float(roc_auc_score(y_test, mean_prob)) if y_test.nunique() > 1 else 0.5,
        "positive_rate": float(y.mean()),
    }
    return {"models": models, "features": list(features), "label_col": label_col, "metrics": metrics}


def train_coarse_model(df: pd.DataFrame, features: Sequence[str], label_col: str = "hotspot", cfg: StageConfig = StageConfig()) -> Dict[str, Any]:
    return _fit_binary_ensemble(df, features, label_col, cfg)


def train_fine_model(df: pd.DataFrame, features: Sequence[str], label_col: str = "hotspot", cfg: StageConfig = StageConfig()) -> Dict[str, Any]:
    return _fit_binary_ensemble(df, features, label_col, cfg)


def predict_hotspot_probability(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    x = df[artifact["features"]]
    probs = np.vstack([m.predict_proba(x)[:, 1] for m in artifact["models"]])
    mean_prob = probs.mean(axis=0)
    std_prob = probs.std(axis=0)

    out = df.copy()
    out["probability"] = mean_prob
    out["uncertainty"] = std_prob
    out["confidence"] = (mean_prob * (1.0 - np.clip(std_prob, 0.0, 1.0))).clip(0.0, 1.0)
    return out


def select_coarse_candidates(pred_df: pd.DataFrame, top_k: int = 20, min_prob: float = 0.5) -> pd.DataFrame:
    keep = pred_df[pred_df["probability"] >= min_prob].copy()
    if keep.empty:
        keep = pred_df.copy()
    return keep.sort_values("probability", ascending=False).head(top_k).reset_index(drop=True)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def generate_search_circles(
    pred_df: pd.DataFrame,
    top_k: int = 5,
    radius_nm: float = 30.0,
    lat_col: str = "grid_lat",
    lon_col: str = "grid_lon",
) -> pd.DataFrame:
    radius_km = radius_nm * 1.852
    ranked = pred_df.sort_values("probability", ascending=False).reset_index(drop=True)

    selected: List[Dict[str, Any]] = []
    for _, row in ranked.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        if any(_haversine_km(lat, lon, c["center_lat"], c["center_lon"]) < radius_km for c in selected):
            continue
        selected.append(
            {
                "center_lat": lat,
                "center_lon": lon,
                "radius_nm": radius_nm,
                "probability": float(row["probability"]),
                "confidence": float(row.get("confidence", row["probability"])),
            }
        )
        if len(selected) >= top_k:
            break

    return pd.DataFrame(selected)


def hit_at_30nm(
    circles: pd.DataFrame,
    truth_points: pd.DataFrame,
    truth_lat_col: str = "lat",
    truth_lon_col: str = "lon",
) -> float:
    if circles.empty or truth_points.empty:
        return 0.0

    hit_count = 0
    for _, truth in truth_points.iterrows():
        tlat, tlon = float(truth[truth_lat_col]), float(truth[truth_lon_col])
        matched = False
        for _, circle in circles.iterrows():
            radius_km = float(circle["radius_nm"]) * 1.852
            dist = _haversine_km(tlat, tlon, float(circle["center_lat"]), float(circle["center_lon"]))
            if dist <= radius_km:
                matched = True
                break
        hit_count += 1 if matched else 0

    return float(hit_count / len(truth_points))


def summarize_top_drivers(feature_importance: Iterable[Tuple[str, float]], top_n: int = 5) -> List[Dict[str, float]]:
    ranked = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": f, "importance": float(v)} for f, v in ranked]
