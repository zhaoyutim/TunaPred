from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
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
_debug_log("H2", "backend/app.py:32", "app module import start", {})
# endregion

from backend.data_pipeline import SPECIES, build_prediction_features, load_or_build_features

try:
    from backend.modeling import load_model, predict_with_confidence, train_model
    # region agent log
    _debug_log("H2", "backend/app.py:39", "backend.modeling import succeeded", {})
    # endregion
except Exception as exc:
    # region agent log
    _debug_log("H2", "backend/app.py:43", "backend.modeling import failed", {"error": str(exc), "error_type": type(exc).__name__})
    # endregion
    raise

app = FastAPI(title="TunaPred Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/")
def root() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.post("/api/fetch")
def fetch_data(force_refresh: bool = False):
    df = load_or_build_features(force_refresh=force_refresh)
    return {
        "rows": int(len(df)),
        "species": sorted(df["species_name"].unique().tolist()),
        "date_min": str(df["event_date"].min()),
        "date_max": str(df["event_date"].max()),
    }


@app.post("/api/train")
def train():
    df = load_or_build_features(force_refresh=False)
    metrics = train_model(df)
    weather_corr_raw = (
        df[["catch_index", "sst", "wind_speed", "pressure", "precipitation"]]
        .corr(numeric_only=True)["catch_index"]
        .to_dict()
    )

    def _finite_or_none(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    metrics_safe = {k: _finite_or_none(v) for k, v in metrics.items()}
    weather_corr = {k: _finite_or_none(v) for k, v in weather_corr_raw.items()}
    return {"metrics": metrics_safe, "weather_correlation": weather_corr}


@app.get("/api/predict")
def predict(
    species: str = "yellowfin",
    predict_date: str = "2025-06-01",
    forecast_date: str | None = None,
    mode: str = "forecast_map",
    region: str = "china_seas",
    resolution: float = 0.25,
    limit: int = 0,
):
    name = SPECIES.get(species, list(SPECIES.values())[0])
    target_date_str = forecast_date or predict_date
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"forecast_map", "legacy"}:
        raise HTTPException(status_code=400, detail="mode must be one of: forecast_map, legacy")

    try:
        target_day = dt.date.fromisoformat(target_date_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="predict_date/forecast_date must be YYYY-MM-DD") from exc

    try:
        use, provider_meta = build_prediction_features(
            name,
            target_day,
            force_refresh=False,
            region=region,
            resolution=resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    artifact = load_model()
    pred, conf = predict_with_confidence(use, artifact)
    use["predicted_catch_index"] = pred
    use["confidence"] = conf
    ranked = use.sort_values("predicted_catch_index", ascending=False)
    if limit > 0:
        clipped = ranked.head(limit)
    elif normalized_mode == "legacy":
        clipped = ranked.head(5000)
    else:
        clipped = ranked

    source_health = provider_meta.get("providers", {})
    return {
        "species": name,
        "predict_date": target_date_str,
        "meta": {
            "mode": normalized_mode,
            "region": region,
            "resolution": resolution,
            "bbox": provider_meta.get("bbox", {}),
            "row_count": int(len(use)),
            "total_points": int(len(use)),
            "returned_points": int(len(clipped)),
            "provider_status": source_health,
            "source_health": source_health,
            "cache_hit": provider_meta.get("cache_hit", False),
            "anchors": provider_meta.get("anchors", 0),
        },
        "points": clipped[
            [
                "grid_lat",
                "grid_lon",
                "event_date",
                "predicted_catch_index",
                "confidence",
                "sst",
                "wind_speed",
            ]
        ].assign(event_date=lambda x: x["event_date"].astype(str)).to_dict(orient="records"),
    }
