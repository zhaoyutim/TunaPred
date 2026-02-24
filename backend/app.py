from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.data_pipeline import SPECIES, load_or_build_features
from backend.modeling import load_model, predict_with_confidence, train_model

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
    weather_corr = (
        df[["catch_index", "sst", "wind_speed", "pressure", "precipitation"]]
        .corr(numeric_only=True)["catch_index"]
        .to_dict()
    )
    return {"metrics": metrics, "weather_correlation": weather_corr}


@app.get("/api/predict")
def predict(species: str = "yellowfin"):
    name = SPECIES.get(species, list(SPECIES.values())[0])
    df = load_or_build_features(force_refresh=False)
    use = df[df["species_name"] == name].copy()
    artifact = load_model()
    pred, conf = predict_with_confidence(use, artifact)
    use["predicted_catch_index"] = pred
    use["confidence"] = conf
    return {
        "species": name,
        "points": use[
            [
                "grid_lat",
                "grid_lon",
                "event_date",
                "predicted_catch_index",
                "confidence",
                "sst",
                "wind_speed",
            ]
        ]
        .sort_values("predicted_catch_index", ascending=False)
        .head(300)
        .assign(event_date=lambda x: x["event_date"].astype(str))
        .to_dict(orient="records"),
    }
