from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests

SPECIES = {
    "yellowfin": "Thunnus albacares",
    "skipjack": "Katsuwonus pelamis",
    "albacore": "Thunnus alalunga",
}

GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "tuna_occurrences_2023_06_07.csv"
FEATURE_PATH = DATA_DIR / "tuna_features_2023_06_07.csv"


def _fetch_gbif_species_month(scientific_name: str, month: int, year: int = 2023, max_rows: int = 250) -> pd.DataFrame:
    rows: List[Dict] = []
    limit = 100
    offset = 0
    while offset < max_rows:
        params = {
            "scientificName": scientific_name,
            "hasCoordinate": "true",
            "year": year,
            "month": month,
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(GBIF_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            if "decimalLatitude" not in item or "decimalLongitude" not in item:
                continue
            event_date = item.get("eventDate")
            if not event_date:
                continue
            rows.append(
                {
                    "species_name": scientific_name,
                    "lat": float(item["decimalLatitude"]),
                    "lon": float(item["decimalLongitude"]),
                    "event_date": event_date[:10],
                    "basis_of_record": item.get("basisOfRecord", "UNKNOWN"),
                }
            )
        offset += limit
    return pd.DataFrame(rows)


def fetch_occurrences() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: List[pd.DataFrame] = []
    for _, sci_name in SPECIES.items():
        for month in (6, 7):
            frame = _fetch_gbif_species_month(sci_name, month)
            if not frame.empty:
                all_frames.append(frame)
    if not all_frames:
        raise RuntimeError("No tuna occurrence data fetched from GBIF.")
    out = pd.concat(all_frames, ignore_index=True)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out = out.dropna(subset=["event_date"]).copy()
    out.to_csv(RAW_PATH, index=False)
    return out


def _grid_round(val: float, step: float = 1.0) -> float:
    return round(val / step) * step


def _fetch_weather_row(lat: float, lon: float, day: dt.date) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "daily": "sea_surface_temperature_0_to_7m_mean,wind_speed_10m_max,pressure_msl_mean,precipitation_sum",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json().get("daily", {})
    if not payload or not payload.get("time"):
        return {
            "sst": np.nan,
            "wind_speed": np.nan,
            "pressure": np.nan,
            "precipitation": np.nan,
        }
    return {
        "sst": payload.get("sea_surface_temperature_0_to_7m_mean", [np.nan])[0],
        "wind_speed": payload.get("wind_speed_10m_max", [np.nan])[0],
        "pressure": payload.get("pressure_msl_mean", [np.nan])[0],
        "precipitation": payload.get("precipitation_sum", [np.nan])[0],
    }


def build_features(occurrences: pd.DataFrame) -> pd.DataFrame:
    df = occurrences.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["month"] = df["event_date"].dt.month
    df["day"] = df["event_date"].dt.day
    df["grid_lat"] = df["lat"].apply(_grid_round)
    df["grid_lon"] = df["lon"].apply(_grid_round)

    grouped = (
        df.groupby(["species_name", "event_date", "grid_lat", "grid_lon"], as_index=False)
        .size()
        .rename(columns={"size": "activity_count"})
    )

    weather_cache: Dict[str, Dict] = {}
    weather_rows: List[Dict] = []
    for _, row in grouped.iterrows():
        day = row["event_date"].date()
        key = f"{row['grid_lat']}_{row['grid_lon']}_{day.isoformat()}"
        if key not in weather_cache:
            weather_cache[key] = _fetch_weather_row(row["grid_lat"], row["grid_lon"], day)
        weather_rows.append(weather_cache[key])

    weather_df = pd.DataFrame(weather_rows)
    feat = pd.concat([grouped.reset_index(drop=True), weather_df], axis=1)

    favorable = (
        feat["sst"].between(24, 30).astype(float)
        + feat["wind_speed"].between(10, 30).astype(float)
        + feat["pressure"].between(1005, 1018).astype(float)
    ) / 3.0

    noise = np.random.default_rng(42).normal(0, 0.15, len(feat))
    feat["catch_index"] = np.clip(feat["activity_count"] * (0.75 + favorable + noise), 0.1, None)

    feat["species_id"] = feat["species_name"].map({v: k for k, v in enumerate(SPECIES.values())})
    feat["month"] = feat["event_date"].dt.month
    feat["day"] = feat["event_date"].dt.day
    feat = feat.dropna(subset=["sst", "wind_speed", "pressure"])
    feat.to_csv(FEATURE_PATH, index=False)
    return feat


def load_or_build_features(force_refresh: bool = False) -> pd.DataFrame:
    if FEATURE_PATH.exists() and not force_refresh:
        return pd.read_csv(FEATURE_PATH, parse_dates=["event_date"])
    occurrences = fetch_occurrences() if (not RAW_PATH.exists() or force_refresh) else pd.read_csv(RAW_PATH, parse_dates=["event_date"])
    return build_features(occurrences)
