from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

SPECIES = {
    "yellowfin": "Thunnus albacares",
    "skipjack": "Katsuwonus pelamis",
    "albacore": "Thunnus alalunga",
}

GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
OBIS_URL = "https://api.obis.org/v3/occurrence"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
NOAA_GFS_POINT_URL = "https://api.open-meteo.com/v1/forecast"
HYCOM_PLACEHOLDER_URL = "https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0"
TRAIN_YEAR = 2023
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "tuna_occurrences_2023.csv"
FEATURE_PATH = DATA_DIR / "tuna_features_2023.csv"
PREDICTION_CACHE_DIR = DATA_DIR / "predictions"
FEATURE_DATASET_PATTERN = "tuna_features_*.csv"
RAW_DATASET_PATTERN = "tuna_occurrences_*.csv"
REALTIME_LOOKBACK_DAYS = 60
HISTORICAL_YEARS = 3

CHINA_SEAS_BBOX = {
    "lat_min": 5.0,
    "lat_max": 45.0,
    "lon_min": 105.0,
    "lon_max": 145.0,
}
CHINA_SEAS_RESOLUTION = 0.25
FORCING_ANCHOR_STEP = 4.0
MAX_FORCE_ANCHORS = 25
_FORCING_CACHE: Dict[Tuple[float, float, str], Dict] = {}
_FORCING_GRID_CACHE: Dict[Tuple[str, float, float, float, float, float, float], Dict] = {}
FORCING_FETCH_TIMEOUT_SECONDS = 8.0


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


def _frange(start: float, end: float, step: float) -> List[float]:
    vals = []
    x = start
    while x <= end + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals


def _normalize_weather(
    source: str,
    sst,
    wind_speed,
    pressure,
    precipitation,
    ok: bool = True,
    detail: str = "",
) -> Dict:
    return {
        "sst": sst,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "precipitation": precipitation,
        "source": source,
        "ok": ok,
        "detail": detail,
    }


def _is_missing(value) -> bool:
    return value is None or pd.isna(value)


def _first_valid(*values, default=np.nan):
    for value in values:
        if not _is_missing(value):
            return value
    return default


def _fetch_noaa_row(lat: float, lon: float, day: dt.date, timeout: int = 2) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "daily": "wind_speed_10m_max,pressure_msl_mean,precipitation_sum",
        "timezone": "UTC",
    }
    resp = requests.get(NOAA_GFS_POINT_URL, params=params, timeout=timeout)
    if resp.status_code >= 400:
        return _normalize_weather("noaa", np.nan, np.nan, np.nan, np.nan, ok=False, detail=f"http {resp.status_code}")
    payload = resp.json().get("daily", {})
    if not payload or not payload.get("time"):
        return _normalize_weather("noaa", np.nan, np.nan, np.nan, np.nan, ok=False, detail="empty")
    return _normalize_weather(
        "noaa",
        np.nan,
        payload.get("wind_speed_10m_max", [np.nan])[0],
        payload.get("pressure_msl_mean", [np.nan])[0],
        payload.get("precipitation_sum", [np.nan])[0],
    )


def _fetch_hycom_row(lat: float, lon: float, day: dt.date, timeout: int = 2) -> Dict:
    # HYCOM NCSS requires dataset-specific query signatures; keep as soft-fail placeholder
    # so the provider chain remains extensible without breaking runtime.
    _ = (lat, lon, day, timeout)
    return _normalize_weather("hycom", np.nan, np.nan, np.nan, np.nan, ok=False, detail="placeholder")


def _fetch_gbif_species_month(scientific_name: str, month: int, year: int = TRAIN_YEAR, max_rows: int = 250) -> pd.DataFrame:
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


def _recent_years(total_years: int = HISTORICAL_YEARS) -> List[int]:
    current_year = dt.date.today().year
    base_years = {TRAIN_YEAR, current_year}
    for shift in range(max(total_years - 1, 0)):
        base_years.add(current_year - shift)
    return sorted(base_years)


def _fetch_gbif_species_recent(scientific_name: str, start_date: dt.date, end_date: dt.date, max_rows: int = 300) -> pd.DataFrame:
    rows: List[Dict] = []
    limit = 100
    offset = 0
    while offset < max_rows:
        params = {
            "scientificName": scientific_name,
            "hasCoordinate": "true",
            "eventDate": f"{start_date.isoformat()},{end_date.isoformat()}",
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
                    "basis_of_record": item.get("basisOfRecord", "GBIF_RECENT"),
                }
            )
        offset += limit
    return pd.DataFrame(rows)


def _fetch_obis_species_recent(scientific_name: str, start_date: dt.date, end_date: dt.date, max_rows: int = 300) -> pd.DataFrame:
    rows: List[Dict] = []
    size = 100
    start = 0
    while start < max_rows:
        params = {
            "scientificname": scientific_name,
            "startdate": start_date.isoformat(),
            "enddate": end_date.isoformat(),
            "size": size,
            "from": start,
            "hascoords": "true",
        }
        resp = requests.get(OBIS_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            lat = item.get("decimalLatitude")
            lon = item.get("decimalLongitude")
            event_date = item.get("eventDate") or item.get("date_mid")
            if lat is None or lon is None or not event_date:
                continue
            rows.append(
                {
                    "species_name": scientific_name,
                    "lat": float(lat),
                    "lon": float(lon),
                    "event_date": str(event_date)[:10],
                    "basis_of_record": item.get("basisOfRecord") or "OBIS_OCCURRENCE",
                }
            )
        start += size
    return pd.DataFrame(rows)


def _collect_near_realtime_occurrences(lookback_days: int = REALTIME_LOOKBACK_DAYS) -> pd.DataFrame:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback_days)
    frames: List[pd.DataFrame] = []
    for sci_name in SPECIES.values():
        try:
            gbif_recent = _fetch_gbif_species_recent(sci_name, start_date=start_date, end_date=end_date)
        except Exception:
            gbif_recent = pd.DataFrame()
        if not gbif_recent.empty:
            frames.append(gbif_recent)

        try:
            obis_recent = _fetch_obis_species_recent(sci_name, start_date=start_date, end_date=end_date)
        except Exception:
            obis_recent = pd.DataFrame()
        if not obis_recent.empty:
            frames.append(obis_recent)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_occurrences() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: List[pd.DataFrame] = []
    for sci_name in SPECIES.values():
        for year in _recent_years():
            for month in range(1, 13):
                frame = _fetch_gbif_species_month(sci_name, month, year=year)
                if not frame.empty:
                    all_frames.append(frame)

    recent = _collect_near_realtime_occurrences()
    if not recent.empty:
        all_frames.append(recent)

    if not all_frames:
        raise RuntimeError("No tuna occurrence data fetched from internet sources.")

    out = pd.concat(all_frames, ignore_index=True)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out = out.dropna(subset=["event_date", "lat", "lon", "species_name"]).copy()
    out = out.drop_duplicates(subset=["species_name", "lat", "lon", "event_date", "basis_of_record"], keep="first")
    out.to_csv(RAW_PATH, index=False)
    return out


def _grid_round(val: float, step: float = 1.0) -> float:
    return round(val / step) * step


def _fetch_weather_row(lat: float, lon: float, day: dt.date) -> Dict:
    default_weather = {
        "sst": np.nan,
        "wind_speed": np.nan,
        "pressure": np.nan,
        "precipitation": np.nan,
    }
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "daily": "sea_surface_temperature_mean,wind_speed_10m_max,pressure_msl_mean,precipitation_sum",
        "timezone": "UTC",
    }
    # region agent log
    _debug_log(
        "H5",
        "backend/data_pipeline.py:115",
        "open-meteo request prepared",
        {"latitude": lat, "longitude": lon, "date": day.isoformat(), "daily": params["daily"]},
    )
    # endregion
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    # region agent log
    _debug_log(
        "H6",
        "backend/data_pipeline.py:125",
        "open-meteo response status",
        {"status_code": resp.status_code, "url": getattr(resp, "url", "")},
    )
    # endregion
    if resp.status_code >= 400:
        # region agent log
        _debug_log(
            "H6",
            "backend/data_pipeline.py:133",
            "open-meteo response error body",
            {"status_code": resp.status_code, "body": (resp.text or "")[:800]},
        )
        # endregion
        return default_weather
    try:
        resp.raise_for_status()
    except requests.RequestException as exc:
        # region agent log
        _debug_log(
            "H6",
            "backend/data_pipeline.py:142",
            "open-meteo raise_for_status failed",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        # endregion
        return default_weather
    payload = resp.json().get("daily", {})
    if not payload or not payload.get("time"):
        return default_weather
    return {
        "sst": payload.get("sea_surface_temperature_mean", [np.nan])[0],
        "wind_speed": payload.get("wind_speed_10m_max", [np.nan])[0],
        "pressure": payload.get("pressure_msl_mean", [np.nan])[0],
        "precipitation": payload.get("precipitation_sum", [np.nan])[0],
    }


def _fetch_ocean_forcing_row(lat: float, lon: float, day: dt.date) -> Dict:
    cache_key = (round(lat, 4), round(lon, 4), day.isoformat())
    if cache_key in _FORCING_CACHE:
        return _FORCING_CACHE[cache_key]

    # NOAA/HYCOM are primary targets; if unavailable, fallback to existing Open-Meteo path.
    try:
        noaa = _fetch_noaa_row(lat, lon, day, timeout=4)
    except Exception:
        noaa = _normalize_weather("noaa", np.nan, np.nan, np.nan, np.nan, ok=False, detail="exception")
    try:
        hycom = _fetch_hycom_row(lat, lon, day, timeout=4)
    except Exception:
        hycom = _normalize_weather("hycom", np.nan, np.nan, np.nan, np.nan, ok=False, detail="exception")

    needs_fallback = (
        _is_missing(hycom["sst"])
        or _is_missing(noaa["wind_speed"])
        or _is_missing(noaa["pressure"])
        or _is_missing(noaa["precipitation"])
    )
    open_meteo = {"sst": np.nan, "wind_speed": np.nan, "pressure": np.nan, "precipitation": np.nan}
    open_meteo_ok = True
    open_meteo_detail = "not_required"
    if needs_fallback:
        try:
            open_meteo = _fetch_weather_row(lat, lon, day)
        except Exception:
            open_meteo_ok = False
            open_meteo_detail = "exception"
            open_meteo = {"sst": np.nan, "wind_speed": np.nan, "pressure": np.nan, "precipitation": np.nan}
        else:
            open_meteo_ok = not all(_is_missing(open_meteo[k]) for k in ["sst", "wind_speed", "pressure", "precipitation"])
            open_meteo_detail = "" if open_meteo_ok else "empty"

    merged = {
        "sst": _first_valid(hycom["sst"], open_meteo["sst"], default=26.0),
        "wind_speed": _first_valid(noaa["wind_speed"], open_meteo["wind_speed"], default=15.0),
        "pressure": _first_valid(noaa["pressure"], open_meteo["pressure"], default=1013.0),
        "precipitation": _first_valid(noaa["precipitation"], open_meteo["precipitation"], default=0.0),
        "provider_status": {
            "noaa": {"ok": noaa.get("ok", False), "detail": noaa.get("detail", "")},
            "hycom": {"ok": hycom.get("ok", False), "detail": hycom.get("detail", "")},
            "open_meteo": {"ok": open_meteo_ok, "detail": open_meteo_detail},
        },
    }
    _FORCING_CACHE[cache_key] = merged
    return merged


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
    # region agent log
    _debug_log("H7", "backend/data_pipeline.py:165", "build_features grouped size", {"rows": int(len(grouped))})
    # endregion
    for _, row in grouped.iterrows():
        day = row["event_date"].date()
        key = f"{row['grid_lat']}_{row['grid_lon']}_{day.isoformat()}"
        if key not in weather_cache:
            weather_cache[key] = _fetch_weather_row(row["grid_lat"], row["grid_lon"], day)
        weather_rows.append(weather_cache[key])

    weather_df = pd.DataFrame(weather_rows)
    feat = pd.concat([grouped.reset_index(drop=True), weather_df], axis=1)

    # Keep rows when weather fields are partially missing.
    weather_defaults = {
        "sst": 26.0,
        "wind_speed": 15.0,
        "pressure": 1013.0,
        "precipitation": 0.0,
    }
    for col, default in weather_defaults.items():
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
        if feat[col].isna().all():
            feat[col] = default
        else:
            feat[col] = feat[col].fillna(feat[col].median())

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
    feat.to_csv(FEATURE_PATH, index=False)
    return feat


def load_or_build_features(force_refresh: bool = False) -> pd.DataFrame:
    def _load_and_merge(pattern: str, parse_dates: List[str]) -> pd.DataFrame:
        paths = sorted(DATA_DIR.glob(pattern))
        if FEATURE_PATH.exists() and pattern == FEATURE_DATASET_PATTERN and FEATURE_PATH not in paths:
            paths.append(FEATURE_PATH)
        if RAW_PATH.exists() and pattern == RAW_DATASET_PATTERN and RAW_PATH not in paths:
            paths.append(RAW_PATH)

        frames: List[pd.DataFrame] = []
        for path in paths:
            try:
                frame = pd.read_csv(path, parse_dates=parse_dates)
            except Exception:
                continue
            if not frame.empty:
                frame["_source_file"] = path.name
                frames.append(frame)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        if "event_date" in merged.columns:
            merged["event_date"] = pd.to_datetime(merged["event_date"], errors="coerce")
        dedupe_cols = [
            c
            for c in ["species_name", "event_date", "grid_lat", "grid_lon", "lat", "lon", "basis_of_record", "activity_count", "catch_index"]
            if c in merged.columns
        ]
        if dedupe_cols:
            merged = merged.drop_duplicates(subset=dedupe_cols, keep="first")
        return merged.drop(columns=["_source_file"], errors="ignore").reset_index(drop=True)

    if not force_refresh:
        merged_features = _load_and_merge(FEATURE_DATASET_PATTERN, parse_dates=["event_date"])
        if not merged_features.empty:
            return merged_features

    if force_refresh:
        occurrences = fetch_occurrences()
    else:
        merged_occurrences = _load_and_merge(RAW_DATASET_PATTERN, parse_dates=["event_date"])
        occurrences = merged_occurrences if not merged_occurrences.empty else fetch_occurrences()

    return build_features(occurrences)


def build_prediction_features(
    species_name: str,
    predict_date: dt.date,
    force_refresh: bool = False,
    region: str = "china_seas",
    resolution: float = CHINA_SEAS_RESOLUTION,
    anchor_step: float = 10.0,
) -> Tuple[pd.DataFrame, Dict]:
    hist = load_or_build_features(force_refresh=force_refresh)
    species_hist = hist[hist["species_name"] == species_name].copy()
    if species_hist.empty:
        raise ValueError(f"No historical features found for species: {species_name}")

    if region != "china_seas":
        raise ValueError("Only region=china_seas is currently supported")

    grid_lats = _frange(CHINA_SEAS_BBOX["lat_min"], CHINA_SEAS_BBOX["lat_max"], resolution)
    grid_lons = _frange(CHINA_SEAS_BBOX["lon_min"], CHINA_SEAS_BBOX["lon_max"], resolution)
    grid = pd.DataFrame([(lat, lon) for lat in grid_lats for lon in grid_lons], columns=["grid_lat", "grid_lon"])

    priors = (
        species_hist.groupby(["grid_lat", "grid_lon"], as_index=False)
        .agg(
            activity_count=("activity_count", "mean"),
            species_id=("species_id", "first"),
            sst_hist=("sst", "median"),
            wind_speed_hist=("wind_speed", "median"),
            pressure_hist=("pressure", "median"),
            precipitation_hist=("precipitation", "median"),
        )
        .copy()
    )

    priors["grid_lat_int"] = priors["grid_lat"].round().astype(float)
    priors["grid_lon_int"] = priors["grid_lon"].round().astype(float)
    prior_keyed = priors.groupby(["grid_lat_int", "grid_lon_int"], as_index=False).median(numeric_only=True)
    prior_keyed = prior_keyed.drop(columns=["grid_lat", "grid_lon"], errors="ignore")
    prior_global = priors[["activity_count", "species_id", "sst_hist", "wind_speed_hist", "pressure_hist", "precipitation_hist"]].median(
        numeric_only=True
    )

    out = grid.copy()
    out["grid_lat_int"] = out["grid_lat"].round().astype(float)
    out["grid_lon_int"] = out["grid_lon"].round().astype(float)
    out = out.merge(prior_keyed, on=["grid_lat_int", "grid_lon_int"], how="left")
    for col, default in {
        "activity_count": float(prior_global.get("activity_count", 1.0)),
        "species_id": float(prior_global.get("species_id", 0.0)),
        "sst_hist": float(prior_global.get("sst_hist", 26.0)),
        "wind_speed_hist": float(prior_global.get("wind_speed_hist", 15.0)),
        "pressure_hist": float(prior_global.get("pressure_hist", 1013.0)),
        "precipitation_hist": float(prior_global.get("precipitation_hist", 0.0)),
    }.items():
        out[col] = out[col].fillna(default)

    cache_key = (
        predict_date.isoformat(),
        float(CHINA_SEAS_BBOX["lat_min"]),
        float(CHINA_SEAS_BBOX["lat_max"]),
        float(CHINA_SEAS_BBOX["lon_min"]),
        float(CHINA_SEAS_BBOX["lon_max"]),
        float(resolution),
        float(anchor_step),
    )
    grid_cache_hit = cache_key in _FORCING_GRID_CACHE
    if grid_cache_hit:
        forcing_grid = _FORCING_GRID_CACHE[cache_key]
        anchors = forcing_grid["anchors"].copy()
        anchor_weather = forcing_grid["anchor_weather"].copy()
        provider_summary = forcing_grid["provider_summary"].copy()
    else:
        anchor_lats = _frange(CHINA_SEAS_BBOX["lat_min"], CHINA_SEAS_BBOX["lat_max"], anchor_step)
        anchor_lons = _frange(CHINA_SEAS_BBOX["lon_min"], CHINA_SEAS_BBOX["lon_max"], anchor_step)
        anchors = pd.DataFrame([(lat, lon) for lat in anchor_lats for lon in anchor_lons], columns=["anchor_lat", "anchor_lon"])
        if len(anchors) > MAX_FORCE_ANCHORS:
            anchors = anchors.iloc[:MAX_FORCE_ANCHORS].copy()
        anchor_weather_rows = []
        noaa_ok = 0
        hycom_ok = 0
        open_meteo_ok = 0
        fetched_anchors = 0
        timed_out = False
        fetch_started_at = time.monotonic()
        for _, row in anchors.iterrows():
            if (time.monotonic() - fetch_started_at) > FORCING_FETCH_TIMEOUT_SECONDS:
                timed_out = True
                break
            forcing = _fetch_ocean_forcing_row(float(row["anchor_lat"]), float(row["anchor_lon"]), predict_date)
            status = forcing.pop("provider_status", {})
            noaa_ok += 1 if status.get("noaa", {}).get("ok") else 0
            hycom_ok += 1 if status.get("hycom", {}).get("ok") else 0
            open_meteo_ok += 1 if status.get("open_meteo", {}).get("ok") else 0
            anchor_weather_rows.append({**row.to_dict(), **forcing})
            fetched_anchors += 1

        if timed_out and fetched_anchors < len(anchors):
            for _, pending in anchors.iloc[fetched_anchors:].iterrows():
                anchor_weather_rows.append(
                    {
                        "anchor_lat": float(pending["anchor_lat"]),
                        "anchor_lon": float(pending["anchor_lon"]),
                        "sst": np.nan,
                        "wind_speed": np.nan,
                        "pressure": np.nan,
                        "precipitation": np.nan,
                    }
                )
        anchor_weather = pd.DataFrame(anchor_weather_rows)
        provider_summary = {
            "noaa_ok_ratio": float(noaa_ok / fetched_anchors) if fetched_anchors else 0.0,
            "hycom_ok_ratio": float(hycom_ok / fetched_anchors) if fetched_anchors else 0.0,
            "open_meteo_ok_ratio": float(open_meteo_ok / fetched_anchors) if fetched_anchors else 0.0,
            "fetched_anchors": int(fetched_anchors),
            "total_anchors": int(len(anchors)),
            "partial_results": bool(timed_out),
        }
        _FORCING_GRID_CACHE[cache_key] = {
            "anchors": anchors.copy(),
            "anchor_weather": anchor_weather.copy(),
            "provider_summary": provider_summary.copy(),
        }

    out["anchor_lat"] = (np.round((out["grid_lat"] - CHINA_SEAS_BBOX["lat_min"]) / anchor_step) * anchor_step) + CHINA_SEAS_BBOX[
        "lat_min"
    ]
    out["anchor_lon"] = (np.round((out["grid_lon"] - CHINA_SEAS_BBOX["lon_min"]) / anchor_step) * anchor_step) + CHINA_SEAS_BBOX[
        "lon_min"
    ]
    out = out.merge(anchor_weather, on=["anchor_lat", "anchor_lon"], how="left")

    out["sst"] = out["sst"].fillna(out["sst_hist"]).fillna(26.0)
    out["wind_speed"] = out["wind_speed"].fillna(out["wind_speed_hist"]).fillna(15.0)
    out["pressure"] = out["pressure"].fillna(out["pressure_hist"]).fillna(1013.0)
    out["precipitation"] = out["precipitation"].fillna(out["precipitation_hist"]).fillna(0.0)

    out["species_name"] = species_name
    out["event_date"] = pd.Timestamp(predict_date)
    out["month"] = int(predict_date.month)
    out["day"] = int(predict_date.day)

    features = out[
        [
            "species_name",
            "event_date",
            "species_id",
            "grid_lat",
            "grid_lon",
            "month",
            "day",
            "sst",
            "wind_speed",
            "pressure",
            "precipitation",
            "activity_count",
        ]
    ].copy()

    provider_meta = {
        "region": region,
        "resolution": resolution,
        "bbox": CHINA_SEAS_BBOX.copy(),
        "cache_hit": grid_cache_hit,
        "anchors": int(len(anchors)),
        "providers": provider_summary,
    }

    PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PREDICTION_CACHE_DIR / f"{species_name.replace(' ', '_').lower()}_{predict_date.isoformat()}_{region}_{resolution:.2f}.csv"
    features.to_csv(cache_path, index=False)

    return features, provider_meta
