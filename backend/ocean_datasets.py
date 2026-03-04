from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

try:
    import xarray as xr
except Exception:  # pragma: no cover - import is optional at runtime
    xr = None


@dataclass(frozen=True)
class DatasetSpec:
    var_name: str
    chinese_name: str
    unit: str
    spatial_resolution: str
    depth_range: str
    dimensionality: str
    filename_template: str
    source_hint: str


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "CHL": DatasetSpec(
        var_name="CHL",
        chinese_name="表层叶绿素",
        unit="mg/m³",
        spatial_resolution="0.0417°",
        depth_range="1层 / 表层",
        dimensionality="2D",
        filename_template="{date}_chla.nc",
        source_hint="Copernicus Marine / OceanColor",
    ),
    "analysed_sst": DatasetSpec(
        var_name="analysed_sst",
        chinese_name="海表温度",
        unit="K",
        spatial_resolution="0.05°",
        depth_range="1层 / 表层",
        dimensionality="2D",
        filename_template="{date}120000-UKMO-...nc",
        source_hint="GHRSST/OSTIA 类产品",
    ),
    "thetao": DatasetSpec(
        var_name="thetao",
        chinese_name="深层海水温度",
        unit="°C",
        spatial_resolution="0.0833°",
        depth_range="30层 / 0.5–380m",
        dimensionality="3D",
        filename_template="{date}_seawater_temperature_deep.nc",
        source_hint="Copernicus Marine reanalysis/analysis",
    ),
    "so": DatasetSpec(
        var_name="so",
        chinese_name="海水盐度",
        unit="PSU",
        spatial_resolution="0.0833°",
        depth_range="30层 / 0.5–380m",
        dimensionality="3D",
        filename_template="{date}_so.nc",
        source_hint="Copernicus Marine reanalysis/analysis",
    ),
    "mlotst": DatasetSpec(
        var_name="mlotst",
        chinese_name="混合层深度",
        unit="m",
        spatial_resolution="0.0833°",
        depth_range="1层 / 表层",
        dimensionality="2D",
        filename_template="{date}_so_mld.nc",
        source_hint="Copernicus Marine reanalysis/analysis",
    ),
    "chl": DatasetSpec(
        var_name="chl",
        chinese_name="3D 叶绿素",
        unit="mg/m³",
        spatial_resolution="0.25°",
        depth_range="38层 / 0.5–411m",
        dimensionality="3D",
        filename_template="{date}_3d_chlorophy.nc",
        source_hint="Biogeochemical products",
    ),
    "o2": DatasetSpec(
        var_name="o2",
        chinese_name="溶解氧浓度",
        unit="mmol/m³",
        spatial_resolution="0.25°",
        depth_range="38层 / 0.5–411m",
        dimensionality="3D",
        filename_template="{date}_o2_pp.nc",
        source_hint="Biogeochemical products",
    ),
    "nppv": DatasetSpec(
        var_name="nppv",
        chinese_name="净初级生产力",
        unit="mg C/m³/day",
        spatial_resolution="0.25°",
        depth_range="38层 / 0.5–411m",
        dimensionality="3D",
        filename_template="{date}_o2_pp.nc",
        source_hint="Biogeochemical products",
    ),
    "WVEL": DatasetSpec(
        var_name="WVEL",
        chinese_name="垂直流速",
        unit="m/s",
        spatial_resolution="0.25°",
        depth_range="22层 / 5–410m",
        dimensionality="3D",
        filename_template="WVEL.1440x720x50.{date}.nc",
        source_hint="MITgcm / ECCO-like products",
    ),
    "sla": DatasetSpec(
        var_name="sla",
        chinese_name="海表高度异常",
        unit="m",
        spatial_resolution="0.125°",
        depth_range="1层 / 表层",
        dimensionality="2D",
        filename_template="dt_global_allsat_phy_l4_{date}_20241017.nc",
        source_hint="CMEMS altimetry L4",
    ),
    "adt": DatasetSpec(
        var_name="adt",
        chinese_name="绝对动力地形",
        unit="m",
        spatial_resolution="0.125°",
        depth_range="1层 / 表层",
        dimensionality="2D",
        filename_template="dt_global_allsat_phy_l4_{date}_adt.nc",
        source_hint="CMEMS altimetry L4",
    ),
}


class OceanDatasetManager:
    def __init__(self, data_root: str = "data/ocean", base_urls: Optional[Dict[str, str]] = None) -> None:
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.base_urls = base_urls or {}

    def render_filename(self, dataset_key: str, date: dt.date) -> str:
        spec = DATASET_SPECS[dataset_key]
        return spec.filename_template.format(date=date.strftime("%Y%m%d"))

    def local_path(self, dataset_key: str, date: dt.date) -> Path:
        return self.data_root / dataset_key / self.render_filename(dataset_key, date)

    def expected_paths(self, dataset_key: str, start: dt.date, end: dt.date) -> List[Path]:
        return [self.local_path(dataset_key, d) for d in daterange(start, end)]

    def download_dataset_for_day(self, dataset_key: str, date: dt.date, overwrite: bool = False) -> Path:
        out = self.local_path(dataset_key, date)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not overwrite:
            return out
        if dataset_key not in self.base_urls:
            raise ValueError(f"No base URL configured for dataset: {dataset_key}")
        url = self.base_urls[dataset_key].rstrip("/") + "/" + self.render_filename(dataset_key, date)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        return out

    def load_dataset(self, dataset_key: str, date: dt.date):
        if xr is None:
            raise ImportError("xarray is required for NetCDF loading. Please install xarray and netCDF4.")
        path = self.local_path(dataset_key, date)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return xr.open_dataset(path)

    def sample_point(self, dataset_key: str, date: dt.date, lat: float, lon: float, depth_m: Optional[float] = None) -> float:
        ds = self.load_dataset(dataset_key, date)
        var = DATASET_SPECS[dataset_key].var_name
        if var not in ds:
            raise KeyError(f"Variable {var} not present in {dataset_key} file")
        da = ds[var]
        lat_name = _first_match(da.dims, ["lat", "latitude", "y"])
        lon_name = _first_match(da.dims, ["lon", "longitude", "x"])
        if lat_name is None or lon_name is None:
            raise KeyError(f"Cannot infer latitude/longitude dims from {da.dims}")

        sel_kwargs = {lat_name: lat, lon_name: lon}
        if depth_m is not None:
            depth_name = _first_match(da.dims, ["depth", "lev", "z"])
            if depth_name is not None:
                sel_kwargs[depth_name] = depth_m

        picked = da.sel(sel_kwargs, method="nearest")
        return float(picked.values)


def _first_match(existing: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    existing_set = set(existing)
    for c in candidates:
        if c in existing_set:
            return c
    return None


def daterange(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def dataset_catalog() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for key, spec in DATASET_SPECS.items():
        rows.append(
            {
                "dataset_key": key,
                "var_name": spec.var_name,
                "chinese_name": spec.chinese_name,
                "unit": spec.unit,
                "spatial_resolution": spec.spatial_resolution,
                "depth_range": spec.depth_range,
                "dimensionality": spec.dimensionality,
                "filename_template": spec.filename_template,
                "source_hint": spec.source_hint,
            }
        )
    return rows
