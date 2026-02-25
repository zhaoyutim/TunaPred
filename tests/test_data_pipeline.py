from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend import data_pipeline


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttpResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.url = "http://example.test"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


class DataPipelineTests(unittest.TestCase):
    def test_fetch_gbif_species_month_filters_and_paginates(self):
        responses = [
            _FakeResponse(
                {
                    "results": [
                        {
                            "decimalLatitude": 1.1,
                            "decimalLongitude": 2.2,
                            "eventDate": "2023-06-01T00:00:00Z",
                            "basisOfRecord": "HUMAN_OBSERVATION",
                        },
                        {
                            "decimalLatitude": 3.3,
                            "decimalLongitude": 4.4,
                            # Missing eventDate, should be filtered out.
                        },
                    ]
                }
            ),
            _FakeResponse({"results": []}),
        ]

        with patch("backend.data_pipeline.requests.get", side_effect=responses) as mock_get:
            df = data_pipeline._fetch_gbif_species_month("Thunnus albacares", month=6, max_rows=250)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["species_name"], "Thunnus albacares")
        self.assertEqual(df.iloc[0]["event_date"], "2023-06-01")
        self.assertEqual(mock_get.call_count, 2)

    def test_fetch_occurrences_writes_raw_csv(self):
        fake_occ = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "lat": 1.0,
                    "lon": 2.0,
                    "event_date": "2023-06-01",
                    "basis_of_record": "HUMAN_OBSERVATION",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data_dir = Path(tmpdir) / "data"
            tmp_raw = tmp_data_dir / "tuna_occurrences_2023_06_07.csv"
            with (
                patch.object(data_pipeline, "DATA_DIR", tmp_data_dir),
                patch.object(data_pipeline, "RAW_PATH", tmp_raw),
                patch("backend.data_pipeline._recent_years", return_value=[2023]),
                patch("backend.data_pipeline._fetch_gbif_species_month", return_value=fake_occ),
                patch("backend.data_pipeline._collect_near_realtime_occurrences", return_value=pd.DataFrame()),
            ):
                out = data_pipeline.fetch_occurrences()

            self.assertTrue(tmp_raw.exists())
            self.assertEqual(len(out), 1)  # duplicate mock rows are deduplicated across loops
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["event_date"]))


    def test_collect_near_realtime_occurrences_combines_gbif_and_obis(self):
        gbif_recent = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "lat": 1.0,
                    "lon": 2.0,
                    "event_date": "2025-01-01",
                    "basis_of_record": "GBIF_RECENT",
                }
            ]
        )
        obis_recent = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "lat": 3.0,
                    "lon": 4.0,
                    "event_date": "2025-01-02",
                    "basis_of_record": "OBIS_OCCURRENCE",
                }
            ]
        )
        with (
            patch("backend.data_pipeline._fetch_gbif_species_recent", return_value=gbif_recent),
            patch("backend.data_pipeline._fetch_obis_species_recent", return_value=obis_recent),
        ):
            out = data_pipeline._collect_near_realtime_occurrences(lookback_days=14)

        self.assertEqual(len(out), 6)
        self.assertIn("basis_of_record", out.columns)

    def test_collect_near_realtime_occurrences_tolerates_provider_errors(self):
        with (
            patch("backend.data_pipeline._fetch_gbif_species_recent", side_effect=RuntimeError("gbif down")),
            patch("backend.data_pipeline._fetch_obis_species_recent", return_value=pd.DataFrame()),
        ):
            out = data_pipeline._collect_near_realtime_occurrences(lookback_days=7)

        self.assertTrue(out.empty)

    def test_load_or_build_features_force_refresh_triggers_download_and_build(self):
        fake_occ = pd.DataFrame(
            [{"species_name": "Thunnus albacares", "lat": 1.0, "lon": 2.0, "event_date": "2023-06-01"}]
        )
        fake_feat = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "event_date": pd.Timestamp("2023-06-01"),
                    "grid_lat": 1.0,
                    "grid_lon": 2.0,
                    "catch_index": 1.2,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data_dir = Path(tmpdir) / "data"
            tmp_raw = tmp_data_dir / "tuna_occurrences_2023_06_07.csv"
            tmp_feat = tmp_data_dir / "tuna_features_2023_06_07.csv"
            with (
                patch.object(data_pipeline, "DATA_DIR", tmp_data_dir),
                patch.object(data_pipeline, "RAW_PATH", tmp_raw),
                patch.object(data_pipeline, "FEATURE_PATH", tmp_feat),
                patch("backend.data_pipeline.fetch_occurrences", return_value=fake_occ) as mock_fetch,
                patch("backend.data_pipeline.build_features", return_value=fake_feat) as mock_build,
            ):
                out = data_pipeline.load_or_build_features(force_refresh=True)

            mock_fetch.assert_called_once()
            mock_build.assert_called_once_with(fake_occ)
            pd.testing.assert_frame_equal(out, fake_feat)

    def test_load_or_build_features_uses_existing_feature_file(self):
        existing = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "event_date": "2023-06-01",
                    "grid_lat": 1.0,
                    "grid_lon": 2.0,
                    "catch_index": 1.2,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_feat = Path(tmpdir) / "tuna_features_2023_06_07.csv"
            existing.to_csv(tmp_feat, index=False)
            with (
                patch.object(data_pipeline, "DATA_DIR", Path(tmpdir)),
                patch.object(data_pipeline, "FEATURE_PATH", tmp_feat),
                patch("backend.data_pipeline.fetch_occurrences") as mock_fetch,
                patch("backend.data_pipeline.build_features") as mock_build,
            ):
                out = data_pipeline.load_or_build_features(force_refresh=False)

            self.assertEqual(len(out), 1)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["event_date"]))
            mock_fetch.assert_not_called()
            mock_build.assert_not_called()

    def test_load_or_build_features_merges_similar_feature_datasets(self):
        first = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "event_date": "2023-06-01",
                    "grid_lat": 1.0,
                    "grid_lon": 2.0,
                    "catch_index": 1.2,
                }
            ]
        )
        second = pd.DataFrame(
            [
                {
                    "species_name": "Katsuwonus pelamis",
                    "event_date": "2023-07-01",
                    "grid_lat": 3.0,
                    "grid_lon": 4.0,
                    "catch_index": 1.6,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            first.to_csv(tmp_data / "tuna_features_2023_06_07.csv", index=False)
            second.to_csv(tmp_data / "tuna_features_2023.csv", index=False)
            with (
                patch.object(data_pipeline, "DATA_DIR", tmp_data),
                patch.object(data_pipeline, "FEATURE_PATH", tmp_data / "tuna_features_2023.csv"),
                patch("backend.data_pipeline.fetch_occurrences") as mock_fetch,
                patch("backend.data_pipeline.build_features") as mock_build,
            ):
                out = data_pipeline.load_or_build_features(force_refresh=False)

            self.assertEqual(len(out), 2)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["event_date"]))
            mock_fetch.assert_not_called()
            mock_build.assert_not_called()

    def test_load_or_build_features_merges_similar_raw_datasets_before_build(self):
        raw_first = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "lat": 1.0,
                    "lon": 2.0,
                    "event_date": "2023-06-01",
                    "basis_of_record": "HUMAN_OBSERVATION",
                }
            ]
        )
        raw_second = pd.DataFrame(
            [
                {
                    "species_name": "Katsuwonus pelamis",
                    "lat": 3.0,
                    "lon": 4.0,
                    "event_date": "2023-06-02",
                    "basis_of_record": "HUMAN_OBSERVATION",
                }
            ]
        )
        fake_built = pd.DataFrame([{"species_name": "merged", "event_date": pd.Timestamp("2023-06-01")}])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            raw_first.to_csv(tmp_data / "tuna_occurrences_2023_06_07.csv", index=False)
            raw_second.to_csv(tmp_data / "tuna_occurrences_2023.csv", index=False)
            with (
                patch.object(data_pipeline, "DATA_DIR", tmp_data),
                patch.object(data_pipeline, "RAW_PATH", tmp_data / "tuna_occurrences_2023.csv"),
                patch.object(data_pipeline, "FEATURE_PATH", tmp_data / "missing_features.csv"),
                patch("backend.data_pipeline.fetch_occurrences") as mock_fetch,
                patch("backend.data_pipeline.build_features", return_value=fake_built) as mock_build,
            ):
                out = data_pipeline.load_or_build_features(force_refresh=False)

            self.assertEqual(len(out), 1)
            mock_fetch.assert_not_called()
            built_input = mock_build.call_args[0][0]
            self.assertEqual(len(built_input), 2)

    def test_build_features_keeps_rows_when_sst_missing(self):
        occ = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "lat": 4.1,
                    "lon": -81.6,
                    "event_date": "2023-06-12",
                    "basis_of_record": "HUMAN_OBSERVATION",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_feat = Path(tmpdir) / "tuna_features_2023_06_07.csv"
            with (
                patch.object(data_pipeline, "FEATURE_PATH", tmp_feat),
                patch(
                    "backend.data_pipeline._fetch_weather_row",
                    return_value={
                        "sst": float("nan"),
                        "wind_speed": 20.0,
                        "pressure": 1012.0,
                        "precipitation": 0.0,
                    },
                ),
            ):
                out = data_pipeline.build_features(occ)

            self.assertEqual(len(out), 1)
            self.assertFalse(pd.isna(out.iloc[0]["sst"]))

    def test_fetch_weather_row_returns_defaults_on_http_error(self):
        response = _FakeHttpResponse(status_code=400, text="bad request")
        with patch("backend.data_pipeline.requests.get", return_value=response):
            out = data_pipeline._fetch_weather_row(1.0, 2.0, pd.Timestamp("2023-06-01").date())

        self.assertTrue(pd.isna(out["sst"]))
        self.assertTrue(pd.isna(out["wind_speed"]))
        self.assertTrue(pd.isna(out["pressure"]))
        self.assertTrue(pd.isna(out["precipitation"]))

    def test_build_prediction_features_for_target_date(self):
        hist = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "event_date": pd.Timestamp("2023-06-01"),
                    "species_id": 0,
                    "grid_lat": 1.0,
                    "grid_lon": 2.0,
                    "month": 6,
                    "day": 1,
                    "sst": 26.2,
                    "wind_speed": 12.0,
                    "pressure": 1010.0,
                    "precipitation": 0.4,
                    "activity_count": 2,
                    "catch_index": 1.2,
                }
            ]
        )
        forcing = {
            "sst": 27.0,
            "wind_speed": 14.0,
            "pressure": 1011.0,
            "precipitation": 0.0,
            "provider_status": {"noaa": {"ok": True, "detail": ""}, "hycom": {"ok": False, "detail": "placeholder"}},
        }
        with (
            patch("backend.data_pipeline.load_or_build_features", return_value=hist),
            patch("backend.data_pipeline._fetch_ocean_forcing_row", return_value=forcing),
            patch.object(data_pipeline, "CHINA_SEAS_BBOX", {"lat_min": 1.0, "lat_max": 1.0, "lon_min": 2.0, "lon_max": 2.0}),
        ):
            out, meta = data_pipeline.build_prediction_features(
                "Thunnus albacares",
                pd.Timestamp("2025-06-01").date(),
                region="china_seas",
                resolution=0.25,
            )

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["species_name"], "Thunnus albacares")
        self.assertEqual(str(out.iloc[0]["event_date"])[:10], "2025-06-01")
        self.assertEqual(out.iloc[0]["month"], 6)
        self.assertEqual(out.iloc[0]["day"], 1)
        self.assertIn("providers", meta)

    def test_fetch_ocean_forcing_row_normalizes_sources(self):
        noaa = {"sst": float("nan"), "wind_speed": 10.0, "pressure": 1009.0, "precipitation": 1.2, "ok": True, "detail": ""}
        hycom = {"sst": 25.5, "wind_speed": float("nan"), "pressure": float("nan"), "precipitation": float("nan"), "ok": True, "detail": ""}
        fallback = {"sst": 24.9, "wind_speed": 12.0, "pressure": 1011.0, "precipitation": 0.0}
        with (
            patch("backend.data_pipeline._fetch_noaa_row", return_value=noaa),
            patch("backend.data_pipeline._fetch_hycom_row", return_value=hycom),
            patch("backend.data_pipeline._fetch_weather_row", return_value=fallback),
        ):
            out = data_pipeline._fetch_ocean_forcing_row(1.0, 2.0, pd.Timestamp("2025-06-01").date())

        self.assertEqual(out["sst"], 25.5)
        self.assertEqual(out["wind_speed"], 10.0)
        self.assertEqual(out["pressure"], 1009.0)
        self.assertEqual(out["precipitation"], 1.2)

    def test_build_prediction_features_generates_expected_grid_shape_and_bounds(self):
        hist = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "event_date": pd.Timestamp("2023-06-01"),
                    "species_id": 0,
                    "grid_lat": 10.0,
                    "grid_lon": 120.0,
                    "month": 6,
                    "day": 1,
                    "sst": 26.2,
                    "wind_speed": 12.0,
                    "pressure": 1010.0,
                    "precipitation": 0.4,
                    "activity_count": 2,
                    "catch_index": 1.2,
                }
            ]
        )
        forcing = {
            "sst": 27.0,
            "wind_speed": 14.0,
            "pressure": 1011.0,
            "precipitation": 0.0,
            "provider_status": {"noaa": {"ok": True, "detail": ""}, "hycom": {"ok": False, "detail": "placeholder"}},
        }
        with (
            patch("backend.data_pipeline.load_or_build_features", return_value=hist),
            patch("backend.data_pipeline._fetch_ocean_forcing_row", return_value=forcing),
            patch.object(data_pipeline, "CHINA_SEAS_BBOX", {"lat_min": 0.0, "lat_max": 0.5, "lon_min": 100.0, "lon_max": 100.5}),
        ):
            out, _ = data_pipeline.build_prediction_features(
                "Thunnus albacares",
                pd.Timestamp("2025-06-01").date(),
                region="china_seas",
                resolution=0.25,
            )

        self.assertEqual(len(out), 9)  # 3 x 3 grid
        self.assertEqual(float(out["grid_lat"].min()), 0.0)
        self.assertEqual(float(out["grid_lat"].max()), 0.5)
        self.assertEqual(float(out["grid_lon"].min()), 100.0)
        self.assertEqual(float(out["grid_lon"].max()), 100.5)

    def test_build_prediction_features_timeout_returns_partial_results(self):
        hist = pd.DataFrame(
            [
                {
                    "species_name": "Thunnus albacares",
                    "event_date": pd.Timestamp("2023-06-01"),
                    "species_id": 0,
                    "grid_lat": 10.0,
                    "grid_lon": 120.0,
                    "month": 6,
                    "day": 1,
                    "sst": 26.2,
                    "wind_speed": 12.0,
                    "pressure": 1010.0,
                    "precipitation": 0.4,
                    "activity_count": 2,
                    "catch_index": 1.2,
                }
            ]
        )
        forcing = {
            "sst": 27.0,
            "wind_speed": 14.0,
            "pressure": 1011.0,
            "precipitation": 0.0,
            "provider_status": {
                "noaa": {"ok": True, "detail": ""},
                "hycom": {"ok": False, "detail": "placeholder"},
                "open_meteo": {"ok": True, "detail": ""},
            },
        }
        with (
            patch("backend.data_pipeline.load_or_build_features", return_value=hist),
            patch("backend.data_pipeline._fetch_ocean_forcing_row", return_value=forcing),
            patch.object(data_pipeline, "FORCING_FETCH_TIMEOUT_SECONDS", 0.0),
            patch.object(data_pipeline, "CHINA_SEAS_BBOX", {"lat_min": 0.0, "lat_max": 1.0, "lon_min": 100.0, "lon_max": 101.0}),
        ):
            out, meta = data_pipeline.build_prediction_features(
                "Thunnus albacares",
                pd.Timestamp("2025-06-01").date(),
                region="china_seas",
                resolution=0.5,
                anchor_step=0.5,
            )

        self.assertGreater(len(out), 0)
        self.assertTrue(meta["providers"]["partial_results"])
        self.assertEqual(meta["providers"]["fetched_anchors"], 0)


if __name__ == "__main__":
    unittest.main()
