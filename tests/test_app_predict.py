from __future__ import annotations

import unittest
import types
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

if "backend.modeling" not in sys.modules:
    fake_modeling = types.ModuleType("backend.modeling")
    fake_modeling.load_model = lambda: {"features": []}
    fake_modeling.predict_with_confidence = lambda df, artifact: (np.zeros(len(df)), np.ones(len(df)))
    fake_modeling.train_model = lambda df: {}
    sys.modules["backend.modeling"] = fake_modeling

from backend.app import app


class AppPredictTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_forecast_mode_returns_dense_shape(self):
        features = pd.DataFrame(
            [
                {
                    "grid_lat": 10.0,
                    "grid_lon": 120.0,
                    "event_date": pd.Timestamp("2025-06-01"),
                    "sst": 26.0,
                    "wind_speed": 12.0,
                    "pressure": 1010.0,
                    "precipitation": 0.2,
                    "species_id": 0,
                    "month": 6,
                    "day": 1,
                    "activity_count": 2.0,
                },
                {
                    "grid_lat": 10.25,
                    "grid_lon": 120.25,
                    "event_date": pd.Timestamp("2025-06-01"),
                    "sst": 26.5,
                    "wind_speed": 13.0,
                    "pressure": 1011.0,
                    "precipitation": 0.1,
                    "species_id": 0,
                    "month": 6,
                    "day": 1,
                    "activity_count": 1.0,
                },
            ]
        )
        provider_meta = {
            "bbox": {"lat_min": 5.0, "lat_max": 45.0, "lon_min": 105.0, "lon_max": 145.0},
            "cache_hit": True,
            "anchors": 4,
            "providers": {"noaa_ok_ratio": 0.75, "hycom_ok_ratio": 0.0, "partial_results": False},
        }
        with (
            patch("backend.app.build_prediction_features", return_value=(features, provider_meta)),
            patch("backend.app.load_model", return_value={"features": []}),
            patch("backend.app.predict_with_confidence", return_value=(np.array([1.0, 0.5]), np.array([0.9, 0.8]))),
        ):
            response = self.client.get(
                "/api/predict",
                params={"species": "yellowfin", "predict_date": "2025-06-01", "mode": "forecast_map", "limit": 0},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["meta"]["mode"], "forecast_map")
        self.assertEqual(payload["meta"]["row_count"], 2)
        self.assertEqual(payload["meta"]["returned_points"], 2)
        self.assertIn("source_health", payload["meta"])
        self.assertEqual(len(payload["points"]), 2)

    def test_predict_rejects_invalid_mode(self):
        response = self.client.get("/api/predict", params={"mode": "bad_mode"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("mode must be one of", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
