from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend import high_precision_modelling as hpm


class _FakeLGBMClassifier:
    def __init__(self, **kwargs):
        self.device_type = kwargs.get("device_type")

    def fit(self, x, y):
        if self.device_type == "gpu":
            raise RuntimeError("gpu unavailable")
        return self

    def predict_proba(self, x):
        p = np.full(len(x), 0.7)
        return np.vstack([1 - p, p]).T


class _PredictModel:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, x):
        p = np.full(len(x), self.p)
        return np.vstack([1 - p, p]).T


class HighPrecisionModellingTests(unittest.TestCase):
    def test_model_name(self):
        self.assertEqual(hpm.model_name(), "高精度modelling")

    def test_build_hotspot_labels_from_cpue(self):
        df = pd.DataFrame(
            {
                "species_name": ["SKJ", "SKJ", "SKJ", "YFT"],
                "region": ["A", "A", "A", "B"],
                "month": [6, 6, 6, 6],
                "cpue": [1.0, 2.0, 3.0, 1.0],
            }
        )
        out = hpm.build_hotspot_labels(df, quantile=0.7)
        self.assertIn("hotspot", out.columns)
        self.assertGreaterEqual(out["hotspot"].sum(), 1)

    def test_train_and_predict_pipeline(self):
        df = pd.DataFrame(
            {
                "f1": np.linspace(0, 1, 40),
                "f2": np.linspace(1, 2, 40),
                "hotspot": [0, 1] * 20,
                "grid_lat": np.linspace(10, 11, 40),
                "grid_lon": np.linspace(120, 121, 40),
            }
        )
        fake_lgb = SimpleNamespace(LGBMClassifier=_FakeLGBMClassifier)
        with patch("backend.high_precision_modelling._get_lightgbm", return_value=fake_lgb):
            artifact = hpm.train_coarse_model(df, features=["f1", "f2"], label_col="hotspot")

        self.assertEqual(len(artifact["models"]), 5)
        out = hpm.predict_hotspot_probability(df, artifact)
        self.assertIn("probability", out.columns)
        self.assertIn("uncertainty", out.columns)
        self.assertIn("confidence", out.columns)

    def test_generate_search_circles_and_hit_rate(self):
        pred = pd.DataFrame(
            {
                "grid_lat": [10.0, 10.01, 11.0],
                "grid_lon": [120.0, 120.01, 121.0],
                "probability": [0.9, 0.8, 0.7],
                "confidence": [0.8, 0.7, 0.6],
            }
        )
        circles = hpm.generate_search_circles(pred, top_k=2, radius_nm=30)
        self.assertGreaterEqual(len(circles), 1)

        truth = pd.DataFrame({"lat": [10.02, 13.0], "lon": [120.02, 130.0]})
        hit = hpm.hit_at_30nm(circles, truth)
        self.assertGreaterEqual(hit, 0.0)
        self.assertLessEqual(hit, 1.0)

    def test_select_candidates(self):
        pred = pd.DataFrame({"probability": [0.2, 0.8, 0.6]})
        out = hpm.select_coarse_candidates(pred, top_k=2, min_prob=0.5)
        self.assertEqual(len(out), 2)

    def test_predict_uncertainty_with_manual_models(self):
        df = pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 2.0]})
        artifact = {"features": ["f1", "f2"], "models": [_PredictModel(0.2), _PredictModel(0.8)]}
        out = hpm.predict_hotspot_probability(df, artifact)
        np.testing.assert_allclose(out["probability"].values, np.array([0.5, 0.5]))
        self.assertTrue((out["uncertainty"] > 0).all())


if __name__ == "__main__":
    unittest.main()
