from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend import modeling


class _FakeLGBMRegressor:
    def __init__(self, **kwargs):
        self.device_type = kwargs.get("device_type")
        self.fitted = False

    def fit(self, X, y):
        if self.device_type == "gpu":
            raise RuntimeError("gpu unavailable")
        self.fitted = True
        return self

    def predict(self, X):
        return np.full(len(X), 1.5)


class _PredictModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class ModelingTests(unittest.TestCase):
    def _sample_training_df(self):
        rows = []
        for i in range(12):
            rows.append(
                {
                    "species_id": i % 3,
                    "grid_lat": float(i),
                    "grid_lon": float(-i),
                    "month": 6 + (i % 2),
                    "day": 1 + i,
                    "sst": 25.0 + (i % 2),
                    "wind_speed": 12.0 + i,
                    "pressure": 1008.0 + (i % 3),
                    "precipitation": float(i % 4),
                    "activity_count": 1 + (i % 5),
                    "catch_index": 0.5 + i * 0.1,
                }
            )
        return pd.DataFrame(rows)

    def test_train_model_falls_back_to_cpu_and_saves_artifact(self):
        df = self._sample_training_df()
        fake_lgb_module = SimpleNamespace(LGBMRegressor=_FakeLGBMRegressor)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_dir = Path(tmpdir) / "models"
            tmp_model_path = tmp_model_dir / "artifact.joblib"
            captured = {}

            def _fake_dump(artifact, path):
                captured["artifact"] = artifact
                captured["path"] = path

            with (
                patch.object(modeling, "MODEL_DIR", tmp_model_dir),
                patch.object(modeling, "MODEL_PATH", tmp_model_path),
                patch("backend.modeling._get_lightgbm", return_value=fake_lgb_module),
                patch("backend.modeling.joblib.dump", side_effect=_fake_dump),
            ):
                metrics = modeling.train_model(df)

        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertEqual(captured["path"], tmp_model_path)
        self.assertEqual(len(captured["artifact"]["models"]), 5)
        self.assertTrue(all(m.device_type == "cpu" for m in captured["artifact"]["models"]))

    def test_load_model_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_path = Path(tmpdir) / "missing.joblib"
            with patch.object(modeling, "MODEL_PATH", tmp_model_path):
                with self.assertRaises(FileNotFoundError):
                    modeling.load_model()

    def test_load_model_reads_saved_artifact(self):
        expected = {"models": [], "features": [], "metrics": {"mae": 1.0, "r2": 0.0}}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_path = Path(tmpdir) / "artifact.joblib"
            tmp_model_path.write_text("placeholder", encoding="utf-8")
            with (
                patch.object(modeling, "MODEL_PATH", tmp_model_path),
                patch("backend.modeling.joblib.load", return_value=expected),
            ):
                out = modeling.load_model()
        self.assertEqual(out, expected)

    def test_predict_with_confidence_computes_mean_and_confidence(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        artifact = {
            "features": ["a", "b"],
            "models": [_PredictModel(2.0), _PredictModel(4.0)],
        }

        pred, conf = modeling.predict_with_confidence(df, artifact)
        np.testing.assert_allclose(pred, np.array([3.0, 3.0]))
        np.testing.assert_allclose(conf, np.array([1.0 / 2.0, 1.0 / 2.0]))


if __name__ == "__main__":
    unittest.main()
