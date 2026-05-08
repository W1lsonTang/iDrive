"""Inference wrapper for physiological drowsiness models."""

from __future__ import annotations

from typing import Dict, Iterable, Union

import joblib
import numpy as np

try:
    from .features import FEATURE_COLUMNS, feature_vector_from_dict
except ImportError:  # Allow direct file execution: python physio_channel/model.py
    from features import FEATURE_COLUMNS, feature_vector_from_dict

FeatureInput = Union[Dict[str, float], Iterable[float], np.ndarray]


class PhysioPredictor:
    """Unified predictor API for physiological models."""

    def __init__(self, model_path: str, scaler_path: str) -> None:
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def _to_vector(self, features: FeatureInput) -> np.ndarray:
        if isinstance(features, dict):
            missing = [name for name in FEATURE_COLUMNS if name not in features]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            return feature_vector_from_dict(features)

        arr = np.asarray(list(features) if not isinstance(features, np.ndarray) else features, dtype=np.float64)
        arr = arr.reshape(-1)
        if arr.size != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Expected {len(FEATURE_COLUMNS)} features, got {arr.size}. "
                f"Order should match: {FEATURE_COLUMNS}"
            )
        return arr

    def predict_proba(self, features: FeatureInput) -> float:
        """Return drowsy probability in [0, 1]."""
        x = self._to_vector(features).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(x_scaled)[0, 1])

        if hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(x_scaled)[0])
            return float(1.0 / (1.0 + np.exp(-score)))

        pred = int(self.model.predict(x_scaled)[0])
        return float(pred)

    def predict(self, features: FeatureInput, threshold: float = 0.5) -> int:
        """Return binary prediction where 1 means drowsy."""
        return int(self.predict_proba(features) >= threshold)


if __name__ == "__main__":
    print(
        "PhysioPredictor module loaded successfully.\n"
        "Use it via imports, or run project tools such as:\n"
        "  python -m physio_channel.train_dd_database\n"
        "  python -m physio_channel.simulator --edf DD-Database/01M_1.edf"
    )
