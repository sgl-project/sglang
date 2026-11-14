"""ML-trained per-iter latency predictor.

Loads a joblib pickle of a sklearn-compatible regressor and predicts forward latency
from batch composition features. Train one with `sglang-simulator-train`.

sim_config.json usage:
    "predictor": {
        "name": "ml",
        "database_path": "/path/to/latency_model.pkl"
    }
"""

import os

import joblib
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class MLTimePredictor(InferTimePredictor):
    """Predict forward latency using the shared 18-feature batch representation."""

    # Feature order is shared by training and inference, regardless of regressor.
    FEATURE_NAMES = list(ML_FEATURE_NAMES)

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        latency_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        database_path = os.path.expandvars(os.path.expanduser(database_path))
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"MLTimePredictor database_path not found: {database_path}. "
                "Train one with `sglang-simulator-train` first."
            )

        bundle = joblib.load(database_path)
        if (
            not isinstance(bundle, dict)
            or "model" not in bundle
            or "features" not in bundle
        ):
            raise ValueError(
                "MLTimePredictor requires a joblib bundle containing both "
                "'model' and ordered 'features' metadata"
            )
        self._model = bundle["model"]
        saved_features = list(bundle["features"])

        if saved_features != self.FEATURE_NAMES:
            raise ValueError(
                "MLTimePredictor feature contract mismatch: "
                f"saved={saved_features}, expected={self.FEATURE_NAMES}. "
                "Retrain or export the model with the exact 18-feature ABI."
            )
        if not callable(getattr(self._model, "predict", None)):
            raise TypeError(
                "MLTimePredictor model must expose a callable predict() method"
            )

        self._latency_scale = float(latency_scale)
        logger.info(
            "MLTimePredictor loaded from %s (model=%s, n_features=%d, latency_scale=%.4f)",
            database_path,
            type(self._model).__name__,
            len(saved_features),
            self._latency_scale,
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        feats = extract_ml_features(exts, pasts)

        return float(self._model.predict([feats])[0]) * self._latency_scale
