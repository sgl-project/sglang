"""ML-trained per-iter latency predictor.

Loads a joblib pickle of a sklearn-compatible regressor and predicts forward latency
from batch composition features. Train one with `sglang-simulator-train`.

Two independent choices pick the regressor:

- By instance role, which bundle file is loaded. A PD prefill instance
  (``disaggregation_mode=prefill``) loads ``prefill_database_path`` when it is
  set; every other instance loads ``database_path``.
- By batch phase, which regressor inside the bundle runs. A bundle holds
  ordered ``features`` and either one ``model`` or separate ``prefill`` and
  ``decode`` regressors. A batch whose requests all extend by one token uses
  ``decode`` and is scaled by ``decode_scale_factor``; every other batch,
  including a mixed one, uses ``prefill`` and ``prefill_scale_factor``. Both
  are also scaled by ``latency_scale``.

sim_config.json usage:
    "predictor": {
        "name": "ml",
        "database_path": "/path/to/latency_model.pkl",
        "prefill_database_path": "/path/to/prefill_model.pkl"
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
        prefill_scale_factor: float = 1.0,
        decode_scale_factor: float = 1.0,
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
        if not isinstance(bundle, dict) or "features" not in bundle:
            raise ValueError(
                "MLTimePredictor requires a joblib bundle containing ordered "
                "'features' metadata"
            )
        if "prefill" in bundle and "decode" in bundle:
            self._prefill_model = bundle["prefill"]
            self._decode_model = bundle["decode"]
        elif "model" in bundle:
            self._prefill_model = self._decode_model = bundle["model"]
        else:
            raise ValueError(
                "MLTimePredictor bundle must contain 'model' or both "
                "'prefill' and 'decode'"
            )
        saved_features = list(bundle["features"])

        if saved_features != self.FEATURE_NAMES:
            raise ValueError(
                "MLTimePredictor feature contract mismatch: "
                f"saved={saved_features}, expected={self.FEATURE_NAMES}. "
                "Retrain or export the model with the exact 18-feature ABI."
            )
        for regressor in (self._prefill_model, self._decode_model):
            if not callable(getattr(regressor, "predict", None)):
                raise TypeError(
                    "MLTimePredictor model must expose a callable predict() method"
                )

        # Prefill and decode are corrected separately: one global scale would
        # trade accuracy on one phase for the other.
        self._prefill_scale = float(latency_scale) * float(prefill_scale_factor)
        self._decode_scale = float(latency_scale) * float(decode_scale_factor)
        logger.info(
            "MLTimePredictor loaded from %s (prefill=%s, decode=%s, n_features=%d, "
            "prefill_scale=%.4f, decode_scale=%.4f)",
            database_path,
            type(self._prefill_model).__name__,
            type(self._decode_model).__name__,
            len(saved_features),
            self._prefill_scale,
            self._decode_scale,
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        feats = extract_ml_features(exts, pasts)

        if all(e == 1 for e in exts):
            regressor, scale = self._decode_model, self._decode_scale
        else:
            regressor, scale = self._prefill_model, self._prefill_scale
        return float(regressor.predict([feats])[0]) * scale
