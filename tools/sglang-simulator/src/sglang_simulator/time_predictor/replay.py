"""Oracle lookup predictor — replays real GPU iter_latency from a pre-built table.

Replay is a diagnostic predictor for separating latency-prediction error from
scheduler, cache, and simulator behavior. A replay table maps a JSON-encoded,
sorted list of ``[extend_input_length, prefix_length]`` pairs to measured
iteration latency in seconds.

Example simulator configuration:
    "predictor": {
        "name": "replay",
        "database_path": "/path/to/replay_table.json",
        "miss_strategy": "knn",       # "zero" (legacy) or "knn" (interpolated)
        "miss_knn_k": 3,              # KNN k for "knn" strategy
        "miss_fallback_seconds": 0.0  # used only when strategy=="zero"
    }
"""

import json
import math
import os

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


def _decode_key(key: str):
    """Parse a sorted-tuple lookup key back to list of (extend, past) pairs."""
    return [tuple(pair) for pair in json.loads(key)]


def _shape_feat(extends, pasts):
    """3-D feature for KNN: (batch_size, sum_extend, sum_past).

    Low-dim and aligned with the dominant latency drivers; avoids the curse of
    dimensionality on the typically small (~1-3K) entries per replay table.
    """
    return (len(extends), sum(extends), sum(pasts))


class ReplayTimePredictor(InferTimePredictor):
    """Oracle lookup predictor: returns real GPU iter_latency for matching batch compositions.

    Compositions are matched exactly by sorted (extend_len, past_kv_len) tuples.
    On lookup miss the behavior is controlled by `miss_strategy`:

      - "zero" (default, legacy): returns `miss_fallback_seconds` (default 0.0).
        Use for "is the gap NOT from the predictor?" diagnostic. Only meaningful
        when miss rate is small AND you accept the bias of dropping miss work.

      - "knn": KNN-interpolated latency from the k batches in the table with
        the closest (batch_size, sum_extend, sum_past) shape (per-dim z-scored
        Euclidean distance). Use when miss rate matters or you want a logically
        complete oracle. Typical k=3.

    For workloads where sim batch composition diverges from real (e.g., bursty
    max-tps), the miss rate can be high — always check the post-run hit ratio.
    """

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        miss_fallback_seconds: float = 0.0,
        miss_strategy: str = "zero",
        miss_knn_k: int = 3,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"ReplayTimePredictor database_path not found: {database_path}"
            )
        with open(database_path) as f:
            self._table = json.load(f)
        if miss_strategy not in ("zero", "knn"):
            raise ValueError(
                f"miss_strategy must be 'zero' or 'knn', got {miss_strategy!r}"
            )
        self._miss_strategy = miss_strategy
        self._miss_fallback = float(miss_fallback_seconds)
        self._miss_knn_k = int(miss_knn_k)
        self._hits = 0
        self._misses = 0

        if self._miss_strategy == "knn":
            self._prep_knn_index()
            logger.info(
                "ReplayTimePredictor loaded %d unique compositions from %s "
                "(miss_strategy=knn, k=%d)",
                len(self._table),
                database_path,
                self._miss_knn_k,
            )
        else:
            self._knn_feats = None
            logger.info(
                "ReplayTimePredictor loaded %d unique compositions from %s "
                "(miss_strategy=zero, fallback=%.4fs)",
                len(self._table),
                database_path,
                self._miss_fallback,
            )

    def _prep_knn_index(self):
        """Build per-feature mean/std + cached (feat, lat) arrays for KNN fallback.

        Uses plain Python (no numpy / sklearn dependency from the predictor side)
        — table size is ~1K-3K so this is fine. Distance is z-scored Euclidean
        across (batch_size, sum_extend, sum_past).
        """
        feats = []
        lats = []
        for key, lat in self._table.items():
            extends_pasts = _decode_key(key)
            extends = [e for e, _ in extends_pasts]
            pasts = [p for _, p in extends_pasts]
            feats.append(_shape_feat(extends, pasts))
            lats.append(float(lat))
        n = len(feats)
        # per-dim mean/std for z-scoring
        means = [sum(f[d] for f in feats) / n for d in range(3)]
        var = [sum((f[d] - means[d]) ** 2 for f in feats) / n for d in range(3)]
        stds = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
        self._knn_feats = feats
        self._knn_lats = lats
        self._knn_mean = means
        self._knn_std = stds

    def _knn_predict(self, query_feat):
        """k nearest neighbors over z-scored 3-D shape feature, simple mean."""
        qz = tuple(
            (query_feat[d] - self._knn_mean[d]) / self._knn_std[d] for d in range(3)
        )
        # compute squared distance to every table entry; pick smallest k
        # n ≤ ~3K so O(n) per query is fine; sim only calls on misses
        dists = []
        for i, f in enumerate(self._knn_feats):
            fz = (
                (f[0] - self._knn_mean[0]) / self._knn_std[0],
                (f[1] - self._knn_mean[1]) / self._knn_std[1],
                (f[2] - self._knn_mean[2]) / self._knn_std[2],
            )
            d2 = (fz[0] - qz[0]) ** 2 + (fz[1] - qz[1]) ** 2 + (fz[2] - qz[2]) ** 2
            dists.append((d2, i))
        dists.sort(key=lambda t: t[0])
        k = min(self._miss_knn_k, len(dists))
        return sum(self._knn_lats[i] for _, i in dists[:k]) / k

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        extends = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]
        key = json.dumps(
            sorted([req.extend_length, req.past_kv_length] for req in batch.reqs)
        )
        v = self._table.get(key)
        if v is None:
            self._misses += 1
            if self._miss_strategy == "knn":
                return self._knn_predict(_shape_feat(extends, pasts))
            return self._miss_fallback
        self._hits += 1
        return float(v)

    def get_metrics(self) -> dict:
        total = self._hits + self._misses
        return {
            "replay_exact_match_steps": self._hits,
            "replay_miss_steps": self._misses,
            "replay_zero_fallback_steps": (
                self._misses if self._miss_strategy == "zero" else 0
            ),
            "replay_knn_fallback_steps": (
                self._misses if self._miss_strategy == "knn" else 0
            ),
            "replay_fallback_rate": self._misses / total if total else 0.0,
        }

    def reset_metrics(self) -> None:
        self._hits = 0
        self._misses = 0
