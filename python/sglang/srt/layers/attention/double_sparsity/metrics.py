"""Prometheus metrics + per-request meta_info for Double Sparsity.

All metric names live under the ``sglang_double_sparsity_*`` namespace per
(deliberately distinct from any ``sglang_hisparse_*`` namespace,
because DS is standalone).

Surfaces:

* ``sglang_double_sparsity_channel_mask_valid`` — Gauge per rank, ``1`` once
  the channel mask file has been loaded and validated, ``0`` otherwise.
* ``sglang_double_sparsity_selected_tokens_sum`` — Counter accumulating the
  count of tokens selected across batches; pair with ``_count`` for the per-
  batch average.
* ``sglang_double_sparsity_selected_tokens_count`` — Counter incrementing
  once per (request, layer, step) selection call.
* ``sglang_double_sparsity_sparsity_rate`` — Histogram of selected /
  total-valid token ratios. Sampled per call.

Per-request fields surfaced via :func:`meta_info_for_request`:

* ``sparsity_rate``: float in ``[0, 1]``
* ``selected_tokens``: int, count of selected tokens.
* ``total_tokens``: int, the true sequence length.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "sglang_double_sparsity"

_lock = threading.Lock()
_metrics_registered: bool = False
_metric_objs: Dict[str, Any] = {}


def _try_register() -> None:
    """Lazy-register metrics with prometheus_client when first used.

    SGLang's metrics infrastructure may or may not be active in any given
    process (CLI tests, calibration scripts, etc.). The lazy path keeps the
    selection hot path import-light.
    """

    global _metrics_registered
    if _metrics_registered:
        return
    with _lock:
        if _metrics_registered:
            return
        try:
            from prometheus_client import Counter, Gauge, Histogram
        except ImportError:
            logger.debug(
                "prometheus_client not available; Double Sparsity metrics are no-ops."
            )
            _metrics_registered = True
            return

        _metric_objs["channel_mask_valid"] = Gauge(
            f"{_METRIC_PREFIX}_channel_mask_valid",
            "1 once the Double Sparsity channel mask file has been loaded and validated.",
        )
        _metric_objs["selected_tokens_sum"] = Counter(
            f"{_METRIC_PREFIX}_selected_tokens_sum",
            "Total count of tokens selected across all DS top-K calls.",
        )
        _metric_objs["selected_tokens_count"] = Counter(
            f"{_METRIC_PREFIX}_selected_tokens_count",
            "Number of DS top-K selection calls.",
        )
        _metric_objs["sparsity_rate"] = Histogram(
            f"{_METRIC_PREFIX}_sparsity_rate",
            "Selected-tokens / total-valid-tokens ratio per DS top-K call.",
            buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
        )
        _metrics_registered = True


def mark_channel_mask_valid(valid: bool) -> None:
    _try_register()
    obj = _metric_objs.get("channel_mask_valid")
    if obj is not None:
        obj.set(1 if valid else 0)


@dataclass
class DoubleSparsityRequestStats:
    sparsity_rate: float  # pruned fraction: 1 - selected_tokens / total_tokens
    selected_tokens: int
    total_tokens: int
    dense_fallback: int = 0


def meta_info_for_request(stats: DoubleSparsityRequestStats) -> Dict[str, Any]:
    """Translate per-request DS stats into the dict surfaced via meta_info.

    Caller wires this into ``ScheduleBatch`` so the values appear in the
    request's ``meta_info`` payload alongside existing fields. ``total_tokens``
    is published EXPLICITLY (the true sequence length) so consumers do not have
    to invert ``sparsity_rate`` (which is the pruned fraction, not selected/total).
    """

    return {
        "sparsity_rate": float(stats.sparsity_rate),
        "selected_tokens": int(stats.selected_tokens),
        "total_tokens": int(stats.total_tokens),
        "dense_fallback": int(stats.dense_fallback),
    }


def reset_for_testing() -> None:
    """Clear registered metric state. Tests only.

    Also unregisters the underlying collectors from the default
    ``prometheus_client.REGISTRY`` so a subsequent ``_try_register()`` does
    not raise ``ValueError: Duplicated timeseries``. The ``unregister`` call
    is best-effort: ``KeyError`` is suppressed for collectors that were
    never registered (e.g. when prometheus_client is unavailable).
    """

    global _metrics_registered
    with _lock:
        try:
            from prometheus_client import REGISTRY
        except ImportError:
            REGISTRY = None
        if REGISTRY is not None:
            for obj in list(_metric_objs.values()):
                try:
                    REGISTRY.unregister(obj)
                except KeyError:
                    pass
        _metric_objs.clear()
        _metrics_registered = False
