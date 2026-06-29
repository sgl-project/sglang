"""Prometheus metrics for Double Sparsity.

Exposes ``sglang_double_sparsity_channel_mask_valid`` — a per-rank Gauge set to
``1`` once the channel mask file has been loaded and validated.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "sglang_double_sparsity"

_lock = threading.Lock()
_metrics_registered: bool = False
_metric_objs: Dict[str, Any] = {}


def _try_register() -> None:
    """Lazy-register the gauge on first use (prometheus_client may be absent in
    CLI / calibration processes; the hot path stays import-light)."""
    global _metrics_registered
    if _metrics_registered:
        return
    with _lock:
        if _metrics_registered:
            return
        try:
            from prometheus_client import Gauge
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
        _metrics_registered = True


def mark_channel_mask_valid(valid: bool) -> None:
    _try_register()
    obj = _metric_objs.get("channel_mask_valid")
    if obj is not None:
        obj.set(1 if valid else 0)


def reset_for_testing() -> None:
    """Clear registered metric state and unregister the collectors so a later
    ``_try_register()`` does not raise ``Duplicated timeseries``. Tests only."""
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
