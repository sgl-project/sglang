"""Prometheus metrics + per-request meta_info for Double Sparsity.

All metric names live under the ``sglang_double_sparsity_*`` namespace per
AC-10 (deliberately distinct from any ``sglang_hisparse_*`` namespace,
because DS is standalone).

Surfaces:

* ``sglang_double_sparsity_channel_mask_valid`` — Gauge per rank, ``1`` once
  the channel mask file has been loaded and validated, ``0`` otherwise.
* ``sglang_double_sparsity_dense_fallback_total`` — Counter; ``0`` in
  healthy runs. Fault-injection paths in tests can drive this without
  introducing a production fallback.
* ``sglang_double_sparsity_selected_pages_sum`` — Counter accumulating the
  count of pages selected across batches; pair with ``_count`` for the per-
  batch average.
* ``sglang_double_sparsity_selected_pages_count`` — Counter incrementing
  once per (request, layer, step) selection call.
* ``sglang_double_sparsity_sparsity_rate`` — Histogram of selected /
  total-valid page ratios. Sampled per call.

Per-request fields surfaced via :func:`meta_info_for_request`:

* ``sparsity_rate``: float in ``[0, 1]``
* ``selected_pages``: int, count of distinct pages used
* ``dense_fallback``: ``0`` healthy, ``1`` on fault-injected fallback
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
        _metric_objs["dense_fallback_total"] = Counter(
            f"{_METRIC_PREFIX}_dense_fallback_total",
            "Number of decode steps that fell back to dense attention. 0 in healthy runs.",
        )
        _metric_objs["selected_pages_sum"] = Counter(
            f"{_METRIC_PREFIX}_selected_pages_sum",
            "Total count of pages selected across all DS top-K calls.",
        )
        _metric_objs["selected_pages_count"] = Counter(
            f"{_METRIC_PREFIX}_selected_pages_count",
            "Number of DS top-K selection calls.",
        )
        _metric_objs["sparsity_rate"] = Histogram(
            f"{_METRIC_PREFIX}_sparsity_rate",
            "Selected-pages / total-valid-pages ratio per DS top-K call.",
            buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
        )
        _metrics_registered = True


def mark_channel_mask_valid(valid: bool) -> None:
    _try_register()
    obj = _metric_objs.get("channel_mask_valid")
    if obj is not None:
        obj.set(1 if valid else 0)


def increment_dense_fallback(n: int = 1) -> None:
    _try_register()
    obj = _metric_objs.get("dense_fallback_total")
    if obj is not None and n > 0:
        obj.inc(n)


def record_selection(
    *,
    selected_pages: int,
    total_valid_pages: int,
) -> None:
    """Record one DS top-K selection call.

    ``selected_pages`` is the number of non-padding entries in ``selected_indices``;
    ``total_valid_pages`` is the number of currently-valid pages in the request's
    KV window. ``sparsity_rate`` is the ratio.
    """

    _try_register()
    sps = _metric_objs.get("selected_pages_sum")
    cnt = _metric_objs.get("selected_pages_count")
    rate = _metric_objs.get("sparsity_rate")
    if sps is not None:
        sps.inc(int(selected_pages))
    if cnt is not None:
        cnt.inc(1)
    if rate is not None and total_valid_pages > 0:
        rate.observe(float(selected_pages) / float(total_valid_pages))


@dataclass
class DoubleSparsityRequestStats:
    sparsity_rate: float
    selected_pages: int
    dense_fallback: int = 0


def meta_info_for_request(stats: DoubleSparsityRequestStats) -> Dict[str, Any]:
    """Translate per-request DS stats into the dict surfaced via meta_info.

    Caller wires this into ``ScheduleBatch`` so the values appear in the
    request's ``meta_info`` payload alongside existing fields.
    """

    return {
        "sparsity_rate": float(stats.sparsity_rate),
        "selected_pages": int(stats.selected_pages),
        "dense_fallback": int(stats.dense_fallback),
    }


def customized_info_for_request(stats: DoubleSparsityRequestStats) -> Dict[str, Any]:
    """Payload to drop into ``recv_obj.customized_info["double_sparsity"]``.

    The tokenizer manager's ``customized_info`` hook (see
    ``tokenizer_manager.py`` near line 1739) auto-surfaces arbitrary
    k/v pairs into the request's ``meta_info``. Wiring this from the
    scheduler is the least-invasive integration point and does not
    require touching the central ``meta_info`` constructor.

    Integration shape (one line in scheduler):

        recv_obj.customized_info.setdefault({}).update({
            "double_sparsity": customized_info_for_request(stats),
        })
    """

    return meta_info_for_request(stats)


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


# ---------------------------------------------------------------------------
# Error taxonomy (AC-3 observability)
# ---------------------------------------------------------------------------

DS_ERROR_CLASSES = (
    "bad_mask",
    "bad_adapter_input",
    "selector_runtime_error",
    "rank_mismatch",
)


def record_error(cls: str, *, message: str = "", request_id: str = "") -> None:
    """Record a DS production failure.

    Increments the Prometheus counter `sglang_double_sparsity_errors_total{cls}`
    (best-effort; silent when `prometheus_client` is unavailable) and emits a
    WARNING-level structured log line so operators see the failure class,
    request_id, and message.

    Raises:
        ValueError: when `cls` is not one of `DS_ERROR_CLASSES`. Callers
        must use the named classes to keep the label set bounded.
    """
    if cls not in DS_ERROR_CLASSES:
        raise ValueError(
            f"Unknown DS error class {cls!r}; must be one of {DS_ERROR_CLASSES!r}."
        )
    try:
        from prometheus_client import Counter as _Counter
        global _DS_ERRORS_COUNTER
        try:
            counter = _DS_ERRORS_COUNTER
        except NameError:
            counter = None
        if counter is None:
            try:
                counter = _Counter(
                    "sglang_double_sparsity_errors_total",
                    "Double Sparsity production failures by class.",
                    ["cls"],
                )
                _DS_ERRORS_COUNTER = counter
            except ValueError:
                # Already registered in another import path.
                counter = None
        if counter is not None:
            counter.labels(cls=cls).inc()
    except ImportError:
        pass

    import logging as _logging
    _logger = _logging.getLogger(__name__)
    _logger.warning(
        "double_sparsity error cls=%s request_id=%s message=%s",
        cls,
        request_id,
        message,
    )

