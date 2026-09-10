# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for multimodal preprocessing in the tokenizer manager.

In the default (co-located) deployment, all multimodal input handling --
downloading media from URLs, base64/PIL/video/audio decoding, and the HF
processor call -- happens in the tokenizer manager process *before* the
request reaches the scheduler. Scheduler-side latency metrics (TTFT, e2e,
queue wait) only start counting afterwards, so without the metrics in this
module the multimodal preprocessing time is invisible and shows up only as
inflated perceived queueing / TTFT.

SGLang serves /metrics via prometheus_client multiprocess mode
(``PROMETHEUS_MULTIPROC_DIR`` + ``MultiProcessCollector``), so Histograms
registered in the default registry of this process are exported
automatically.

All metrics are lazily initialized on the first observation and only when
the server is started with ``--enable-metrics`` -- there are no import-time
side effects and no overhead when metrics are disabled.
"""

import threading

_metrics_lock = threading.Lock()
_metrics_initialized = False
_metrics_enabled = False

_media_download_seconds = None
_media_download_bytes = None
_media_load_seconds = None
_load_data_seconds = None
_processor_seconds = None

# Per-item latencies: an item is usually fetched+decoded in tens of
# milliseconds, but large videos over slow links can take tens of seconds.
_ITEM_LATENCY_BUCKETS = [
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
]

# Per-request stage latencies (all items of one request, or the HF processor
# call for one request).
_STAGE_LATENCY_BUCKETS = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
]

_SIZE_BUCKETS = [
    1024,
    10240,
    102400,
    524288,
    1048576,
    5242880,
    10485760,
    52428800,
    104857600,
]


def _metrics_turned_on() -> bool:
    """Return True iff the server was started with --enable-metrics.

    Returns False (instead of raising) when the global server args have not
    been published in this process (unit tests, offline tools).
    """
    try:
        from sglang.srt.runtime_context import get_server_args

        return bool(get_server_args().enable_metrics)
    except Exception:
        return False


def _init_metrics(enabled: bool, registry=None) -> None:
    """(Re)initialize the metric objects. Holds no lock; callers synchronize."""
    global _media_download_seconds, _media_download_bytes, _media_load_seconds
    global _load_data_seconds, _processor_seconds
    global _metrics_initialized, _metrics_enabled

    _media_download_seconds = None
    _media_download_bytes = None
    _media_load_seconds = None
    _load_data_seconds = None
    _processor_seconds = None

    if enabled:
        from prometheus_client import Histogram

        _media_download_seconds = Histogram(
            name="sglang:mm_media_download_seconds",
            documentation=(
                "Histogram of HTTP(S) media download latency in seconds "
                "(per media item)."
            ),
            buckets=_ITEM_LATENCY_BUCKETS,
            labelnames=["modality"],
            registry=registry,
        )
        _media_download_bytes = Histogram(
            name="sglang:mm_media_download_bytes",
            documentation=(
                "Histogram of downloaded media size in bytes (per media item)."
            ),
            buckets=_SIZE_BUCKETS,
            labelnames=["modality"],
            registry=registry,
        )
        _media_load_seconds = Histogram(
            name="sglang:mm_media_load_seconds",
            documentation=(
                "Histogram of the full load latency of one media item in "
                "seconds (download + decode). Subtract "
                "sglang:mm_media_download_seconds to isolate decode time."
            ),
            buckets=_ITEM_LATENCY_BUCKETS,
            labelnames=["modality"],
            registry=registry,
        )
        _load_data_seconds = Histogram(
            name="sglang:mm_load_data_seconds",
            documentation=(
                "Histogram of the media loading stage latency in seconds "
                "(all items of one request, loaded concurrently)."
            ),
            buckets=_STAGE_LATENCY_BUCKETS,
            labelnames=[],
            registry=registry,
        )
        _processor_seconds = Histogram(
            name="sglang:mm_processor_seconds",
            documentation=(
                "Histogram of the HF multimodal processor call latency in "
                "seconds (per request)."
            ),
            buckets=_STAGE_LATENCY_BUCKETS,
            labelnames=[],
            registry=registry,
        )

    _metrics_enabled = enabled
    _metrics_initialized = True


def _ensure_metrics() -> bool:
    """Initialize metrics on first call; return True iff they are enabled."""
    global _metrics_initialized, _metrics_enabled
    if _metrics_initialized:
        return _metrics_enabled
    with _metrics_lock:
        if _metrics_initialized:
            return _metrics_enabled
        try:
            _init_metrics(enabled=_metrics_turned_on())
        except Exception:
            # Never let metrics break request serving.
            _metrics_initialized = True
            _metrics_enabled = False
    return _metrics_enabled


def _reset_for_test(enabled: bool = True, registry=None) -> None:
    """Reinitialize metrics with an explicit state. For unit tests only."""
    with _metrics_lock:
        _init_metrics(enabled=enabled, registry=registry)


def observe_mm_media_download(
    modality: str, elapsed_seconds: float, num_bytes: int
) -> None:
    """Record the HTTP(S) download latency and size of one media item."""
    if not _ensure_metrics():
        return
    _media_download_seconds.labels(modality=modality).observe(elapsed_seconds)
    _media_download_bytes.labels(modality=modality).observe(num_bytes)


def observe_mm_media_load(modality: str, elapsed_seconds: float) -> None:
    """Record the full load (download + decode) latency of one media item."""
    if not _ensure_metrics():
        return
    _media_load_seconds.labels(modality=modality).observe(elapsed_seconds)


def observe_mm_load_data(elapsed_seconds: float) -> None:
    """Record the media loading stage latency of one request (all items)."""
    if not _ensure_metrics():
        return
    _load_data_seconds.observe(elapsed_seconds)


def observe_mm_processor(elapsed_seconds: float) -> None:
    """Record the HF multimodal processor call latency of one request."""
    if not _ensure_metrics():
        return
    _processor_seconds.observe(elapsed_seconds)
