"""
Records startup latency breakdown by context using gauge metrics in seconds
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional

logger = logging.getLogger(__name__)

enable_startup_metrics = False
STARTUP_LATENCY_SECONDS = None
# Track maximum durations for each context
_max_durations: Dict[str, float] = {}


def enable_startup_timer():
    """Initialize startup latency metrics when metrics are enabled"""
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import Gauge

    global enable_startup_metrics, STARTUP_LATENCY_SECONDS
    enable_startup_metrics = True

    STARTUP_LATENCY_SECONDS = Gauge(
        "sglang:startup_latency_breakdown_seconds_max",
        "Startup latency breakdown in seconds by context, only records the maximum duration if the context is called multiple times.",
        labelnames=["context"],
        multiprocess_mode="mostrecent",
    )


def set_startup_metric(context: str, value: float, should_log: bool = True):
    """Set the startup metric for a given context"""
    if should_log:
        logger.info(f"Setting startup metric: {context} took {value:.3f}s")

    if not enable_startup_metrics:
        return
    current_max = _max_durations.get(context, 0.0)
    if value > current_max:
        _max_durations[context] = value
        STARTUP_LATENCY_SECONDS.labels(context=context).set(value)


def reset_startup_timers():
    """Reset all recorded maximum durations. Useful for testing or reinitialization."""
    global _max_durations
    _max_durations.clear()


def get_max_duration(context: str) -> Optional[float]:
    """Get the maximum recorded duration for a context name."""
    return _max_durations.get(context)


@contextmanager
def startup_timer(name: str, log_only: bool = False) -> Generator[None, None, None]:
    """
    Context manager to measure startup latency for arbitrary code blocks.
    Only records the maximum duration if the context is called multiple times.

    Usage:
        with startup_timer("model_loading"):
            # model loading code
            model = load_model()

        with startup_timer("memory_allocation"):
            # memory setup code
            allocate_memory()
    """
    start_time = time.monotonic()
    try:
        yield
    finally:
        duration_seconds = time.monotonic() - start_time

        # Track the maximum duration for this context name
        current_max = _max_durations.get(name, 0.0)
        is_new_max = duration_seconds > current_max

        if is_new_max:
            _max_durations[name] = duration_seconds

            # Only update Prometheus gauge if this is a new maximum
            if enable_startup_metrics and not log_only:
                STARTUP_LATENCY_SECONDS.labels(context=name).set(duration_seconds)

        # Log with indication if this was a new max
        logger.info(f"Startup timing: {name} took {duration_seconds:.3f}s")


def time_startup_latency(
    func: Callable = None, name: Optional[str] = None, log_only: bool = False
) -> Callable[..., Any]:
    """
    A decorator to measure startup context latency and record it in seconds.
    Only records the maximum duration if the context is called multiple times.

    Usage:
        @time_startup_latency
        def load_model():
            # model loading code

        @time_startup_latency(name="custom_init")
        def initialize_something():
            # initialization code

        @time_startup_latency(name="debug_only", log_only=True)
        def debug_function():
            # This will only log, not record to Prometheus
    """

    def measure(func: Callable[..., Any]) -> Callable[..., Any]:
        nonlocal name
        name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_seconds = time.monotonic() - start_time

                # Track the maximum duration for this context name
                current_max = _max_durations.get(name, 0.0)
                is_new_max = duration_seconds > current_max

                if is_new_max:
                    _max_durations[name] = duration_seconds

                    # Only update Prometheus gauge if this is a new maximum
                    if enable_startup_metrics and not log_only:
                        STARTUP_LATENCY_SECONDS.labels(context=name).set(
                            duration_seconds
                        )

                # Log the timing
                logger.info(f"Startup timing: {name} took {duration_seconds:.3f}s")

        return wrapper

    if func:
        return measure(func)
    else:
        return measure
