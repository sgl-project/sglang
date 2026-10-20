"""Explicit, process-local Python thread scheduling for scheduler workers."""

import logging
import math
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def validate_python_thread_switch_interval(value: Optional[float]) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("--python-thread-switch-interval must be positive and finite")


def configure_python_thread_scheduling(interval: Optional[float]) -> None:
    """Leave interpreter defaults untouched unless explicitly requested.

    Call in the spawned scheduler process, after runtime-context publication;
    setting a parent interpreter does not configure spawned children. This is
    process-wide, not a thread priority or a CPU-affinity control.
    """
    validate_python_thread_switch_interval(interval)
    if interval is None:
        return
    sys.setswitchinterval(interval)
    logger.info(
        "Python scheduler thread switch interval pid=%d requested_seconds=%.9g "
        "actual_seconds=%.9g",
        os.getpid(),
        interval,
        sys.getswitchinterval(),
    )
