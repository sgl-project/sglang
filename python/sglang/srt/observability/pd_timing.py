"""Opt-in PD stage logs. No device synchronization or payload logging."""

import json
import logging
import os
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def log_pd_timing(req, stage: str, **details) -> None:
    if not envs.SGLANG_ENABLE_PD_TIMING_LOG.get():
        return
    room = getattr(req, "bootstrap_room", None)
    if room is None:
        return
    record = {
        "stage": stage,
        "rid": getattr(req, "rid", None),
        "bootstrap_room": room,
        "bootstrap_host": getattr(req, "bootstrap_host", None),
        "pid": os.getpid(),
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        **details,
    }
    logger.info("PD_TIMING %s", json.dumps(record, separators=(",", ":"), default=str))
