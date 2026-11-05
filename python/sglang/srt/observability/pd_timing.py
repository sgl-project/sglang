"""Opt-in PD stage logs. No device synchronization or payload logging."""

import json
import logging
import os
import time
from contextvars import ContextVar
from types import SimpleNamespace

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


# Request-local correlation follows explicit transfer calls, not engine globals.
_transfer_room = ContextVar("pd_transfer_timing_room", default=None)


def run_pd_transfer(room, transfer, *args, **kwargs):
    """Bind a room around a transfer worker call; always restore prior context."""
    if not envs.SGLANG_ENABLE_PD_TIMING_LOG.get():
        return transfer(*args, **kwargs)
    token = _transfer_room.set(room)
    req = SimpleNamespace(bootstrap_room=room)
    log_pd_timing(req, "transfer_worker_call_begin")
    try:
        return transfer(*args, **kwargs)
    finally:
        log_pd_timing(req, "transfer_worker_call_end")
        _transfer_room.reset(token)


def run_pd_engine_transfer(transfer, session_id, buffers, peers, lengths):
    """Time the synchronous engine call without logging memory addresses."""
    room = _transfer_room.get()
    if room is None or not envs.SGLANG_ENABLE_PD_TIMING_LOG.get():
        return transfer(session_id, buffers, peers, lengths)
    import threading

    req = SimpleNamespace(bootstrap_room=room)
    started = time.monotonic_ns()
    details = {
        "call_id": f"{os.getpid()}:{threading.get_ident()}:{started}",
        "session_id": session_id,
        "segments": len(lengths),
        "bytes": sum(lengths),
    }
    log_pd_timing(req, "engine_call_begin", **details)
    status = None
    try:
        status = transfer(session_id, buffers, peers, lengths)
        return status
    finally:
        log_pd_timing(
            req,
            "engine_call_end",
            **details,
            status=status,
            elapsed_ms=(time.monotonic_ns() - started) / 1e6,
        )
