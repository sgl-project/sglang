"""Opt-in metadata diagnostics; usable with compression disabled as a control."""

import json
import logging
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_last_cache_trace = 0.0


def trace_handoff(stage, req, token_id, **extra):
    if not envs.SGLANG_KV_COMPRESSION_TRACE_HANDOFF.get():
        return
    logger.info(
        "KV_COMPRESSION_HANDOFF %s",
        json.dumps(
            dict(
                stage=stage,
                rid=req.rid,
                room=req.bootstrap_room,
                token_id=int(token_id),
                metadata_index=getattr(req, "metadata_buffer_index", None),
                cached_tokens=getattr(req, "cached_tokens", None),
                cached_device=getattr(req, "cached_tokens_device", None),
                cached_host=getattr(req, "cached_tokens_host", None),
                **extra,
            )
        ),
    )


def trace_native_cache_state(cache):
    """Expose completion of the uncompressed HiCache control during validation."""
    global _last_cache_trace
    if not envs.SGLANG_KV_COMPRESSION_TRACE_HANDOFF.get():
        return
    if (
        getattr(cache, "get_kv_compression_context", lambda: (None, None))()[0]
        is not None
    ):
        return  # The HiCache controller emits its detailed transfer snapshot.
    if not hasattr(cache, "ongoing_write_through"):
        return
    now = time.monotonic()
    if now - _last_cache_trace < 5:
        return
    _last_cache_trace = now
    logger.info(
        "KV_COMPRESSION_CACHE_STATE %s",
        json.dumps(
            dict(
                native_control=True,
                writes=len(cache.ongoing_write_through),
                loads=len(cache.ongoing_load_back),
            )
        ),
    )
