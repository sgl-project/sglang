"""Shared helpers for mem_cache unit tests."""

from sglang.srt.managers.schedule_batch import FINISH_LENGTH


def finish_req(cache, req, up_to):
    """What release_kv_cache does for a request that keeps its KV: mark it
    finished, checkpoint, free the rest, drop the lock."""
    if getattr(req, "finished_reason", None) is None:
        req.finished_reason = FINISH_LENGTH(length=0)
    req.refresh_fill_ids()
    cache.checkpoint(req, up_to=up_to)
    cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, up_to)])
    cache.unlock(req.lock)
    req.lock = None
