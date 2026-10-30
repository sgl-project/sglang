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


def routed(pool):
    """A stand-in pool whose layers its own plain router hands out, so a backend
    that reads and writes through ``kv_layers`` reaches the stand-in's methods."""
    from unittest.mock import Mock

    from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
    from sglang.srt.mem_cache.kv.router import PlainPoolRouter

    # A Mock answers every attribute; give it the two the router reads.
    if not isinstance(getattr(pool, "start_layer", None), int):
        pool.start_layer = 0
    if isinstance(getattr(pool, "layer_transfer_counter", None), Mock) or not hasattr(
        pool, "layer_transfer_counter"
    ):
        pool.layer_transfer_counter = None
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool
