"""Shared helpers for mem_cache unit tests."""


def finish_req(cache, req, up_to):
    """insert_req, then what release_kv_cache does after it: free the rest, drop the lock."""
    cache.insert_req(req, up_to=up_to)
    cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, up_to)])
    cache.unpin(req)
