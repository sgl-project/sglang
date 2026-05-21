"""Process-lifetime KV pool global context.

Provides ``set_kv_pools()`` / ``get_token_to_kv_pool()`` / ``get_req_to_token_pool()``
as module-level globals, following the same pattern as ``get_global_server_args()``.

``ModelRunner.__init__`` calls ``set_kv_pools()`` once after ``init_memory_pool()``,
and ``ModelRunner._forward_raw()`` resets the globals to ``self``'s pools at
entry and restores the prior values on exit so that the active runner's pools
are visible to model-layer code regardless of init ordering. Callers that need
a temporary pool swap (e.g. frozen-KV MTP) override ``runner.token_to_kv_pool``
directly so the ``_forward_raw`` set picks up the override. No locking is
required: all runners live in the same process and forward calls are
synchronous, so the global is mutated and read serially.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool

_req_to_token_pool: Optional["ReqToTokenPool"] = None
_token_to_kv_pool: Optional["KVCache"] = None


def set_kv_pools(
    req_to_token_pool: "ReqToTokenPool",
    token_to_kv_pool: "KVCache",
) -> Tuple[Optional["ReqToTokenPool"], Optional["KVCache"]]:
    """Set pool refs; return the previous (req, token) tuple for save/restore."""
    global _req_to_token_pool, _token_to_kv_pool
    prev = (_req_to_token_pool, _token_to_kv_pool)
    _req_to_token_pool = req_to_token_pool
    _token_to_kv_pool = token_to_kv_pool
    return prev


def get_token_to_kv_pool() -> "KVCache":
    assert (
        _token_to_kv_pool is not None
    ), "token_to_kv_pool not initialized — call set_kv_pools() after init_memory_pool()."
    return _token_to_kv_pool


def get_req_to_token_pool() -> "ReqToTokenPool":
    assert (
        _req_to_token_pool is not None
    ), "req_to_token_pool not initialized — call set_kv_pools() after init_memory_pool()."
    return _req_to_token_pool
