# python/sglang/srt/model_executor/pool_context.py
"""Process-lifetime KV pool global context.

Provides set_kv_pools() / get_token_to_kv_pool() / get_req_to_token_pool()
as module-level globals, following the same pattern as get_global_server_args().

ModelRunner.__init__ calls set_kv_pools() once after init_memory_pool().
Code that needs a temporary pool swap (e.g. frozen-KV MTP) does explicit
save/restore around the affected forward call. No locking is required: all
runners live in the same process and forward calls are synchronous, so the
global is mutated and read serially.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool

_req_to_token_pool: Optional["ReqToTokenPool"] = None
_token_to_kv_pool: Optional["KVCache"] = None


def set_kv_pools(
    req_to_token_pool: "ReqToTokenPool",
    token_to_kv_pool: "KVCache",
) -> None:
    """Set process-lifetime pool refs. Called once after ModelRunner.init_memory_pool()."""
    global _req_to_token_pool, _token_to_kv_pool
    _req_to_token_pool = req_to_token_pool
    _token_to_kv_pool = token_to_kv_pool


def get_token_to_kv_pool() -> "KVCache":
    """Return the process-lifetime token-to-KV pool."""
    assert _token_to_kv_pool is not None, (
        "token_to_kv_pool not initialized. "
        "Call set_kv_pools() in ModelRunner.__init__ after init_memory_pool()."
    )
    return _token_to_kv_pool


def get_req_to_token_pool() -> "ReqToTokenPool":
    """Return the process-lifetime request-to-token pool."""
    assert _req_to_token_pool is not None, (
        "req_to_token_pool not initialized. "
        "Call set_kv_pools() in ModelRunner.__init__ after init_memory_pool()."
    )
    return _req_to_token_pool
