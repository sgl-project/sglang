from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def engine_stats(ctx: ScriptedContext) -> Dict[str, int]:
    s = ctx.scheduler
    return {
        "kv_pool_free": s.token_to_kv_pool_allocator.available_size(),
        "req_pool_free": s.req_to_token_pool.available_size(),
        "req_pool_total": s.req_to_token_pool.size,
        "page_size": s.page_size,
    }
