"""Compatibility exports for router-directed HiCache reuse."""

from sglang.srt.mem_cache.router_kv_plan import (  # noqa: F401
    REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY,
    REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY,
    REMOTE_KV_REUSE_PLAN_VERSION,
    RemoteKvReusePlan,
)
from sglang.srt.mem_cache.router_kv_reuse import (  # noqa: F401
    G2plusManager,
    RemoteG2ReuseHandler,
    RouterKVReuseManager,
)
from sglang.srt.mem_cache.router_kv_source import (  # noqa: F401
    ResolvedHostPage,
    resolve_host_pages,
)
