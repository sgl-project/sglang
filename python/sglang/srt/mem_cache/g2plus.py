"""Compatibility exports for the router-driven HiCache reuse prototype."""

from sglang.srt.mem_cache.router_kv_reuse import (  # noqa: F401
    REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY,
    REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY,
    REMOTE_KV_REUSE_PLAN_VERSION,
    G2plusManager,
    RemoteG2ReuseHandler,
    RemoteKvReusePlan,
    ResolvedHostPage,
    RouterKVReuseManager,
    resolve_host_pages,
)
