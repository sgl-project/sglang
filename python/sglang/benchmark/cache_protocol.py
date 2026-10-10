"""Benchmark radix-cache protocol recorded in each serving JSONL row.

The serving harness flushes only when ``--flush-cache`` is set, or when the
sglang backend runs with ``SGLANG_IS_IN_CI``. Default local runs leave the
server cache uncontrolled. These fields make that choice visible in the
result row without changing the default flush behavior.
"""

from typing import Optional, TypedDict


class BenchmarkCacheProtocol(TypedDict):
    flushed_cache: bool
    cache_flush_reason: str
    cache_state: str


def resolve_benchmark_cache_protocol(
    backend: str,
    flush_cache: bool,
    *,
    in_ci: Optional[bool] = None,
    ci_env: Optional[str] = None,
) -> BenchmarkCacheProtocol:
    """Return the cache protocol this benchmark invocation will run.

    ``in_ci`` is the resolved boolean. When omitted, ``ci_env`` is parsed the
    same way ``SGLANG_IS_IN_CI`` is elsewhere in the serving harness
    (``true`` or ``1``, case-insensitive). Pass ``ci_env`` from
    ``os.getenv("SGLANG_IS_IN_CI")`` at the call site so this helper stays
    free of process-global reads during tests.
    """
    if in_ci is None:
        value = "false" if ci_env is None else ci_env
        in_ci = value.lower() in ("true", "1")
    ci_flush = "sglang" in backend and in_ci
    flushed_cache = bool(ci_flush or flush_cache)
    if flush_cache and ci_flush:
        reason = "cli_and_ci"
    elif flush_cache:
        reason = "cli"
    elif ci_flush:
        reason = "ci"
    else:
        reason = "none"
    return {
        "flushed_cache": flushed_cache,
        "cache_flush_reason": reason,
        "cache_state": "cold-flushed" if flushed_cache else "uncontrolled",
    }
