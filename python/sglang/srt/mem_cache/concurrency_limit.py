"""Attribution for the resolved `max_running_requests`.

Independent bounds (user request, KV capacity, hybrid state pool, ...) compete
for the concurrency ceiling. Carrying them as data instead of nested `min()`
calls lets the server report which one bound the result and how to raise it.

The remedies name user-facing CLI flags, so renaming a flag means updating them
here too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Minimum KV tokens assumed per running request.
KV_TOKENS_PER_REQUEST = 2
# Heuristic ceiling, used only when the user did not ask for a value.
HEURISTIC_TOKENS_PER_REQUEST = 512
HEURISTIC_BOUNDS = (2048, 4096)


@dataclass(frozen=True)
class ConcurrencyLimit:
    """One upper bound on `max_running_requests`."""

    source: str
    value: int
    # How the value was derived, e.g. "max_mamba_cache_size=131 per shard / 5 ...".
    detail: str
    # Flags that raise this bound; None for the user's own request.
    remedy: Optional[str] = None


def user_request_limit(
    max_running_requests: int, attn_dp_size: int
) -> ConcurrencyLimit:
    detail = f"--max-running-requests={max_running_requests}"
    if attn_dp_size > 1:
        detail += f" / {attn_dp_size} dp workers"
    return ConcurrencyLimit(
        source="max_running_requests",
        value=max_running_requests // attn_dp_size,
        detail=detail,
    )


def kv_capacity_limit(token_capacity: int) -> ConcurrencyLimit:
    return ConcurrencyLimit(
        source="kv_capacity",
        value=token_capacity // KV_TOKENS_PER_REQUEST,
        detail=f"max_total_num_tokens={token_capacity} / {KV_TOKENS_PER_REQUEST}",
        remedy="raise --mem-fraction-static or use GPUs with more memory",
    )


def heuristic_limit(token_capacity: int, context_len: int) -> ConcurrencyLimit:
    lo, hi = HEURISTIC_BOUNDS
    estimated = int(token_capacity / context_len * HEURISTIC_TOKENS_PER_REQUEST)
    return ConcurrencyLimit(
        source="heuristic_estimate",
        value=max(min(estimated, hi), lo),
        detail=f"token_capacity / context_len * {HEURISTIC_TOKENS_PER_REQUEST}, "
        f"clamped to [{lo}, {hi}]",
        remedy="set --max-running-requests explicitly, or lower the context length",
    )


def state_pool_limit(
    per_shard_pool_size: int,
    slots_per_request: int,
    attn_dp_size: int,
    target: Optional[int] = None,
) -> ConcurrencyLimit:
    """Hybrid (mamba / linear-attention) state pool bound.

    `per_shard_pool_size` is the resolved per-DP-shard slot count, so the remedy
    scales back up: --max-mamba-cache-size is a global value that the server
    divides by `attn_dp_size`. `target` is the concurrency to size for; without
    one the remedy names the flags but no number, since the only sizes we could
    invent are either one request more than today or far past what fits.
    """
    if target is None:
        sizing = "--max-running-requests <target> (the log then reports the exact size)"
    else:
        sizing = (
            f"--max-mamba-cache-size "
            f"{target * slots_per_request * attn_dp_size} (memory permitting)"
        )
    return ConcurrencyLimit(
        source="mamba_state_pool",
        value=per_shard_pool_size // slots_per_request,
        detail=f"max_mamba_cache_size={per_shard_pool_size} per shard / "
        f"{slots_per_request} slots per request",
        remedy=f"try one of: {sizing}, a larger --mamba-full-memory-ratio, "
        f"or --mamba-ssm-dtype bfloat16",
    )


def resolve_concurrency_limit(limits: List[ConcurrencyLimit]) -> ConcurrencyLimit:
    """Return the limit that bound the ceiling; its value is the resolution.
    Ties resolve to the first entry, so list limits in reporting order."""
    assert limits, "at least one concurrency limit is required"
    return min(limits, key=lambda limit: limit.value)


def format_concurrency_report(
    binding: ConcurrencyLimit,
    limits: List[ConcurrencyLimit],
    requested: Optional[ConcurrencyLimit] = None,
) -> Tuple[bool, str]:
    """Render the resolution as (is_downgrade, message). A downgrade means the
    user asked for more than they got; log those at WARNING, the rest at INFO."""
    others = ", ".join(
        f"{limit.source}={limit.value}" for limit in limits if limit is not binding
    )
    others = f"; other limits: {others}" if others else ""
    remedy = f" To raise it: {binding.remedy}." if binding.remedy else ""

    if requested is not None and binding.value < requested.value:
        return True, (
            f"max_running_requests reduced from the requested {requested.value} to "
            f"{binding.value} (per dp worker; {requested.detail}), bound by "
            f"{binding.source} ({binding.detail}){others}.{remedy}"
        )
    return False, (
        f"max_running_requests={binding.value} (per dp worker), bound by "
        f"{binding.source} ({binding.detail}){others}.{remedy}"
    )
