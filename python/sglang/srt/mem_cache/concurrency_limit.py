"""Attribution for the resolved `max_running_requests`.

Independent bounds (user request, KV capacity, hybrid state pool, ...) compete
for the concurrency ceiling. Carrying them as data instead of nested `min()`
calls lets the server report which one bound the result and how to raise it.
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
    # How the value was derived, e.g. "max_mamba_cache_size=131 / 5 per request".
    detail: str
    # Flags that raise this bound; None for the user's own request.
    remedy: Optional[str] = None


def user_request_limit(
    max_running_requests: int, attn_dp_size: int
) -> ConcurrencyLimit:
    return ConcurrencyLimit(
        source="max_running_requests",
        value=max_running_requests // attn_dp_size,
        detail=f"--max-running-requests={max_running_requests}",
    )


def kv_capacity_limit(token_capacity: int) -> ConcurrencyLimit:
    return ConcurrencyLimit(
        source="kv_capacity",
        value=token_capacity // KV_TOKENS_PER_REQUEST,
        detail=f"max_total_num_tokens={token_capacity} / {KV_TOKENS_PER_REQUEST}",
        remedy="raise --mem-fraction-static or lower the context length",
    )


def heuristic_limit(token_capacity: int, context_len: int) -> ConcurrencyLimit:
    lo, hi = HEURISTIC_BOUNDS
    estimated = int(token_capacity / context_len * HEURISTIC_TOKENS_PER_REQUEST)
    return ConcurrencyLimit(
        source="estimate",
        value=max(min(estimated, hi), lo),
        detail=f"token_capacity / context_len * {HEURISTIC_TOKENS_PER_REQUEST}, "
        f"clamped to [{lo}, {hi}]",
        remedy="set --max-running-requests explicitly",
    )


def state_pool_limit(
    max_mamba_cache_size: int, slots_per_request: int, target: Optional[int] = None
) -> ConcurrencyLimit:
    """Hybrid (mamba / linear-attention) state pool bound. `target` is the
    concurrency the remedy sizes for, defaulting to one more than fits."""
    value = max_mamba_cache_size // slots_per_request
    target = target or (value + 1)
    return ConcurrencyLimit(
        source="mamba_state_pool",
        value=value,
        detail=f"max_mamba_cache_size={max_mamba_cache_size} / "
        f"{slots_per_request} slots per request",
        remedy=f"set --max-mamba-cache-size {target * slots_per_request}, raise "
        f"--mamba-full-memory-ratio, or halve the state size with "
        f"--mamba-ssm-dtype bfloat16",
    )


def resolve_concurrency_limit(
    limits: List[ConcurrencyLimit],
) -> Tuple[int, ConcurrencyLimit]:
    """Return the effective ceiling and the limit that bound it. Ties resolve
    to the first entry, so list limits in the order they should be reported."""
    assert limits, "at least one concurrency limit is required"
    binding = min(limits, key=lambda limit: limit.value)
    return binding.value, binding


def format_concurrency_report(
    resolved: int,
    binding: ConcurrencyLimit,
    limits: List[ConcurrencyLimit],
    requested: Optional[int] = None,
) -> Tuple[bool, str]:
    """Render the resolution as (is_downgrade, message). A downgrade means the
    user asked for more than they got; log those at WARNING, the rest at INFO."""
    others = ", ".join(
        f"{limit.source}={limit.value}" for limit in limits if limit is not binding
    )
    others = f"; other limits: {others}" if others else ""

    is_downgrade = requested is not None and resolved < requested
    if is_downgrade:
        remedy = f" To raise it: {binding.remedy}." if binding.remedy else ""
        return True, (
            f"max_running_requests reduced from the requested {requested} to "
            f"{resolved} (per dp worker), bound by {binding.source} "
            f"({binding.detail}){others}.{remedy}"
        )
    return False, (
        f"max_running_requests={resolved} (per dp worker), bound by "
        f"{binding.source} ({binding.detail}){others}."
    )
