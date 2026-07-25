"""Attribution for the resolved `max_running_requests`.

Several independent bounds compete for the concurrency ceiling (the user's
request, KV capacity, the hybrid state pool, ...). Collecting them as data
rather than folding them into nested `min()` calls lets the server report
*which* one bound the result and how to raise it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ConcurrencyLimit:
    """One upper bound on `max_running_requests`."""

    source: str
    value: int
    # Where the number comes from, e.g. "max_mamba_cache_size=131 / 5 per request".
    detail: str
    # Flags that raise this bound; None for the user's own request.
    remedy: Optional[str] = None


def resolve_concurrency_limit(
    limits: List[ConcurrencyLimit],
) -> Tuple[int, ConcurrencyLimit]:
    """Return the effective ceiling and the limit that bound it.

    Ties resolve to the first entry, so callers should list limits in the
    order they want reported.
    """
    assert limits, "at least one concurrency limit is required"
    binding = min(limits, key=lambda limit: limit.value)
    return binding.value, binding


def format_concurrency_report(
    resolved: int,
    binding: ConcurrencyLimit,
    limits: List[ConcurrencyLimit],
    requested: Optional[int] = None,
) -> Tuple[bool, str]:
    """Render the resolution as (is_downgrade, message).

    `is_downgrade` is True when the user explicitly asked for more than they
    got; callers should log those at WARNING and the rest at INFO.
    """
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
