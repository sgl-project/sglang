"""Test shorthand for an allocator's `TokenDemand`."""

from __future__ import annotations

from typing import Mapping, Optional

from sglang.srt.mem_cache.allocator.base import TokenDemand


def _target_window(allocator) -> Optional[str]:
    target = getattr(allocator, "swa_attn_allocator", None)
    if target is not None and hasattr(target, "sub_pool_name"):
        return target.sub_pool_name
    names = allocator.window_pool_names()
    return names[0] if names else None


def token_demand(
    allocator,
    full_tokens: int,
    target_window_tokens: int,
    *,
    state_slots: int = 0,
    windows: Optional[Mapping[str, int]] = None,
) -> TokenDemand:
    """A demand naming every window ``allocator`` keeps, in the tests'
    two-number form: the target model's window takes
    ``target_window_tokens``; any other window (a draft window of another
    length) keeps the whole demand as the owner does, ``full_tokens``;
    ``windows`` gives any window its own."""
    windows = dict(windows or {})
    names = allocator.window_pool_names()
    unknown = set(windows) - set(names)
    if unknown:
        raise ValueError(f"the allocator keeps no window {sorted(unknown)}")
    target = _target_window(allocator)
    return TokenDemand(
        full_tokens,
        {
            name: windows.get(
                name, target_window_tokens if name == target else full_tokens
            )
            for name in names
        },
        state_slots,
    )


def demand_windows(allocator, full_tokens: int, target_window_tokens: int) -> dict:
    """`token_demand`'s windows alone, for `admits`."""
    return dict(
        token_demand(allocator, full_tokens, target_window_tokens).window_tokens
    )
