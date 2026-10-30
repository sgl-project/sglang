"""The unified buffer's byte budget and its bs=1 floor, from a plan."""

from typing import Optional, Sequence, Tuple

import msgspec

from sglang.srt.mem_cache.kv.decl import FullLifetime, StateLifetime, WindowLifetime
from sglang.srt.mem_cache.kv.plan import KVStoragePlan

# One running request holds 1 active state slot plus 2 radix checkpoints; a
# floor, not headroom.
_BS1_STATE_SLOTS = 3


def check_bs1_floor(
    *, total_bytes: int, floor_terms: Sequence[Tuple[str, int]], where: str
) -> None:
    """Reject pools that cannot hold one worst-case request at boot.

    Retraction cannot make progress below that floor, even at batch size one.
    """
    floor = sum(b for _, b in floor_terms)
    if total_bytes >= floor:
        return
    detail = " + ".join(f"{name}={b}" for name, b in floor_terms)
    raise RuntimeError(
        f"[unified-memory-pool] {where}: byte budget {total_bytes} cannot fit "
        f"ONE worst-case request (bs=1 floor {floor} = {detail}). A pool this "
        f"size retract-livelocks at runtime. Raise --mem-fraction-static, lower "
        f"the model context length, or reduce reserved memory."
    )


class KVSizingInputs(msgspec.Struct, frozen=True, kw_only=True):
    # Bytes for the token sub-pools; the state slots ride on top. None sizes the
    # token side from the token capacities instead.
    token_budget_bytes: Optional[int] = None
    full_tokens: Optional[int] = None
    window_tokens: Optional[int] = None
    state_slots: int = 0
    # None skips the bs=1 floor check.
    context_len: Optional[int] = None
    # The window the bs=1 floor charges: the model config's window, which some
    # models state one token above the window their layers attend over (the
    # plan's). Required when the plan has a window and the floor is checked.
    floor_window: Optional[int] = None


class SizedKVStorage(msgspec.Struct, frozen=True, kw_only=True):
    total_bytes: int
    bs1_floor_terms: Tuple[Tuple[str, int], ...] = ()


def _token_side_bytes(plan: KVStoragePlan, inputs: KVSizingInputs) -> int:
    if inputs.token_budget_bytes is not None:
        return inputs.token_budget_bytes
    full = plan.of_lifetime(FullLifetime)
    window = plan.of_lifetime(WindowLifetime)
    if inputs.full_tokens is None or (
        window is not None and inputs.window_tokens is None
    ):
        raise ValueError(
            "a token byte budget or the token capacity of every token sub-pool "
            "must be provided"
        )
    total = inputs.full_tokens * full.spec.entry_bytes()
    if window is not None:
        total += inputs.window_tokens * window.spec.entry_bytes()
    return total


def _bs1_floor_terms(
    plan: KVStoragePlan, context_len: int, floor_window: Optional[int]
) -> Tuple[Tuple[str, int], ...]:
    """The bytes ONE worst-case request needs alone: one window of window KV (plus
    a page of slack for the page-granular walk, clamped to the context), the
    state slots it locks, and the slot-0 sink. The full side is not charged:
    `max_req_len` already clamps it to the pool."""
    terms = []
    window = plan.of_lifetime(WindowLifetime)
    if window is not None:
        size = floor_window
        tokens = (
            min(context_len, size + plan.page_size) if size is not None else context_len
        )
        terms.append(("swa_window_kv", tokens * window.spec.entry_bytes()))
    state = plan.of_lifetime(StateLifetime)
    if state is not None:
        terms.append(("bs1_state_slots", _BS1_STATE_SLOTS * state.spec.entry_bytes()))
    terms.append(("sink", plan.sink_bytes()))
    return tuple(terms)


def size_kv_storage(plan: KVStoragePlan, inputs: KVSizingInputs) -> SizedKVStorage:
    """Total buffer bytes for a plan; refuses a budget below the bs=1 floor."""
    total_bytes = _token_side_bytes(plan, inputs)
    state = plan.of_lifetime(StateLifetime)
    if state is not None:
        total_bytes += inputs.state_slots * state.spec.entry_bytes()
    terms: Tuple[Tuple[str, int], ...] = ()
    if inputs.context_len is not None:
        terms = _bs1_floor_terms(plan, inputs.context_len, inputs.floor_window)
        check_bs1_floor(
            total_bytes=total_bytes, floor_terms=terms, where="build_unified_pools"
        )
    if total_bytes <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes}")
    return SizedKVStorage(total_bytes=total_bytes, bs1_floor_terms=terms)
