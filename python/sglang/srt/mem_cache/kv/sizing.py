"""The unified buffer's byte budget and its bs=1 floor, from a plan."""

from typing import Optional, Sequence, Tuple

import msgspec

from sglang.srt.mem_cache.kv.decl import StateLifetime, WindowLifetime
from sglang.srt.mem_cache.kv.plan import KVStoragePlan
from sglang.srt.mem_cache.kv.state_policy import StateSlotStrategy


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
    budget_bytes: Optional[int] = None
    # Token capacities are physical rows per rank; state capacities are slots.
    # Explicit capacities are useful to construct deterministic small pools.
    full_tokens: Optional[int] = None
    window_tokens: Optional[int] = None
    state_slots: Optional[int] = None
    state_share: float = 0.1
    state_strategy: Optional[StateSlotStrategy] = None
    radix_cache: bool = True
    context_len: Optional[int] = None
    floor_window: Optional[int] = None
    window_ratio: float = 1.0
    window_request_cap: bool = False
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    pre_alloc_rows: int = 0
    extra_context_len: int = 0
    alignment: int = 4096


class FixedAllocation(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    bytes: int


class SizedKVStorage(msgspec.Struct, frozen=True, kw_only=True):
    total_bytes: int
    budget_bytes: int
    full_tokens: int
    window_tokens: int
    state_units: Tuple[Tuple[str, int], ...]
    max_running_requests: int
    scratch_rows: int
    request_rows: int
    fixed: Tuple[FixedAllocation, ...]
    full_cap_tokens: Optional[int] = None
    bs1_floor_terms: Tuple[Tuple[str, int], ...] = ()
    state_strategy: StateSlotStrategy = StateSlotStrategy()

    @property
    def fixed_bytes(self):
        return sum(term.bytes for term in self.fixed)

    def units(self, name):
        return dict(self.state_units)[name]


def _request_estimate(plan, inp):
    if inp.max_running_requests is not None:
        return inp.max_running_requests
    if inp.full_tokens is not None:
        tokens = inp.full_tokens
    else:
        tokens = inp.budget_bytes // plan.owner().spec.entry_bytes()
    return max(
        1,
        min(
            tokens // 2,
            max(2048, min(4096, tokens * 512 // max(1, inp.context_len or 1))),
        ),
    )


def _fixed_terms(plan, inp, running, strategy):
    scratch_rows = running + 1
    request_rows = scratch_rows + inp.pre_alloc_rows
    row_bytes = 4 * ((inp.context_len or 0) + inp.extra_context_len)
    row_bytes += sum(
        4 + 8 * strategy.tracking_slots
        for s in plan.sub_pools
        if isinstance(s.lifetime, StateLifetime)
    )
    terms = [FixedAllocation(name="request_table", bytes=row_bytes * request_rows)]
    terms.extend(
        FixedAllocation(name=d.part.name, bytes=d.part.row_bytes * scratch_rows)
        for d in plan.scratch
    )
    terms.extend(
        FixedAllocation(name=d.part.name, bytes=d.part.row_bytes * request_rows)
        for d in plan.request_parts
    )
    return tuple(terms)


def _derived_terms(plan, chain_bytes):
    return tuple(
        FixedAllocation(
            name=f"{d.sub_pool}.{d.part.name}",
            bytes=(
                chain_bytes
                // (
                    plan.sub_pool(d.sub_pool).unit
                    * plan.sub_pool(d.sub_pool).spec.entry_bytes()
                )
            )
            * d.part.row_bytes,
        )
        for d in plan.derived_parts
    )


def _chain_budget(plan, budget, fixed, alignment):
    # Page-indexed storage spans every physical page of the reserved arena.
    low, high = 0, max(0, budget - sum(t.bytes for t in fixed)) // alignment
    while low < high:
        mid = (low + high + 1) // 2
        chain = mid * alignment
        if (
            chain
            + sum(t.bytes for t in _derived_terms(plan, chain))
            + sum(t.bytes for t in fixed)
            <= budget
        ):
            low = mid
        else:
            high = mid - 1
    return low * alignment


def _window_units(sub_pool, full, inp, running, page):
    if inp.window_request_cap:
        window = sub_pool.lifetime.window
        return (
            running
            * -(
                -(
                    min(inp.context_len, window + page)
                    if window is not None
                    else inp.context_len
                )
                // page
            )
            * page
        )
    if inp.window_tokens is not None and inp.full_tokens is not None:
        return int(full * inp.window_tokens / max(1, inp.full_tokens)) // page * page
    return int(full * inp.window_ratio) // page * page


def _bs1_floor_terms(plan, context_len, floor_window, strategy):
    terms = []
    for sub_pool in plan.sub_pools:
        if isinstance(sub_pool.lifetime, WindowLifetime):
            window = (
                floor_window
                if floor_window is not None
                and sub_pool is plan.of_lifetime(WindowLifetime)
                else sub_pool.lifetime.window
            )
            tokens = (
                context_len
                if window is None
                else min(context_len, window + sub_pool.unit)
            )
            tokens = -(-tokens // sub_pool.unit) * sub_pool.unit
            terms.append(
                (f"{sub_pool.name}_window_kv", tokens * sub_pool.spec.entry_bytes())
            )
        elif isinstance(sub_pool.lifetime, StateLifetime):
            terms.append(
                (
                    f"{sub_pool.name}_bs1_state_slots",
                    strategy.slots_per_request() * sub_pool.spec.entry_bytes(),
                )
            )
    terms.append(("sink", plan.sink_bytes()))
    return tuple(terms)


def _state_targets(plan, inp, chain, running, states, frozen, strategy):
    units = {}
    state_budget = int(chain * inp.state_share)
    for sub_pool in states:
        count = inp.state_slots
        if count is None:
            count = (
                running * strategy.slots_per_request()
                if not inp.radix_cache
                else state_budget // sub_pool.spec.entry_bytes()
            )
        if frozen is not None:
            count = min(count, frozen.units(sub_pool.name))
        units[sub_pool.name] = count
    return units


def _token_targets(plan, inp, chain, running, states, windows, units, frozen):
    tokens_budget = (
        chain
        - plan.sink_bytes()
        - sum(units[s.name] * s.spec.entry_bytes() for s in states)
    )
    owner = plan.owner()
    page = plan.page_size
    low, high = 0, max(0, tokens_budget) // (page * owner.spec.entry_bytes())
    while low < high:
        mid = (low + high + 1) // 2
        full = mid * page
        cost = full * owner.spec.entry_bytes() + sum(
            _window_units(s, full, inp, running, page) * s.spec.entry_bytes()
            for s in windows
        )
        if cost <= tokens_budget:
            low = mid
        else:
            high = mid - 1
    full = low * page
    caps = [
        x
        for x in (
            inp.max_total_tokens,
            inp.full_tokens,
            frozen.full_tokens if frozen else None,
        )
        if x is not None
    ]
    if caps:
        full = min(full, *caps) // page * page
    window = max(
        (_window_units(s, full, inp, running, page) for s in windows), default=0
    )
    return full, window


def _targets_for_running(
    plan, inp, budget, running, strategy, states, windows, explicit_chain, frozen
):
    fixed = (
        _fixed_terms(plan, inp, running, strategy) if frozen is None else frozen.fixed
    )
    if frozen is None:
        chain = (
            _chain_budget(plan, budget, fixed, inp.alignment)
            if explicit_chain is None
            else explicit_chain
        )
    else:
        chain = min(
            frozen.total_bytes,
            max(0, budget - frozen.fixed_bytes) // inp.alignment * inp.alignment,
        )
    units = _state_targets(plan, inp, chain, running, states, frozen, strategy)
    full, window = _token_targets(
        plan, inp, chain, running, states, windows, units, frozen
    )
    return chain, fixed, units, full, window


def size_plan(
    plan: KVStoragePlan, inp: KVSizingInputs, *, frozen: Optional[SizedKVStorage] = None
) -> SizedKVStorage:
    """Solve all declared storage against one unsubtracted byte budget.

    State units are boot targets. The arena may give state more slots at runtime;
    only the explicit FULL limit caps allocation. Captured fixed extents never grow.
    """
    if inp.alignment <= 0 or not 0 <= inp.state_share < 1:
        raise ValueError("invalid sizing alignment or state share")
    running = _request_estimate(plan, inp)
    strategy = inp.state_strategy or StateSlotStrategy(radix_cache=inp.radix_cache)
    fixed = _fixed_terms(plan, inp, running, strategy)
    states = [s for s in plan.sub_pools if isinstance(s.lifetime, StateLifetime)]
    windows = [s for s in plan.sub_pools if isinstance(s.lifetime, WindowLifetime)]
    budget = inp.budget_bytes
    explicit_chain = None
    if budget is None:
        if inp.full_tokens is None:
            raise ValueError("sizing needs the total byte budget")
        chain = inp.full_tokens * plan.owner().spec.entry_bytes() + plan.sink_bytes()
        chain += sum((inp.window_tokens or 0) * s.spec.entry_bytes() for s in windows)
        chain += sum((inp.state_slots or 0) * s.spec.entry_bytes() for s in states)
        explicit_chain = chain // inp.alignment * inp.alignment
        budget = chain + sum(t.bytes for t in (*fixed, *_derived_terms(plan, chain)))
    terms = (
        ()
        if inp.context_len is None
        else _bs1_floor_terms(plan, inp.context_len, inp.floor_window, strategy)
    )
    if frozen is not None:
        running = min(running, frozen.max_running_requests)
    # Both initial and final sizing solve concurrency with its state demand.
    # Capture freezes addresses/extents, not the number of active requests.
    low, high = 0, running
    floor = sum(cost for _, cost in terms)
    while low < high:
        candidate = (low + high + 1) // 2
        chain, fixed, units, full, window = _targets_for_running(
            plan,
            inp,
            budget,
            candidate,
            strategy,
            states,
            windows,
            explicit_chain,
            frozen,
        )
        capacity = min(
            [full // 2]
            + [units[s.name] // strategy.slots_per_request() for s in states]
        )
        if candidate <= capacity and chain >= floor:
            low = candidate
        else:
            high = candidate - 1
    running = low
    chain, fixed, units, full, window = _targets_for_running(
        plan,
        inp,
        budget,
        max(1, running),
        strategy,
        states,
        windows,
        explicit_chain,
        frozen,
    )
    if frozen is None:
        fixed += _derived_terms(plan, chain)
        if explicit_chain is not None:
            budget = chain + sum(t.bytes for t in fixed)
        scratch_rows, request_rows = running + 1, running + inp.pre_alloc_rows + 1
    else:
        scratch_rows, request_rows = frozen.scratch_rows, frozen.request_rows
    check_bs1_floor(total_bytes=chain, floor_terms=terms, where="size_plan")
    if full <= 0 or running <= 0:
        raise RuntimeError("the declared storage budget cannot serve one request")
    return SizedKVStorage(
        total_bytes=chain,
        budget_bytes=budget,
        full_tokens=full,
        window_tokens=window,
        state_units=tuple(units.items()),
        max_running_requests=running,
        scratch_rows=scratch_rows,
        request_rows=request_rows,
        fixed=fixed,
        full_cap_tokens=inp.max_total_tokens,
        bs1_floor_terms=terms,
        state_strategy=strategy,
    )
