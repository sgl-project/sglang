"""What an allocation takes from each sub-pool, and what a request needs."""

from __future__ import annotations

import enum
import math
from typing import Callable, Iterable, Mapping, NamedTuple, Optional

import msgspec

from sglang.srt.utils.common import ceil_align


class KVDemand(msgspec.Struct, frozen=True):
    """What an allocation takes: tokens of the token sub-pool, tokens of each
    window sub-pool by its name, and state slots. Admission and the take must
    name every window (`checked`); an eviction request may name only the
    sub-pools it is short of (`completed`)."""

    full_tokens: int = 0
    window_tokens: Mapping[str, int] = msgspec.field(default_factory=dict)
    state_slots: int = 0

    @classmethod
    def same_tokens(cls, allocator, tokens: int, *, state_slots: int = 0) -> KVDemand:
        """Every token sub-pool takes ``tokens``: the new tokens of an extend
        or decode step, which every window keeps."""
        return cls(
            tokens,
            {name: tokens for name in allocator.window_pool_names()},
            state_slots,
        )

    def checked(self, window_pool_names: Iterable[str]) -> KVDemand:
        """This demand, once it names exactly the windows listed."""
        names = set(window_pool_names)
        if set(self.window_tokens) != names:
            raise ValueError(
                f"a demand must name every window pool: it names "
                f"{sorted(self.window_tokens)}, the allocator keeps {sorted(names)}"
            )
        if self.full_tokens < 0 or self.state_slots < 0:
            raise ValueError(f"negative demand {self}")
        if any(t < 0 for t in self.window_tokens.values()):
            raise ValueError(f"negative window demand {self}")
        return self

    def completed(self, window_pool_names: Iterable[str]) -> KVDemand:
        """This demand with every listed window named, those it leaves out
        taking nothing; a window it names that is not listed is an error."""
        names = tuple(window_pool_names)
        unknown = set(self.window_tokens) - set(names)
        if unknown:
            raise ValueError(
                f"no window pools {sorted(unknown)}; the allocator keeps "
                f"{sorted(names)}"
            )
        windows = {name: int(self.window_tokens.get(name, 0)) for name in names}
        return msgspec.structs.replace(self, window_tokens=windows).checked(names)

    def is_empty(self) -> bool:
        return not (
            self.full_tokens > 0
            or self.state_slots > 0
            or any(t > 0 for t in self.window_tokens.values())
        )


def retained_window_tokens(
    sliding_window_size: int, page_size: int, *, supports_prefix_sharing: bool = True
) -> int:
    """Rows behind a request's frontier that `retire_window_rows` keeps: for a
    prefix-sharing cache, max(window, page) rounded up to whole pages."""
    if not supports_prefix_sharing:
        return sliding_window_size
    return ceil_align(max(sliding_window_size, page_size), page_size)


def request_demand(
    input_tokens: int,
    max_new_tokens: int,
    *,
    windows: Mapping[str, Optional[int]],
    page_size: int,
    prefill_retire_lag: int,
    eviction_interval: int,
    checkpoint_interval: int = 0,
    decode_retire_lag: int = 0,
    chunk_size: Optional[int] = None,
    state_slots: int = 0,
    host_hit_lengths: Optional[Mapping[str, int]] = None,
    extra_tokens: int = 0,
    cached_tokens: int = 0,
) -> KVDemand:
    """What one request still needs to place ``input_tokens`` prompt tokens in
    chunks of at most ``chunk_size`` and then decode ``max_new_tokens``.

    The token sub-pool takes the page-ceiled prompt, the decode, the rows a
    decode step allocates ahead (``decode_retire_lag``: the overlap scheduler's
    next step or a speculative reserve) and one rounding page, plus
    ``extra_tokens``. Each window of ``windows`` (name to retained window) takes
    one bound over every phase: the request's own rows, capped by the retained
    window plus the larger of a prefill chunk with what retirement keeps beside
    it and what decode keeps past the window, less the rows ``cached_tokens``
    already hold; plus a rounding page and the pages a host hit restores.

    The bound never falls as the chunk grows. Every site that prices a request
    (the ceiling clip, admission, a chunked request's commitment) calls this,
    so a request the clip lets through is one its chunks can finish."""
    hits = host_hit_lengths or {}

    if (
        prefill_retire_lag < 0
        or decode_retire_lag < 0
        or eviction_interval < 1
        or checkpoint_interval < 0
    ):
        raise ValueError(
            f"retirement lags {prefill_retire_lag} and {decode_retire_lag}, "
            f"interval {eviction_interval} and checkpoint interval "
            f"{checkpoint_interval}"
        )
    decodes = max_new_tokens > 0
    # Rows a decode keeps past the window, and rows a decode step allocates
    # ahead of its processed length: none without a decode.
    retention = max(eviction_interval, checkpoint_interval) if decodes else 0
    ahead = decode_retire_lag if decodes else 0
    rows = input_tokens + max_new_tokens
    last_chunk = input_tokens if chunk_size is None else min(input_tokens, chunk_size)

    def window_bound(name: str, window: Optional[int]) -> int:
        hit = hits.get(name, 0)
        hit_pages = -(-hit // page_size) * page_size if hit > 0 else 0
        held = rows + ahead
        if window is not None and window > 0:
            # TODO(ch-wan): deliberately conservative. Window rows retire by
            # scheduler-side rules keyed on the processed length (maybe_evict_swa,
            # retire_window_rows); once retirement moves into the memory layer
            # and keys on the allocated frontier, derive this from that policy.
            # TODO(ch-wan): assumes a checkpoint every checkpoint_interval rows
            # that the prefix-sharing cache takes, retiring the window behind it.
            # Two stalls keep an older floor (or none), more rows than counted:
            # (A) lazy Mamba tracking skips a checkpoint when no second tracking
            # slot is free after eviction (`mamba_lazy_prealloc_at_boundary`);
            # (B) with no replacement state slot free after eviction,
            # `MambaComponent.prepare_for_caching_req` returns 0 and
            # `checkpoint_into_tree` retires no row. (Sub-grid chunks are cut on
            # the grid by `mamba_prefill_chunk_grid`.) Admission over this bound
            # is an estimate: it does not guarantee an admitted request finishes
            # when a checkpoint stalls.
            retained = retained_window_tokens(window, page_size)
            in_prefix = min(retained, cached_tokens)
            # Past its latest checkpoint a chunk keeps up to an interval less a
            # row, or the whole chunk when it takes a branch early in it: a state
            # branch (`branch_checkpoint`) or a window-only cache's branch window.
            tail = max(checkpoint_interval - 1, last_chunk)
            beside = max(prefill_retire_lag + last_chunk + tail, retention - 1)
            # The last chunk is still live when the first decode allocates
            # ahead, so they add up; the cached prefix's rows are held already.
            held = min(held, retained + beside + ahead - in_prefix)
        return held + page_size + hit_pages

    return KVDemand(
        -(-input_tokens // page_size) * page_size
        + max_new_tokens
        + max(0, ahead - 1)
        + page_size
        + extra_tokens,
        {name: window_bound(name, window) for name, window in windows.items()},
        state_slots,
    )


class DemandPurpose(enum.Enum):
    """Why a request's demand is asked. Each purpose builds the inputs of
    `request_demand` by its row of `DEMAND_RULES`."""

    # A lone request in an empty pool: whether it can ever run. The memory
    # clip of an arriving request and a decode node's capacity check.
    CEILING = "ceiling"
    # A prefill's admission, its final shape check and its commitment.
    ADMIT = "admit"
    # The rows a selected admission realizes before its host load-back.
    LOAD_BACK = "load_back"
    # A decode node's preallocation of a transferred request.
    DECODE_PREALLOC = "decode_prealloc"


class StateSlots(enum.Enum):
    # What acquire allocates for a request that holds no state yet.
    FRESH = "fresh"
    # What the request still lacks.
    PENDING = "pending"


class DemandRule(msgspec.Struct, frozen=True, kw_only=True):
    """How one purpose differs from the others; everything else is built alike."""

    # Price the scheduler's decode estimate; a ceiling prices all of
    # ``max_new_tokens``, since the generation it lets through must fit.
    estimates_decode: bool
    state_slots: StateSlots
    # Whether evictable cached entries count as room. A load-back reclaims
    # first and asks what is free then; a ceiling asks an empty pool.
    credits_evictable: bool


DEMAND_RULES: Mapping[DemandPurpose, DemandRule] = {
    DemandPurpose.CEILING: DemandRule(
        estimates_decode=False,
        state_slots=StateSlots.FRESH,
        credits_evictable=False,
    ),
    DemandPurpose.ADMIT: DemandRule(
        estimates_decode=True,
        state_slots=StateSlots.PENDING,
        credits_evictable=True,
    ),
    DemandPurpose.LOAD_BACK: DemandRule(
        estimates_decode=True,
        state_slots=StateSlots.PENDING,
        credits_evictable=False,
    ),
    DemandPurpose.DECODE_PREALLOC: DemandRule(
        estimates_decode=True,
        state_slots=StateSlots.PENDING,
        credits_evictable=True,
    ),
}


def decode_estimate_tokens(max_new_tokens: int) -> int:
    """The decode the scheduler expects of a request that may generate
    ``max_new_tokens``: at most SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION."""
    from sglang.srt.environ import envs

    return min(max_new_tokens, envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get())


class WindowRetirement(NamedTuple):
    """When the scheduler retires a request's window rows, as
    `request_demand` takes it (`window_retirement`)."""

    prefill_retire_lag: int
    decode_retire_lag: int
    eviction_interval: int


def windows_retire_by_forward() -> bool:
    """Whether decode retires window rows every SGLANG_SWA_EVICTION_INTERVAL
    forwards rather than rows, as Aiter MegaMoE on ROCm does
    (`ScheduleBatch._window_eviction_trigger`)."""
    from sglang.srt.utils import is_hip

    if not is_hip():
        return False
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    return get_moe_a2a_backend().is_megamoe()


def window_retirement(*, shares_prefixes: bool) -> WindowRetirement:
    """When the scheduler retires a request's window rows
    (`ScheduleBatch.maybe_evict_swa`). Without prefix sharing, the overlap
    scheduler retires a prefill's rows a chunk behind its frontier; decode
    retires once SGLANG_SWA_EVICTION_INTERVAL rows fell out of the window,
    behind the processed length, while each step allocates ahead of it (the
    overlap lag, or the speculative reserve `get_alloc_reserve_per_decode`)."""
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
    from sglang.srt.runtime_context import get_schedule, get_spec

    schedule = get_schedule()
    chunk = schedule.chunked_prefill_size or 0
    overlap = not schedule.disable_overlap_schedule
    prefill_lag = chunk if overlap and not shares_prefixes and chunk > 0 else 0
    if get_spec().speculative_algorithm:
        decode_lag = get_alloc_reserve_per_decode()
    else:
        decode_lag = 1 if overlap else 0
    interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())
    if windows_retire_by_forward():
        # Between two retirements every forward but the retiring one commits
        # its rows past the window, up to all of its draft tokens each.
        from sglang.srt.runtime_context import max_speculative_num_draft_tokens

        per_forward = (
            max_speculative_num_draft_tokens() or 1
            if get_spec().speculative_algorithm
            else 1
        )
        interval *= per_forward
    return WindowRetirement(prefill_lag, decode_lag, interval)


class DemandInputs(msgspec.Struct, frozen=True, kw_only=True):
    """The inputs of `request_demand` for one purpose (`demand_inputs`)."""

    purpose: DemandPurpose
    input_tokens: int
    max_new_tokens: int
    windows: Mapping[str, Optional[int]]
    page_size: int
    prefill_retire_lag: int
    eviction_interval: int
    checkpoint_interval: int
    decode_retire_lag: int
    chunk_size: Optional[int]
    state_slots: int
    host_hit_lengths: Mapping[str, int]
    extra_tokens: int
    cached_tokens: int

    @property
    def rule(self) -> DemandRule:
        return DEMAND_RULES[self.purpose]

    def demand(self) -> KVDemand:
        return request_demand(
            self.input_tokens,
            self.max_new_tokens,
            windows=self.windows,
            page_size=self.page_size,
            prefill_retire_lag=self.prefill_retire_lag,
            eviction_interval=self.eviction_interval,
            checkpoint_interval=self.checkpoint_interval,
            decode_retire_lag=self.decode_retire_lag,
            chunk_size=self.chunk_size,
            state_slots=self.state_slots,
            host_hit_lengths=self.host_hit_lengths,
            extra_tokens=self.extra_tokens,
            cached_tokens=self.cached_tokens,
        )


def demand_inputs(
    purpose: DemandPurpose,
    input_tokens: int,
    max_new_tokens: int,
    *,
    windows: Mapping[str, Optional[int]],
    page_size: int,
    chunk_limit: Optional[int],
    prefill_retire_lag: int,
    eviction_interval: int,
    checkpoint_interval: int = 0,
    decode_retire_lag: int = 0,
    state_slots: int = 0,
    cached_tokens: int = 0,
    host_hit_lengths: Optional[Mapping[str, int]] = None,
    extra_tokens: int = 0,
) -> DemandInputs:
    """The `request_demand` inputs every pricing site builds alike; the
    purpose's `DEMAND_RULES` row sets what differs. ``chunk_limit`` is the
    chunk the caller admitted, so a request's admission, shape check and
    commitment price the same chunk."""
    rule = DEMAND_RULES[purpose]
    if purpose is DemandPurpose.CEILING and (cached_tokens or host_hit_lengths):
        raise ValueError("a ceiling prices a request with nothing cached")
    if rule.estimates_decode:
        max_new_tokens = decode_estimate_tokens(max_new_tokens)
    return DemandInputs(
        purpose=purpose,
        input_tokens=input_tokens,
        max_new_tokens=max_new_tokens,
        windows=dict(windows),
        page_size=page_size,
        prefill_retire_lag=prefill_retire_lag,
        eviction_interval=eviction_interval,
        checkpoint_interval=checkpoint_interval,
        decode_retire_lag=decode_retire_lag,
        chunk_size=chunk_limit,
        state_slots=state_slots,
        host_hit_lengths=dict(host_hit_lengths or {}),
        extra_tokens=extra_tokens,
        cached_tokens=cached_tokens,
    )


def largest_fitting_chunk(
    fits: Callable[[int], bool], chunk_limit: int, granule: int
) -> int:
    """The largest multiple of ``granule`` up to ``chunk_limit`` that ``fits``,
    or 0; a binary search, since demand never falls as the chunk grows."""
    low, high = 0, max(0, chunk_limit) // granule
    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid * granule):
            low = mid
        else:
            high = mid - 1
    return low * granule


def chunk_granule(page_size: int, align: Optional[int] = None) -> int:
    """The step a truncated chunk is cut on: a page, or lcm(page, ``align``)
    where the scheduler sets a truncation alignment."""
    return page_size if not align else math.lcm(page_size, align)


def smallest_admissible_chunk(
    chunk_size: Optional[int], page_size: int, align: Optional[int] = None
) -> Optional[int]:
    """The smallest chunk admission may cut a prompt to: the `chunk_granule`
    step, or 0 when ``chunk_size`` is below it; None without chunking. A
    request fits at some searched chunk exactly when it fits at this one."""
    if chunk_size is None:
        return None
    granule = chunk_granule(page_size, align)
    return granule if chunk_size >= granule else 0
