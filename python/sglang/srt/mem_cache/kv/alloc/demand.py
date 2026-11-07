"""What an allocation takes from each sub-pool, and what a request needs."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Callable, Iterable, Mapping, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class KVDemand:
    """What an allocation takes: tokens of the token sub-pool, tokens of each
    window sub-pool by its name, and state slots.

    Admission and the take need every window named (`checked`): a window left
    out is an error, not a window that takes nothing, so no window's share is
    guessed. An eviction request may name only the sub-pools it is short of
    (`completed` fills the rest with zero)."""

    full_tokens: int = 0
    window_tokens: Mapping[str, int] = dataclasses.field(default_factory=dict)
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
        return dataclasses.replace(self, window_tokens=windows).checked(names)

    def is_empty(self) -> bool:
        return not (
            self.full_tokens > 0
            or self.state_slots > 0
            or any(t > 0 for t in self.window_tokens.values())
        )


def estimate_window_tokens(
    extend_input_len: int,
    max_new_tokens: int,
    *,
    sliding_window_size: Optional[int],
    page_size: int,
    allocation_limit: Optional[int] = None,
    host_hit_length: int = 0,
) -> int:
    """Peak window reservation for one prefill/decode request."""
    if sliding_window_size is None or sliding_window_size <= 0:
        reserved = extend_input_len + max_new_tokens + page_size
    else:
        allocated = (
            extend_input_len
            if allocation_limit is None
            else min(extend_input_len, allocation_limit)
        )
        allocated_tail = max(allocated - sliding_window_size, 0)
        # With a roughly two-window SWA pool, a cached prefix can already lock
        # one window. Charging another whole window for a short resume can then
        # block admission forever on an idle pool. Reserve only its uncached
        # tail plus decode headroom, capped at the window.
        # Including the extension also keeps the reservation large enough for
        # this pass's prefill allocation.
        reserved = (
            allocated_tail
            + min(extend_input_len + max_new_tokens, sliding_window_size)
            + page_size
        )
    if host_hit_length > 0:
        reserved += -(-host_hit_length // page_size) * page_size
    return reserved


def request_demand(
    input_tokens: int,
    max_new_tokens: int,
    *,
    windows: Mapping[str, Optional[int]],
    page_size: int,
    prefill_retire_lag: int,
    eviction_interval: int,
    checkpoint_interval: int = 0,
    chunk_size: Optional[int] = None,
    state_slots: int = 0,
    host_hit_lengths: Optional[Mapping[str, int]] = None,
    extra_tokens: int = 0,
    new_window: bool = False,
    cached_tokens: int = 0,
) -> KVDemand:
    """What one request still needs to place ``input_tokens`` prompt tokens,
    in chunks of at most ``chunk_size``, and then decode ``max_new_tokens``:
    the page-ceiled prompt, its decode and one page of allocation rounding in
    the token sub-pool (plus ``extra_tokens`` its caller charges on top), the
    peak each window sub-pool of ``windows`` (name to retained window) holds,
    and ``state_slots``.

    A window's peak follows when the scheduler retires its rows, which the
    caller states: a prefill chunk retires the rows behind its frontier less
    ``prefill_retire_lag`` tokens (the overlap scheduler keeps one configured
    chunk more for the prefill still running), and decode retires them only
    once ``eviction_interval`` tokens fell out of the window. A cache that
    keeps state checkpoints every ``checkpoint_interval`` tokens (0 without)
    keeps the window behind the latest one too (its retention floor), which
    trails a decoding frontier by up to an interval less one row. So decode
    holds up to the window and the larger interval less one row, and with
    ``new_window`` (a first chunk) a request of more than one chunk keeps, between its chunks, the
    window and the lag that end at its last row beside the next chunk. Its
    cached prefix of ``cached_tokens`` already holds up to a window of those
    rows, so it takes the rest: the retained rows, less the window rows the
    prefix holds, plus the next chunk. With no prefix it takes the whole
    window; a prefix of a whole window or more leaves it the chunk alone.

    One function prices a request wherever its demand is asked: the memory
    clip of a lone request in an empty pool, its admission, and the
    commitment a chunked request keeps until its last chunk. Priced alike,
    a request the clip lets through is one its chunks can finish."""
    hits = host_hit_lengths or {}

    if prefill_retire_lag < 0 or eviction_interval < 1 or checkpoint_interval < 0:
        raise ValueError(
            f"retirement lag {prefill_retire_lag}, interval {eviction_interval} "
            f"and checkpoint interval {checkpoint_interval}"
        )
    # Rows a decode keeps past the window: until an eviction interval fell out
    # of it, and behind the latest state checkpoint, which only a decode
    # moves on (a request with no decode keeps the interval alone).
    retention = (
        max(eviction_interval, checkpoint_interval)
        if max_new_tokens > 0
        else eviction_interval
    )

    def window_peak(name: str, window: Optional[int]) -> int:
        hit = hits.get(name, 0)
        peak = estimate_window_tokens(
            input_tokens,
            max_new_tokens,
            sliding_window_size=window,
            page_size=page_size,
            allocation_limit=chunk_size,
            host_hit_length=hit,
        )
        if window is None or window <= 0:
            return peak
        hit_pages = -(-hit // page_size) * page_size if hit > 0 else 0
        # Decode retires a window's rows once an interval of them fell out of
        # it, and never past the window behind the latest checkpoint: until
        # then it holds up to the larger interval less one row more.
        decode = (
            min(input_tokens + max_new_tokens, window + retention - 1)
            + page_size
            + hit_pages
        )
        peak = max(peak, decode)
        if not (new_window and chunk_size is not None and input_tokens > chunk_size):
            return peak
        retained = min(
            window + prefill_retire_lag, cached_tokens + input_tokens - chunk_size
        )
        held_by_prefix = min(window, cached_tokens)
        return max(peak, retained - held_by_prefix + chunk_size + page_size + hit_pages)

    return KVDemand(
        -(-input_tokens // page_size) * page_size
        + max_new_tokens
        + page_size
        + extra_tokens,
        {name: window_peak(name, window) for name, window in windows.items()},
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


@dataclasses.dataclass(frozen=True)
class DemandRule:
    """How one purpose differs from the others. Everything else (the chunk,
    the windows, the new window, the token sub-pool's rounding page) is
    built alike for every purpose."""

    # The decode priced is the scheduler's estimate, at most
    # SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION tokens. A ceiling prices all of
    # ``max_new_tokens``: the generation it lets through must fit.
    estimates_decode: bool
    state_slots: StateSlots
    # Whether cached entries the prefix cache could evict count as room. A
    # load-back reclaims first and asks what is free then; a ceiling asks an
    # empty pool.
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


def window_retirement(*, shares_prefixes: bool) -> Tuple[int, int]:
    """When the scheduler retires a request's window rows, as
    `request_demand` takes it: ``(prefill_retire_lag, eviction_interval)``.
    Without prefix sharing, the overlap scheduler retires a prefill's rows a
    configured chunk behind its frontier, which the prefill still running
    reads (`ScheduleBatch.maybe_evict_swa`); decode retires once
    SGLANG_SWA_EVICTION_INTERVAL rows fell out of the window."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_schedule

    schedule = get_schedule()
    chunk = schedule.chunked_prefill_size or 0
    overlap = not schedule.disable_overlap_schedule
    lag = chunk if overlap and not shares_prefixes and chunk > 0 else 0
    return lag, max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())


@dataclasses.dataclass(frozen=True)
class DemandInputs:
    """The inputs of `request_demand` for one purpose (`demand_inputs`)."""

    purpose: DemandPurpose
    input_tokens: int
    max_new_tokens: int
    windows: Mapping[str, Optional[int]]
    page_size: int
    prefill_retire_lag: int
    eviction_interval: int
    checkpoint_interval: int
    chunk_size: Optional[int]
    state_slots: int
    host_hit_lengths: Mapping[str, int]
    extra_tokens: int
    new_window: bool
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
            chunk_size=self.chunk_size,
            state_slots=self.state_slots,
            host_hit_lengths=self.host_hit_lengths,
            extra_tokens=self.extra_tokens,
            new_window=self.new_window,
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
    state_slots: int = 0,
    cached_tokens: int = 0,
    continuation: bool = False,
    host_hit_lengths: Optional[Mapping[str, int]] = None,
    extra_tokens: int = 0,
) -> DemandInputs:
    """The inputs every site that prices a request builds alike: the
    ``input_tokens`` it still places in chunks of at most ``chunk_limit``,
    the ``windows`` it keeps (sub-pool name to retained window), one rounding
    page in the token sub-pool (``extra_tokens`` are the caller's charges on
    top of it), the scheduler's window retirement (``prefill_retire_lag``,
    ``eviction_interval``, as `request_demand` takes them), and the window a
    first chunk keeps beside its next chunk (`request_demand`'s
    ``new_window``), less what its ``cached_tokens`` already hold; a later
    chunk (``continuation``) holds the window its earlier chunks took. The purpose's row of `DEMAND_RULES` sets what differs:
    the decode estimate and which ``state_slots`` the caller counts (and, for
    the caller to apply, whether eviction is credited). ``chunk_limit`` is the
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
        chunk_size=chunk_limit,
        state_slots=state_slots,
        host_hit_lengths=dict(host_hit_lengths or {}),
        extra_tokens=extra_tokens,
        new_window=not continuation,
        cached_tokens=cached_tokens,
    )


def largest_fitting_chunk(
    fits: Callable[[int], bool], chunk_limit: int, granule: int
) -> int:
    """The largest multiple of ``granule`` at most ``chunk_limit`` for which
    ``fits`` holds, or 0 when none does. A request's demand only grows with
    its chunk (`request_demand`), so ``fits`` is monotone and a binary search
    finds it."""
    low, high = 0, max(0, chunk_limit) // granule
    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid * granule):
            low = mid
        else:
            high = mid - 1
    return low * granule


def chunk_granule(page_size: int, align: Optional[int] = None) -> int:
    """The step a truncated chunk is cut on: one page, or the least common
    multiple of a page and the scheduler's truncation alignment where it
    sets one. A chunk limit off this grid truncates to a shorter chunk, and
    one below it to none."""
    return page_size if not align else math.lcm(page_size, align)


def smallest_admissible_chunk(
    chunk_size: Optional[int], page_size: int, align: Optional[int] = None
) -> Optional[int]:
    """The smallest chunk admission may cut a prompt to: the grid step
    (`chunk_granule`), when the configured ``chunk_size`` reaches it; 0 when
    it does not, since no positive chunk of the grid is admissible then, as
    `largest_fitting_chunk` finds none. None without chunking. Since demand
    only grows with the chunk, a request fits at some chunk
    `largest_fitting_chunk` searches exactly when it fits at this one."""
    if chunk_size is None:
        return None
    granule = chunk_granule(page_size, align)
    return granule if chunk_size >= granule else 0
