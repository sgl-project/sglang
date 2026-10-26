"""Capacity pricing over a host-only snapshot of a chain's bands.

A snapshot holds counts and byte frontiers, never hole positions (those live in
device tensors), so taking one and pricing over it never synchronizes with the
device. Every capacity decision of `ChainAllocator` is one of the pure functions
below applied to a fresh snapshot; the chain then performs the realization the
decision names (end flushes, a float relocation) and the takes.

The per-shape rules are the chains' current ones: a chain of two token ends
prices the shared gap jointly and credits compaction only when no member's moves
are blocked; a chain whose window floats between the owner and a state end
prices its float band in place and credits end flushes per member.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import msgspec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator


class BandSnapshot(msgspec.Struct, frozen=True, kw_only=True):
    """One band's counts and byte frontiers; pages are this band's own pages."""

    name: str
    grow: str  # "up", "down" or "float"
    page_bytes: int  # bytes of one page entry
    entry_bytes: int  # bytes of one token or slot entry
    page_size: int
    num_pages: int
    min_page: int
    lazy: bool
    blocked: bool
    holes: int  # free physical pages inside the span
    live: int  # live pages
    span: (
        int  # allocated pages: the watermark span of an end, the live pages of a float
    )
    low_byte: int
    high_byte: int
    free_virtual_pages: int
    num_virtual_pages: int
    low_page: int = 0  # float only
    high_page: int = 0  # float only

    @property
    def transparent(self) -> bool:
        """An empty float: its neighbours' frontier walks see through it."""
        return self.grow == "float" and self.live == 0

    @property
    def drains(self) -> bool:
        """An urgent flush would release this band's holes."""
        return self.lazy and not self.blocked

    @property
    def index_room(self) -> int:
        return self.num_pages - self.min_page - self.span

    def pages_in_band(self, *, low_byte: int, high_byte: int) -> int:
        """Pages of this band's grid inside ``[low_byte, high_byte)``."""
        lo = max((low_byte + self.page_bytes - 1) // self.page_bytes, self.min_page)
        hi = min(high_byte // self.page_bytes, self.num_pages)
        return max(0, hi - lo)


class ChainSnapshot(msgspec.Struct, frozen=True, kw_only=True):
    """Bands in chain order, low byte to high byte."""

    bands: Tuple[BandSnapshot, ...]
    page_size: int
    total_bytes: int

    def index(self, name: str) -> int:
        for i, band in enumerate(self.bands):
            if band.name == name:
                return i
        raise KeyError(name)

    def band(self, name: str) -> BandSnapshot:
        return self.bands[self.index(name)]

    def high_frontier_below(self, name: str) -> int:
        """High byte frontier of the nearest opaque band below; 0 if none."""
        for band in reversed(self.bands[: self.index(name)]):
            if not band.transparent:
                return band.high_byte
        return 0

    def low_frontier_above(self, name: str) -> int:
        """Low byte frontier of the nearest opaque band above; the top if none."""
        for band in self.bands[self.index(name) + 1 :]:
            if not band.transparent:
                return band.low_byte
        return self.total_bytes

    def gap_bytes(self, name: str) -> int:
        """Free bytes between an end's frontier and the nearest opaque band."""
        band = self.band(name)
        if band.grow == "up":
            return max(0, self.low_frontier_above(name) - band.high_byte)
        return max(0, band.low_byte - self.high_frontier_below(name))


def snapshot_band(band: MultiEndedAllocator) -> BandSnapshot:
    floats = band.grow_direction == "float"
    return BandSnapshot(
        name=band.sub_pool_name,
        grow=band.grow_direction,
        page_bytes=band.entry_bytes_per_page,
        entry_bytes=band.entry_bytes,
        page_size=band.page_size,
        num_pages=band.num_pages,
        min_page=band.min_page_index,
        lazy=band.lazy_compaction,
        blocked=band.moves_blocked(),
        holes=int(band._free_phys_pages.numel()),
        live=band.allocated_count() // band.page_size,
        span=band._allocated_pages(),
        low_byte=band._byte_low_frontier(),
        high_byte=band._byte_high_frontier(),
        # Only the id owner mints virtual pages; a window binds the owner's.
        free_virtual_pages=(
            0 if band.free_virtual_ids is None else int(band.free_virtual_ids.numel())
        ),
        num_virtual_pages=band.num_virtual_ids,
        low_page=band.low_wm_page if floats else 0,
        high_page=band.high_wm_page if floats else 0,
    )


def snapshot_chain(
    bands: Tuple[MultiEndedAllocator, ...], *, page_size: int, total_bytes: int
) -> ChainSnapshot:
    return ChainSnapshot(
        bands=tuple(snapshot_band(band) for band in bands),
        page_size=page_size,
        total_bytes=total_bytes,
    )


def _first_fit(high: int, predicate: Callable[[int], bool]) -> int:
    """Smallest value in [0, high] for which the monotone predicate holds."""
    low = 0
    while low < high:
        mid = (low + high) // 2
        if predicate(mid):
            high = mid
        else:
            low = mid + 1
    return low


# ---------------------------------------------------------------------------
# Two token ends: the owner and its window, sharing the gap between them
# ---------------------------------------------------------------------------


class TokenEnds(msgspec.Struct, frozen=True, kw_only=True):
    """Constants of a two-end chain fixed at construction (or post-capture resize)."""

    owner: str
    window: str
    empty_gap_bytes: int  # the shared gap of the empty chain
    size_full: int  # tokens
    size_swa: int  # tokens


def ends_compaction_allowed(snap: ChainSnapshot, ends: TokenEnds) -> bool:
    """Compaction is credited only when no member's moves are blocked."""
    return not (snap.band(ends.owner).blocked or snap.band(ends.window).blocked)


def ends_fit(
    snap: ChainSnapshot,
    ends: TokenEnds,
    full_pages: int,
    swa_pages: int,
    *,
    full_reclaim_pages: int = 0,
    swa_reclaim_pages: int = 0,
    compacted: bool,
    empty_pool: bool = False,
) -> bool:
    """Whether one owner/window page demand fits, as the chain stands or after
    both ends pack to their live pages (``compacted``)."""
    if min(full_pages, swa_pages) < 0:
        return False
    fa, sa = snap.band(ends.owner), snap.band(ends.window)
    if empty_pool:
        full_live = swa_live = 0
        full_reclaim_pages = swa_reclaim_pages = 0
    else:
        full_live, swa_live = fa.live, sa.live
        full_reclaim_pages = min(full_live, max(0, int(full_reclaim_pages)))
        swa_reclaim_pages = min(swa_live, max(0, int(swa_reclaim_pages)))
    full_live -= full_reclaim_pages
    swa_live -= swa_reclaim_pages
    virtual_capacity = fa.num_virtual_pages - fa.min_page
    full_capacity = min(virtual_capacity, fa.num_pages - fa.min_page)
    swa_capacity = min(virtual_capacity, sa.num_pages - sa.min_page)
    if full_live + full_pages > full_capacity or swa_live + swa_pages > swa_capacity:
        return False

    if compacted:
        full_virtual_room = virtual_capacity - full_live
        full_holes = swa_holes = 0
        full_index_room = full_capacity - full_live
        swa_index_room = swa_capacity - swa_live
        gap_bytes = (
            ends.empty_gap_bytes - full_live * fa.page_bytes - swa_live * sa.page_bytes
        )
    else:
        full_virtual_room = fa.free_virtual_pages + full_reclaim_pages
        full_holes = fa.holes + full_reclaim_pages
        swa_holes = sa.holes + swa_reclaim_pages
        full_index_room = fa.index_room
        swa_index_room = sa.index_room
        gap_bytes = snap.gap_bytes(ends.owner)

    if full_pages > full_virtual_room:
        return False
    if full_pages > full_holes + full_index_room:
        return False
    if swa_pages > swa_holes + swa_index_room:
        return False
    full_extensions = max(0, full_pages - full_holes)
    swa_extensions = max(0, swa_pages - swa_holes)
    return full_extensions * fa.page_bytes + swa_extensions * sa.page_bytes <= max(
        0, gap_bytes
    )


def ends_available_tokens(snap: ChainSnapshot, ends: TokenEnds) -> int:
    """Tokens of equal owner/window demand that fit as the chain stands."""
    fa, sa = snap.band(ends.owner), snap.band(ends.window)
    e_f, e_s = fa.page_bytes, sa.page_bytes
    if fa.grow == "up":
        gap_bytes = max(0, sa.low_byte - fa.high_byte)
    else:
        gap_bytes = max(0, fa.low_byte - sa.high_byte)
    r_f, r_s = fa.index_room, sa.index_room
    if not fa.lazy:
        return min(gap_bytes // (e_f + e_s), r_f, r_s) * snap.page_size

    h_f, h_s = fa.holes, sa.holes
    both_drain = min(h_f, h_s)
    # The side with fewer holes extends while the other keeps draining.
    e_extend, drain_max = (e_f, h_s) if h_f <= h_s else (e_s, h_f)
    room = drain_max - both_drain
    one_extends = min(room, gap_bytes // e_extend) if e_extend > 0 else room
    gap_bytes -= one_extends * e_extend
    both_extend = gap_bytes // (e_f + e_s)
    total = min(both_drain + one_extends + both_extend, h_f + r_f, h_s + r_s)
    return total * snap.page_size


def ends_reclaim_plan(
    snap: ChainSnapshot,
    ends: TokenEnds,
    full_tokens: float,
    swa_tokens: float,
    *,
    full_evictable_tokens: int = 0,
    swa_evictable_tokens: int = 0,
    empty_pool: bool = False,
    prefer_swa: bool = False,
) -> Optional[Tuple[int, int]]:
    """Cumulative owner/window eviction targets in tokens, or None if impossible.

    The tree evicts owner pages before window pages: minimize the window reclaim
    with every evictable owner page, then trim the owner's (reversed under
    ``prefer_swa``).
    """
    if full_tokens < 0 or swa_tokens < 0:
        return None
    ps = snap.page_size
    full_pages = (math.ceil(full_tokens) + ps - 1) // ps
    swa_pages = (math.ceil(swa_tokens) + ps - 1) // ps
    fa = snap.band(ends.owner)
    compacted = empty_pool or not fa.lazy or ends_compaction_allowed(snap, ends)

    def fits(full_reclaim: int, swa_reclaim: int) -> bool:
        return ends_fit(
            snap,
            ends,
            full_pages,
            swa_pages,
            full_reclaim_pages=full_reclaim,
            swa_reclaim_pages=swa_reclaim,
            compacted=compacted,
            empty_pool=empty_pool,
        )

    if empty_pool:
        return (0, 0) if fits(0, 0) else None
    if fits(0, 0):
        return (0, 0)
    sa = snap.band(ends.window)
    max_full = min(fa.live, max(0, int(full_evictable_tokens)) // ps)
    max_swa = min(sa.live, max(0, int(swa_evictable_tokens)) // ps)
    if not fits(max_full, max_swa):
        return None
    if prefer_swa:
        full_reclaim = _first_fit(max_full, lambda v: fits(v, max_swa))
        swa_reclaim = _first_fit(max_swa, lambda v: fits(full_reclaim, v))
    else:
        swa_reclaim = _first_fit(max_swa, lambda v: fits(max_full, v))
        full_reclaim = _first_fit(max_full, lambda v: fits(v, swa_reclaim))
    return (full_reclaim * ps, swa_reclaim * ps)


def ends_can_reserve(
    snap: ChainSnapshot,
    ends: TokenEnds,
    full_tokens: float,
    swa_tokens: float,
    *,
    full_evictable_tokens: int = 0,
    swa_evictable_tokens: int = 0,
    empty_pool: bool = False,
    require_token_slack: bool = False,
) -> bool:
    """Admission's question: does the demand fit once evictable pages are reclaimed."""
    if full_tokens < 0 or swa_tokens < 0:
        return False
    if require_token_slack and (
        full_tokens >= ends.size_full or swa_tokens >= ends.size_swa
    ):
        return False
    return (
        ends_reclaim_plan(
            snap,
            ends,
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
            empty_pool=empty_pool,
        )
        is not None
    )


class EndsRealization(msgspec.Struct, frozen=True):
    """What allocating a demand on a two-end chain needs: nothing, both ends
    flushed first, or it cannot fit."""

    fits: bool
    flush_ends: bool = False


def ends_realization(
    snap: ChainSnapshot, ends: TokenEnds, full_pages: int, swa_pages: int
) -> EndsRealization:
    if ends_fit(snap, ends, full_pages, swa_pages, compacted=False):
        return EndsRealization(fits=True)
    fa = snap.band(ends.owner)
    if not fa.lazy or not ends_compaction_allowed(snap, ends):
        return EndsRealization(fits=False)
    if not ends_fit(snap, ends, full_pages, swa_pages, compacted=True):
        return EndsRealization(fits=False)
    return EndsRealization(fits=True, flush_ends=True)


# ---------------------------------------------------------------------------
# A state end, the window floating in the middle, and the owner end
# ---------------------------------------------------------------------------


class FloatChain(msgspec.Struct, frozen=True, kw_only=True):
    owner: str
    window: str
    state: str


def float_fit(
    snap: ChainSnapshot, chain: FloatChain, full_pages: int, swa_pages: int
) -> bool:
    """The owner takes first, then the window extends one side of its band."""
    fa, sa = snap.band(chain.owner), snap.band(chain.window)
    h_f = fa.holes if fa.lazy else 0
    h_s = sa.holes
    if full_pages > h_f + fa.index_room or swa_pages > h_s + sa.index_room:
        return False
    full_bytes = max(0, full_pages - h_f) * fa.page_bytes
    if full_bytes > snap.gap_bytes(chain.owner):
        return False
    ext_s = max(0, swa_pages - h_s)
    full_low_after = fa.low_byte - full_bytes
    below = snap.high_frontier_below(chain.window)
    if sa.transparent:
        return ext_s <= sa.pages_in_band(low_byte=below, high_byte=full_low_after)
    low_room = sa.pages_in_band(low_byte=below, high_byte=sa.low_byte)
    high_room = sa.pages_in_band(low_byte=sa.high_byte, high_byte=full_low_after)
    return ext_s <= max(low_room, high_room)


def float_available_tokens(snap: ChainSnapshot, chain: FloatChain) -> int:
    """Tokens of equal owner/window demand that fit as the chain stands."""
    fa, sa = snap.band(chain.owner), snap.band(chain.window)
    h_f = fa.holes if fa.lazy else 0
    h_s = sa.holes
    full_gap = snap.gap_bytes(chain.owner)
    below = snap.high_frontier_below(chain.window)
    if sa.transparent:
        high_band_low, low_room = below, 0
    else:
        high_band_low = sa.high_byte
        low_room = sa.pages_in_band(low_byte=below, high_byte=sa.low_byte)
    lo, hi = 0, min(h_f + fa.index_room, h_s + sa.index_room)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        full_bytes = max(0, mid - h_f) * fa.page_bytes
        room = max(
            low_room,
            sa.pages_in_band(
                low_byte=high_band_low, high_byte=fa.low_byte - full_bytes
            ),
        )
        if full_bytes <= full_gap and max(0, mid - h_s) <= room:
            lo = mid
        else:
            hi = mid - 1
    return lo * snap.page_size


def float_can_reserve(
    snap: ChainSnapshot,
    chain: FloatChain,
    full_tokens: float,
    swa_tokens: float,
    *,
    full_evictable_tokens: int = 0,
    swa_evictable_tokens: int = 0,
    empty_pool: bool = False,
) -> bool:
    """The demand fits as the chain stands; evictable credit and the empty-pool
    projection are not priced on this chain."""
    if (
        full_tokens < 0
        or swa_tokens < 0
        or full_evictable_tokens
        or swa_evictable_tokens
        or empty_pool
    ):
        return False
    ps = snap.page_size
    return float_fit(
        snap, chain, math.ceil(full_tokens / ps), math.ceil(swa_tokens / ps)
    )


def float_token_id_capacity(
    snap: ChainSnapshot, chain: FloatChain, num_tokens: int
) -> bool:
    pages = num_tokens // snap.page_size
    sa = snap.band(chain.window)
    return (
        pages <= snap.band(chain.owner).free_virtual_pages
        and pages <= sa.num_pages - sa.min_page - sa.live
    )


def _float_end_frontiers(
    snap: ChainSnapshot, chain: FloatChain, num_tokens: int, *, flush_ends: bool
) -> Tuple[int, int, int]:
    """Owner before and after the take and the state frontier after the end
    flushes a member's gate allows; pending reuse is never credited."""
    fa, ma = snap.band(chain.owner), snap.band(chain.state)

    def end_credit(band: BandSnapshot) -> int:
        if not flush_ends or not band.lazy or band.blocked:
            return 0
        return band.holes

    full_credit, state_credit = end_credit(fa), end_credit(ma)
    full_holes = fa.holes if fa.lazy else 0
    pages = num_tokens // snap.page_size
    full_extension = max(0, pages - (full_holes - full_credit))
    full_before = fa.low_byte + full_credit * fa.page_bytes
    full_after = full_before - full_extension * fa.page_bytes
    state_after = ma.high_byte - state_credit * ma.page_bytes
    return full_before, full_after, state_after


def float_end_certificate(
    snap: ChainSnapshot, chain: FloatChain, num_tokens: int
) -> bool:
    """Equal demand fits after the end flushes the members' gates allow."""
    if not float_token_id_capacity(snap, chain, num_tokens):
        return False
    if num_tokens <= float_available_tokens(snap, chain):
        return True
    fa, sa = snap.band(chain.owner), snap.band(chain.window)
    _, full_after, state_after = _float_end_frontiers(
        snap, chain, num_tokens, flush_ends=True
    )
    pages = num_tokens // snap.page_size
    if full_after < fa.min_page * fa.page_bytes:
        return False
    if sa.transparent:
        return pages <= sa.pages_in_band(low_byte=state_after, high_byte=full_after)
    if full_after < sa.high_byte:
        return False
    extension = max(0, pages - sa.holes)
    return extension <= max(
        sa.pages_in_band(low_byte=state_after, high_byte=sa.low_byte),
        sa.pages_in_band(low_byte=sa.high_byte, high_byte=full_after),
    )


def float_high_target(
    snap: ChainSnapshot, chain: FloatChain, num_tokens: int, *, flush_ends: bool = True
) -> Optional[int]:
    """Bytes the float must open on its high side after the end flushes, so the
    owner's take and the window's pages fit; None when no retreat serves."""
    if not float_token_id_capacity(snap, chain, num_tokens):
        return None
    fa, sa = snap.band(chain.owner), snap.band(chain.window)
    if sa.transparent or sa.blocked:
        return None
    full_before, full_after, state_after = _float_end_frontiers(
        snap, chain, num_tokens, flush_ends=flush_ends
    )
    if full_after < fa.min_page * fa.page_bytes:
        return None
    e_s = sa.page_bytes
    low = max(-(-state_after // e_s), sa.min_page)
    high = min(full_before // e_s, sa.num_pages)
    high_after = min(full_after // e_s, sa.num_pages)
    limit = high_after - num_tokens // snap.page_size
    retreat = sa.high_page - limit
    if retreat <= 0 or sa.low_page - low < retreat:
        return None
    return (high - limit) * e_s


def float_reclaim_satisfied(
    snap: ChainSnapshot, chain: FloatChain, num_tokens: int
) -> bool:
    """A sufficient allocation certificate; False leaves the float geometry unknown."""
    return float_end_certificate(snap, chain, num_tokens) or (
        float_high_target(snap, chain, num_tokens) is not None
    )


def float_byte_shortfall(
    snap: ChainSnapshot,
    chain: FloatChain,
    num_tokens: int,
    *,
    full_reclaim: int = 0,
    swa_reclaim: int = 0,
    state_reclaim: int = 0,
) -> int:
    """Bytes missing even if every band packed; zero does not certify the float."""
    fa, sa, ma = (snap.band(n) for n in (chain.owner, chain.window, chain.state))
    live_bytes = sum(
        max(0, band.live * band.page_size - reclaim) * band.entry_bytes
        for band, reclaim in (
            (fa, full_reclaim),
            (sa, swa_reclaim),
            (ma, state_reclaim),
        )
    )
    sink_bytes = min(band.min_page * band.page_bytes for band in (fa, sa, ma))
    required = num_tokens // snap.page_size * (fa.page_bytes + sa.page_bytes)
    return max(0, required + live_bytes + sink_bytes - snap.total_bytes)
