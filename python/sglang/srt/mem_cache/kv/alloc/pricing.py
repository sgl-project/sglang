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
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import msgspec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv.alloc.band import Band


class BandSnapshot(msgspec.Struct, frozen=True, gc=False):
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


class ChainSnapshot(msgspec.Struct, frozen=True, gc=False):
    """Bands in chain order, low byte to high byte."""

    bands: Tuple[BandSnapshot, ...]
    page_size: int
    total_bytes: int
    positions: Dict[str, int]  # band name -> index in `bands`

    def index(self, name: str) -> int:
        return self.positions[name]

    def band(self, name: str) -> BandSnapshot:
        return self.bands[self.positions[name]]

    def high_frontier_below(self, name: str) -> int:
        """High byte frontier of the nearest opaque band below; 0 if none."""
        bands = self.bands
        for i in range(self.positions[name] - 1, -1, -1):
            band = bands[i]
            if band.live or band.grow != "float":
                return band.high_byte
        return 0

    def low_frontier_above(self, name: str) -> int:
        """Low byte frontier of the nearest opaque band above; the top if none."""
        bands = self.bands
        for i in range(self.positions[name] + 1, len(bands)):
            band = bands[i]
            if band.live or band.grow != "float":
                return band.low_byte
        return self.total_bytes

    def gap_bytes(self, name: str) -> int:
        """Free bytes between an end's frontier and the nearest opaque band."""
        band = self.bands[self.positions[name]]
        if band.grow == "up":
            return max(0, self.low_frontier_above(name) - band.high_byte)
        return max(0, band.low_byte - self.high_frontier_below(name))


def snapshot_band(band: Band) -> BandSnapshot:
    floats = band.grow_direction == "float"
    span = band._allocated_pages()
    free_virtual_ids = band.free_virtual_ids
    return BandSnapshot(
        band.sub_pool_name,
        band.grow_direction,
        band.entry_bytes_per_page,
        band.entry_bytes,
        band.page_size,
        band.num_pages,
        band.min_page_index,
        band.lazy_compaction,
        band.moves_blocked(),
        band._free_phys_pages.numel(),
        # `allocated_count() // page_size`: lazy ends count live pages apart
        # from the holes and pending pages inside their span.
        band.live_page_count if band.lazy_compaction else span,
        span,
        band._byte_low_frontier(),
        band._byte_high_frontier(),
        # Only the id owner mints virtual pages; a window binds the owner's.
        0 if free_virtual_ids is None else free_virtual_ids.numel(),
        band.ids.num_virtual_ids,
        band.low_wm_page if floats else 0,
        band.high_wm_page if floats else 0,
    )


def snapshot_chain(
    bands: Tuple[Band, ...],
    *,
    page_size: int,
    total_bytes: int,
    positions: Optional[Dict[str, int]] = None,
) -> ChainSnapshot:
    """``positions`` maps each band's name to its index, when the caller keeps one."""
    snaps = tuple([snapshot_band(band) for band in bands])
    if positions is None:
        positions = {b.name: i for i, b in enumerate(snaps)}
    return ChainSnapshot(snaps, page_size, total_bytes, positions)


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


def _ends_bands(
    snap: ChainSnapshot, ends: TokenEnds
) -> Tuple[BandSnapshot, BandSnapshot]:
    positions, bands = snap.positions, snap.bands
    return bands[positions[ends.owner]], bands[positions[ends.window]]


def ends_compaction_allowed(snap: ChainSnapshot, ends: TokenEnds) -> bool:
    """Compaction is credited only when no member's moves are blocked."""
    fa, sa = _ends_bands(snap, ends)
    return not (fa.blocked or sa.blocked)


def _ends_fit(
    fa: BandSnapshot,
    sa: BandSnapshot,
    gap_bytes: int,
    empty_gap_bytes: int,
    full_pages: int,
    swa_pages: int,
    full_reclaim_pages: int,
    swa_reclaim_pages: int,
    compacted: bool,
    empty_pool: bool,
) -> bool:
    """`ends_fit` on resolved bands; ``gap_bytes`` is the owner's current gap."""
    if min(full_pages, swa_pages) < 0:
        return False
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
            empty_gap_bytes - full_live * fa.page_bytes - swa_live * sa.page_bytes
        )
    else:
        full_virtual_room = fa.free_virtual_pages + full_reclaim_pages
        full_holes = fa.holes + full_reclaim_pages
        swa_holes = sa.holes + swa_reclaim_pages
        full_index_room = fa.num_pages - fa.min_page - fa.span
        swa_index_room = sa.num_pages - sa.min_page - sa.span

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
    fa, sa = _ends_bands(snap, ends)
    return _ends_fit(
        fa,
        sa,
        snap.gap_bytes(ends.owner),
        ends.empty_gap_bytes,
        full_pages,
        swa_pages,
        full_reclaim_pages,
        swa_reclaim_pages,
        compacted,
        empty_pool,
    )


def ends_available_tokens(snap: ChainSnapshot, ends: TokenEnds) -> int:
    """Tokens of equal owner/window demand that fit as the chain stands."""
    fa, sa = _ends_bands(snap, ends)
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
    fa, sa = _ends_bands(snap, ends)
    compacted = empty_pool or not fa.lazy or not (fa.blocked or sa.blocked)
    # A compacted projection prices from the empty chain's gap, not the current one.
    gap_bytes = 0 if compacted else snap.gap_bytes(ends.owner)
    empty_gap_bytes = ends.empty_gap_bytes

    def fits(full_reclaim: int, swa_reclaim: int) -> bool:
        return _ends_fit(
            fa,
            sa,
            gap_bytes,
            empty_gap_bytes,
            full_pages,
            swa_pages,
            full_reclaim,
            swa_reclaim,
            compacted,
            empty_pool,
        )

    if empty_pool:
        return (0, 0) if fits(0, 0) else None
    if fits(0, 0):
        return (0, 0)
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


_FITS = EndsRealization(fits=True)
_REFUSED = EndsRealization(fits=False)
_FITS_AFTER_FLUSH = EndsRealization(fits=True, flush_ends=True)


def ends_realization(
    snap: ChainSnapshot, ends: TokenEnds, full_pages: int, swa_pages: int
) -> EndsRealization:
    fa, sa = _ends_bands(snap, ends)
    gap_bytes, empty_gap_bytes = snap.gap_bytes(ends.owner), ends.empty_gap_bytes
    if _ends_fit(
        fa, sa, gap_bytes, empty_gap_bytes, full_pages, swa_pages, 0, 0, False, False
    ):
        return _FITS
    if not fa.lazy or fa.blocked or sa.blocked:
        return _REFUSED
    if not _ends_fit(
        fa, sa, gap_bytes, empty_gap_bytes, full_pages, swa_pages, 0, 0, True, False
    ):
        return _REFUSED
    return _FITS_AFTER_FLUSH


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
    # `sa.pages_in_band(low_byte=high_band_low, high_byte=...)`, its low end fixed.
    e_f, e_s, full_low = fa.page_bytes, sa.page_bytes, fa.low_byte
    band_lo = max((high_band_low + e_s - 1) // e_s, sa.min_page)
    num_pages = sa.num_pages
    lo, hi = 0, min(h_f + fa.index_room, h_s + sa.index_room)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        full_bytes = (mid - h_f) * e_f if mid > h_f else 0
        if full_bytes > full_gap:
            hi = mid - 1
            continue
        high_end = (full_low - full_bytes) // e_s
        high_room = (high_end if high_end < num_pages else num_pages) - band_lo
        room = low_room if low_room > high_room else high_room
        if mid - h_s <= room or mid <= h_s:
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


def sequential_fit(
    snapshot: ChainSnapshot, demands: Tuple[Tuple[str, int], ...]
) -> bool:
    """Price physical takes in their actual order, without relocation credit.

    Empty floats choose the midpoint of their current neighbours, so changing
    an earlier demand can change a later float's placement. This predicate is
    exact for the no-relief takes, but is not assumed monotone in total demand.
    """
    bands = snapshot.bands
    lows = [b.low_byte for b in bands]
    highs = [b.high_byte for b in bands]
    live = [b.live for b in bands]
    for name, pages in demands:
        if pages < 0:
            return False
        if not pages:
            continue
        i = snapshot.index(name)
        b = bands[i]
        holes = b.holes if b.lazy or b.grow == "float" else 0
        extension = max(0, pages - holes)
        if b.live + pages > b.num_pages - b.min_page:
            return False
        low = next(
            (
                highs[j]
                for j in range(i - 1, -1, -1)
                if live[j] or bands[j].grow != "float"
            ),
            0,
        )
        high = next(
            (
                lows[j]
                for j in range(i + 1, len(bands))
                if live[j] or bands[j].grow != "float"
            ),
            snapshot.total_bytes,
        )
        lo = max(b.min_page, -(-low // b.page_bytes))
        hi = min(b.num_pages, high // b.page_bytes)
        if extension:
            if b.grow == "up":
                end = highs[i] // b.page_bytes + extension
                if end > hi:
                    return False
                highs[i] = end * b.page_bytes
            elif b.grow == "down":
                start = lows[i] // b.page_bytes - extension
                if start < lo:
                    return False
                lows[i] = start * b.page_bytes
            elif not live[i]:
                if extension > hi - lo:
                    return False
                start = lo + (hi - lo - extension) // 2
                lows[i], highs[i] = (
                    start * b.page_bytes,
                    (start + extension) * b.page_bytes,
                )
            else:
                start, end = lows[i] // b.page_bytes, highs[i] // b.page_bytes
                below, above = start - lo, hi - end
                if extension > max(below, above):
                    return False
                if above >= below:
                    highs[i] += extension * b.page_bytes
                else:
                    lows[i] -= extension * b.page_bytes
        live[i] += pages
    return True
