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
    # A pin source is busy: the whole band stays where it is.
    pinned: bool = False
    # Pages ``[first, end)`` holding every page that cannot move (a load
    # reservation, a quarantined page); the band's extent must keep covering
    # them. None when there is none.
    fixed: Optional[Tuple[int, int]] = None

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
        band.pinned_by_source(),
        band.fixed_extent(),
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


def reclaimed_snapshot(
    snapshot: ChainSnapshot, reclaim_pages: Dict[str, int], *, empty_pool: bool = False
) -> ChainSnapshot:
    """Conservative geometry after reclaim, without guessing where holes lie.

    Unblocked ends can realize their compacted frontiers by an urgent flush.
    Pinned ends retain their spans. Partial float frees remain holes in place;
    empty floats become transparent. No live float relocation is credited.
    """
    bands = []
    for b in snapshot.bands:
        reclaimed = (
            b.live if empty_pool else min(b.live, max(0, reclaim_pages.get(b.name, 0)))
        )
        live = b.live - reclaimed
        low, high, span = b.low_byte, b.high_byte, b.span
        holes = b.holes + reclaimed
        if empty_pool or (not live and (b.grow == "float" or not b.blocked)):
            span = holes = 0
            low = high = (
                b.num_pages if b.grow == "down" else b.min_page
            ) * b.page_bytes
        elif b.grow != "float" and not b.blocked:
            span, holes = live, 0
            if b.grow == "up":
                high = (b.min_page + live) * b.page_bytes
            else:
                low = (b.num_pages - live) * b.page_bytes
        bands.append(
            msgspec.structs.replace(
                b,
                live=live,
                span=span,
                holes=holes,
                low_byte=low,
                high_byte=high,
                free_virtual_pages=(
                    b.num_virtual_pages - b.min_page
                    if empty_pool
                    else b.free_virtual_pages + reclaimed
                ),
            )
        )
    return msgspec.structs.replace(snapshot, bands=tuple(bands))


class PhysicalTake(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """A holes-first take, with an explicit extension and its source geometry."""

    pages: int
    holes: int
    live: int
    low_byte: int
    high_byte: int
    fresh: Optional[Tuple[int, int]] = None

    def prefix(self, pages: int, *, page_bytes: int) -> PhysicalTake:
        """Consume part of this take while staying inside its certified span."""
        if not 0 <= pages <= self.pages:
            raise ValueError("a physical take prefix exceeds its reservation")
        fresh = None
        if self.fresh is not None:
            start, end = self.fresh
            reuse = self.pages - (end - start)
            extension = max(0, pages - reuse)
            if extension:
                # Low-side growth must remain adjacent to the live span.
                fresh = (
                    (end - extension, end)
                    if end * page_bytes == self.low_byte
                    else (start, start + extension)
                )
        return msgspec.structs.replace(self, pages=pages, fresh=fresh)


class NoReliefPlan(msgspec.Struct, frozen=True, gc=False):
    takes: Dict[str, PhysicalTake]


def no_relief_plan(
    snapshot: ChainSnapshot, demands: Tuple[Tuple[str, int], ...]
) -> Optional[NoReliefPlan]:
    """Place takes without moving live pages or releasing boundary holes.

    A live float may extend on either side; empty floats pack in chain order
    between fixed spans. Of the feasible choices for a fixed span, the lowest
    high frontier dominates every other choice for the remaining bands. The
    walk is linear in the number of bands and feasibility is downward closed
    in demand. Allocation must use the returned extensions, not its usual
    midpoint/roomier-side heuristic.
    """
    counts = dict(demands)
    if len(counts) != len(demands) or any(
        name not in snapshot.positions for name in counts
    ):
        raise ValueError("physical demand must name distinct chain bands")
    takes = {}
    pending = []
    frontier = 0

    def take(b, pages, fresh=None):
        return PhysicalTake(
            pages=pages,
            holes=b.holes,
            live=b.live,
            low_byte=b.low_byte,
            high_byte=b.high_byte,
            fresh=fresh,
        )

    def pack_empty(high):
        edge, packed = frontier, {}
        for b, pages in pending:
            start = max(b.min_page, -(-edge // b.page_bytes))
            end = start + pages
            if end > b.num_pages or end * b.page_bytes > high:
                return None
            packed[b.name] = take(b, pages, (start, end))
            edge = end * b.page_bytes
        return packed

    for b in snapshot.bands:
        pages = counts.get(b.name, 0)
        if pages < 0 or b.live + pages > b.num_pages - b.min_page:
            return None
        if b.transparent:
            if b.holes:
                raise ValueError("an empty float must have released its hole span")
            if pages:
                pending.append((b, pages))
            else:
                takes[b.name] = take(b, 0)
            continue
        holes = b.holes if b.lazy or b.grow == "float" else 0
        extension = max(0, pages - holes)
        delta = extension * b.page_bytes
        if b.grow == "up":
            choices = ((b.low_byte, b.high_byte + delta),)
        elif b.grow == "down":
            choices = ((b.low_byte - delta, b.high_byte),)
        elif extension:
            choices = (
                (b.low_byte - delta, b.high_byte),
                (b.low_byte, b.high_byte + delta),
            )
        else:
            choices = ((b.low_byte, b.high_byte),)
        for low, high in choices:
            if low < b.min_page * b.page_bytes or high > b.num_pages * b.page_bytes:
                continue
            if low < frontier:
                continue
            packed = pack_empty(low)
            if packed is None:
                continue
            fresh = None
            if extension:
                fresh = (
                    (low // b.page_bytes, b.low_byte // b.page_bytes)
                    if low < b.low_byte
                    else (b.high_byte // b.page_bytes, high // b.page_bytes)
                )
            takes.update(packed)
            takes[b.name] = take(b, pages, fresh)
            frontier = high
            pending = []
            break
        else:
            return None
    packed = pack_empty(snapshot.total_bytes)
    if packed is None:
        return None
    takes.update(packed)
    return NoReliefPlan(takes)


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


# ---------------------------------------------------------------------------
# One planner for every chain: whether a demand fits, and the layout serving it
# ---------------------------------------------------------------------------

# How far a plan may rearrange the chain: nothing moves; unblocked ends release
# their holes by compaction; unblocked floats also relocate.
NO_MOVE = "NO_MOVE"
FLUSH_ENDS = "FLUSH_ENDS"
RELAYOUT = "RELAYOUT"
PLAN_STEPS = (NO_MOVE, FLUSH_ENDS, RELAYOUT)


class PlanTake(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """One band's part of a plan, on its own page grid: ``reuse`` holes taken
    in place, then ``pages`` ``[first, end)`` extended on ``side`` ("low",
    "high", or "place" for an empty float placed afresh). ``low_byte`` and
    ``high_byte`` are the band's extent after the take."""

    reuse: int = 0
    side: Optional[str] = None
    pages: Optional[Tuple[int, int]] = None
    low_byte: int = 0
    high_byte: int = 0


class TakePlan(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """A layout serving a demand. ``flush``: ends compacted first. ``targets``:
    the page block ``[first, end)`` each relocated float ends up in, holding its
    live pages and, as holes, its demand (and, when its fixed pages reach
    further, more holes). ``takes``: every demanded band's take (a relocated
    float takes ``reuse`` of its block's holes)."""

    step: str
    flush: Tuple[str, ...]
    targets: Dict[str, Tuple[int, int]]
    takes: Dict[str, PlanTake]


def _align_up(x: int, e: int) -> int:
    return -(-x // e) * e


def _fixed_takes(
    b: BandSnapshot, pages: int, *, step: str, total_bytes: int
) -> Tuple[PlanTake, ...]:
    """The takes a band that stays in place can make, as extents on bytes. An
    end compacted first (any step but NO_MOVE, when not pinned) holds just its
    live pages, reaching as far as its farthest fixed page with holes between;
    otherwise it keeps its span and reuses its holes. A float in place reuses
    holes and extends one side, either one."""
    e = b.page_bytes
    if b.grow == "float":
        low, high = b.low_byte // e, b.high_byte // e
        holes = b.holes
    elif step != NO_MOVE and not b.pinned and (b.lazy or b.fixed is None):
        if b.grow == "up":
            low, high = b.min_page, b.min_page + b.live
            if b.fixed is not None:
                high = max(high, b.fixed[1])
        else:
            low, high = b.num_pages - b.live, b.num_pages
            if b.fixed is not None:
                low = min(low, b.fixed[0])
        holes = high - low - b.live
    else:
        holes = b.holes if b.lazy else 0
        low, high = b.low_byte // e, b.high_byte // e
    reuse = min(pages, holes)
    ext = pages - reuse
    if not ext:
        return (PlanTake(reuse=reuse, low_byte=low * e, high_byte=high * e),)
    out = []
    if b.grow != "up" and low - ext >= b.min_page:
        out.append(
            PlanTake(
                reuse=reuse,
                side="low",
                pages=(low - ext, low),
                low_byte=(low - ext) * e,
                high_byte=high * e,
            )
        )
    if b.grow != "down" and (high + ext) * e <= min(b.num_pages * e, total_bytes):
        out.append(
            PlanTake(
                reuse=reuse,
                side="high",
                pages=(high, high + ext),
                low_byte=low * e,
                high_byte=(high + ext) * e,
            )
        )
    return tuple(out)


def _block_end(b: BandSnapshot, pages: int, start: int) -> int:
    """End byte of a placed float's block starting at byte ``start``: room for
    its live pages and demand, reaching past its last fixed page."""
    end = start + (b.live + pages) * b.page_bytes
    if b.fixed is not None:
        end = max(end, b.fixed[1] * b.page_bytes)
    return end


def _pack(
    low_byte: int, movers: Tuple[Tuple[BandSnapshot, int], ...]
) -> Optional[Tuple[Tuple[int, ...], int]]:
    """The placed floats of a segment packed leftmost in chain order, each on
    its own grid within its index room: their start bytes and the frontier the
    last leaves, or None when one runs past its index room or starts past its
    first fixed page. A block's end never falls as its start falls, so the
    leftmost packing leaves every later float the most room."""
    x, starts = low_byte, []
    for b, pages in movers:
        start = max(_align_up(x, b.page_bytes), b.min_page * b.page_bytes)
        if b.fixed is not None and start > b.fixed[0] * b.page_bytes:
            return None
        x = _block_end(b, pages, start)
        if x > b.num_pages * b.page_bytes:
            return None
        starts.append(start)
    return tuple(starts), x


def _segment_fits(
    low_byte: int, movers: Tuple[Tuple[BandSnapshot, int], ...], high_byte: int
) -> bool:
    packed = _pack(low_byte, movers)
    return packed is not None and packed[1] <= high_byte


def _segment_starts(
    low_byte: int,
    movers: Tuple[Tuple[BandSnapshot, int], ...],
    high_byte: int,
    weights: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Start bytes of a segment's placed floats: the demand first, then the
    slack split between the segment's gaps by ``weights`` (one per gap, evenly
    when all are zero), each float clamped between its leftmost packed start
    and its latest start that still lets the floats after it fit (never past
    its first fixed page)."""
    _, end = _pack(low_byte, movers)
    slack = high_byte - end
    if not any(weights):
        weights = (1,) * len(weights)
    total = sum(weights)
    latest = [0] * len(movers)
    y = high_byte
    for i in range(len(movers) - 1, -1, -1):
        b, pages = movers[i]
        e = b.page_bytes
        y = (min(y, b.num_pages * e) - (b.live + pages) * e) // e * e
        if b.fixed is not None:
            y = min(y, b.fixed[0] * e)
        latest[i] = y
    starts, x = [], low_byte
    for i, (b, pages) in enumerate(movers):
        e = b.page_bytes
        earliest = max(_align_up(x, e), b.min_page * e)
        gap = slack * weights[i] // total
        start = min(max(_align_up(x + gap, e), earliest), latest[i])
        starts.append(start)
        x = _block_end(b, pages, start)
    return tuple(starts)


def _gap_weights(
    snapshot: ChainSnapshot,
    left: Optional[str],
    segment: Tuple[Tuple[BandSnapshot, int], ...],
    right: Optional[str],
    growth: Dict[str, int],
) -> Tuple[int, ...]:
    """Expected growth into each gap of a segment, in half-units: a float's
    growth splits between its two gaps, an end's goes to its one gap."""
    weights = [0] * (len(segment) + 1)
    for name, gap in ((left, 0), (right, len(segment))):
        if name is not None:
            whole = snapshot.band(name).grow != "float"
            weights[gap] += growth.get(name, 0) * (2 if whole else 1)
    for i, (b, _) in enumerate(segment):
        weights[i] += growth.get(b.name, 0)
        weights[i + 1] += growth.get(b.name, 0)
    return tuple(weights)


def plan(
    snapshot: ChainSnapshot,
    demands: Dict[str, int],
    step: str,
    *,
    growth: Optional[Dict[str, int]] = None,
) -> Optional[TakePlan]:
    """A layout of the chain serving ``demands`` (pages per band) with the
    rearrangement ``step`` allows, or None if there is none.

    Counts and frontiers only: a band in place reuses holes wherever they are
    and extends one side; it never relies on releasing boundary holes. Bands are
    walked in chain order; a band in place is a fixed point, and the floats
    between two fixed points are placed in their segment. Packing a segment
    leftmost on every float's own grid decides whether it fits, and the fit
    only gets easier as the segment's low frontier falls, so of a fixed band's
    takes the one with the lowest high byte is kept: the walk is exact for
    this world and linear in the number of bands. A pinned band is always in
    place; a band's fixed pages only bound where it may be laid out: a
    compacted end reaches its farthest fixed page, a relocated float's block
    covers them.
    """
    if step not in PLAN_STEPS:
        raise ValueError(f"unknown plan step {step!r}")
    if any(name not in snapshot.positions or n < 0 for name, n in demands.items()):
        raise ValueError("a demand must name a chain band and be non-negative")
    total = snapshot.total_bytes
    takes: Dict[str, PlanTake] = {}
    targets: Dict[str, Tuple[int, int]] = {}
    movers = []
    segments = []
    frontier = 0
    for b in snapshot.bands:
        pages = demands.get(b.name, 0)
        if b.live + pages > b.num_pages - b.min_page:
            return None
        placed = b.transparent or (
            b.grow == "float" and step == RELAYOUT and not b.pinned
        )
        if placed:
            if b.live or pages:
                movers.append((b, pages))
            continue
        movers_here = tuple(movers)
        choices = [
            t
            for t in _fixed_takes(b, pages, step=step, total_bytes=total)
            if t.low_byte >= frontier
            and _segment_fits(frontier, movers_here, t.low_byte)
        ]
        if not choices:
            return None
        chosen = min(choices, key=lambda t: t.high_byte)
        segments.append((frontier, movers_here, chosen.low_byte, b.name))
        takes[b.name] = chosen
        frontier = chosen.high_byte
        movers = []
    movers_here = tuple(movers)
    if not _segment_fits(frontier, movers_here, total):
        return None
    segments.append((frontier, movers_here, total, None))
    growth = growth or {}
    left = None
    for low_byte, segment, high_byte, right in segments:
        if segment:
            weights = _gap_weights(snapshot, left, segment, right, growth)
            starts = _segment_starts(low_byte, segment, high_byte, weights)
            for (b, pages), start in zip(segment, starts):
                first = start // b.page_bytes
                if b.live:
                    end = _block_end(b, pages, start)
                    targets[b.name] = (first, end // b.page_bytes)
                    takes[b.name] = PlanTake(reuse=pages, low_byte=start, high_byte=end)
                else:
                    takes[b.name] = PlanTake(
                        side="place",
                        pages=(first, first + pages),
                        low_byte=start,
                        high_byte=(first + pages) * b.page_bytes,
                    )
        left = right
    flush = (
        ()
        if step == NO_MOVE
        else tuple(b.name for b in snapshot.bands if b.grow != "float" and not b.pinned)
    )
    return TakePlan(step=step, flush=flush, targets=targets, takes=takes)


def feasible(snapshot: ChainSnapshot, demands: Dict[str, int]) -> bool:
    """Whether some layout reachable from ``snapshot`` serves ``demands``: the
    one capacity question. Pass a snapshot with evictable pages reclaimed
    (`reclaimed_snapshot`) to ask it with eviction."""
    return plan(snapshot, demands, RELAYOUT) is not None


def first_plan(
    snapshot: ChainSnapshot,
    demands: Dict[str, int],
    *,
    growth: Optional[Dict[str, int]] = None,
) -> Optional[TakePlan]:
    """The least rearrangement serving ``demands``: in place, then with the
    ends compacted, then with floats relocated. None exactly when not
    `feasible`."""
    for step in PLAN_STEPS:
        found = plan(snapshot, demands, step, growth=growth)
        if found is not None:
            return found
    return None


def move_order(
    spans: Dict[str, Tuple[int, int]], targets: Dict[str, Tuple[int, int]]
) -> Tuple[str, ...]:
    """An order to relocate floats so that none is copied into a block another
    float still occupies: a float waits for every other float whose current
    byte span ``spans`` meets its target byte block. Spans and targets both keep
    chain order, so the order exists."""
    names = list(targets)
    waits = {
        n: {
            m
            for m in names
            if m != n and spans[m][0] < targets[n][1] and targets[n][0] < spans[m][1]
        }
        for n in names
    }
    order = []
    while len(order) < len(names):
        ready = next(
            (n for n in names if n not in order and not waits[n] - set(order)), None
        )
        if ready is None:
            raise ValueError("relocation targets do not keep chain order")
        order.append(ready)
    return tuple(order)
