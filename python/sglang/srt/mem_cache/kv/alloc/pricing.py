"""Capacity pricing over a host-only snapshot of a chain's bands.

A snapshot holds counts and byte frontiers, never hole positions (those live in
device tensors), so taking one and pricing over it never synchronizes with the
device. Every capacity decision of `ChainAllocator` is one of the pure functions
below applied to a fresh snapshot; the chain then performs the realization the
decision names (end flushes, float relocations) and the takes. One planner
serves every chain shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, Optional, Tuple

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
    blocked: bool  # holds fixed pages, which neither compaction nor a move passes
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
    # Pages ``[first, end)`` holding every page that cannot move (a load
    # reservation); the band's extent must keep covering them. None when there
    # is none.
    fixed: Optional[Tuple[int, int]] = None
    # Pages pending until their step is processed: counted in `holes` (a take
    # uses them), yet their bytes stay the band's until they are settled.
    pending: int = 0

    @property
    def transparent(self) -> bool:
        """An empty float, none of its pages fixed or pending until their step
        is processed: its neighbours' frontier walks see through it. A fixed or
        pending page keeps a float's span over it."""
        return (
            self.grow == "float"
            and self.live == 0
            and self.fixed is None
            and self.pending == 0
        )

    @property
    def index_room(self) -> int:
        return self.num_pages - self.min_page - self.span

    def pages_in_band(self, *, low_byte: int, high_byte: int) -> int:
        """Pages of this band's grid inside ``[low_byte, high_byte)``."""
        lo = max((low_byte + self.page_bytes - 1) // self.page_bytes, self.min_page)
        hi = min(high_byte // self.page_bytes, self.num_pages)
        return max(0, hi - lo)


class ChainSnapshot(msgspec.Struct, frozen=True, gc=False):
    """Bands in chain order, low byte to high byte. ``held``: sorted, disjoint
    byte ranges ``[lo, hi)`` outside every band that an unreleased network
    send still reads (the bytes a last-resort move took out of a band under
    it): a plan neither extends a band nor places a float there. A snapshot
    for `feasible` holds none: there every byte is usable, waiting on the host
    for the send when it must."""

    bands: Tuple[BandSnapshot, ...]
    page_size: int
    total_bytes: int
    positions: Dict[str, int]  # band name -> index in `bands`
    held: Tuple[Tuple[int, int], ...] = ()

    def index(self, name: str) -> int:
        return self.positions[name]

    def band(self, name: str) -> BandSnapshot:
        return self.bands[self.positions[name]]

    def high_frontier_below(self, name: str) -> int:
        """High byte frontier of the nearest opaque band below; 0 if none."""
        bands = self.bands
        for i in range(self.positions[name] - 1, -1, -1):
            band = bands[i]
            if not band.transparent:
                return band.high_byte
        return 0

    def low_frontier_above(self, name: str) -> int:
        """Low byte frontier of the nearest opaque band above; the top if none."""
        bands = self.bands
        for i in range(self.positions[name] + 1, len(bands)):
            band = bands[i]
            if not band.transparent:
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
        # Holes, and pages pending until their step is processed: a take reaches
        # those after it settles them (`Band.settle_pending`).
        band._reusable_pages(),
        # `allocated_count() // page_size`: lazy ends count live pages apart
        # from the holes and pending pages inside their span; a pending page
        # that must stay put counts as occupied.
        band.occupied_pages() if band.lazy_compaction else span,
        span,
        band._byte_low_frontier(),
        band._byte_high_frontier(),
        # Only the id owner mints virtual pages; a window binds the owner's.
        0 if free_virtual_ids is None else free_virtual_ids.numel(),
        band.ids.num_virtual_ids,
        band.low_wm_page if floats else 0,
        band.high_wm_page if floats else 0,
        band.fixed_extent(),
        band._pending_until_retire_pages,
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


def reclaimed_snapshot(
    snapshot: ChainSnapshot, reclaim_pages: Dict[str, int], *, empty_pool: bool = False
) -> ChainSnapshot:
    """Conservative geometry after reclaim, without guessing where holes lie.

    Ends without fixed pages can realize their compacted frontiers by an
    urgent flush; ends holding fixed pages keep their spans. Partial float
    frees remain holes in place; empty floats become transparent unless a page
    of theirs is fixed, which keeps the span as holes. Only the
    frees are credited here: `feasible` then lays the result out, relocating
    floats where its plan step allows.
    """
    bands = []
    for b in snapshot.bands:
        reclaimed = (
            b.live if empty_pool else min(b.live, max(0, reclaim_pages.get(b.name, 0)))
        )
        live = b.live - reclaimed
        low, high, span = b.low_byte, b.high_byte, b.span
        holes, pending = b.holes + reclaimed, b.pending
        # A float's fixed page keeps its span, as an end's does. Recovery
        # settles every pending page first: a float left with no live page
        # parks, and an unblocked end's urgent flush absorbs them.
        empties = b.fixed is None if b.grow == "float" else not b.blocked
        if empty_pool or (not live and empties):
            span = holes = pending = 0
            low = high = (
                b.num_pages if b.grow == "down" else b.min_page
            ) * b.page_bytes
        elif b.grow != "float" and not b.blocked:
            span, holes, pending = live, 0, 0
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
                pending=pending,
                low_byte=low,
                high_byte=high,
                free_virtual_pages=(
                    b.num_virtual_pages - b.min_page
                    if empty_pool
                    else b.free_virtual_pages + reclaimed
                ),
                # An empty pool holds nothing to keep in place, on any band.
                fixed=None if empty_pool else b.fixed,
            )
        )
    return msgspec.structs.replace(snapshot, bands=tuple(bands))


# ---------------------------------------------------------------------------
# One planner for every chain: whether a demand fits, and the layout serving it
# ---------------------------------------------------------------------------

# How far a plan may rearrange the chain: nothing moves; ends release their
# holes by compaction; floats also relocate. Fixed pages never move: a
# compaction or a relocation moves the other pages around them, and keeps them
# inside their band.
NO_MOVE = "NO_MOVE"
FLUSH_ENDS = "FLUSH_ENDS"
RELAYOUT = "RELAYOUT"
PLAN_STEPS = (NO_MOVE, FLUSH_ENDS, RELAYOUT)

# The part of its growth-proportional share of the slack every gap keeps when
# a rearrangement places floats; the rest may go to keeping floats in place.
_SHARE_KEPT = (7, 8)


class PlanTake(msgspec.Struct, frozen=True, gc=False, kw_only=True):
    """One band's part of a plan, on its own page grid: ``reuse`` holes taken
    in place, then ``pages`` ``[first, end)`` extended on ``side`` ("low",
    "high", or "place" for an empty float placed afresh). ``low_byte`` and
    ``high_byte`` are the band's extent after the take, for a take a plan
    returns; a shorter `prefix` of it carries no extent (both 0)."""

    reuse: int = 0
    side: Optional[str] = None
    pages: Optional[Tuple[int, int]] = None
    low_byte: int = 0
    high_byte: int = 0

    def count(self) -> int:
        """Pages the take hands out: reused holes plus the extension."""
        extension = 0 if self.pages is None else self.pages[1] - self.pages[0]
        return self.reuse + extension

    def prefix(self, count: int) -> PlanTake:
        """The first ``count`` pages of this take: holes first, then the part
        of the extension next to the band's current span, so what is left is
        still a take from the new span. The whole take is the take itself,
        its extent included; a shorter prefix names only its pages."""
        if not 0 <= count <= self.count():
            raise ValueError("a take prefix exceeds its take")
        if count == self.count():
            return self
        reuse = min(count, self.reuse)
        extension = count - reuse
        if self.pages is None or not extension:
            return PlanTake(reuse=reuse, side=self.side)
        first, end = self.pages
        pages = (
            (end - extension, end) if self.side == "low" else (first, first + extension)
        )
        return PlanTake(reuse=reuse, side=self.side, pages=pages)


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


def _held_meeting(
    held: Tuple[Tuple[int, int], ...], lo: int, hi: int
) -> Optional[Tuple[int, int]]:
    """The first of the ``held`` byte ranges that meets ``[lo, hi)``, if any."""
    for first, end in held:
        if first >= hi:
            return None
        if lo < end:
            return first, end
    return None


def _fixed_takes(
    b: BandSnapshot,
    pages: int,
    *,
    step: str,
    total_bytes: int,
    held: Tuple[Tuple[int, int], ...] = (),
) -> Tuple[PlanTake, ...]:
    """The takes a band that stays in place can make, as extents on bytes. An
    end compacted first (any step but NO_MOVE) holds just its
    live pages, reaching as far as its farthest fixed page with holes between;
    otherwise it keeps its span and reuses its holes. A float in place reuses
    holes and extends one side, either one. No extension meets ``held``."""
    e = b.page_bytes
    if b.grow == "float":
        low, high = b.low_byte // e, b.high_byte // e
        holes = b.holes
    elif step != NO_MOVE and (b.lazy or b.fixed is None):
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
    return tuple(
        t for t in out if _held_meeting(held, t.pages[0] * e, t.pages[1] * e) is None
    )


def _block_end(b: BandSnapshot, pages: int, start: int) -> int:
    """End byte of a placed float's block starting at byte ``start``: room for
    its live pages and demand, reaching past its last fixed page."""
    end = start + (b.live + pages) * b.page_bytes
    if b.fixed is not None:
        end = max(end, b.fixed[1] * b.page_bytes)
    return end


def _pack(
    low_byte: int,
    movers: Tuple[Tuple[BandSnapshot, int], ...],
    held: Tuple[Tuple[int, int], ...] = (),
) -> Optional[Tuple[Tuple[int, ...], int]]:
    """The placed floats of a segment packed leftmost in chain order, each on
    its own grid within its index room and past every ``held`` range its block
    would meet: their start bytes and the frontier the last leaves, or None
    when one runs past its index room or starts past its first fixed page. A
    block's end never falls as its start falls, so the leftmost packing leaves
    every later float the most room."""
    x, starts = low_byte, []
    for b, pages in movers:
        start = max(_align_up(x, b.page_bytes), b.min_page * b.page_bytes)
        blocker = _held_meeting(held, start, _block_end(b, pages, start))
        while blocker is not None:
            start = _align_up(blocker[1], b.page_bytes)
            blocker = _held_meeting(held, start, _block_end(b, pages, start))
        if b.fixed is not None and start > b.fixed[0] * b.page_bytes:
            return None
        x = _block_end(b, pages, start)
        if x > b.num_pages * b.page_bytes:
            return None
        starts.append(start)
    return tuple(starts), x


def _segment_fits(
    low_byte: int,
    movers: Tuple[Tuple[BandSnapshot, int], ...],
    high_byte: int,
    held: Tuple[Tuple[int, int], ...] = (),
) -> bool:
    packed = _pack(low_byte, movers, held)
    return packed is not None and packed[1] <= high_byte


def _segment_blocks(
    low_byte: int,
    movers: Tuple[Tuple[BandSnapshot, int], ...],
    high_byte: int,
    weights: Tuple[int, ...],
    held: Tuple[Tuple[int, int], ...] = (),
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Start and end bytes of a segment's placed floats.

    The slack (what the leftmost packing leaves) is a store of room for later
    takes. Split between the segment's gaps in proportion to ``weights`` (each
    gap's expected growth; evenly when all are zero), it lasts the most takes
    before the next rearrangement. Each gap keeps at least
    `_SHARE_KEPT` of that proportional share; within that, a float that already
    has live pages stays as close to where it is as it can, so a rearrangement
    copies as few pages as the shares allow. An empty float is placed at its
    proportional start. Every start stays between the float's leftmost packed
    start and its latest start that still lets the floats after it fit, and
    never passes its first fixed page. A float with live pages also keeps its
    far edge where its span ends, holding its holes, when the gap after it
    still keeps its share: only the pages its new start leaves behind are
    copied. When one of these blocks would meet a ``held`` range, the segment
    is packed leftmost around them instead, which always fits."""
    packed_starts, end = _pack(low_byte, movers, held)
    slack = high_byte - end
    if not any(weights):
        weights = (1,) * len(weights)
    total = sum(weights)
    share = [slack * w // total for w in weights]
    kept = [s * _SHARE_KEPT[0] // _SHARE_KEPT[1] for s in share]
    n = len(movers)
    latest = [0] * n
    latest_kept = [0] * n
    y = y_kept = high_byte
    y_kept -= kept[n]
    for i in range(n - 1, -1, -1):
        b, pages = movers[i]
        e = b.page_bytes
        block = (b.live + pages) * e
        y = (min(y, b.num_pages * e) - block) // e * e
        y_kept = (min(y_kept, b.num_pages * e) - block) // e * e
        if b.fixed is not None:
            y = min(y, b.fixed[0] * e)
            y_kept = min(y_kept, b.fixed[0] * e)
        latest[i] = y
        latest_kept[i] = min(y_kept, y)
        y_kept = latest_kept[i] - kept[i]
    starts, x = [], low_byte
    for i, (b, pages) in enumerate(movers):
        e = b.page_bytes
        earliest = max(_align_up(x, e), b.min_page * e)
        proportional = min(max(_align_up(x + share[i], e), earliest), latest[i])
        low = max(_align_up(x + kept[i], e), earliest)
        high = latest_kept[i]
        if b.live and low <= high:
            start = min(max(b.low_byte, low), high)
        else:
            start = proportional
        starts.append(start)
        x = _block_end(b, pages, start)
    ends = []
    for i, (b, pages) in enumerate(movers):
        e = b.page_bytes
        end = _block_end(b, pages, starts[i])
        if b.live:
            # Keep the block's far edge where the float's span ends, its holes
            # inside, when the next gap still keeps its share: the pages beyond
            # a compacted block would otherwise be copied in.
            limit = (starts[i + 1] if i + 1 < n else high_byte) - kept[i + 1]
            keep_to = min(b.high_byte, limit, b.num_pages * e) // e * e
            end = max(end, keep_to)
        ends.append(end)
    if any(
        _held_meeting(held, start, end) is not None for start, end in zip(starts, ends)
    ):
        return packed_starts, tuple(
            _block_end(b, pages, start)
            for (b, pages), start in zip(movers, packed_starts)
        )
    return tuple(starts), tuple(ends)


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
    keep_ends: FrozenSet[str] = frozenset(),
) -> Optional[TakePlan]:
    """A layout of the chain serving ``demands`` (pages per band) with the
    rearrangement ``step`` allows, or None if there is none. Ends named in
    ``keep_ends`` stay as they are (no compaction) whatever the step.

    Counts and frontiers only: a band in place reuses holes wherever they are
    and extends one side; it never relies on releasing boundary holes. Bands are
    walked in chain order; a band in place is a fixed point, and the floats
    between two fixed points are placed in their segment. Packing a segment
    leftmost on every float's own grid decides whether it fits, and the fit
    only gets easier as the segment's low frontier falls, so of a fixed band's
    takes the one with the lowest high byte is kept: the walk is exact for
    this world and linear in the number of bands. A band's fixed pages only
    bound where it may be laid out: a
    compacted end reaches its farthest fixed page, a relocated float's block
    covers them. Bytes the snapshot holds for a network send (``held``) bound
    it too: no band extends into them and no float is placed over them, so a
    plan never waits for a send.
    """
    if step not in PLAN_STEPS:
        raise ValueError(f"unknown plan step {step!r}")
    if any(name not in snapshot.positions or n < 0 for name, n in demands.items()):
        raise ValueError("a demand must name a chain band and be non-negative")
    total = snapshot.total_bytes
    held = snapshot.held
    takes: Dict[str, PlanTake] = {}
    targets: Dict[str, Tuple[int, int]] = {}
    movers = []
    segments = []
    frontier = 0
    for b in snapshot.bands:
        pages = demands.get(b.name, 0)
        if b.live + pages > b.num_pages - b.min_page:
            return None
        placed = b.transparent or (b.grow == "float" and step == RELAYOUT)
        if placed:
            # A relocation settles pages pending until their step is
            # processed first: they are holes, and a float with no live page
            # parks (`reclaimed_snapshot`'s rule), so they need no block.
            if b.live or pages or b.fixed is not None:
                movers.append((b, pages))
            continue
        movers_here = tuple(movers)
        choices = [
            t
            for t in _fixed_takes(
                b,
                pages,
                step=NO_MOVE if b.name in keep_ends else step,
                total_bytes=total,
                held=held,
            )
            if t.low_byte >= frontier
            and _segment_fits(frontier, movers_here, t.low_byte, held)
        ]
        if not choices:
            return None
        chosen = min(choices, key=lambda t: t.high_byte)
        segments.append((frontier, movers_here, chosen.low_byte, b.name))
        takes[b.name] = chosen
        frontier = chosen.high_byte
        movers = []
    movers_here = tuple(movers)
    if not _segment_fits(frontier, movers_here, total, held):
        return None
    segments.append((frontier, movers_here, total, None))
    growth = growth or {}
    left = None
    for low_byte, segment, high_byte, right in segments:
        if segment:
            weights = _gap_weights(snapshot, left, segment, right, growth)
            starts, ends = _segment_blocks(low_byte, segment, high_byte, weights, held)
            for (b, pages), start, end in zip(segment, starts, ends):
                first = start // b.page_bytes
                if b.live or b.fixed is not None:
                    # Relocated into a block holding its live and fixed pages
                    # (its pending pages are settled into holes first).
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
        else tuple(
            b.name
            for b in snapshot.bands
            if b.grow != "float" and b.name not in keep_ends
        )
    )
    return TakePlan(step=step, flush=flush, targets=targets, takes=takes)


def feasible(snapshot: ChainSnapshot, demands: Dict[str, int]) -> bool:
    """Whether some layout reachable from ``snapshot`` serves ``demands``: the
    one capacity question. Pass a snapshot with evictable pages reclaimed
    (`reclaimed_snapshot`) to ask it with eviction. Every byte counts, those
    held for a network send too: a layout that needs them waits for it."""
    if snapshot.held:
        snapshot = msgspec.structs.replace(snapshot, held=())
    return plan(snapshot, demands, RELAYOUT) is not None


def first_plan(
    snapshot: ChainSnapshot,
    demands: Dict[str, int],
    *,
    growth: Optional[Dict[str, int]] = None,
) -> Optional[TakePlan]:
    """The least rearrangement serving ``demands``: in place, then with the
    ends compacted, then with floats relocated. Compacting the ends only, it
    compacts just the ends it needs (each end, in chain order, is left as it
    is when the demand still fits that way). Relocating floats, it compacts
    every end: the floats are placed in the room between the ends, and an end
    left with its holes would leave them less of it, so the next relocation
    would come sooner. None exactly when not `feasible`."""
    for step in PLAN_STEPS:
        found = plan(snapshot, demands, step, growth=growth)
        if found is None:
            continue
        if step == RELAYOUT:
            return found
        keep: FrozenSet[str] = frozenset()
        for name in found.flush:
            fewer = plan(
                snapshot, demands, step, growth=growth, keep_ends=keep | {name}
            )
            if fewer is not None:
                keep, found = keep | {name}, fewer
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
