"""Shared page plans, windows, ownership and round schedules.

This module is direction-neutral bookkeeping; it performs no tensor I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from zlib import crc32

from sglang.srt.mem_cache.layer_split.layer_split_config import StagingBufferConfig

# Canonical components of a page, in the order the exchange walks them.
#
# They are exchanged as separate collectives because they live in separate slabs,
# and they live in separate slabs because the L3 object needs each component
# contiguous -- interleaving them into one send buffer would make neither of them
# a single span. So a round costs one collective per component.
#
# Every rank must walk them in this order, for the same reason rounds are
# submitted in plan order: ``all_to_all_single`` carries no tag, so position in
# the sequence is the only thing pairing a rank's collective with its peers'.
#
# A window is not finished until every component has been exchanged. A page
# holding only its target is unusable, and the published prefix promises both.
EXCHANGE_COMPONENTS = ("target", "indexer")

# Version of the plan/exchange protocol the fingerprint covers: ownership
# derivation, window sizing, round-to-page mapping, component order. Bumping it
# makes ranks running different code disagree at the per-window agreement instead
# of exchanging bytes under incompatible assumptions -- which is the failure a
# rolling upgrade would otherwise produce silently.
_PLAN_SCHEMA_VERSION = 1


def rotation_base(request_id: str) -> int:
    """Rank-invariant seed that rotates page ownership between operations.

    Ownership has to be agreed by every rank *without communication*, so the seed
    must not come from a rank-local counter such as ``op_id`` or
    ``StorageOperation.counter``. Any divergence -- one rank skipping an
    operation, or failing between taking an id and building the plan -- would
    offset that rank's owner table from then on, so pages would be fetched twice
    or not at all and the exchange split sizes would stop matching. That failure
    is a hang or silent corruption, not a clean error.

    ``request_id`` is the same string on every rank: the existing MIN all-reduce
    over hit counts already depends on ranks agreeing which request they serve,
    so this adds no new assumption. It also varies per request, which keeps the
    uneven tail of a partial window off the same ranks every time.

    ``crc32`` rather than the builtin ``hash()``: string hashing is salted per
    process, so eight worker processes would each compute a different value.
    """
    return crc32(request_id.encode("utf-8"))


def page_owner(page_ordinal: int, window_base: int, shard_size: int) -> int:
    """CP rank that performs L3 I/O for ``page_ordinal``.

    A pure function of the plan inputs, so every rank computes the same owner
    table locally and the table never has to be broadcast. ``window_base``
    rotates the assignment between operations so that the uneven tail of a
    partial window does not always land on the same ranks.

    The owner only labels the current transfer; it is not part of the L3 key, so
    the same page may be owned by a different rank next time.
    """
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    if page_ordinal < 0:
        raise ValueError(f"page_ordinal must not be negative, got {page_ordinal}")
    return (window_base + page_ordinal) % shard_size


@dataclass(frozen=True)
class PageWindow:
    """A contiguous run of pages exchanged as one unit.

    ``page_start`` is an ordinal into the owning plan's page list, not an L2
    slot index: physical slots stay rank-local and are never shared.
    """

    index: int
    page_start: int
    page_count: int

    @property
    def page_end(self) -> int:
        return self.page_start + self.page_count

    def ordinals(self) -> range:
        return range(self.page_start, self.page_end)


@dataclass(frozen=True)
class PageTransferPlan:
    """Immutable page/ownership description for either transfer direction.

    Nothing rank-local (L2 slot indices, host pointers, buffer addresses) may be
    added here -- see the "local allocation" invariant in the design doc.

    ``op_id`` is the one deliberate exception: it is *this rank's* submission
    counter, so it may differ between ranks and must never feed a cross-rank
    decision. It identifies the native operation locally, not a buffer lease.
    Everything that has to agree -- ``page_hashes``,
    ``page_owners``, ``window_base``, ``shard_size``, ``pages_per_window`` -- is
    derived from the request, see :func:`rotation_base`.
    """

    op_id: int
    request_id: str
    page_hashes: Tuple[str, ...]
    page_owners: Tuple[int, ...]
    shard_size: int
    pages_per_window: int
    window_base: int

    def __post_init__(self) -> None:
        if len(self.page_hashes) != len(self.page_owners):
            raise ValueError(
                f"page_hashes ({len(self.page_hashes)}) and page_owners "
                f"({len(self.page_owners)}) must have equal length"
            )
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {self.shard_size}")
        if self.pages_per_window <= 0:
            raise ValueError(
                f"pages_per_window must be positive, got {self.pages_per_window}"
            )

    @property
    def page_count(self) -> int:
        return len(self.page_hashes)

    def windows(self) -> List[PageWindow]:
        """Split the plan into transfer windows, in submission order.

        The last window may be short; ownership within it is unbalanced by at
        most one page, which ``window_base`` rotation spreads over time.
        """
        result: List[PageWindow] = []
        for index, start in enumerate(range(0, self.page_count, self.pages_per_window)):
            count = min(self.pages_per_window, self.page_count - start)
            result.append(PageWindow(index=index, page_start=start, page_count=count))
        return result

    def owned_ordinals(
        self, rank: int, window: Optional[PageWindow] = None
    ) -> List[int]:
        """Page ordinals whose L3 I/O ``rank`` is responsible for.

        Restricted to ``window`` when given. The returned order is the order in
        which the owner should issue its GETs, which is also the order the
        round-based exchange consumes them in.
        """
        ordinals = window.ordinals() if window is not None else range(self.page_count)
        return [i for i in ordinals if self.page_owners[i] == rank]

    def window_fingerprint(self, window: PageWindow) -> int:
        """Stable digest of *what* this window is, identical on every rank.

        Folded into the per-window agreement so a divergence is caught before any
        data collective runs. Gloo pairs the Nth call on a group with its peers'
        Nth call and nothing more: if a bug put two ranks on different requests or
        different windows, the agreement would still complete, and a later exchange
        of the same shape could return plausible-looking wrong bytes. Comparing a
        digest turns that into a loud failure.

        Covers everything the exchange's shape and meaning depend on: which request
        and which pages, where the window sits, how wide the split is, the
        component order, exchange schema, and the round count.
        ``op_id`` is deliberately absent -- it is rank-local -- and so is anything
        derived from Python's ``hash``, which is salted per process.
        """
        parts = [
            self.request_id,
            str(self.shard_size),
            str(self.pages_per_window),
            # window_base decides ownership rotation, so two ranks disagreeing on
            # it would own different pages while agreeing on everything else.
            str(self.window_base),
            str(window.index),
            str(window.page_start),
            str(window.page_end),
            str(self.rounds(window)),
            ",".join(EXCHANGE_COMPONENTS),
            str(_PLAN_SCHEMA_VERSION),
            # The pages themselves, not just their count: two requests can agree
            # on length and still be different prefixes.
            *self.page_hashes[window.page_start : window.page_end],
        ]
        return crc32("\x1f".join(parts).encode()) & 0x7FFFFFFF

    def owner_page_counts(self) -> List[int]:
        """Pages owned per rank, for load-balance assertions and metrics."""
        counts = [0] * self.shard_size
        for owner in self.page_owners:
            counts[owner] += 1
        return counts

    def rounds(self, window: PageWindow) -> int:
        """Number of collective rounds needed to exchange ``window``.

        One round exchanges at most one owned page per rank, so the round count
        is set by the busiest owner. Rounds are submitted in a globally
        consistent order; a per-page, completion-driven order would let ranks
        issue collectives in different sequences and mismatch.
        """
        counts = [0] * self.shard_size
        for i in window.ordinals():
            counts[self.page_owners[i]] += 1
        return max(counts) if counts else 0


def build_transfer_plan(
    *,
    op_id: int,
    request_id: str,
    page_hashes: Sequence[str],
    staging_buffer_config: StagingBufferConfig,
    window_base: Optional[int] = None,
) -> PageTransferPlan:
    """Plan the entire already-admitted page list for either transfer direction.

    Hit-prefix selection belongs to query/L2 admission, not this builder.
    Ownership rotates by the common request ID, never a rank-local operation ID.
    """
    staging_buffer_config.require_host_layout()
    base = rotation_base(request_id) if window_base is None else window_base
    hashes = tuple(page_hashes)
    shard_size = staging_buffer_config.shard_size
    return PageTransferPlan(
        op_id=op_id,
        request_id=request_id,
        page_hashes=hashes,
        page_owners=tuple(page_owner(i, base, shard_size) for i in range(len(hashes))),
        shard_size=shard_size,
        pages_per_window=staging_buffer_config.pages_per_window,
        window_base=base,
    )


# ---------------------------------------------------------------------------
# Plan output: the collective schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExchangeRound:
    """One collective round of a window's exchange.

    ``contributions[r]`` is the page ordinal owned by rank ``r`` this round,
    or ``None`` if it owns no page. Prefetch scatters that owner's full page;
    backup gathers shards into it. A non-owner still participates in shard
    transfers. Each rank owns at most one full page per round.

    Rounds are numbered and submitted in this order on *every* rank. Driving the
    exchange off local I/O completion order instead would let ranks issue collectives
    in different sequences, which mismatches the collective and hangs.

    ``index`` addresses the window-local fixed buffer. It restarts at zero only
    after the previous window has finished using that buffer.
    """

    index: int
    contributions: Tuple[Optional[int], ...]

    def pages(self) -> List[int]:
        """Page ordinals carried by this round, ascending."""
        return sorted(p for p in self.contributions if p is not None)

    def contributor_ranks(self) -> List[int]:
        return [r for r, p in enumerate(self.contributions) if p is not None]

    def is_full(self) -> bool:
        """Whether every rank contributes, i.e. the collective is balanced."""
        return all(p is not None for p in self.contributions)


def build_exchange_rounds(
    plan: PageTransferPlan, window: PageWindow
) -> List[ExchangeRound]:
    """Turn a window of a plan into its ordered collective schedule.

    Round ``k`` takes each rank's ``k``-th owned page in the window. The number
    of rounds is set by the busiest owner; ranks that run out of owned pages
    contribute ``None`` and idle for the remaining rounds.

    Derived purely from the plan, so all ranks produce an identical schedule.
    """
    per_rank = [plan.owned_ordinals(rank, window) for rank in range(plan.shard_size)]
    round_count = max((len(owned) for owned in per_rank), default=0)
    rounds: List[ExchangeRound] = []
    for k in range(round_count):
        contributions = tuple(
            owned[k] if k < len(owned) else None for owned in per_rank
        )
        rounds.append(ExchangeRound(index=k, contributions=contributions))
    return rounds
