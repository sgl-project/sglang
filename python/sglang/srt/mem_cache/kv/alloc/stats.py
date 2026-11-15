"""Observer-side numbers of a unified-memory chain: each band's page layout,
the free gaps between bands, and cumulative counts of the work the allocator
did to serve takes. Every number is host state the allocator already keeps;
reading them never waits on the device and never feeds a capacity decision."""

from __future__ import annotations

from typing import Dict

import msgspec

from sglang.srt.mem_cache.kv.alloc.pricing import PLAN_STEPS, ChainSnapshot

# The plan count of a demand no layout serves.
REFUSED = "REFUSED"
# Every key a chain's plan counts may take.
PLAN_OUTCOMES = (*PLAN_STEPS, REFUSED)


class BandCounters(msgspec.Struct, kw_only=True):
    """Cumulative work one band did, never reset."""

    # Compaction passes that moved at least one page, and the pages they
    # moved: urgent passes serve a take, opportunistic ones run at
    # scheduler idle points.
    urgent_compactions: int = 0
    urgent_compaction_moves: int = 0
    opportunistic_compactions: int = 0
    opportunistic_compaction_moves: int = 0
    # A float's relocations into a planned block, and the pages they copied.
    float_relocations: int = 0
    float_relocated_pages: int = 0

    def note_compaction(self, *, urgent: bool, moves: int) -> None:
        if urgent:
            self.urgent_compactions += 1
            self.urgent_compaction_moves += moves
        else:
            self.opportunistic_compactions += 1
            self.opportunistic_compaction_moves += moves

    def note_relocation(self, moves: int) -> None:
        self.float_relocations += 1
        self.float_relocated_pages += moves


class ChainCounters(msgspec.Struct, kw_only=True):
    """Cumulative work of a chain's takes, never reset."""

    # Relief plans by the step that served them (`pricing.PLAN_STEPS`), and
    # demands no layout served.
    plans: Dict[str, int] = msgspec.field(default_factory=dict)
    # Reliefs that found no plan with the rows network transfers still touch
    # pinned in place, so planned with them movable (a move waits on the host).
    unpinned_network_reliefs: int = 0
    # Planned takes and the host time they spent: planning, laying the chain
    # out and taking, up to the kernels they launch.
    takes: int = 0
    take_seconds: float = 0.0

    def note_plan(self, step: str) -> None:
        self.plans[step] = self.plans.get(step, 0) + 1


class BandLayout(msgspec.Struct, frozen=True, kw_only=True):
    """One band's pages as the planner sees them now."""

    # Pages the band keeps: live pages, and pending pages that must stay put.
    live_pages: int
    # Free pages inside the span a take can reuse at once.
    hole_pages: int
    # Freed pages still waiting for their step to be processed.
    pending_pages: int
    counters: BandCounters


class ChainStats(msgspec.Struct, frozen=True, kw_only=True):
    """A chain's layout now and its cumulative work, bands by sub-pool name."""

    bands: Dict[str, BandLayout]
    # Free bytes between neighbouring bands (empty floats are seen through):
    # the room a band can grow into without any page moving.
    gap_bytes: int
    largest_gap_bytes: int
    counters: ChainCounters


def free_gaps(snapshot: ChainSnapshot) -> list[int]:
    """The free byte gap between each pair of neighbouring opaque bands."""
    opaque = [b for b in snapshot.bands if not b.transparent]
    return [
        max(0, above.low_byte - below.high_byte)
        for below, above in zip(opaque, opaque[1:])
    ]
