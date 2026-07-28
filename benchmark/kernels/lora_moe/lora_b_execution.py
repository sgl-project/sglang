"""Typed identity of one LoRA-B execution candidate (Step 4, plan §64.1).

Mirrors the A-side discipline: the workload case carries no schedule
identity; a candidate is this spec; different candidates compare against
the SAME workload ``case_id``; a timing record carries ``spec.key()`` so
arm identity is structured, not a naming convention; impossible
combinations are rejected here rather than falling through a dispatcher.
"""

from __future__ import annotations

import msgspec

SITES = ("gate_up", "down")
OWNERSHIPS = ("grouped", "indexed")
# per_slice = one stock-kernel launch per slice (the incumbent: two at
# gate/up, using the generic fused-MoE body); lean_per_slice = the LEAN
# kernel body, still one launch per slice — exists to isolate the
# launch-fusion effect from the body-leanness effect (gate-4 review
# finding 2); one_launch_sliced = ONE launch whose N tiles are laid out
# slice-major (a tile never crosses the slice boundary; the slice is
# derived per tile and selects the bridge K-range and the destination
# column offset). Named for what it is — it is NOT a contiguous-flat
# layout: when BLOCK_N divides the slice width the slice-major tile set
# and order are IDENTICAL to a contiguous layout, and when it does not,
# a contiguous layout would need boundary-straddling tiles (illegal for
# one tensor-core dot without splitting it).
SLICINGS = ("per_slice", "lean_per_slice", "one_launch_sliced")
REDUCTIONS = ("whole_rank", "deterministic_rank_split")


class LoraBExecutionSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One point in the B-schedule candidate space."""

    site: str
    ownership: str
    slicing: str = "per_slice"
    reduction: str = "whole_rank"

    def __post_init__(self):
        for field_name, (value, vocabulary) in {
            "site": (self.site, SITES),
            "ownership": (self.ownership, OWNERSHIPS),
            "slicing": (self.slicing, SLICINGS),
            "reduction": (self.reduction, REDUCTIONS),
        }.items():
            if value not in vocabulary:
                raise ValueError(f"{field_name}={value!r} is not one of {vocabulary}")
        if self.ownership == "indexed":
            if self.slicing != "one_launch_sliced":
                raise ValueError(
                    "the indexed family derives keys per pair and computes "
                    "every slice in its one launch; "
                    "slicing='one_launch_sliced' is its only honest "
                    "description"
                )
            if self.reduction != "whole_rank":
                raise ValueError(
                    "the indexed family is deterministic by construction "
                    "(serial K loop); a rank-split variant of it does not "
                    "exist"
                )
        if self.reduction == "deterministic_rank_split" and self.slicing != (
            "one_launch_sliced"
        ):
            raise ValueError(
                "deterministic_rank_split partials use the one-launch "
                "sliced body; declare slicing='one_launch_sliced'"
            )

    def key(self) -> str:
        parts = [self.site, "b", self.ownership]
        if self.slicing == "lean_per_slice":
            parts.append("lean2launch")
        elif self.slicing == "one_launch_sliced":
            parts.append("1launch")
        if self.reduction != "whole_rank":
            parts.append("ranksplit")
        return "_".join(parts)
