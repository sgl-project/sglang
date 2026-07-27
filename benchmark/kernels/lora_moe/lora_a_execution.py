"""Typed identity of one LoRA-A execution candidate (Step-3 review fix).

The first cut put a single ``A_SCHEDULES`` string on the workload case; the
review showed it conflated four independent dimensions (ownership, reduction,
implementation, shared-form) — it could not express indexed+CuTeDSL or
grouped+split-K, both required comparisons, and it accepted impossible
declarations like a token-dedup DOWN site.  The case stays workload-oriented;
the candidate is this spec, and different candidates compare against the SAME
workload ``case_id``.  A timing record carries ``spec.key()`` as its
candidate string, so arm identity is structured, not a naming convention.
"""

from __future__ import annotations

import msgspec

SITES = ("gate_up", "down")
OWNERSHIPS = ("grouped", "indexed", "segmented")
REDUCTIONS = ("whole_rank", "deterministic_split_k")
IMPLEMENTATIONS = ("triton", "cutedsl")
SHARED_HANDLINGS = ("repeated_pairs", "token_dedup")
# The segmented family's kernel variants (sixth S3 review: free-form
# variant strings let a recorded identity drift from any real kernel).
SEGMENTED_VARIANTS = ("chunked", "unchunked")


class LoraAExecutionSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One point in the A-schedule candidate space.

    ``shared_handling`` is meaningful only when the workload declares a
    shared-outer factor for the site; ``repeated_pairs`` is the identity
    (today's control form).
    """

    site: str
    ownership: str
    reduction: str = "whole_rank"
    implementation: str = "triton"
    shared_handling: str = "repeated_pairs"
    # Sub-variant within an ownership family (fifth S3 review: the two
    # segmented kernels needed typed identity, not hand strings) — e.g.
    # "chunked" / "unchunked" for ownership="segmented". Empty = none.
    variant: str = ""

    def __post_init__(self):
        for field_name, (value, vocabulary) in {
            "site": (self.site, SITES),
            "ownership": (self.ownership, OWNERSHIPS),
            "reduction": (self.reduction, REDUCTIONS),
            "implementation": (self.implementation, IMPLEMENTATIONS),
            "shared_handling": (self.shared_handling, SHARED_HANDLINGS),
        }.items():
            if value not in vocabulary:
                raise ValueError(f"{field_name}={value!r} is not one of {vocabulary}")
        if self.ownership == "segmented":
            if self.variant not in SEGMENTED_VARIANTS:
                raise ValueError(
                    "a segmented spec must name its kernel variant, one of "
                    f"{SEGMENTED_VARIANTS}; got {self.variant!r} (sixth S3 "
                    "review: identity is typed dispatch, not a label)"
                )
        elif self.variant:
            raise ValueError(
                "variant is only meaningful for the segmented ownership "
                f"family, got variant={self.variant!r} with "
                f"ownership={self.ownership!r}"
            )
        if self.site == "down" and self.shared_handling == "token_dedup":
            raise ValueError(
                "token dedup collapses K pairs of one (token, adapter) into "
                "one row; the down site's input is inherently per-pair "
                "(each pair activates a different expert), so a token-dedup "
                "down-A does not exist (plan section 41.1)"
            )

    def key(self) -> str:
        """Candidate string for timing records; omits identity defaults."""
        parts = [self.site, self.ownership]
        if self.reduction != "whole_rank":
            parts.append("splitk")
        if self.implementation != "triton":
            parts.append(self.implementation)
        if self.shared_handling != "repeated_pairs":
            parts.append(self.shared_handling)
        if self.variant:
            parts.append(self.variant)
        return "_".join(parts)


class LegScheduleSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Typed composite of one MoE leg's two A-site candidates.

    Site-set timing arms carry this as their identity (second S3 review:
    ad-hoc set names could diverge from what the thunk executed), so a
    route-inclusive record names EXACTLY the per-site specs it ran.
    """

    gate_up: LoraAExecutionSpec
    down: LoraAExecutionSpec

    def __post_init__(self):
        if self.gate_up.site != "gate_up" or self.down.site != "down":
            raise ValueError(
                "LegScheduleSpec fields must carry specs for their own site"
            )

    def key(self) -> str:
        return f"set__{self.gate_up.key()}__{self.down.key()}"
