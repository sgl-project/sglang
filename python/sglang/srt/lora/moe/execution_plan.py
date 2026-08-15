"""Typed execution plans for the BF16 MoE-LoRA pipeline.

This module describes *what* one forward executes.  It deliberately contains
no CUDA imports, launch configuration, device thresholds, or selection
logic.
The runner and benchmark can therefore validate a forced whole-pipeline plan
before allocating a workspace or launching any kernel.

An A kernel writes a rank bridge and the matching B kernel reads it.  The
bridge contract is explicit at each site:

* ``PAIR_MAJOR`` is one row per routed ``(token, expert)`` pair.
* ``TOKEN_MAJOR`` is the shared-outer gate/up form, one row per token.

Fusion is represented by ownership, not by pretending that a consumed stage
still runs independently.  For example, ``B_ACTIVATION`` carries the
``consumed_gate_b`` factor contract and requires ``plan.gate_b is None``.
This makes illegal combinations such as gate-A+B overlap plus B+activation
fusion fail before CUDA work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FactorSite(str, Enum):
    GATE_UP = "gate_up"
    DOWN = "down"


class FactorLayout(str, Enum):
    """Logical row layout of the rank bridge between A and B."""

    PAIR_MAJOR = "pair_major"
    TOKEN_MAJOR = "token_major"


class RouteRequirement(str, Enum):
    """Route representations consumed by a whole execution plan.

    ``RAW`` materializes no derived metadata; consumers derive keys directly
    from the source tensors.  The other values are distinct products and may
    coexist.  In particular, a shared-outer forward can require both aligned
    per-expert and aligned shared-outer pair plans.
    """

    RAW = "raw"
    FUSED_PER_EXPERT = "fused_per_expert"
    FUSED_SHARED_OUTER = "fused_shared_outer"
    ALIGNED_PER_EXPERT = "aligned_per_expert"
    ALIGNED_SHARED_OUTER = "aligned_shared_outer"
    SHARED_TOKEN_PLAN = "shared_token_plan"


class RouteBuilderFamily(str, Enum):
    """Implementation used to build the required route products."""

    STANDARD = "standard"
    JOINT_SHARED_OUTER = "joint_shared_outer"


class LoraAFamily(str, Enum):
    GROUPED = "grouped"
    INDEXED = "indexed"
    TOKEN_DEDUP_GROUPED = "token_dedup_grouped"


class LoraBFamily(str, Enum):
    ONE_LAUNCH_SLICED = "one_launch_sliced"
    INDEXED_PAIRS = "indexed_pairs"


class ActivationFamily(str, Enum):
    SWIGLU = "swiglu"
    RELU2 = "relu2"


class MiddleFamily(str, Enum):
    MATERIALIZED = "materialized"
    B_ACTIVATION = "b_activation"


class FinalizeFamily(str, Enum):
    MATERIALIZED = "materialized"
    SHARED_RANK_REDUCE = "shared_rank_reduce"


class EarlyOverlap(str, Enum):
    NONE = "none"
    GATE_A = "gate_a"
    GATE_A_B = "gate_a_b"


class LateOverlap(str, Enum):
    NONE = "none"
    DOWN_A = "down_a"
    DOWN_B = "down_b"
    DOWN_A_B = "down_a_b"


def _require_enum(value: object, enum_type: type[Enum], field: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"{field} must be {enum_type.__name__}, got {type(value).__name__}"
        )


def _require_bool(value: object, field: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool, got {type(value).__name__}")


def _aligned_requirement(is_shared_outer: bool) -> RouteRequirement:
    if is_shared_outer:
        return RouteRequirement.ALIGNED_SHARED_OUTER
    return RouteRequirement.ALIGNED_PER_EXPERT


@dataclass(frozen=True, slots=True)
class FactorContract:
    """The factor and bridge contract of one logical A or B stage."""

    site: FactorSite
    is_shared_outer: bool
    layout: FactorLayout

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> FactorContract:
        _require_enum(self.site, FactorSite, "site")
        _require_bool(self.is_shared_outer, "is_shared_outer")
        _require_enum(self.layout, FactorLayout, "layout")
        if self.site is FactorSite.DOWN and self.layout is FactorLayout.TOKEN_MAJOR:
            raise ValueError(
                "the down bridge is inherently pair-major: each routed expert "
                "produces a different activation"
            )
        return self


@dataclass(frozen=True, slots=True)
class LoraASpec:
    """One standalone LoRA-A execution stage."""

    site: FactorSite
    family: LoraAFamily
    is_shared_outer: bool = False
    output_layout: FactorLayout = FactorLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    @property
    def contract(self) -> FactorContract:
        return FactorContract(self.site, self.is_shared_outer, self.output_layout)

    def validate(self) -> LoraASpec:
        _require_enum(self.site, FactorSite, "site")
        _require_enum(self.family, LoraAFamily, "family")
        _require_bool(self.is_shared_outer, "is_shared_outer")
        _require_enum(self.output_layout, FactorLayout, "output_layout")
        self.contract.validate()

        if self.site is FactorSite.DOWN and self.is_shared_outer:
            raise ValueError(
                "down A is always per-expert; only gate/up A may be shared-outer"
            )
        if self.family is LoraAFamily.GROUPED:
            if self.output_layout is not FactorLayout.PAIR_MAJOR:
                raise ValueError("grouped A writes a pair-major bridge")
        elif self.family is LoraAFamily.INDEXED:
            # Step-3 qualified indexed A only as the down-site small-decode
            # frontier; every other site keeps its aligned general kernel.
            if self.site is not FactorSite.DOWN:
                raise ValueError("indexed A is retained only at the down site")
            if self.is_shared_outer:
                raise ValueError("indexed A is qualified only for per-expert factors")
            if self.output_layout is not FactorLayout.PAIR_MAJOR:
                raise ValueError("indexed A writes a pair-major bridge")
        else:
            if self.site is not FactorSite.GATE_UP:
                raise ValueError(f"{self.family.value} is a shared gate/up-A family")
            if not self.is_shared_outer:
                raise ValueError(
                    f"{self.family.value} requires shared-outer A ownership"
                )
            if self.output_layout is not FactorLayout.TOKEN_MAJOR:
                raise ValueError(f"{self.family.value} writes a token-major bridge")
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is LoraAFamily.INDEXED:
            return frozenset((RouteRequirement.RAW,))
        if self.family is LoraAFamily.TOKEN_DEDUP_GROUPED:
            return frozenset((RouteRequirement.SHARED_TOKEN_PLAN,))
        return frozenset((_aligned_requirement(self.is_shared_outer),))


@dataclass(frozen=True, slots=True)
class LoraBSpec:
    """One standalone LoRA-B execution stage."""

    site: FactorSite
    family: LoraBFamily
    is_shared_outer: bool = False
    input_layout: FactorLayout = FactorLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    @property
    def contract(self) -> FactorContract:
        return FactorContract(self.site, self.is_shared_outer, self.input_layout)

    def validate(self) -> LoraBSpec:
        _require_enum(self.site, FactorSite, "site")
        _require_enum(self.family, LoraBFamily, "family")
        _require_bool(self.is_shared_outer, "is_shared_outer")
        _require_enum(self.input_layout, FactorLayout, "input_layout")
        self.contract.validate()

        if self.site is FactorSite.GATE_UP and self.is_shared_outer:
            raise ValueError(
                "gate/up B is always per-expert; only down B may be shared-outer"
            )
        if (
            self.input_layout is FactorLayout.TOKEN_MAJOR
            and self.site is not FactorSite.GATE_UP
        ):
            raise ValueError("a token-major B input exists only at gate/up")
        if self.family is LoraBFamily.INDEXED_PAIRS:
            # The pair-indexed decode expand derives each pair's virtual
            # expert key inline from the raw route; it has no shared-outer
            # or token-major qualification.
            if self.is_shared_outer:
                raise ValueError(
                    "pair-indexed B is qualified only for per-expert factors"
                )
            if self.input_layout is not FactorLayout.PAIR_MAJOR:
                raise ValueError("pair-indexed B consumes a pair-major bridge")
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is LoraBFamily.INDEXED_PAIRS:
            # Descriptor-only: keys are derived inline from the raw source
            # tensors; no aligned pair plan is required for this stage.
            return frozenset((RouteRequirement.RAW,))
        return frozenset((_aligned_requirement(self.is_shared_outer),))


@dataclass(frozen=True, slots=True)
class MiddleSpec:
    """Activation boundary and the gate/up B stage optionally fused into it.

    A consumed factor names its data contract, while ``family`` names the
    fused implementation.  It is intentionally not an executable
    ``LoraBSpec`` because the standalone family does not run.
    """

    family: MiddleFamily
    activation: ActivationFamily
    consumed_gate_b: FactorContract | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MiddleSpec:
        _require_enum(self.family, MiddleFamily, "family")
        _require_enum(self.activation, ActivationFamily, "activation")
        if self.consumed_gate_b is not None:
            self.consumed_gate_b.validate()
            if self.consumed_gate_b.site is not FactorSite.GATE_UP:
                raise ValueError("consumed_gate_b must describe the gate/up site")
            if self.consumed_gate_b.is_shared_outer:
                raise ValueError("consumed gate/up B must be per-expert")

        expected_gate_b = self.family is MiddleFamily.B_ACTIVATION
        if (self.consumed_gate_b is not None) != expected_gate_b:
            raise ValueError(
                f"middle family {self.family.value} "
                f"{'requires' if expected_gate_b else 'does not consume'} gate B"
            )
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.consumed_gate_b is None:
            return frozenset()
        return frozenset((_aligned_requirement(self.consumed_gate_b.is_shared_outer),))


@dataclass(frozen=True, slots=True)
class FinalizeSpec:
    """Final combine family and an optional down-B stage consumed by it."""

    family: FinalizeFamily
    consumed_down_b: FactorContract | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> FinalizeSpec:
        _require_enum(self.family, FinalizeFamily, "family")
        consumes_down_b = self.family is not FinalizeFamily.MATERIALIZED
        if (self.consumed_down_b is not None) != consumes_down_b:
            raise ValueError(
                f"finalize family {self.family.value} "
                f"{'requires' if consumes_down_b else 'does not consume'} down B"
            )
        if self.consumed_down_b is not None:
            self.consumed_down_b.validate()
            if self.consumed_down_b.site is not FactorSite.DOWN:
                raise ValueError("consumed_down_b must describe the down site")
        if self.family is FinalizeFamily.SHARED_RANK_REDUCE:
            consumed_down_b = self.consumed_down_b
            if consumed_down_b is None:
                raise ValueError(f"{self.family.value} requires down B")
            if not consumed_down_b.is_shared_outer:
                raise ValueError(
                    f"{self.family.value} requires shared-outer down-B ownership"
                )
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is FinalizeFamily.MATERIALIZED:
            return frozenset()
        # The shared-rank finalizer derives its fixed-top-k keys from the raw
        # route; it never consumes LoRABatchInfo.
        return frozenset((RouteRequirement.RAW,))


@dataclass(frozen=True, slots=True, kw_only=True)
class MoeLoraExecutionPlan:
    """One immutable whole-pipeline MoE-LoRA execution strategy."""

    gate_a: LoraASpec
    middle: MiddleSpec
    finalize: FinalizeSpec
    gate_b: LoraBSpec | None = None
    down_a: LoraASpec | None = None
    down_b: LoraBSpec | None = None
    early_overlap: EarlyOverlap = EarlyOverlap.NONE
    late_overlap: LateOverlap = LateOverlap.NONE
    route_builder: RouteBuilderFamily = RouteBuilderFamily.STANDARD
    # None means PDL-off for the standard fused-align builder. Explicit
    # off/on controls remain separate composed candidates until end-to-end
    # evidence is strong enough to select one by default.
    route_pdl: bool | None = None
    # Programmatic dependent launch is an execution edge, not a device-global
    # kernel switch.  It is legal only when the named A producer and B consumer
    # are consecutive launches on the same stream and both families implement
    # the matching GDC signal/wait protocol.
    gate_a_to_b_pdl: bool = False
    down_a_to_b_pdl: bool = False
    # Base-provider PDL is modeled as two concrete execution edges. The
    # CuTeDSL base GEMM signals; the selected middle/finalize kernel is the
    # dependent consumer. Overlap schedules insert an event join, so these
    # controls admit only direct same-stream handoffs.
    base_gateup_to_middle_pdl: bool = False
    # Down-tail reordering experiment: the standalone one-launch sliced
    # down-B runs AFTER the base down GEMM and read-modify-write adds its
    # unweighted delta into the provider's down output rows through the
    # provider's pair-to-row mapping (indirect row addressing; the GEMM
    # tiling itself is unchanged), and the materialized finalize then runs in
    # no-pair-delta mode.  The [T, K, H] pair-major delta buffer is never
    # allocated on this path.
    down_b_scatter: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def has_overlap(self) -> bool:
        """Whether any stage is scheduled concurrently with the base path."""

        return (
            self.early_overlap is not EarlyOverlap.NONE
            or self.late_overlap is not LateOverlap.NONE
        )

    def _gate_b_contract(self) -> FactorContract:
        if self.gate_b is not None:
            return self.gate_b.contract
        consumed = self.middle.consumed_gate_b
        if consumed is None:
            raise ValueError("the execution plan has no gate-B owner")
        return consumed

    def _down_a_contract(self) -> FactorContract:
        if self.down_a is None:
            raise ValueError("the execution plan has no down-A owner")
        return self.down_a.contract

    def _down_b_contract(self) -> FactorContract:
        if self.down_b is not None:
            return self.down_b.contract
        consumed = self.finalize.consumed_down_b
        if consumed is None:
            raise ValueError("the execution plan has no down-B owner")
        return consumed

    def validate(self) -> MoeLoraExecutionPlan:
        if not isinstance(self.gate_a, LoraASpec):
            raise TypeError("gate_a must be LoraASpec")
        if not isinstance(self.middle, MiddleSpec):
            raise TypeError("middle must be MiddleSpec")
        if not isinstance(self.finalize, FinalizeSpec):
            raise TypeError("finalize must be FinalizeSpec")
        for field, value, expected in (
            ("gate_b", self.gate_b, LoraBSpec),
            ("down_a", self.down_a, LoraASpec),
            ("down_b", self.down_b, LoraBSpec),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{field} must be {expected.__name__} or None")
        _require_enum(self.early_overlap, EarlyOverlap, "early_overlap")
        _require_enum(self.late_overlap, LateOverlap, "late_overlap")
        _require_enum(self.route_builder, RouteBuilderFamily, "route_builder")
        if self.route_pdl is not None and not isinstance(self.route_pdl, bool):
            raise TypeError("route_pdl must be bool or None")
        if (
            self.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER
            and self.route_pdl is None
        ):
            raise ValueError(
                "joint shared-outer routing requires an explicit route_pdl "
                "control until its PDL chain is promoted by composed evidence"
            )
        if not isinstance(self.gate_a_to_b_pdl, bool):
            raise TypeError("gate_a_to_b_pdl must be bool")
        if not isinstance(self.down_a_to_b_pdl, bool):
            raise TypeError("down_a_to_b_pdl must be bool")
        if not isinstance(self.base_gateup_to_middle_pdl, bool):
            raise TypeError("base_gateup_to_middle_pdl must be bool")
        if not isinstance(self.down_b_scatter, bool):
            raise TypeError("down_b_scatter must be bool")

        self.gate_a.validate()
        self.middle.validate()
        self.finalize.validate()
        if self.gate_b is not None:
            self.gate_b.validate()
        if self.down_a is not None:
            self.down_a.validate()
        if self.down_b is not None:
            self.down_b.validate()

        if self.gate_a.site is not FactorSite.GATE_UP:
            raise ValueError("gate_a must describe the gate/up site")
        if self.gate_b is not None and self.gate_b.site is not FactorSite.GATE_UP:
            raise ValueError("gate_b must describe the gate/up site")
        if self.down_a is not None and self.down_a.site is not FactorSite.DOWN:
            raise ValueError("down_a must describe the down site")
        if self.down_b is not None and self.down_b.site is not FactorSite.DOWN:
            raise ValueError("down_b must describe the down site")

        gate_b_consumed = self.middle.consumed_gate_b is not None
        if gate_b_consumed == (self.gate_b is not None):
            raise ValueError(
                "gate B must have exactly one owner: standalone gate_b or middle"
            )
        if self.down_a is None:
            raise ValueError(
                "down A is always a standalone stage: no retained middle "
                "family consumes it"
            )
        down_b_consumed = self.finalize.consumed_down_b is not None
        if down_b_consumed == (self.down_b is not None):
            raise ValueError(
                "down B must have exactly one owner: standalone down_b or finalize"
            )

        gate_b_contract = self._gate_b_contract()
        down_a_contract = self._down_a_contract()
        down_b_contract = self._down_b_contract()
        if self.gate_a.output_layout is not gate_b_contract.layout:
            raise ValueError("gate A output layout must match the gate B input layout")
        if down_a_contract.layout is not down_b_contract.layout:
            raise ValueError("down A output layout must match the down B input layout")

        if self.early_overlap is EarlyOverlap.GATE_A_B and self.gate_b is None:
            raise ValueError(
                "gate-A+B overlap requires standalone gate B; the middle owns it"
            )
        if (
            self.late_overlap
            in (
                LateOverlap.DOWN_B,
                LateOverlap.DOWN_A_B,
            )
            and self.down_b is None
        ):
            raise ValueError(
                f"{self.late_overlap.value} overlap requires standalone down B"
            )

        if self.gate_a_to_b_pdl and not self.gate_a_to_b_pdl_eligible():
            raise ValueError(
                "gate-A -> gate-B PDL requires a grouped A producer, a "
                "one-launch sliced B consumer, and a same-stream consecutive "
                "schedule (serial or gate-A+B overlap)"
            )
        if self.down_a_to_b_pdl and not self.down_a_to_b_pdl_eligible():
            raise ValueError(
                "down-A -> down-B PDL requires standalone grouped A and "
                "one-launch sliced B stages on a same-stream consecutive "
                "schedule (serial or down-A+B overlap)"
            )
        if (
            self.base_gateup_to_middle_pdl
            and not self.base_gateup_to_middle_pdl_eligible()
        ):
            raise ValueError(
                "base gate/up -> middle PDL requires a direct serial handoff "
                "to a PDL-aware Triton middle consumer"
            )

        if self.down_b_scatter and not self.down_b_scatter_eligible():
            raise ValueError(
                "down-B scatter requires a standalone one-launch sliced "
                "down-B stage, the materialized finalize (run in "
                "no-pair-delta mode), no late overlap window (the scatter "
                "read-modify-writes the base down output), and no down-site "
                "or base-down PDL edge (the base down GEMM sits between "
                "down-A and down-B, and the scatter launch sits between the "
                "base down GEMM and the finalize)"
            )

        requirements = self._route_requirements_unchecked()
        if self.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER:
            needed = {
                RouteRequirement.ALIGNED_PER_EXPERT,
                RouteRequirement.ALIGNED_SHARED_OUTER,
            }
            if not needed.issubset(requirements):
                raise ValueError(
                    "the joint shared-outer route builder requires both aligned "
                    "per-expert and aligned shared-outer pair plans"
                )
        return self

    def is_fully_serial(self) -> bool:
        """Whether the schedule is a plain ordered same-stream pipeline.

        True iff the schedule has no early/late overlap windows, down-A is
        GROUPED over the canonical pair activation, and no stage is stitched
        to another through a PDL edge.  Such a plan drives the provider seam
        (prepare / gateup / middle / down / finalize) as ordered same-stream
        calls with no cross-stage coupling, which is the schedule shape
        row-domain conversions key on; the finalize family is judged
        separately (:meth:`is_fully_serial_materialized`).
        """
        return (
            self.early_overlap is EarlyOverlap.NONE
            and self.late_overlap is LateOverlap.NONE
            and self.down_a is not None
            and self.down_a.family is LoraAFamily.GROUPED
            and not self.gate_a_to_b_pdl
            and not self.down_a_to_b_pdl
            and not self.base_gateup_to_middle_pdl
            # The scatter reordering couples down-B to the base down output;
            # it is applied ON TOP of a fully serial materialized shape and
            # must not re-qualify for shape-keyed conversions.
            and not self.down_b_scatter
        )

    def is_fully_serial_materialized(self) -> bool:
        """A fully serial schedule whose finalize is also MATERIALIZED.

        The MATERIALIZED finalize recombines base and LoRA delta in one
        standalone launch, so this is the shape the middle-swap and
        scatter config steps key on.
        """
        return (
            self.is_fully_serial()
            and self.finalize.family is FinalizeFamily.MATERIALIZED
        )

    def down_b_scatter_eligible(self) -> bool:
        """Whether the down tail admits the scatter-into-base reordering."""

        return (
            self.down_b is not None
            and self.down_b.family is LoraBFamily.ONE_LAUNCH_SLICED
            and self.finalize.family is FinalizeFamily.MATERIALIZED
            and self.late_overlap is LateOverlap.NONE
            and not self.down_a_to_b_pdl
        )

    def gate_a_to_b_pdl_eligible(self) -> bool:
        """Whether the gate/up rank bridge has a real same-stream PDL edge."""

        return (
            self.gate_b is not None
            and self.gate_a.family
            in (LoraAFamily.GROUPED, LoraAFamily.TOKEN_DEDUP_GROUPED)
            and self.gate_b.family is LoraBFamily.ONE_LAUNCH_SLICED
            and self.early_overlap in (EarlyOverlap.NONE, EarlyOverlap.GATE_A_B)
        )

    def down_a_to_b_pdl_eligible(self) -> bool:
        """Whether the down rank bridge has a real same-stream PDL edge."""

        return (
            self.down_a is not None
            and self.down_b is not None
            and self.down_a.family is LoraAFamily.GROUPED
            and self.down_b.family is LoraBFamily.ONE_LAUNCH_SLICED
            and self.late_overlap in (LateOverlap.NONE, LateOverlap.DOWN_A_B)
        )

    def base_gateup_to_middle_pdl_eligible(self) -> bool:
        """Whether base GEMM1 directly precedes a PDL-aware middle kernel."""

        return self.early_overlap is EarlyOverlap.NONE

    def _route_requirements_unchecked(self) -> frozenset[RouteRequirement]:
        requirements: set[RouteRequirement] = set()
        for stage in (self.gate_a, self.gate_b, self.down_a, self.down_b):
            if stage is not None:
                requirements.update(stage.route_requirements())
        requirements.update(self.middle.route_requirements())
        requirements.update(self.finalize.route_requirements())
        return frozenset(requirements)

    def route_requirements(self) -> frozenset[RouteRequirement]:
        """Return the exact union of route products consumed by this plan."""

        # The plan and every nested stage are frozen dataclasses, and
        # __post_init__ validates the complete dependency graph. Re-running
        # that proof at each call site would charge immutable plan validation
        # multiple times per layer/forward.
        return self._route_requirements_unchecked()

    def validate_ownership(self, is_shared_outer: bool) -> MoeLoraExecutionPlan:
        """Validate plan identity against the resident A/B weight layout.

        One flag covers both outer factors: an adapter that shares gate/up A
        across experts shares down B too.
        """

        _require_bool(is_shared_outer, "is_shared_outer")
        if self.gate_a.is_shared_outer is not is_shared_outer:
            raise ValueError(
                "plan gate-A ownership does not match resident gate/up-A weights"
            )
        if self._down_b_contract().is_shared_outer is not is_shared_outer:
            raise ValueError(
                "plan down-B ownership does not match resident down-B weights"
            )
        return self


SERIAL_MATERIALIZED_REFERENCE = MoeLoraExecutionPlan(
    gate_a=LoraASpec(
        FactorSite.GATE_UP,
        LoraAFamily.GROUPED,
        False,
        FactorLayout.PAIR_MAJOR,
    ),
    gate_b=LoraBSpec(
        FactorSite.GATE_UP,
        LoraBFamily.ONE_LAUNCH_SLICED,
        False,
        FactorLayout.PAIR_MAJOR,
    ),
    middle=MiddleSpec(
        family=MiddleFamily.MATERIALIZED,
        activation=ActivationFamily.SWIGLU,
    ),
    down_a=LoraASpec(
        FactorSite.DOWN,
        LoraAFamily.GROUPED,
        False,
        FactorLayout.PAIR_MAJOR,
    ),
    down_b=LoraBSpec(
        FactorSite.DOWN,
        LoraBFamily.ONE_LAUNCH_SLICED,
        False,
        FactorLayout.PAIR_MAJOR,
    ),
    finalize=FinalizeSpec(family=FinalizeFamily.MATERIALIZED),
)


def materialized_reference_plan(
    *,
    activation: ActivationFamily,
    is_shared_outer: bool,
) -> MoeLoraExecutionPlan:
    """Build the serial correctness plan for one resident layer contract."""
    _require_bool(is_shared_outer, "is_shared_outer")
    return MoeLoraExecutionPlan(
        gate_a=LoraASpec(
            FactorSite.GATE_UP,
            LoraAFamily.GROUPED,
            is_shared_outer,
            FactorLayout.PAIR_MAJOR,
        ),
        gate_b=LoraBSpec(
            FactorSite.GATE_UP,
            LoraBFamily.ONE_LAUNCH_SLICED,
            False,
            FactorLayout.PAIR_MAJOR,
        ),
        middle=MiddleSpec(
            family=MiddleFamily.MATERIALIZED,
            activation=activation,
        ),
        down_a=LoraASpec(
            FactorSite.DOWN,
            LoraAFamily.GROUPED,
            False,
            FactorLayout.PAIR_MAJOR,
        ),
        down_b=LoraBSpec(
            FactorSite.DOWN,
            LoraBFamily.ONE_LAUNCH_SLICED,
            is_shared_outer,
            FactorLayout.PAIR_MAJOR,
        ),
        finalize=FinalizeSpec(family=FinalizeFamily.MATERIALIZED),
    )
