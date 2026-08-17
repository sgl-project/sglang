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
``consumed_gate_up_b`` factor contract and requires ``plan.gate_up_b is None``.
This makes illegal combinations such as gate/up-A+B overlap plus B+activation
fusion fail before CUDA work.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import cache

import pydantic
from pydantic.dataclasses import dataclass as pydantic_dataclass

logger = logging.getLogger(__name__)

# The spec classes are pydantic dataclasses: field types (enums, bools,
# nested specs) are enforced at construction, so the validate() methods
# carry only cross-field semantics.  Strict mode keeps the fail-closed
# posture — no silent coercion of strings or ints into enums and bools —
# and extra="forbid" keeps unknown constructor kwargs a hard error the
# way stdlib dataclasses were (pydantic's default silently drops them).
_STRICT = pydantic.ConfigDict(strict=True, extra="forbid")


class Site(str, Enum):
    GATE_UP = "gate_up"
    DOWN = "down"


class BridgeLayout(str, Enum):
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
    GATE_UP_A = "gate_up_a"
    GATE_UP_A_B = "gate_up_a_b"


class LateOverlap(str, Enum):
    NONE = "none"
    DOWN_A = "down_a"
    DOWN_B = "down_b"
    DOWN_A_B = "down_a_b"


def _require_bool(value: object, field: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool, got {type(value).__name__}")


def _aligned_requirement(is_shared_outer: bool) -> RouteRequirement:
    if is_shared_outer:
        return RouteRequirement.ALIGNED_SHARED_OUTER
    return RouteRequirement.ALIGNED_PER_EXPERT


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class StageContract:
    """The factor and bridge contract of one logical A or B stage."""

    site: Site
    is_shared_outer: bool
    layout: BridgeLayout

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> StageContract:
        if self.site is Site.DOWN and self.layout is BridgeLayout.TOKEN_MAJOR:
            raise ValueError(
                "the down bridge is inherently pair-major: each routed expert "
                "produces a different activation"
            )
        return self


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class LoraASpec:
    """One standalone LoRA-A execution stage."""

    site: Site
    family: LoraAFamily
    is_shared_outer: bool = False
    output_layout: BridgeLayout = BridgeLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    @property
    def contract(self) -> StageContract:
        return StageContract(self.site, self.is_shared_outer, self.output_layout)

    def validate(self) -> LoraASpec:
        self.contract.validate()
        if self.site is Site.DOWN and self.is_shared_outer:
            raise ValueError(
                "down A is always per-expert; only gate/up A may be shared-outer"
            )
        if self.family is LoraAFamily.GROUPED:
            if self.output_layout is not BridgeLayout.PAIR_MAJOR:
                raise ValueError("grouped A writes a pair-major bridge")
        elif self.family is LoraAFamily.INDEXED:
            # Step-3 qualified indexed A only as the down-site small-decode
            # frontier; every other site keeps its aligned general kernel.
            if self.site is not Site.DOWN:
                raise ValueError("indexed A is retained only at the down site")
            if self.is_shared_outer:
                raise ValueError("indexed A is qualified only for per-expert factors")
            if self.output_layout is not BridgeLayout.PAIR_MAJOR:
                raise ValueError("indexed A writes a pair-major bridge")
        else:
            if self.site is not Site.GATE_UP:
                raise ValueError(f"{self.family.value} is a shared gate/up-A family")
            if not self.is_shared_outer:
                raise ValueError(
                    f"{self.family.value} requires shared-outer A ownership"
                )
            if self.output_layout is not BridgeLayout.TOKEN_MAJOR:
                raise ValueError(f"{self.family.value} writes a token-major bridge")
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is LoraAFamily.INDEXED:
            return frozenset((RouteRequirement.RAW,))
        if self.family is LoraAFamily.TOKEN_DEDUP_GROUPED:
            return frozenset((RouteRequirement.SHARED_TOKEN_PLAN,))
        return frozenset((_aligned_requirement(self.is_shared_outer),))


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class LoraBSpec:
    """One standalone LoRA-B execution stage."""

    site: Site
    family: LoraBFamily
    is_shared_outer: bool = False
    input_layout: BridgeLayout = BridgeLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    @property
    def contract(self) -> StageContract:
        return StageContract(self.site, self.is_shared_outer, self.input_layout)

    def validate(self) -> LoraBSpec:
        self.contract.validate()
        if self.site is Site.GATE_UP and self.is_shared_outer:
            raise ValueError(
                "gate/up B is always per-expert; only down B may be shared-outer"
            )
        if (
            self.input_layout is BridgeLayout.TOKEN_MAJOR
            and self.site is not Site.GATE_UP
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
            if self.input_layout is not BridgeLayout.PAIR_MAJOR:
                raise ValueError("pair-indexed B consumes a pair-major bridge")
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is LoraBFamily.INDEXED_PAIRS:
            # Descriptor-only: keys are derived inline from the raw source
            # tensors; no aligned pair plan is required for this stage.
            return frozenset((RouteRequirement.RAW,))
        return frozenset((_aligned_requirement(self.is_shared_outer),))


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class MiddleSpec:
    """Activation boundary and the gate/up B stage optionally fused into it.

    A consumed factor names its data contract, while ``family`` names the
    fused implementation.  It is intentionally not an executable
    ``LoraBSpec`` because the standalone family does not run.
    """

    family: MiddleFamily
    activation: ActivationFamily
    consumed_gate_up_b: StageContract | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MiddleSpec:
        if self.consumed_gate_up_b is not None:
            if self.consumed_gate_up_b.site is not Site.GATE_UP:
                raise ValueError("consumed_gate_up_b must describe the gate/up site")
            if self.consumed_gate_up_b.is_shared_outer:
                raise ValueError("consumed gate/up B must be per-expert")

        expected_gate_up_b = self.family is MiddleFamily.B_ACTIVATION
        if (self.consumed_gate_up_b is not None) != expected_gate_up_b:
            raise ValueError(
                f"middle family {self.family.value} "
                f"{'requires' if expected_gate_up_b else 'does not consume'} gate/up B"
            )
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.consumed_gate_up_b is None:
            return frozenset()
        return frozenset(
            (_aligned_requirement(self.consumed_gate_up_b.is_shared_outer),)
        )


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class FinalizeSpec:
    """Final combine family and an optional down-B stage consumed by it."""

    family: FinalizeFamily
    consumed_down_b: StageContract | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> FinalizeSpec:
        consumes_down_b = self.family is not FinalizeFamily.MATERIALIZED
        if (self.consumed_down_b is not None) != consumes_down_b:
            raise ValueError(
                f"finalize family {self.family.value} "
                f"{'requires' if consumes_down_b else 'does not consume'} down B"
            )
        if self.consumed_down_b is not None:
            if self.consumed_down_b.site is not Site.DOWN:
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


@pydantic_dataclass(frozen=True, slots=True, kw_only=True, config=_STRICT)
class MoeLoraExecutionPlan:
    """One immutable whole-pipeline MoE-LoRA execution strategy."""

    gate_up_a: LoraASpec
    middle: MiddleSpec
    finalize: FinalizeSpec
    gate_up_b: LoraBSpec | None = None
    down_a: LoraASpec | None = None
    down_b: LoraBSpec | None = None
    early_overlap: EarlyOverlap = EarlyOverlap.NONE
    late_overlap: LateOverlap = LateOverlap.NONE
    route_builder: RouteBuilderFamily = RouteBuilderFamily.STANDARD
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

    def _gate_up_b_contract(self) -> StageContract:
        if self.gate_up_b is not None:
            return self.gate_up_b.contract
        consumed = self.middle.consumed_gate_up_b
        if consumed is None:
            raise ValueError("the execution plan has no gate/up-B owner")
        return consumed

    def _down_a_contract(self) -> StageContract:
        if self.down_a is None:
            raise ValueError("the execution plan has no down-A owner")
        return self.down_a.contract

    def _down_b_contract(self) -> StageContract:
        if self.down_b is not None:
            return self.down_b.contract
        consumed = self.finalize.consumed_down_b
        if consumed is None:
            raise ValueError("the execution plan has no down-B owner")
        return consumed

    def validate(self) -> MoeLoraExecutionPlan:
        if self.gate_up_a.site is not Site.GATE_UP:
            raise ValueError("gate_up_a must describe the gate/up site")
        if self.gate_up_b is not None and self.gate_up_b.site is not Site.GATE_UP:
            raise ValueError("gate_up_b must describe the gate/up site")
        if self.down_a is not None and self.down_a.site is not Site.DOWN:
            raise ValueError("down_a must describe the down site")
        if self.down_b is not None and self.down_b.site is not Site.DOWN:
            raise ValueError("down_b must describe the down site")

        gate_up_b_consumed = self.middle.consumed_gate_up_b is not None
        if gate_up_b_consumed == (self.gate_up_b is not None):
            raise ValueError(
                "gate/up B must have exactly one owner: standalone gate_up_b or middle"
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

        gate_up_b_contract = self._gate_up_b_contract()
        down_a_contract = self._down_a_contract()
        down_b_contract = self._down_b_contract()
        if self.gate_up_a.output_layout is not gate_up_b_contract.layout:
            raise ValueError(
                "gate/up A output layout must match the gate/up B input layout"
            )
        if down_a_contract.layout is not down_b_contract.layout:
            raise ValueError("down A output layout must match the down B input layout")

        if self.early_overlap is EarlyOverlap.GATE_UP_A_B and self.gate_up_b is None:
            raise ValueError(
                "gate/up-A+B overlap requires standalone gate/up B; the middle owns it"
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

        if self.down_b_scatter and not self.down_b_scatter_eligible():
            raise ValueError(
                "down-B scatter requires a standalone one-launch sliced "
                "down-B stage, the materialized finalize (run in "
                "no-pair-delta mode), no late overlap window (the scatter "
                "read-modify-writes the base down output)"
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

        True iff the schedule has no early/late overlap windows and down-A
        is GROUPED over the canonical pair activation.  Such a plan drives
        the provider seam
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
        )

    def _route_requirements_unchecked(self) -> frozenset[RouteRequirement]:
        requirements: set[RouteRequirement] = set()
        for stage in (self.gate_up_a, self.gate_up_b, self.down_a, self.down_b):
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
        if self.gate_up_a.is_shared_outer is not is_shared_outer:
            raise ValueError(
                "plan gate/up-A ownership does not match resident gate/up-A weights"
            )
        if self._down_b_contract().is_shared_outer is not is_shared_outer:
            raise ValueError(
                "plan down-B ownership does not match resident down-B weights"
            )
        return self


SERIAL_MATERIALIZED_REFERENCE = MoeLoraExecutionPlan(
    gate_up_a=LoraASpec(
        Site.GATE_UP,
        LoraAFamily.GROUPED,
        False,
        BridgeLayout.PAIR_MAJOR,
    ),
    gate_up_b=LoraBSpec(
        Site.GATE_UP,
        LoraBFamily.ONE_LAUNCH_SLICED,
        False,
        BridgeLayout.PAIR_MAJOR,
    ),
    middle=MiddleSpec(
        family=MiddleFamily.MATERIALIZED,
        activation=ActivationFamily.SWIGLU,
    ),
    down_a=LoraASpec(
        Site.DOWN,
        LoraAFamily.GROUPED,
        False,
        BridgeLayout.PAIR_MAJOR,
    ),
    down_b=LoraBSpec(
        Site.DOWN,
        LoraBFamily.ONE_LAUNCH_SLICED,
        False,
        BridgeLayout.PAIR_MAJOR,
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
        gate_up_a=LoraASpec(
            Site.GATE_UP,
            LoraAFamily.GROUPED,
            is_shared_outer,
            BridgeLayout.PAIR_MAJOR,
        ),
        gate_up_b=LoraBSpec(
            Site.GATE_UP,
            LoraBFamily.ONE_LAUNCH_SLICED,
            False,
            BridgeLayout.PAIR_MAJOR,
        ),
        middle=MiddleSpec(
            family=MiddleFamily.MATERIALIZED,
            activation=activation,
        ),
        down_a=LoraASpec(
            Site.DOWN,
            LoraAFamily.GROUPED,
            False,
            BridgeLayout.PAIR_MAJOR,
        ),
        down_b=LoraBSpec(
            Site.DOWN,
            LoraBFamily.ONE_LAUNCH_SLICED,
            is_shared_outer,
            BridgeLayout.PAIR_MAJOR,
        ),
        finalize=FinalizeSpec(family=FinalizeFamily.MATERIALIZED),
    )


# ---------------------------------------------------------------------------
# Plan tables: pydantic-validated JSON, loaded once per architecture.
#
# Per-forward selection is a phase lookup: layout and the pool-padded rank
# are server-lifetime constants, so ``resolve_plans`` runs once per layer at
# bind time and returns one selected plan per phase. Rows are matched first
# hit in order; ``max_rank`` is the only predicate (the measured H200
# shared-prefill kernel band). Launch tiles live in the separate tile tables
# (see ``launch_config``): plans say WHAT runs, tiles say HOW it launches.
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    DECODE = "decode"
    PREFILL = "prefill"


class DeviceArchitecture(str, Enum):
    H200 = "h200"
    GB300 = "gb300"
    DEFAULT = "default"


def architecture_for_capability(major: int, minor: int) -> DeviceArchitecture:
    if major == 9:
        return DeviceArchitecture.H200
    if major >= 10:
        return DeviceArchitecture.GB300
    return DeviceArchitecture.DEFAULT


class _PlanSpecModel(pydantic.BaseModel):
    """One row's plan spec. Enum-typed, so a malformed value in ANY row
    fails at table load, not only when the row is selected."""

    model_config = pydantic.ConfigDict(extra="forbid")

    gate_up_a_family: LoraAFamily = LoraAFamily.GROUPED
    down_a_family: LoraAFamily = LoraAFamily.GROUPED
    gate_up_b_family: LoraBFamily = LoraBFamily.ONE_LAUNCH_SLICED
    down_b_family: LoraBFamily = LoraBFamily.ONE_LAUNCH_SLICED
    middle_family: MiddleFamily = MiddleFamily.MATERIALIZED
    finalize_family: FinalizeFamily = FinalizeFamily.MATERIALIZED
    early_overlap: EarlyOverlap = EarlyOverlap.NONE
    late_overlap: LateOverlap = LateOverlap.NONE
    route_builder: RouteBuilderFamily = RouteBuilderFamily.STANDARD
    down_b_scatter: bool = False


class _PlanRowModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    layout: str | None = None  # per_expert | shared; None matches both
    phase: str | None = None  # decode | prefill; None matches both
    max_rank: int | None = None
    provider: str
    plan: _PlanSpecModel


class _PlansFileModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    arch: str
    domain: dict[str, int] = pydantic.Field(default_factory=dict)
    scenarios: list[_PlanRowModel] = pydantic.Field(default_factory=list)
    fallback: list[_PlanRowModel]


_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _read_table(name: str) -> dict | None:
    from sglang.srt.environ import envs

    override_dir = envs.SGLANG_LORA_MOE_CONFIG_DIR.get()
    for directory in filter(None, (override_dir, _CONFIG_DIR)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            if directory == override_dir:
                logger.info(
                    "MoE LoRA table %r loaded from override dir %s", name, directory
                )
            with open(path) as handle:
                return json.load(handle)
    return None


@cache
def load_plans(architecture: DeviceArchitecture) -> _PlansFileModel:
    raw = _read_table(f"{architecture.value}.plans.json")
    if raw is None:
        logger.warning(
            "no MoE LoRA plan table for architecture %s; serving the "
            "conservative default plans",
            architecture.value,
        )
        raw = _read_table("default.plans.json")
    if raw is None:
        raise RuntimeError("MoE LoRA plan tables are missing from the package")
    return _PlansFileModel.model_validate(raw)


@dataclass(frozen=True, slots=True)
class SelectedPlan:
    """One phase's resolved menu entry: identity, provider, validated plan."""

    key: str
    name: str
    provider: str
    plan: MoeLoraExecutionPlan


def build_plan(
    spec: _PlanSpecModel,
    *,
    activation: ActivationFamily,
    is_shared_outer: bool,
) -> MoeLoraExecutionPlan:
    """Materialize one row's plan spec into a validated execution plan.

    The activation is a property of the layer, injected at construction —
    rows are activation-agnostic by decision (2026-08-16).
    """
    gate_up_a_family = spec.gate_up_a_family
    gate_up_layout = (
        BridgeLayout.TOKEN_MAJOR
        if gate_up_a_family is LoraAFamily.TOKEN_DEDUP_GROUPED
        else BridgeLayout.PAIR_MAJOR
    )
    middle_family = spec.middle_family
    finalize_family = spec.finalize_family
    gate_up_b_contract = StageContract(Site.GATE_UP, False, gate_up_layout)
    down_b_contract = StageContract(Site.DOWN, is_shared_outer, BridgeLayout.PAIR_MAJOR)
    consumes_gate_up_b = middle_family is MiddleFamily.B_ACTIVATION
    consumes_down_b = finalize_family is not FinalizeFamily.MATERIALIZED
    plan = MoeLoraExecutionPlan(
        gate_up_a=LoraASpec(
            Site.GATE_UP, gate_up_a_family, is_shared_outer, gate_up_layout
        ),
        gate_up_b=(
            None
            if consumes_gate_up_b
            else LoraBSpec(Site.GATE_UP, spec.gate_up_b_family, False, gate_up_layout)
        ),
        middle=MiddleSpec(
            middle_family,
            activation,
            gate_up_b_contract if consumes_gate_up_b else None,
        ),
        down_a=LoraASpec(
            Site.DOWN,
            spec.down_a_family,
            False,
            BridgeLayout.PAIR_MAJOR,
        ),
        down_b=(
            None
            if consumes_down_b
            else LoraBSpec(
                Site.DOWN,
                spec.down_b_family,
                is_shared_outer,
                BridgeLayout.PAIR_MAJOR,
            )
        ),
        finalize=FinalizeSpec(
            finalize_family, down_b_contract if consumes_down_b else None
        ),
        early_overlap=spec.early_overlap,
        late_overlap=spec.late_overlap,
        route_builder=spec.route_builder,
        down_b_scatter=spec.down_b_scatter,
    )
    return plan.validate_ownership(is_shared_outer)


def resolve_plans(
    *,
    architecture: DeviceArchitecture,
    is_shared_outer: bool,
    physical_rank: int,
    activation: ActivationFamily,
    hidden_size: int,
    num_local_experts: int,
) -> dict[Phase, SelectedPlan]:
    """Resolve the layer's one plan per phase, once, at bind time.

    Every input is a server-lifetime constant, so nothing about plan
    selection remains for the forward path. Out-of-domain geometry serves
    the fallback rows.
    """
    table = load_plans(architecture)
    layout_name = "shared" if is_shared_outer else "per_expert"
    in_domain = hidden_size <= table.domain.get(
        "max_hidden", 1 << 30
    ) and num_local_experts <= table.domain.get("max_local_experts", 1 << 30)
    rows = table.scenarios if in_domain else []
    if not in_domain:
        logger.warning(
            "MoE LoRA geometry (hidden=%d, local_experts=%d) is outside the "
            "tuned domain of table %r; serving the serial fallback",
            hidden_size,
            num_local_experts,
            architecture.value,
        )
    selected: dict[Phase, SelectedPlan] = {}
    for phase in Phase:
        row = next(
            (
                candidate
                for candidate in (*rows, *table.fallback)
                if candidate.layout in (None, layout_name)
                and candidate.phase in (None, phase.value)
                and (candidate.max_rank is None or physical_rank <= candidate.max_rank)
            ),
            None,
        )
        if row is None:
            raise RuntimeError(
                f"no MoE LoRA plan row matches ({layout_name}, {phase.value}) "
                f"on {architecture.value}"
            )
        selected[phase] = SelectedPlan(
            key=f"{architecture.value}.{layout_name}.{row.name}",
            name=row.name,
            provider=row.provider,
            plan=build_plan(
                row.plan,
                activation=activation,
                is_shared_outer=is_shared_outer,
            ),
        )
    return selected


def iter_selected_plans(
    *,
    architecture: DeviceArchitecture,
    is_shared_outer: bool,
    activation: ActivationFamily,
) -> list[SelectedPlan]:
    """Every buildable row for one layout — the menu, rank-unfiltered.

    Tests and the offline tuner enumerate this; serving uses
    :func:`resolve_plans`, which picks one row per phase.
    """
    table = load_plans(architecture)
    layout_name = "shared" if is_shared_outer else "per_expert"
    out: list[SelectedPlan] = []
    for row in (*table.scenarios, *table.fallback):
        if row.layout not in (None, layout_name):
            continue
        out.append(
            SelectedPlan(
                key=f"{architecture.value}.{layout_name}.{row.name}",
                name=row.name,
                provider=row.provider,
                plan=build_plan(
                    row.plan,
                    activation=activation,
                    is_shared_outer=is_shared_outer,
                ),
            )
        )
    return out
