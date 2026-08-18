"""Typed execution plans for the BF16 MoE-LoRA pipeline, and the tables
that pick one.

This module describes *what* one forward executes and resolves which plan a
layer runs: rows load from ``{arch}.plans.json`` and ``resolve_plans`` picks
one per phase, once, at weight bind.  It holds no CUDA imports and no launch
tiles (those are ``launch_config``), so a plan can be validated before any
workspace is allocated.

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
from typing import Any, Literal

import pydantic
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sglang.srt.lora.moe.activation import ActivationFn

logger = logging.getLogger(__name__)

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

    The ``RAW_*`` pair materializes no derived metadata; consumers derive keys
    directly from the source tensors.  Every value names its ownership, so a
    plan that consumes only one form does not get the other built, and asking
    for the form it did not request raises.  The values are distinct products
    and may coexist.  In particular, a shared-outer forward can require both aligned
    per-expert and aligned shared-outer pair plans.
    """

    RAW_PER_EXPERT = "raw_per_expert"
    RAW_SHARED_OUTER = "raw_shared_outer"
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


def _raw_requirement(is_shared_outer: bool) -> RouteRequirement:
    if is_shared_outer:
        return RouteRequirement.RAW_SHARED_OUTER
    return RouteRequirement.RAW_PER_EXPERT


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
        if self.family is LoraAFamily.GROUPED:
            if self.output_layout is not BridgeLayout.PAIR_MAJOR:
                raise ValueError("grouped A writes a pair-major bridge")
        elif self.family is LoraAFamily.INDEXED:
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
            return frozenset((_raw_requirement(self.is_shared_outer),))
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
        if self.family is LoraBFamily.INDEXED_PAIRS:
            # The pair-indexed expand visits one routed pair at a time, so
            # its bridge is inherently pair-major.
            if self.input_layout is not BridgeLayout.PAIR_MAJOR:
                raise ValueError("pair-indexed B consumes a pair-major bridge")
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is LoraBFamily.INDEXED_PAIRS:
            # Descriptor-only: keys are derived inline from the raw source
            # tensors; no aligned pair plan is required for this stage.
            return frozenset((_raw_requirement(self.is_shared_outer),))
        return frozenset((_aligned_requirement(self.is_shared_outer),))


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class MiddleSpec:
    """Activation boundary and the gate/up B stage optionally fused into it.

    A consumed factor names its data contract, while ``family`` names the
    fused implementation.  It is intentionally not an executable
    ``LoraBSpec`` because the standalone family does not run.
    """

    family: MiddleFamily
    activation: ActivationFn
    consumed_gate_up_b: StageContract | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MiddleSpec:
        if self.consumed_gate_up_b is not None:
            if self.consumed_gate_up_b.site is not Site.GATE_UP:
                raise ValueError("consumed_gate_up_b must describe the gate/up site")

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
        if (
            self.family is FinalizeFamily.SHARED_RANK_REDUCE
            and not self.consumed_down_b.is_shared_outer
        ):
            raise ValueError(
                f"{self.family.value} requires shared-outer down-B ownership"
            )
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is FinalizeFamily.MATERIALIZED:
            return frozenset()
        # The shared-rank finalizer derives its fixed-top-k keys from the raw
        # route; it never consumes LoRABatchInfo. consumed_down_b is set here
        # by the exactly-one-owner rule, and its ownership is the one the
        # finalizer reads back.
        return frozenset((_raw_requirement(self.consumed_down_b.is_shared_outer),))


@pydantic_dataclass(frozen=True, slots=True, kw_only=True, config=_STRICT)
class MoeLoraExecutionPlan:
    """One immutable whole-pipeline MoE-LoRA execution strategy."""

    gate_up_a: LoraASpec
    down_a: LoraASpec
    middle: MiddleSpec
    finalize: FinalizeSpec
    gate_up_b: LoraBSpec | None = None
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

    def _gate_up_b_contract(self) -> StageContract:
        if self.gate_up_b is not None:
            return self.gate_up_b.contract
        # The exactly-one-owner parity check runs first, so falling past
        # the branch above means the middle consumed it.
        return self.middle.consumed_gate_up_b

    def _down_a_contract(self) -> StageContract:
        return self.down_a.contract

    def _down_b_contract(self) -> StageContract:
        if self.down_b is not None:
            return self.down_b.contract
        return self.finalize.consumed_down_b

    def validate(self) -> MoeLoraExecutionPlan:
        if self.gate_up_a.site is not Site.GATE_UP:
            raise ValueError("gate_up_a must describe the gate/up site")
        if self.gate_up_b is not None and self.gate_up_b.site is not Site.GATE_UP:
            raise ValueError("gate_up_b must describe the gate/up site")
        if self.down_a.site is not Site.DOWN:
            raise ValueError("down_a must describe the down site")
        if self.down_b is not None and self.down_b.site is not Site.DOWN:
            raise ValueError("down_b must describe the down site")

        gate_up_b_consumed = self.middle.consumed_gate_up_b is not None
        if gate_up_b_consumed == (self.gate_up_b is not None):
            raise ValueError(
                "gate/up B must have exactly one owner: standalone gate_up_b or middle"
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
                "down-B scatter requires a standalone down-B stage and no "
                "late overlap window (the scatter read-modify-writes the "
                "base down output)"
            )

        return self

    def is_fully_serial(self) -> bool:
        """Whether the schedule is a plain ordered same-stream pipeline.

        True iff the schedule has no early/late overlap windows and down-A
        is GROUPED over the pair activation.  Such a plan drives
        the provider seam
        (prepare / gateup / middle / down / finalize) as ordered same-stream
        calls with no cross-stage coupling, which is the schedule shape
        row-domain conversions key on; the finalize family is judged
        separately (:meth:`is_fully_serial_materialized`).
        """
        return (
            self.early_overlap is EarlyOverlap.NONE
            and self.late_overlap is LateOverlap.NONE
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

        # A standalone down-B implies the materialized finalize (any other
        # finalize consumes it).  Which B kernel implements the epilogue is a
        # provider capability, checked in MoeLoraRunner.validate_plan.
        return self.down_b is not None and self.late_overlap is LateOverlap.NONE

    def route_requirements(self) -> frozenset[RouteRequirement]:
        """Return the exact union of route products consumed by this plan.

        Deliberately does not validate: build_routes calls this per forward
        per layer, and the plan and every nested stage are frozen dataclasses
        whose __post_init__ already proved the whole dependency graph.
        """
        return self._requirements_of(
            self.gate_up_a, self.gate_up_b, self.down_a, self.down_b
        )

    def downstream_route_requirements(self) -> frozenset[RouteRequirement]:
        """The same union with gate/up-A left out.

        build_routes asks this to learn whether the SHARED per-expert
        aligned route exists ONLY for gate/up-A, which -- when gate/up-A runs
        at its own block size -- means the shared one need not be built.

        It cannot be derived by subtracting gate_up_a's requirements from the
        full union: that would also drop a requirement a downstream stage
        shares with it, and skip a route something still reads.
        """
        return self._requirements_of(self.gate_up_b, self.down_a, self.down_b)

    def _requirements_of(
        self, *stages: LoraASpec | LoraBSpec | None
    ) -> frozenset[RouteRequirement]:
        # The middle and finalize stages are never optional, so they join
        # every union; only the standalone A/B stages vary.
        requirements: set[RouteRequirement] = set()
        for stage in stages:
            if stage is not None:
                requirements.update(stage.route_requirements())
        requirements.update(self.middle.route_requirements())
        requirements.update(self.finalize.route_requirements())
        return frozenset(requirements)


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
    # Typed, not str: these three are the row's MATCH keys, and a typo in one
    # loads clean and then never matches -- the row goes dead and its layers
    # silently serve the fallback. None still means "matches both".
    layout: Literal["per_expert", "shared"] | None = None
    phase: Phase | None = None
    max_rank: int | None = None
    # Row order of the activation buffer reaching the base GEMM -- NOT the
    # adapter ``layout`` above. expert_major = padded [E, m_max, K] slabs,
    # route_major = one flat buffer of aligned segments. Which vendor
    # implements it is --moe-lora-base-gemm, not a table value.
    base_gemm_rows: Literal["expert_major", "route_major"]
    plan: _PlanSpecModel
    # Free-form annotation the offline tuner stamps on rows it emits or
    # sweeps.  Declared so extra="forbid" still rejects genuine typos while
    # a tuned table stays loadable (see tune_lora_config.py --emit-seed).
    provenance: str | None = None


class _DomainModel(pydantic.BaseModel):
    """Geometry box the scenario rows were tuned inside.

    Typed rather than a bare dict because the reads used to be
    ``domain.get("max_hidden", 1 << 30)``: a typo'd key did not fail, it
    silently left that gate wide open and admitted every geometry to rows
    that were never measured for it. ``None`` means genuinely unbounded;
    the shipped ``default`` table uses 0 to admit nothing.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    max_hidden: int | None = None
    max_local_experts: int | None = None

    def admits(self, *, hidden_size: int, num_local_experts: int) -> bool:
        return (self.max_hidden is None or hidden_size <= self.max_hidden) and (
            self.max_local_experts is None
            or num_local_experts <= self.max_local_experts
        )


class _PlansFileModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    arch: DeviceArchitecture
    domain: _DomainModel = pydantic.Field(default_factory=_DomainModel)
    scenarios: list[_PlanRowModel] = pydantic.Field(default_factory=list)
    fallback: list[_PlanRowModel]
    # Same: the geometry the tuner widened this table's domain for.
    seeded_for: dict[str, Any] | None = None


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
    """One phase's resolved menu entry: identity, row order, validated plan.

    Carries the row order the plan requires, not a provider name: the vendor
    that implements it comes from serving config, so a table cannot pin it.
    """

    key: str
    name: str
    base_gemm_rows: str
    plan: MoeLoraExecutionPlan


def build_plan(
    spec: _PlanSpecModel,
    *,
    activation: ActivationFn,
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
    return plan


@cache
def _warn_out_of_domain(
    architecture: DeviceArchitecture, hidden_size: int, num_local_experts: int
) -> None:
    """Warn once per geometry, not once per layer (60-94 layers per model)."""
    logger.warning(
        "MoE LoRA geometry (hidden=%d, local_experts=%d) is outside the tuned "
        "domain of table %r; serving the serial fallback",
        hidden_size,
        num_local_experts,
        architecture.value,
    )


def resolve_plans(
    *,
    architecture: DeviceArchitecture,
    is_shared_outer: bool,
    physical_rank: int,
    activation: ActivationFn,
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
    in_domain = table.domain.admits(
        hidden_size=hidden_size, num_local_experts=num_local_experts
    )
    rows = table.scenarios if in_domain else []
    if not in_domain:
        _warn_out_of_domain(architecture, hidden_size, num_local_experts)
    selected: dict[Phase, SelectedPlan] = {}
    for phase in Phase:
        row = next(
            (
                candidate
                for candidate in (*rows, *table.fallback)
                if candidate.layout in (None, layout_name)
                and candidate.phase in (None, phase)
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
            base_gemm_rows=row.base_gemm_rows,
            plan=build_plan(
                row.plan,
                activation=activation,
                is_shared_outer=is_shared_outer,
            ),
        )
    return selected
