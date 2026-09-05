"""Select and validate MoE LoRA plans without importing CUDA code."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Literal

import pydantic
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sglang.srt.lora.moe.activation import ActivationFn

logger = logging.getLogger(__name__)

_STRICT = pydantic.ConfigDict(strict=True, extra="forbid")


class Site(str, Enum):
    GATE_UP = "gate_up"
    DOWN = "down"


class BridgeLayout(str, Enum):
    PAIR_MAJOR = "pair_major"
    TOKEN_MAJOR = "token_major"


class RouteRequirement(str, Enum):
    RAW_PER_EXPERT = "raw_per_expert"
    RAW_SHARED_OUTER = "raw_shared_outer"
    ALIGNED_PER_EXPERT = "aligned_per_expert"
    ALIGNED_SHARED_OUTER = "aligned_shared_outer"
    SHARED_TOKEN_PLAN = "shared_token_plan"


class RouteBuilderFamily(str, Enum):
    STANDARD = "standard"
    # Build the shared route on the workspace side stream.
    PARALLEL_SHARED_OUTER = "parallel_shared_outer"


class LoraAFamily(str, Enum):
    GROUPED = "grouped"
    PER_PAIR = "per_pair"
    TOKEN_GROUPED = "token_grouped"


class LoraBFamily(str, Enum):
    GROUPED = "grouped"
    PER_PAIR = "per_pair"


class ActFamily(str, Enum):
    MATERIALIZED = "materialized"
    B_ACTIVATION = "b_activation"


class FinalizeFamily(str, Enum):
    MATERIALIZED = "materialized"
    SHARED_RANK_REDUCE = "shared_rank_reduce"


class GateUpOverlap(str, Enum):
    NONE = "none"
    GATE_UP_A = "gate_up_a"
    GATE_UP_A_B = "gate_up_a_b"


class DownOverlap(str, Enum):
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


def _check_down_bridge_layout(site: Site, layout: BridgeLayout) -> None:
    if site is Site.DOWN and layout is BridgeLayout.TOKEN_MAJOR:
        raise ValueError(
            "the down bridge is inherently pair-major: each routed expert "
            "produces a different activation"
        )


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class LoraASpec:
    site: Site
    family: LoraAFamily
    is_shared_outer: bool = False
    output_layout: BridgeLayout = BridgeLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> LoraASpec:
        _check_down_bridge_layout(self.site, self.output_layout)
        if self.family is LoraAFamily.GROUPED:
            if self.output_layout is not BridgeLayout.PAIR_MAJOR:
                raise ValueError("grouped A writes a pair-major bridge")
        elif self.family is LoraAFamily.PER_PAIR:
            if self.output_layout is not BridgeLayout.PAIR_MAJOR:
                raise ValueError("per_pair A writes a pair-major bridge")
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
        if self.family is LoraAFamily.PER_PAIR:
            return frozenset((_raw_requirement(self.is_shared_outer),))
        if self.family is LoraAFamily.TOKEN_GROUPED:
            return frozenset((RouteRequirement.SHARED_TOKEN_PLAN,))
        return frozenset((_aligned_requirement(self.is_shared_outer),))


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class LoraBSpec:
    site: Site
    family: LoraBFamily
    is_shared_outer: bool = False
    input_layout: BridgeLayout = BridgeLayout.PAIR_MAJOR

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> LoraBSpec:
        _check_down_bridge_layout(self.site, self.input_layout)
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is LoraBFamily.PER_PAIR:
            return frozenset((_raw_requirement(self.is_shared_outer),))
        return frozenset((_aligned_requirement(self.is_shared_outer),))


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class ActSpec:
    """B_ACTIVATION owns gate/up B and consumes the A bridge."""

    family: ActFamily
    activation: ActivationFn

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is not ActFamily.B_ACTIVATION:
            return frozenset()
        return frozenset((_aligned_requirement(False),))


@pydantic_dataclass(frozen=True, slots=True, config=_STRICT)
class FinalizeSpec:
    family: FinalizeFamily
    is_shared_outer: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> FinalizeSpec:
        if (
            self.family is FinalizeFamily.SHARED_RANK_REDUCE
            and not self.is_shared_outer
        ):
            raise ValueError(
                f"{self.family.value} requires shared-outer down-B ownership"
            )
        return self

    def route_requirements(self) -> frozenset[RouteRequirement]:
        if self.family is FinalizeFamily.MATERIALIZED:
            return frozenset()
        # The shared-rank finalizer builds its fixed top-k keys from the raw
        # route.
        return frozenset((_raw_requirement(self.is_shared_outer),))


@pydantic_dataclass(frozen=True, slots=True, kw_only=True, config=_STRICT)
class MoeLoraExecutionPlan:
    gate_up_a: LoraASpec
    gate_up_b: LoraBSpec | None = None
    act: ActSpec
    down_a: LoraASpec
    down_b: LoraBSpec | None = None
    finalize: FinalizeSpec
    gate_up_overlap: GateUpOverlap = GateUpOverlap.NONE
    down_overlap: DownOverlap = DownOverlap.NONE
    down_b_into_base: bool = False
    route_builder: RouteBuilderFamily = RouteBuilderFamily.STANDARD

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MoeLoraExecutionPlan:
        if self.gate_up_a.site is not Site.GATE_UP:
            raise ValueError("gate_up_a must describe the gate/up site")
        if self.gate_up_b is not None and self.gate_up_b.site is not Site.GATE_UP:
            raise ValueError("gate_up_b must describe the gate/up site")
        if self.down_a.site is not Site.DOWN:
            raise ValueError("down_a must describe the down site")
        if self.down_b is not None and self.down_b.site is not Site.DOWN:
            raise ValueError("down_b must describe the down site")

        gate_up_b_consumed = self.act.family is ActFamily.B_ACTIVATION
        if gate_up_b_consumed == (self.gate_up_b is not None):
            raise ValueError(
                "gate/up B must have exactly one owner: standalone gate_up_b or the act stage"
            )
        down_b_consumed = self.finalize.family is not FinalizeFamily.MATERIALIZED
        if down_b_consumed == (self.down_b is not None):
            raise ValueError(
                "down B must have exactly one owner: standalone down_b or finalize"
            )

        # Fused consumers inherit A's layout; standalone B must match it.
        if (
            self.gate_up_b is not None
            and self.gate_up_a.output_layout is not self.gate_up_b.input_layout
        ):
            raise ValueError(
                "gate/up A output layout must match the gate/up B input layout"
            )
        if (
            self.down_b is not None
            and self.down_a.output_layout is not self.down_b.input_layout
        ):
            raise ValueError("down A output layout must match the down B input layout")

        if self.gate_up_overlap is GateUpOverlap.GATE_UP_A_B and self.gate_up_b is None:
            raise ValueError(
                "gate/up-A+B overlap requires standalone gate/up B; the act stage owns it"
            )
        if (
            self.down_overlap
            in (
                DownOverlap.DOWN_B,
                DownOverlap.DOWN_A_B,
            )
            and self.down_b is None
        ):
            raise ValueError(
                f"{self.down_overlap.value} overlap requires standalone down B"
            )

        if self.down_b_into_base and not self.down_b_into_base_eligible():
            raise ValueError(
                "down-B into-base requires a standalone down-B stage and no "
                "late overlap window (it read-modify-writes the base down "
                "output)"
            )

        return self

    def is_fully_serial(self) -> bool:
        return (
            self.gate_up_overlap is GateUpOverlap.NONE
            and self.down_overlap is DownOverlap.NONE
            and self.down_a.family is LoraAFamily.GROUPED
            # An into-base plan also runs in order, but its down B writes
            # into the base down output. The row-domain conversions read this
            # answer, and they must not treat an into-base plan as serial.
            and not self.down_b_into_base
        )

    def is_fully_serial_materialized(self) -> bool:
        return (
            self.is_fully_serial()
            and self.finalize.family is FinalizeFamily.MATERIALIZED
        )

    def down_b_into_base_eligible(self) -> bool:
        # In-place B must wait for the base down GEMM to finish writing.
        return self.down_b is not None and self.down_overlap in (
            DownOverlap.NONE,
            DownOverlap.DOWN_A,
        )

    def route_requirements(self) -> frozenset[RouteRequirement]:
        return (
            self._requirements_of(self.gate_up_a, self.gate_up_b, self.down_a)
            | self._down_b_route_requirements()
        )

    def _down_b_route_requirements(self) -> frozenset[RouteRequirement]:
        # The in-place B kernel replaces the selected family and needs aligned rows.
        if self.down_b is None:
            return frozenset()
        if self.down_b_into_base:
            return frozenset((_aligned_requirement(self.down_b.is_shared_outer),))
        return self.down_b.route_requirements()

    def _requirements_of(
        self, *stages: LoraASpec | LoraBSpec | None
    ) -> frozenset[RouteRequirement]:
        requirements: set[RouteRequirement] = set()
        for stage in stages:
            if stage is not None:
                requirements.update(stage.route_requirements())
        requirements.update(self.act.route_requirements())
        requirements.update(self.finalize.route_requirements())
        return frozenset(requirements)


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
    model_config = pydantic.ConfigDict(extra="forbid")

    gate_up_a_family: LoraAFamily = LoraAFamily.GROUPED
    down_a_family: LoraAFamily = LoraAFamily.GROUPED
    gate_up_b_family: LoraBFamily = LoraBFamily.GROUPED
    down_b_family: LoraBFamily = LoraBFamily.GROUPED
    act_family: ActFamily = ActFamily.MATERIALIZED
    finalize_family: FinalizeFamily = FinalizeFamily.MATERIALIZED
    gate_up_overlap: GateUpOverlap = GateUpOverlap.NONE
    down_overlap: DownOverlap = DownOverlap.NONE
    route_builder: RouteBuilderFamily = RouteBuilderFamily.STANDARD
    down_b_into_base: bool = False


class _PlanRowModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    layout: Literal["per_expert", "shared"] | None = None
    phase: Phase | None = None
    max_rank: int | None = None
    base_gemm_rows: Literal["expert_major", "route_major"]
    plan: _PlanSpecModel


class _DomainModel(pydantic.BaseModel):
    """Geometry covered by the tuned scenario rows."""

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
    gate_up_a_family = spec.gate_up_a_family
    gate_up_layout = (
        BridgeLayout.TOKEN_MAJOR
        if gate_up_a_family is LoraAFamily.TOKEN_GROUPED
        else BridgeLayout.PAIR_MAJOR
    )
    act_family = spec.act_family
    finalize_family = spec.finalize_family
    consumes_gate_up_b = act_family is ActFamily.B_ACTIVATION
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
        act=ActSpec(act_family, activation),
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
            finalize_family, is_shared_outer if consumes_down_b else False
        ),
        gate_up_overlap=spec.gate_up_overlap,
        down_overlap=spec.down_overlap,
        route_builder=spec.route_builder,
        down_b_into_base=spec.down_b_into_base,
    )
    return plan


@cache
def _warn_out_of_domain(
    architecture: DeviceArchitecture, hidden_size: int, num_local_experts: int
) -> None:
    logger.warning(
        "MoE LoRA geometry (hidden=%d, local_experts=%d) is outside the tuned "
        "domain of table %r; serving the fallback rows",
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
        # Earlier rows take precedence, including wildcard rows.
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
            key=f"{architecture.value}.{row.name}",
            name=row.name,
            base_gemm_rows=row.base_gemm_rows,
            plan=build_plan(
                row.plan,
                activation=activation,
                is_shared_outer=is_shared_outer,
            ),
        )
    return selected
