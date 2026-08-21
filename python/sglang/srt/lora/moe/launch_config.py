from __future__ import annotations

from dataclasses import field
from functools import cache
from typing import Any, Mapping

import pydantic
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sglang.srt.lora.moe.execution_plan import (
    ActFamily,
    DeviceArchitecture,
    MoeLoraExecutionPlan,
    RouteRequirement,
    Site,
)
from sglang.srt.lora.moe.kernels.masked_finalize import (
    SHARED_RANK_DEFAULT_CONFIG,
)
from sglang.srt.lora.moe.kernels.masked_fused_act import (
    FUSED_B_ACT_DEFAULT_CONFIG,
)


def _a_default() -> dict[str, int]:
    return {
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


def _b_default() -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


def _copy_nested(
    source: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {section: dict(values) for section, values in source.items()}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_flat_config(name: str, config: Mapping[str, int]) -> None:
    # The strict pydantic fields already check the key and value types.
    if not config:
        raise ValueError(f"{name} must be a non-empty launch-config mapping")


@pydantic_dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
    config=pydantic.ConfigDict(strict=True, extra="forbid"),
)
class MoeLoraLaunchConfig:
    # Every grouped LoRA kernel in a plan reads one aligned route. This value
    # is the row tile of those kernels, and of nothing else. See
    # configs/README.md for the shipped values and the measurements behind them.
    routing_block_size: int = 16
    gate_up_a: dict[str, int] = field(default_factory=_a_default)
    gate_up_b: dict[str, int] = field(default_factory=_b_default)
    down_a: dict[str, int] = field(default_factory=_a_default)
    down_b: dict[str, int] = field(default_factory=_b_default)
    b_activation: dict[str, int] = field(
        default_factory=lambda: dict(FUSED_B_ACT_DEFAULT_CONFIG)
    )
    shared_finalize: dict[str, dict[str, int]] = field(
        default_factory=lambda: _copy_nested(SHARED_RANK_DEFAULT_CONFIG)
    )

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.routing_block_size):
            raise ValueError("routing_block_size must be a positive power of two")
        for name in (
            "gate_up_a",
            "gate_up_b",
            "down_a",
            "down_b",
            "b_activation",
        ):
            _validate_flat_config(name, getattr(self, name))
        if set(self.shared_finalize) != {"reduce", "tail"}:
            raise ValueError("shared_finalize must contain exactly 'reduce' and 'tail'")
        for name, config in self.shared_finalize.items():
            _validate_flat_config(f"shared_finalize.{name}", config)

    def for_a(self, site: Site) -> Mapping[str, int]:
        return self.gate_up_a if site is Site.GATE_UP else self.down_a

    def for_b(self, site: Site) -> Mapping[str, int]:
        return self.gate_up_b if site is Site.GATE_UP else self.down_b

    def for_act(self, family: ActFamily) -> Mapping[str, int]:
        if family is ActFamily.B_ACTIVATION:
            return self.b_activation
        raise ValueError(f"{family.value} has no fused-act launch config")

    def validate_for_plan(self, plan: MoeLoraExecutionPlan) -> None:
        requirements = plan.route_requirements()
        aligned = {
            RouteRequirement.ALIGNED_PER_EXPERT,
            RouteRequirement.ALIGNED_SHARED_OUTER,
        }
        if requirements.intersection(aligned) and self.routing_block_size < 16:
            raise ValueError(
                "routing_block_size must be at least 16 for aligned "
                "tensor-core LoRA consumers"
            )


# Each plan row has a list of rules, and the runtime uses the first rule that
# matches. ``max_rank`` resolves once at bind time, because the pool-padded
# rank is a server constant. ``max_tokens`` resolves on every forward.


class _TileRuleModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    max_rank: int | None = None
    max_tokens: int | None = None
    sites: dict[str, Any] = pydantic.Field(default_factory=dict)


class _TilesFileModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    arch: DeviceArchitecture
    rules: dict[str, list[_TileRuleModel]]


@cache
def _load_tiles(architecture_value: str) -> _TilesFileModel | None:
    from sglang.srt.lora.moe.execution_plan import _read_table

    raw = _read_table(f"{architecture_value}.tiles.json")
    if raw is None:
        return None
    return _TilesFileModel.model_validate(raw)


def _config_from_sites(sites: Mapping[str, Any]) -> MoeLoraLaunchConfig:
    return MoeLoraLaunchConfig(**{name: value for name, value in sites.items()})


class TileTable:
    def __init__(self, rules: list[tuple[int, MoeLoraLaunchConfig]]) -> None:
        # ``resolve_tiles`` is the only caller, and it always passes at least
        # one rule. ``config_for`` can therefore always read the last rule.
        self._rules = rules

    def config_for(self, num_tokens: int) -> MoeLoraLaunchConfig:
        for bound, config in self._rules:
            if num_tokens <= bound:
                return config
        return self._rules[-1][1]

    def validate_for_plan(self, plan: MoeLoraExecutionPlan) -> None:
        for _, config in self._rules:
            config.validate_for_plan(plan)


def resolve_tiles(
    *,
    architecture_value: str,
    plan_key_name: str,
    physical_rank: int,
) -> TileTable:
    table = _load_tiles(architecture_value)
    rules = table.rules.get(plan_key_name) if table is not None else None
    if not rules:
        return TileTable([(1 << 30, MoeLoraLaunchConfig())])
    resolved: list[tuple[int, MoeLoraLaunchConfig]] = []
    for rule in rules:
        if rule.max_rank is not None and physical_rank > rule.max_rank:
            continue
        bound = rule.max_tokens if rule.max_tokens is not None else 1 << 30
        resolved.append((bound, _config_from_sites(rule.sites)))
        if rule.max_tokens is None:
            break  # a rule with no token bound is the last rule in the ladder
    if not resolved:
        resolved.append((1 << 30, MoeLoraLaunchConfig()))
    return TileTable(resolved)
