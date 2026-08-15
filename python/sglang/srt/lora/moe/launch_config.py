"""Explicit launch parameters for a forced MoE-LoRA execution plan.

This is configuration transport, not a selector.  Defaults are correctness
baselines shared by serving and the composed benchmark; a benchmark may
replace any site independently with a promoted Step-3/4/5/6 configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from sglang.srt.lora.moe.base_gemm_provider.masked_finalize import (
    SHARED_RANK_DEFAULT_CONFIG,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_fused_middle import (
    FUSED_B_ACT_DEFAULT_CONFIG,
)
from sglang.srt.lora.moe.execution_plan import (
    FactorSite,
    LoraAFamily,
    MiddleFamily,
    MoeLoraExecutionPlan,
    RouteRequirement,
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
    if not isinstance(config, Mapping) or not config:
        raise ValueError(f"{name} must be a non-empty launch-config mapping")
    for key, value in config.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise TypeError(f"{name} launch parameters must be string -> int")


@dataclass(frozen=True, slots=True, kw_only=True)
class MoeLoraLaunchConfig:
    """Site-specific launch settings; no token/rank/device config."""

    # The canonical aligned route is shared by gate-B, down-A, and down-B.
    # Gate-A may request a second aligned plan with a different M tile.  Its
    # output is written by original pair ID, so the canonical B route can
    # consume that bridge without a layout conversion.  Keeping the two
    # values explicit also charges the extra route build in composed timing.
    routing_block_size: int = 16
    gate_a_routing_block_size: int = 16
    gate_a: Mapping[str, int] = field(default_factory=_a_default)
    gate_b: Mapping[str, int] = field(default_factory=_b_default)
    down_a: Mapping[str, int] = field(default_factory=_a_default)
    down_b: Mapping[str, int] = field(default_factory=_b_default)
    b_activation: Mapping[str, int] = field(
        default_factory=lambda: dict(FUSED_B_ACT_DEFAULT_CONFIG)
    )
    shared_finalize: Mapping[str, Mapping[str, int]] = field(
        default_factory=lambda: _copy_nested(SHARED_RANK_DEFAULT_CONFIG)
    )

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.routing_block_size):
            raise ValueError("routing_block_size must be a positive power of two")
        if not _is_power_of_two(self.gate_a_routing_block_size):
            raise ValueError(
                "gate_a_routing_block_size must be a positive power of two"
            )
        for name in (
            "gate_a",
            "gate_b",
            "down_a",
            "down_b",
            "b_activation",
        ):
            _validate_flat_config(name, getattr(self, name))
        if set(self.shared_finalize) != {"reduce", "tail"}:
            raise ValueError("shared_finalize must contain exactly 'reduce' and 'tail'")
        for name, config in self.shared_finalize.items():
            _validate_flat_config(f"shared_finalize.{name}", config)

    def for_a(self, site: FactorSite) -> Mapping[str, int]:
        return self.gate_a if site is FactorSite.GATE_UP else self.down_a

    def for_b(self, site: FactorSite) -> Mapping[str, int]:
        return self.gate_b if site is FactorSite.GATE_UP else self.down_b

    def for_middle(self, family: MiddleFamily) -> Mapping[str, int]:
        """Return the explicit config for one fused-middle kernel family."""
        if family is MiddleFamily.B_ACTIVATION:
            return self.b_activation
        raise ValueError(f"{family.value} has no fused-middle launch config")

    def validate_for_plan(self, plan: MoeLoraExecutionPlan) -> None:
        """Validate route tiles against the selected tensor-core consumers."""
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
        separate_gate_route = self.gate_a_routing_block_size != self.routing_block_size
        if separate_gate_route:
            gate_uses_separate_aligned = (
                plan.gate_a.family is LoraAFamily.GROUPED
                and not plan.gate_a.is_shared_outer
            )
            if not gate_uses_separate_aligned:
                raise ValueError(
                    "a distinct gate_a_routing_block_size is valid only for "
                    "grouped per-expert gate/up-A"
                )
            if self.gate_a_routing_block_size < 16:
                raise ValueError(
                    "gate_a_routing_block_size must be at least 16 for grouped "
                    "gate/up-A"
                )

    @property
    def lora_a(self) -> Mapping[str, int]:
        """Step-1–7 compatibility view for the common gate-A config."""
        return self.gate_a

    @property
    def lora_b(self) -> Mapping[str, int]:
        """Step-1–7 compatibility view for the common gate-B config."""
        return self.gate_b

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe complete identity used by composed benchmark records."""
        return {
            "routing_block_size": self.routing_block_size,
            "gate_a_routing_block_size": self.gate_a_routing_block_size,
            "gate_a": dict(self.gate_a),
            "gate_b": dict(self.gate_b),
            "down_a": dict(self.down_a),
            "down_b": dict(self.down_b),
            "b_activation": dict(self.b_activation),
            "shared_finalize": _copy_nested(self.shared_finalize),
        }


PROVISIONAL_LAUNCH_CONFIG = MoeLoraLaunchConfig()
