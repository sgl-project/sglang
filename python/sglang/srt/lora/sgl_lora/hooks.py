"""Typed provider boundary for the SGL LoRA MoE execution engine.

The builder runs while the canonical dispatch tensors are still valid.  A
provider may dispose the canonical hidden-state storage during pre-permute, so
callbacks must retain derived state (for example a LoRA-A result), not the
dispatch hidden-state tensor itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend


PAIR_CONTRIBUTION_STATE_KEY = "sgl_lora_pair_contribution"
OUTPUT_DTYPE_STATE_KEY = "sgl_lora_output_dtype"


@dataclass(frozen=True)
class MoeLoRAHookBuildContext:
    """Canonical inputs available only during provider hook construction.

    The builder may launch gate/up LoRA-A here and retain its result.  It must
    not retain ``dispatch_output.hidden_states`` for a later callback.
    """

    dispatch_output: DispatchOutput
    lora_info: Any
    runner_backend: MoeRunnerBackend


@dataclass(frozen=True)
class GateUpHookContext:
    """DeepGEMM gate/up output in expert-masked provider-row order."""

    provider_gate_up: torch.Tensor
    pair_to_provider_row: torch.Tensor
    provider_layout: Literal["gate_then_up"]
    provider_row_domain: Literal["expert_masked"] = "expert_masked"


@dataclass(frozen=True)
class DownHookContext:
    """Post-activation DeepGEMM rows and their canonical pair mapping."""

    provider_activation: torch.Tensor
    pair_to_provider_row: torch.Tensor
    provider_row_domain: Literal["expert_masked"] = "expert_masked"


@dataclass(frozen=True)
class PairDomainLoRAContribution:
    """Unweighted LoRA down delta materialized in canonical pair order.

    The provider combine owns router weighting, routed scaling, FP32
    accumulation, and the final output allocation/cast.
    """

    values: torch.Tensor

    def validate_for(
        self,
        *,
        expected_shape: tuple[int, int, int],
        expected_device: torch.device,
    ) -> None:
        if tuple(self.values.shape) != expected_shape:
            raise ValueError(
                "LoRA pair contribution must have shape "
                f"{expected_shape}, got {tuple(self.values.shape)}"
            )
        if self.values.device != expected_device:
            raise ValueError("LoRA pair contribution must be on the provider device")
        if self.values.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("LoRA pair contribution must be BF16 or FP32")
        if not self.values.is_contiguous():
            raise ValueError("LoRA pair contribution must be contiguous")


@dataclass
class SglMoeLoRAHooks:
    """Typed callbacks implemented by the SGL LoRA factor/kernels layer."""

    inject_gate_up: Callable[[GateUpHookContext], None] | None = None
    build_down_pair_contribution: (
        Callable[[DownHookContext], PairDomainLoRAContribution | None] | None
    ) = None


class MoeLoRAHookBuilder(Protocol):
    def __call__(self, context: MoeLoRAHookBuildContext) -> SglMoeLoRAHooks | None: ...


__all__ = [
    "DownHookContext",
    "GateUpHookContext",
    "MoeLoRAHookBuildContext",
    "MoeLoRAHookBuilder",
    "OUTPUT_DTYPE_STATE_KEY",
    "PAIR_CONTRIBUTION_STATE_KEY",
    "PairDomainLoRAContribution",
    "SglMoeLoRAHooks",
]
