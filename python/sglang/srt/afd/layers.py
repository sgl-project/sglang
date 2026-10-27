"""AFD model-side layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from sglang.srt.afd.afd_config import AFDRole
from sglang.srt.afd.dispatcher import get_afd_dispatcher
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.base_config import QuantizationConfig


@dataclass(frozen=True)
class AFDMoELayer:
    layer_id: int
    experts: nn.Module
    hidden_size: int
    intermediate_size: int
    top_k: int
    num_experts: int
    num_fused_shared_experts: int
    routed_scaling_factor: Optional[float]


class AFDAttnMoE(nn.Module):
    """Attention-side routing endpoint with no expert parameters."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_local_experts = 0
        # Send unscaled routing weights; F returns a scale-final routed output.
        self.should_fuse_routed_scaling_factor_in_topk = False
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=0,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            activation=activation,
            inplace=False,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.quant_method = None
        self.runner_backend = None
        self.routing_method_type = None
        self.supports_deferred_finalize = False

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise RuntimeError(
                "AFD attention instance requires STANDARD topk output with explicit "
                f"topk_ids/topk_weights; got {type(topk_output).__name__}."
            )
        # Autotune forwards precede the handshake, so there is no remote peer yet.
        if get_in_autotune_dummy_run():
            return hidden_states
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)

        dispatcher = get_afd_dispatcher()
        handle = dispatcher.dispatch(
            self.layer_id,
            hidden_states,
            topk_output.topk_ids,
            topk_output.topk_weights,
        )
        return dispatcher.combine(handle)


def collect_afd_moe_layers(model: nn.Module, role: AFDRole) -> list[AFDMoELayer]:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

    experts_type = AFDAttnMoE if role.is_attn() else FusedMoE
    experts_by_layer = sorted(
        (module for module in model.modules() if isinstance(module, experts_type)),
        key=lambda experts: experts.layer_id,
    )
    moe_layers = [
        AFDMoELayer(
            layer_id=experts.layer_id,
            experts=experts,
            hidden_size=experts.hidden_size,
            intermediate_size=(
                experts.moe_runner_config.intermediate_size_per_partition
            ),
            top_k=experts.top_k,
            num_experts=experts.num_experts,
            num_fused_shared_experts=experts.num_fused_shared_experts,
            routed_scaling_factor=experts.moe_runner_config.routed_scaling_factor,
        )
        for experts in experts_by_layer
    ]
    if not moe_layers:
        raise RuntimeError("AFD found no MoE expert layers")
    return moe_layers
