"""Draft FFN modules built from DraftMixFFNSpec: DraftMLP and DraftMoE."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch
from torch import nn

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.draft.ffn_spec import DraftMoeFFNSpec
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe import reduce_moe_output
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

SWIGLUOAI_ALPHA = 1.702  # gpt-oss's fixed sigmoid slope


def _silu_and_mul_clamp(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """DeepSeek-V4's SwiGLU: gate capped at limit, up clamped to [-limit, limit]."""
    from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp

    out = gate_up.new_empty(gate_up.shape[:-1] + (gate_up.shape[-1] // 2,))
    silu_and_mul_clamp(gate_up, out, limit)
    return out


def gated_activation(
    hidden_act: str,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The act(gate) * up of a draft FFN, from the spec's activation fields."""
    if hidden_act == "swigluoai":
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            swiglu_no_interleaved_with_alpha_and_limit,
        )

        if swiglu_limit is None:
            raise ValueError("Draft FFN: hidden_act='swigluoai' needs swiglu_limit.")
        return partial(
            swiglu_no_interleaved_with_alpha_and_limit,
            gemm1_alpha=float(swiglu_alpha or SWIGLUOAI_ALPHA),
            gemm1_limit=float(swiglu_limit),
        )
    if hidden_act != "silu":
        raise ValueError(
            f"Draft FFN: hidden_act={hidden_act!r} is not implemented; "
            "available are ('silu', 'swigluoai')."
        )
    if swiglu_limit is None:
        return SiluAndMul()
    return partial(_silu_and_mul_clamp, limit=float(swiglu_limit))


class DraftMLP(nn.Module):
    """Gated dense FFN of a draft layer; also the shared expert of an MoE layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        *,
        swiglu_limit: Optional[float] = None,
        swiglu_alpha: Optional[float] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = gated_activation(hidden_act, swiglu_limit, swiglu_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out


class DraftMoE(nn.Module):
    """Routed experts plus an optional shared expert, built from a DraftMoeFFNSpec."""

    def __init__(
        self,
        spec: DraftMoeFFNSpec,
        hidden_size: int,
        *,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
        from sglang.srt.layers.moe.topk import TopK

        _check_parallel_fit(spec)
        self.spec = spec
        self.hidden_size = hidden_size
        self.gate = ReplicatedLinear(
            hidden_size,
            spec.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        # The trained fp32 selection bias lives on the gate, where the checkpoint keeps it.
        self.gate.expert_bias = (
            nn.Parameter(
                torch.empty(spec.num_experts, dtype=torch.float32), requires_grad=False
            )
            if spec.expert_bias
            else None
        )
        grouped = spec.num_expert_group is not None
        self.topk = TopK(
            spec.top_k,
            layer_id=layer_id,
            renormalize=spec.norm_topk_prob,
            use_grouped_topk=grouped,
            num_expert_group=spec.num_expert_group,
            topk_group=spec.topk_group,
            scoring_func=spec.scoring_func,
            correction_bias=self.gate.expert_bias,
            quant_config=quant_config,
            routed_scaling_factor=(
                None
                if spec.routed_scaling_factor == 1.0
                else spec.routed_scaling_factor
            ),
        )
        # The fused softmax router ignores the bias; the torch path adds it to the scores.
        if spec.expert_bias and spec.scoring_func == "softmax" and not grouped:
            self.topk.topk_config.torch_native = True
        swigluoai = spec.hidden_act == "swigluoai"
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            hidden_size=hidden_size,
            intermediate_size=spec.intermediate_size,
            layer_id=layer_id,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation=spec.hidden_act,
            gemm1_alpha=(spec.swiglu_alpha or SWIGLUOAI_ALPHA) if swigluoai else None,
            gemm1_clamp_limit=spec.swiglu_limit if swigluoai else None,
            swiglu_limit=None if swigluoai else spec.swiglu_limit,
            routing_method_type=_routing_method(spec),
        )
        self.topk.bind_runner_backend(
            self.experts.runner_backend,
            routing_method_type=self.experts.routing_method_type,
        )
        self.shared_expert = (
            DraftMLP(
                hidden_size,
                spec.shared_expert_intermediate_size,
                spec.hidden_act,
                swiglu_limit=spec.swiglu_limit,
                swiglu_alpha=spec.swiglu_alpha,
                reduce_results=False,
                quant_config=quant_config,
                prefix=add_prefix("shared_expert", prefix),
            )
            if spec.shared_expert_intermediate_size
            else None
        )
        self.shared_expert_gate = (
            nn.Linear(hidden_size, 1, bias=False) if spec.shared_expert_gated else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(-1, self.hidden_size)
        if x.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(x.device)
        else:
            router_logits, _ = self.gate(x)
            topk_output = self.topk(x, router_logits.float())  # fp32 scores, as trained
        out = self.experts(x, topk_output)
        if self.shared_expert is not None:
            shared = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
            out = out + shared
        return reduce_moe_output(out).view_as(hidden_states)


def _routing_method(spec: DraftMoeFFNSpec):
    """The RoutingMethodType the fused routers key on; TopK itself reads the spec fields."""
    from sglang.srt.layers.moe.utils import RoutingMethodType

    if spec.scoring_func != "softmax" or spec.num_expert_group is not None:
        return RoutingMethodType.DeepSeekV3
    return (
        RoutingMethodType.Renormalize
        if spec.norm_topk_prob
        else RoutingMethodType.Default
    )


def _check_parallel_fit(spec: DraftMoeFFNSpec) -> None:
    """FusedMoE shards the expert width over MoE TP and the experts over MoE EP."""
    parallel = get_parallel()
    if spec.intermediate_size % parallel.moe_tp_size:
        raise ValueError(
            f"Draft FFN: moe_intermediate_size={spec.intermediate_size} is not "
            f"divisible by the MoE tensor-parallel size {parallel.moe_tp_size}."
        )
    if spec.num_experts % parallel.moe_ep_size:
        raise ValueError(
            f"Draft FFN: num_experts={spec.num_experts} is not divisible by the "
            f"MoE expert-parallel size {parallel.moe_ep_size}."
        )
