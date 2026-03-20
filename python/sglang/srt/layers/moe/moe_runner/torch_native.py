from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

from torch.nn import functional as F


@dataclass
class TorchNativeRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_tokens: torch.Tensor
    tokens_per_expert: torch.Tensor
    idxs: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NATIVE


@dataclass
class TorchNativeRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NATIVE


@dataclass
class TorchNativeMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


class TorchNativeRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        from sglang.srt.layers.activation import GeluAndMul, SiluAndMul

        if config.activation == "silu":
            self.act = SiluAndMul()
        elif config.activation == "gelu":
            self.act = GeluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {config.activation=}")

    def run(
        self,
        runner_input: TorchNativeRunnerInput,
        quant_info: TorchNativeMoeQuantInfo,
        running_state: dict,
    ) -> TorchNativeRunnerOutput:

        out_hidden_states = moe_forward_native(
            x=runner_input.hidden_states,
            sorted_tokens=runner_input.sorted_tokens,
            tokens_per_expert=runner_input.tokens_per_expert,
            topk_weights=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            act=self.act,
            w13_weight=quant_info.w13_weight,
            w2_weight=quant_info.w2_weight,
            idxs=runner_input.idxs,
        )

        return TorchNativeRunnerOutput(
            hidden_states=out_hidden_states,
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NATIVE


@register_fused_func("none", "torch_native")
def fused_experts_none_to_torch_native(
    dispatch_output: StandardDispatchOutput,
    quant_info: TorchNativeMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:

    output = fused_moe_forward_native(
        w13_weight=quant_info.w13_weight,
        w2_weight=quant_info.w2_weight,
        moe_runner_config=runner_config,
        dispatch_output=dispatch_output,
    )

    return output


@register_pre_permute("standard", "torch_native")
def pre_permute_standard_to_torch_native(
    dispatch_output: StandardDispatchOutput,
    quant_info: TorchNativeMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TorchNativeRunnerInput:

    if runner_config.apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = dispatch_output.topk_output

    # Ref code from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589
    len_experts = runner_config.num_experts

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = dispatch_output.hidden_states[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    return TorchNativeRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_tokens=sorted_tokens,
        tokens_per_expert=tokens_per_expert,
        idxs=idxs,
    )


@register_post_permute("torch_native", "standard")
def post_permute_torch_native_to_standard(
    runner_output: TorchNativeRunnerOutput,
    quant_info: TorchNativeMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )


"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""


def fused_moe_forward_native(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    moe_runner_config: MoeRunnerConfig,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    x, x_scale, topk_output = dispatch_output

    if moe_runner_config.apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    w13_weights = w13_weight[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = w2_weight[topk_ids]
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
    if moe_runner_config.activation == "silu":
        x1 = F.silu(x1)
    elif moe_runner_config.activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {moe_runner_config.activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    expert_outs = torch.einsum(
        "tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype)
    )
    return StandardCombineInput(hidden_states=expert_outs)


def moe_forward_native(
    x: torch.Tensor,
    sorted_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    idxs: torch.Tensor,
) -> torch.Tensor:

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        layer_w13_weight = w13_weight[i]
        layer_w2_weight = w2_weight[i]

        gate_up = F.linear(tokens_for_this_expert, layer_w13_weight)
        gate_up = act(gate_up)
        expert_out = F.linear(gate_up, layer_w2_weight)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weights.dtype)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out
