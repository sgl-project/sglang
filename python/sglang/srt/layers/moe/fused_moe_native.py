"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""

import torch
from torch.nn import functional as F

from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


def fused_moe_forward_native(
    layer: torch.nn.Module,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:

    x, x_scale, topk_output = dispatch_output
    moe_runner_config = layer.moe_runner_config

    if moe_runner_config.apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    w13_weights = layer.w13_weight[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = layer.w2_weight[topk_ids]
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
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
) -> torch.Tensor:

    if moe_runner_config.apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    # Ref code from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589
    len_experts = layer.num_experts

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    if moe_runner_config.activation == "silu":
        act = SiluAndMul()
    elif moe_runner_config.activation == "gelu":
        act = GeluAndMul()
    else:
        raise ValueError(f"Unsupported activation: {moe_runner_config.activation=}")

    # Get bias terms if available
    w13_bias = getattr(layer, "w13_weight_bias", None)
    w2_bias = getattr(layer, "w2_weight_bias", None)
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        layer_w13_weight = layer.w13_weight[i]
        layer_w2_weight = layer.w2_weight[i]

        # Store original dtype
        original_dtype = tokens_for_this_expert.dtype

        # Get bias terms if available for this expert
        layer_w13_bias = w13_bias[i] if w13_bias is not None else None
        layer_w2_bias = w2_bias[i] if w2_bias is not None else None

        # Apply w13 linear with bias for models like GPT-OSS
        if layer_w13_bias is not None:
            # BF16 matmul with FP32 accumulation
            gate_up = F.linear(tokens_for_this_expert, layer_w13_weight, bias=None)
            gate_up_fp32 = gate_up.float() + layer_w13_bias
            gate_up = gate_up_fp32.to(original_dtype)

            # Split the interleaved gate and up projections as per GPT-OSS weight layout:
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]

            # Apply clamping and modified SiLU activation
            if (
                moe_runner_config.activation == "silu"
                and moe_runner_config.gemm1_alpha is not None
            ):
                assert moe_runner_config.gemm1_clamp_limit is not None
                gate = gate.clamp(min=None, max=moe_runner_config.gemm1_clamp_limit)
                up = up.clamp(
                    min=-moe_runner_config.gemm1_clamp_limit,
                    max=moe_runner_config.gemm1_clamp_limit,
                )
                glu = gate * torch.sigmoid(gate * moe_runner_config.gemm1_alpha)
                gate_up = glu * (up + 1)
            else:
                gate_up = act(gate_up)
        else:
            # Direct path for models without bias (e.g., Mixtral, DeepSeek)
            gate_up = F.linear(tokens_for_this_expert, layer_w13_weight)
            gate_up = act(gate_up)

        # Apply w2 linear
        if layer_w2_bias is not None:
            expert_out = F.linear(gate_up, layer_w2_weight, bias=None)
            expert_out = expert_out.float() + layer_w2_bias
            expert_out = expert_out.to(original_dtype)
        else:
            # Direct path for models without bias (e.g., Mixtral, DeepSeek)
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
