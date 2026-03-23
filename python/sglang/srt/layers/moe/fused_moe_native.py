"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""

from typing import Callable

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

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        layer_w13_weight = layer.w13_weight[i]
        layer_w2_weight = layer.w2_weight[i]

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

def fused_moe_forward_native_grouped_mm(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    activation_fn: Callable,
    activation_fn_args: Callable,
    weights_pre_transposed: bool = False,
) -> torch.Tensor:
    """
    Native grouped_mm MoE forward pass with configurable activation.

    Args:
        weights_pre_transposed: If True, w13_weight is already [E, H, 2*I] and
            w2_weight is already [E, I, H] (e.g. triton_kernel layout), so the
            transpose is skipped.

    Returns:
        torch.Tensor: (num_tokens, hidden_size)
    """
    topk_weights, topk_ids, _ = topk_output
    alpha = moe_runner_config.gemm1_alpha
    limit = moe_runner_config.gemm1_clamp_limit
    device = hidden_states.device
    num_tokens, hidden_size = hidden_states.shape
    num_top_k = topk_ids.size(-1)
    num_experts = layer.num_experts

    # Flatten topk_ids to get expert_ids per selected sample
    expert_ids = topk_ids.reshape(-1)  # (num_tokens * top_k,)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )

    # Get routing weights per selected sample
    sample_weights = topk_weights.reshape(-1)  # (S,)

    # TODO: handle apply_router_weight_on_input. When
    # moe_runner_config.apply_router_weight_on_input is True, the routing weight should
    # be multiplied into hidden_states BEFORE GEMM1 (and NOT after GEMM2). Currently we
    # always apply it after GEMM2. Because a nonlinear activation sits between the two
    # GEMMs, pre- vs post-multiply is not mathematically equivalent:
    #   σ(w·x @ W1) @ W2  ≠  σ(x @ W1) @ W2 · w

    # Get current hidden states for selected samples
    current_hidden_states = hidden_states[token_idx]  # (S, hidden_size)

    # Get permutation to group by expert
    perm = torch.argsort(expert_ids)
    # O(n) inverse permutation via scatter instead of a second argsort
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device, dtype=perm.dtype)

    # Group by expert for grouped_mm
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    current_states_g = current_hidden_states[perm]

    # Compute offsets for grouped_mm
    # expert_ids_g is sorted, so searchsorted gives cumulative counts directly,
    # replacing the histc + cumsum chain with a single op.
    boundaries = torch.arange(
        1, num_experts + 1, device=device, dtype=expert_ids_g.dtype
    )
    offsets = torch.searchsorted(expert_ids_g, boundaries).to(torch.int32)

    # --- Up projection per expert (grouped_mm) ---
    # grouped_mm computes input @ weight, so weight must be (E, H, 2*I).
    # Normal layout stores [E, 2*I, H] and needs a transpose;
    # triton_kernel layout already stores [E, H, 2*I].
    w13 = layer.w13_weight if weights_pre_transposed else layer.w13_weight.transpose(-1, -2)
    gate_up_out = torch._grouped_mm(current_states_g, w13, offsets)

    # Add bias if present
    w13_bias = getattr(layer, "w13_weight_bias", None)
    if w13_bias is not None:
        gate_up_out = gate_up_out + w13_bias[expert_ids_g]

    hidden_after_activation = activation_fn(*activation_fn_args(gate_up_out, alpha, limit))
    # Cast back to original dtype (swiglu_with_alpha_and_limit may return float32 due to torch.compile)
    hidden_after_activation = hidden_after_activation.to(hidden_states.dtype)

    # --- Down projection per expert (grouped_mm) ---
    # grouped_mm computes input @ weight, so weight must be (E, I, H).
    # Normal layout stores [E, H, I] and needs a transpose;
    # triton_kernel layout already stores [E, I, H].
    w2 = layer.w2_weight if weights_pre_transposed else layer.w2_weight.transpose(-1, -2)
    out_per_sample_g = torch._grouped_mm(hidden_after_activation, w2, offsets)

    # Add bias if present
    w2_bias = getattr(layer, "w2_weight_bias", None)
    if w2_bias is not None:
        out_per_sample_g = out_per_sample_g + w2_bias[expert_ids_g]

    # Apply routing weights
    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)

    # Restore original order
    out_per_sample = out_per_sample_g[inv_perm]

    # TODO: apply routed_scaling_factor. The Triton path multiplies the combined output
    # by moe_runner_config.routed_scaling_factor (defaults to 1.0 when None). Models like
    # DeepSeek-V3 set this to a non-trivial value (e.g. 2.5), so omitting it here will
    # produce incorrect results for those models.
    # Reference: fused_moe.py fused_experts_impl, moe_sum_reduce(..., routed_scaling_factor)

    return out_per_sample.view(num_tokens, num_top_k, hidden_size).sum(dim=1).to(hidden_states.dtype)
