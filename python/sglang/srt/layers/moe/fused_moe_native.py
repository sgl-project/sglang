"""
Torch-native implementation for FusedMoE. This is used for torch.compile.
It is based on https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/mixtral-moe/model.py#L204
"""

from typing import Callable, Optional

import torch
from torch.nn import functional as F

from sglang.srt.layers.activation import GeluAndMul, SiluAndMul
from sglang.srt.layers.moe.topk import select_experts


def fused_moe_forward_native(
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    inplace: bool = True,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:

    if apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        num_fused_shared_experts=num_fused_shared_experts,
        custom_routing_function=custom_routing_function,
        correction_bias=correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        torch_native=True,
    )

    w13_weights = layer.w13_weight[topk_ids]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = layer.w2_weight[topk_ids]
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
    if activation == "silu":
        x1 = F.silu(x1)
    elif activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))


def moe_forward_native(
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        num_fused_shared_experts=num_fused_shared_experts,
        custom_routing_function=custom_routing_function,
        correction_bias=correction_bias,
        torch_native=True,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Ref code from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589
    len_experts = layer.num_experts

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    if activation == "silu":
        act = SiluAndMul()
    elif activation == "gelu":
        act = GeluAndMul()
    else:
        raise ValueError(f"Unsupported activation: {activation=}")

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
