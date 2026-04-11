import torch
import torch.nn as nn
from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai
from sglang.srt.layers.activation import GeluAndMul

# ==========================================
# Helper Functions
# ==========================================

def _reshape_to_2d(tensor):
    """Reshape [B, S, H] -> [B*S, H]; returns (reshaped, original_shape)."""
    original_shape = tensor.shape
    if len(original_shape) == 3:
        tensor = tensor.view(-1, original_shape[-1])
    return tensor, original_shape

def _init_routing_v1(hidden_states, topk_ids, topk_weights, top_k):
    """
    Standard routing (v1) used by unquant, wna16, w8a8.
    Returns (routed_hidden, expanded_row_idx, expanded_expert_idx).
    """
    num_tokens = hidden_states.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    routed, expanded_row_idx, expanded_expert_idx = torch.ops.npu.npu_moe_init_routing(
        hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
    )
    return routed, expanded_row_idx, expanded_expert_idx

def _init_routing_v2(hidden_states, topk_ids, top_k, num_experts, quant_mode=1):
    """
    Alternative routing (v2) used by w4a8 and sometimes w4a4.
    Returns (sorted_hidden, expanded_row_idx, expert_tokens, pertoken_scale).
    """
    original_shape = hidden_states.shape
    num_tokens = original_shape[:-1].numel()
    first_expert_idx, last_expert_idx = 0, num_experts
    sorted_hidden, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=quant_mode,
        )
    )
    # Reshape expanded_row_idx for later use
    expanded_row_idx = expanded_row_idx.view(-1, top_k).permute(1, 0).reshape(-1)
    return sorted_hidden, expanded_row_idx, expert_tokens, pertoken_scale

def _compute_expert_tokens(expanded_expert_idx, num_experts):
    """Convert expanded expert indices to per-expert token counts."""
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    return expert_tokens.to(torch.int64)

def _finalize_routing_v1(hidden_states, topk_weights, expanded_row_idx, topk_ids,
                         drop_pad_mode=None):
    """
    Standard finalize routing (v1) with optional drop_pad_mode.
    """
    kwargs = {
        "skip1": None,
        "skip2": None,
        "bias": None,
        "scales": topk_weights,
        "expanded_src_to_dst_row": expanded_row_idx,
        "export_for_source_row": topk_ids,
    }
    if drop_pad_mode is not None:
        kwargs["drop_pad_mode"] = drop_pad_mode
    return torch.ops.npu.npu_moe_finalize_routing(hidden_states, **kwargs)

def _apply_activation(hidden_states, activation, w13=None):
    """Route to the correct activation function."""
    if activation == "npu_swiglu_oai":
        # Create a dummy module for swiglu_oai
        layer = nn.ModuleList()
        layer.register_parameter("w13_weight", w13)
        return swiglu_oai(layer, hidden_states)
    elif activation == "silu":
        return torch.ops.npu.npu_swiglu(hidden_states)
    else:
        return GeluAndMul()(hidden_states)


# ==========================================
# Main Expert Functions
# ==========================================

def npu_fused_experts_unquant(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    activation: str,
    input_shape=None,
):
    """
    Unquantized MoE forward pass.
    """
    # Reshape input if needed
    hidden_states, orig_shape = _reshape_to_2d(hidden_states)
    original_dtype = hidden_states.dtype
    topk_weights = topk_weights.to(original_dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w13.shape[0]

    routed, expanded_row_idx, expanded_expert_idx = _init_routing_v1(
        hidden_states, topk_ids, topk_weights, top_k
    )
    expert_tokens = _compute_expert_tokens(expanded_expert_idx, num_experts)

    w13_bias = [w13_weight_bias] if w13_weight_bias is not None else None
    w2_bias = [w2_weight_bias] if w2_weight_bias is not None else None

    # gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[routed],
        weight=[w13],
        bias=w13_bias,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # activation
    hidden_states = _apply_activation(hidden_states, activation, w13=w13)

    # down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        bias=w2_bias,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    out = _finalize_routing_v1(
        hidden_states, topk_weights, expanded_row_idx, topk_ids
    )
    return out.view(orig_shape) if len(orig_shape) == 3 else out


def npu_fused_experts_wna16(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    input_shape=None,
):
    """
    WNA16 (int4 weights, bf16/fp16 activations) MoE forward pass.
    """
    hidden_states, orig_shape = _reshape_to_2d(hidden_states)
    original_dtype = hidden_states.dtype
    topk_weights = topk_weights.to(original_dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w13.shape[0]

    routed, expanded_row_idx, expanded_expert_idx = _init_routing_v1(
        hidden_states, topk_ids, topk_weights, top_k
    )
    expert_tokens = _compute_expert_tokens(expanded_expert_idx, num_experts)

    # gate_up_proj with anti-quant
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[routed],
        weight=[w13],
        antiquant_scale=[w13_scale],
        antiquant_offset=[w13_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # activation (always SiLU)
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

    # down_proj with anti-quant
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    out = _finalize_routing_v1(
        hidden_states, topk_weights, expanded_row_idx, topk_ids
    )
    return out.view(orig_shape) if len(orig_shape) == 3 else out


def npu_fused_experts_w4a4(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    input_shape=None,
):
    """
    W4A4 (int4 weights, int4 activations) MoE forward pass.
    """
    hidden_states, orig_shape = _reshape_to_2d(hidden_states)
    original_dtype = hidden_states.dtype
    topk_weights = topk_weights.to(original_dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w13.shape[0]

    # Use v2 routing (quant_mode=-1)
    routed, expanded_row_idx, expert_tokens, pertoken_scale = _init_routing_v2(
        hidden_states, topk_ids, top_k, num_experts, quant_mode=-1
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # gate_up_proj with dynamic quant to quint4x2
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(
        routed, dst_type=torch.quint4x2
    )
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        scale=[w13_scale],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # activation
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

    # down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    out = _finalize_routing_v1(
        hidden_states, topk_weights, expanded_row_idx, topk_ids, drop_pad_mode=2
    )
    return out.view(orig_shape) if len(orig_shape) == 3 else out


def npu_fused_experts_w4a8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_bias: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    input_shape=None,
):
    """
    W4A8 (int4 weights, int8 activations) MoE forward pass.
    """
    hidden_states, orig_shape = _reshape_to_2d(hidden_states)
    original_dtype = hidden_states.dtype
    topk_weights = topk_weights.to(original_dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w13.shape[0]

    # v2 routing with quant_mode=1
    routed, expanded_row_idx, expert_tokens, pertoken_scale = _init_routing_v2(
        hidden_states, topk_ids, top_k, num_experts, quant_mode=1
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[routed],
        weight=[w13],
        scale=[w13_scale],
        bias=[w13_scale_bias],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=1,
        output_dtype=original_dtype,
    )[0]

    # activation + dynamic quant
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states, swiglu_out_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

    # down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        bias=[w2_scale_bias],
        per_token_scale=[swiglu_out_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=1,
        output_dtype=original_dtype,
    )[0]

    # final unpermute (instead of finalize_routing)
    out = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=hidden_states,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )
    return out.view(orig_shape) if len(orig_shape) == 3 else out


def npu_fused_experts_w8a8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    input_shape=None,
):
    """
    W8A8 (int8 weights, int8 activations) MoE forward pass.
    """
    hidden_states, orig_shape = _reshape_to_2d(hidden_states)
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    topk_weights = topk_weights.to(original_dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w13.shape[0]

    # Use v1 routing (standard)
    routed, expanded_row_idx, expanded_expert_idx = _init_routing_v1(
        hidden_states, topk_ids, topk_weights, top_k
    )
    expert_tokens = _compute_expert_tokens(expanded_expert_idx, num_experts)

    # gate_up_proj with dynamic quant
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(routed)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        scale=[w13_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # activation (using fused dequant+swiglu+quant)
    hidden_states, _ = torch.ops.npu.npu_dequant_swiglu_quant(
        hidden_states, quant_mode=1, activate_left=True
    )
    # Note: npu_dequant_swiglu_quant returns (quantized_output, scale)
    # The output is already quantized to int8; we just use it.

    # down_proj with dynamic quant (again)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    out = _finalize_routing_v1(
        hidden_states, topk_weights, expanded_row_idx, topk_ids
    )
    return out.view(orig_shape) if len(orig_shape) == 3 else out
