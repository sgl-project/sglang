from __future__ import annotations

"""
FLOP and memory utility functions for Marconi FLOP-efficiency-weighted cache eviction.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.configs.mamba_utils import BaseLinearStateParams
    from sglang.srt.configs.model_config import ModelConfig


def get_full_attn_flops(
    seqlen: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> float:
    """FLOPs for one full-attention (GQA) layer on seqlen tokens."""
    proj_flops = 4 * seqlen * hidden_size * (hidden_size + num_kv_heads * head_dim)
    attn_flops = 4 * seqlen**2 * num_heads * head_dim
    return proj_flops + attn_flops


def get_linear_attn_flops(
    seqlen: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    state_size: int,
) -> float:
    """FLOPs for one Mamba2 / linear-attention layer on seqlen tokens."""
    intermediate_size = num_heads * head_dim
    ssm_scan_flops = 2 * seqlen * intermediate_size * state_size
    proj_flops = 8 * seqlen * hidden_size * intermediate_size
    return ssm_scan_flops + proj_flops


def get_moe_flops(
    seqlen: int,
    hidden_size: int,
    num_experts_per_tok: int,
    expert_intermediate_size: int,
    shared_expert_intermediate_size: int,
) -> float:
    """FLOPs for one MoE FFN layer on seqlen tokens."""
    routed_flops = (
        6 * seqlen * hidden_size * expert_intermediate_size * num_experts_per_tok
    )
    shared_flops = 6 * seqlen * hidden_size * shared_expert_intermediate_size
    return routed_flops + shared_flops


def get_dense_mlp_flops(
    seqlen: int,
    hidden_size: int,
    intermediate_size: int,
) -> float:
    """FLOPs for one dense SwiGLU MLP layer on seqlen tokens (up+gate+down)."""
    return 6 * seqlen * hidden_size * intermediate_size


def compute_flops_saved(
    seqlen_child: int,
    seqlen_total: int,
    config: Any,
    seqlen_parent: int | None = None,
) -> float:
    """FLOPs saved by reusing this node's cached state.

    ``seqlen_child`` is the marginal token span whose work is avoided relative to
    the reusable ancestor. By default the reusable ancestor is inferred as the
    immediate parent (``seqlen_total - seqlen_child``). Callers can pass ``seqlen_parent`` explicitly when
    the true reusable ancestor is higher in the tree.
    """
    if seqlen_parent is None:
        seqlen_parent = max(0, seqlen_total - seqlen_child)
    else:
        seqlen_parent = max(0, min(seqlen_parent, seqlen_total))
        seqlen_child = max(0, seqlen_total - seqlen_parent)

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Mamba2 / linear-attention params
    linear_num_heads = config.linear_num_value_heads
    linear_head_dim = config.linear_value_head_dim
    linear_state_size = config.linear_key_head_dim  # state_size == key_head_dim

    mlp_only_layers = set(getattr(config, "mlp_only_layers", []))
    is_moe = getattr(config, "num_experts", 1) > 1

    flops = 0.0
    for layer_idx, layer_type in enumerate(config.layers_block_type):
        # Attention / SSM layer
        if layer_type == "attention":
            flops += get_full_attn_flops(
                seqlen_total, hidden_size, num_heads, num_kv_heads, head_dim
            ) - get_full_attn_flops(
                seqlen_parent, hidden_size, num_heads, num_kv_heads, head_dim
            )
        else:
            # Mamba2 / GatedDeltaNet: savings are only for this node's own segment
            flops += get_linear_attn_flops(
                seqlen_child,
                hidden_size,
                linear_num_heads,
                linear_head_dim,
                linear_state_size,
            )

        # FFN layer: marginal savings relative to parent
        if is_moe and layer_idx not in mlp_only_layers:
            flops += get_moe_flops(
                seqlen_total,
                hidden_size,
                config.num_experts_per_tok,
                config.moe_intermediate_size,
                config.shared_expert_intermediate_size,
            ) - get_moe_flops(
                seqlen_parent,
                hidden_size,
                config.num_experts_per_tok,
                config.moe_intermediate_size,
                config.shared_expert_intermediate_size,
            )
        else:
            flops += get_dense_mlp_flops(
                seqlen_total, hidden_size, config.intermediate_size
            ) - get_dense_mlp_flops(
                seqlen_parent, hidden_size, config.intermediate_size
            )

    return flops


def compute_memory_bytes(
    prefix_len: int,
    cache_params: "BaseLinearStateParams",
    config: Any,
    model_config: "ModelConfig",
    tp_world_size: int,
    kv_dtype_bytes: int = 2,
) -> float:
    """Total per-GPU memory occupied by one cached node (SSM state + KV cache)."""
    ssm_bytes = cache_params.mamba_cache_per_req
    num_kv_heads_per_gpu = model_config.get_num_kv_heads(tp_world_size)
    num_attn_layers = len(config.full_attention_layer_ids)
    # factor of 2: K and V are both stored
    kv_bytes = (
        prefix_len
        * num_kv_heads_per_gpu
        * config.head_dim
        * 2
        * kv_dtype_bytes
        * num_attn_layers
    )
    return ssm_bytes + kv_bytes


def compute_flop_efficiency(
    seqlen_child: int,
    seqlen_total: int,
    cache_params: "BaseLinearStateParams",
    config: Any,
    model_config: "ModelConfig",
    tp_world_size: int,
    kv_dtype_bytes: int = 2,
) -> float:
    """Marconi efficiency score: FLOPs_saved / memory_bytes.
    Higher score = prefer to keep.
    """
    return compute_flops_saved(seqlen_child, seqlen_total, config) / (
        compute_memory_bytes(
            seqlen_total,
            cache_params,
            config,
            model_config,
            tp_world_size,
            kv_dtype_bytes,
        )
        + 1e-8
    )


def compute_flop_efficiency_with_parent(
    seqlen_total: int,
    seqlen_parent: int,
    cache_params: "BaseLinearStateParams",
    config: Any,
    model_config: "ModelConfig",
    tp_world_size: int,
    kv_dtype_bytes: int = 2,
) -> float:
    """Marconi efficiency score with an explicit reusable ancestor length."""
    return compute_flops_saved(
        seqlen_child=max(0, seqlen_total - seqlen_parent),
        seqlen_total=seqlen_total,
        config=config,
        seqlen_parent=seqlen_parent,
    ) / (
        compute_memory_bytes(
            seqlen_total,
            cache_params,
            config,
            model_config,
            tp_world_size,
            kv_dtype_bytes,
        )
        + 1e-8
    )
