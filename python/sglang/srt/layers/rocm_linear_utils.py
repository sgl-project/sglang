import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic

from sglang.srt.utils import BumpAllocator

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    gemm_output_zero_allocator: BumpAllocator = None,
):
    M = hidden_states.shape[0]
    N = weight.shape[0]
    y = None

    if M <= 256:
        # TODO (cagri): convert to bfloat16 as part of another kernel to save time
        # for now it is also coupled with zero allocator.
        if gemm_output_zero_allocator != None:
            y = gemm_output_zero_allocator.allocate(M * N).view(M, N)
        else:
            y = torch.zeros((M, N), dtype=torch.float32, device=hidden_states.device)

    if y is not None:
        logits = gemm_a16w16_atomic(hidden_states, weight, y=y).to(hidden_states.dtype)
    else:
        logits = gemm_a16w16(hidden_states, weight)

    return logits


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
