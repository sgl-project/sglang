import torch
from typing import Optional
import functools
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size, try_get_optimal_moe_config)
from vllm.scalar_type import scalar_types
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (marlin_quantize)
from vllm.scalar_type import scalar_types
# from sglang.srt.layers.moe.topk import select_experts
# from sgl_kernel import fused_marlin_moe

def get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.

    assert hidden_states.shape[
        1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2), "Hidden size mismatch w2"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]

    has_no_act_order = (g_idx1 is None and g_idx2 is None
                        and sort_indices1 is None and sort_indices2 is None)
    has_all_act_order = (g_idx1 is not None and g_idx2 is not None
                         and sort_indices1 is not None
                         and sort_indices2 is not None)
    assert has_no_act_order or has_all_act_order, (
        "g_idx and sorted_indices "
        "must be all not None or must be all None")

    has_no_zp = w1_zeros is None and w2_zeros is None
    has_all_zp = w1_zeros is not None and w2_zeros is not None
    assert has_no_zp or has_all_zp, ("zero points must be both not None or "
                                     "must be both None")

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        None,
        is_marlin=True,
    )
    config = get_config_func(M)

    block_size_m = config["BLOCK_SIZE_M"]

    sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m, E)

    max_workspace_size = (max(2 * N, K) // 64) * 16
    workspace = torch.zeros(max_workspace_size,
                            dtype=torch.int,
                            device=hidden_states.device,
                            requires_grad=False)

    if has_no_zp:
        w1_zeros = torch.empty((0, 0),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device,
                               requires_grad=False)
        w2_zeros = torch.empty((0, 0),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device,
                               requires_grad=False)

    if has_no_act_order:
        g_idx1 = torch.empty((0, 0),
                             dtype=torch.int32,
                             device=hidden_states.device,
                             requires_grad=False)
        g_idx2 = torch.empty((0, 0),
                             dtype=torch.int32,
                             device=hidden_states.device,
                             requires_grad=False)
        sort_indices1 = torch.empty((0),
                                    dtype=torch.int32,
                                    device=hidden_states.device,
                                    requires_grad=False)
        sort_indices2 = torch.empty((0, 0),
                                    dtype=torch.int32,
                                    device=hidden_states.device,
                                    requires_grad=False)

    scalar_type1 = get_scalar_type(num_bits, has_all_zp)
    scalar_type2 = get_scalar_type(num_bits, has_all_zp)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    intermediate_cache1 = torch.ops.sgl_kernel.marlin_gemm_moe.default(
        hidden_states,
        w1,
        sorted_token_ids,
        topk_weights,
        topk_ids,
        w1_scale,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        scalar_type1.id,
        M,
        2 * N,
        K,
        is_k_full,
        E,
        topk,
        block_size_m,
        True,
        False,
    )

    torch.ops._C.silu_and_mul(intermediate_cache2,
                              intermediate_cache1.view(-1, 2 * N))

    intermediate_cache3 = torch.ops.sgl_kernel.marlin_gemm_moe.default(
        intermediate_cache2,
        w2,
        sorted_token_ids,
        topk_weights,
        topk_ids,
        w2_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        scalar_type2.id,
        M,
        K,
        N,
        is_k_full,
        E,
        topk,
        block_size_m,
        False,
        True,
    )

    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1)


def fused_marlin_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)
