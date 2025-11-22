import functools
from typing import Optional

import torch

from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import silu_and_mul


def get_scalar_type(num_bits: int, has_zp: bool):
    from sgl_kernel.scalar_type import scalar_types

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
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: float = None,
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
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import (
        moe_align_block_size,
        try_get_optimal_moe_config,
    )

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert (
        hidden_states.dtype == w1_scale.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
    assert (
        hidden_states.dtype == w2_scale.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]

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

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    )

    intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        hidden_states,
        intermediate_cache1,
        w1,
        w1_scale,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type1.id,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        w2_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type2.id,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)
    torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1, out=output)
    if routed_scaling_factor is not None:
        output *= routed_scaling_factor
    return output


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
    inplace: bool = False,
    routed_scaling_factor: float = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def fused_marlin_moe_deepep_ll(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    masked_m: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    routed_scaling_factor: float = None,
) -> torch.Tensor:
    """
    Marlin MoE kernel for DeepEP low latency mode with masked operations.
    
    Args:
        hidden_states: Input tensor [num_experts, max_tokens_per_expert, hidden_size]
        w1: First weight tensor [num_experts, ...]
        w2: Second weight tensor [num_experts, ...]
        w1_scale: Scale for w1
        w2_scale: Scale for w2
        masked_m: Number of valid tokens per expert [num_experts]
        g_idx1, g_idx2: Group indices for act_order
        sort_indices1, sort_indices2: Sorting indices for act_order
        w1_zeros, w2_zeros: Zero points
        num_bits: Number of quantization bits
        is_k_full: Whether K dimension is full
        routed_scaling_factor: Optional scaling factor
        
    Returns:
        Output tensor [num_experts, max_tokens_per_expert, hidden_size]
    """
    assert hidden_states.ndim == 3, "hidden_states must be 3D for DeepEP LL mode"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert num_bits in [4, 8]
    
    num_experts, max_tokens, hidden_size = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # Verify dimensions
    assert hidden_size == w1.shape[1] * 16, f"Hidden size mismatch: {hidden_size} != {w1.shape[1] * 16}"
    assert hidden_size == w2.shape[2] // (num_bits // 2), "Hidden size mismatch w2"
    assert w1.shape[0] == num_experts, "Number of experts mismatch"
    assert w2.shape[0] == num_experts, "Number of experts mismatch"
    assert masked_m.shape[0] == num_experts, "masked_m must have num_experts elements"
    
    N = w2.shape[1] * 16  # Intermediate dimension
    
    # Allocate output tensor
    output = torch.zeros_like(hidden_states)
    
    # Get scalar types
    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)
    
    # Process each expert separately, only on valid tokens
    for expert_id in range(num_experts):
        valid_tokens = int(masked_m[expert_id].item())
        
        if valid_tokens == 0:
            continue
            
        # Extract valid tokens for this expert
        expert_input = hidden_states[expert_id, :valid_tokens, :].contiguous()  # [valid_tokens, hidden_size]
        
        # Prepare workspace
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        workspace = torch.zeros(sms * 4, dtype=torch.int, device=device, requires_grad=False)
        
        # First GEMM: expert_input @ w1[expert_id] -> intermediate (gate_up projection)
        intermediate1 = torch.empty(
            (valid_tokens, 2 * N),
            device=device,
            dtype=dtype,
        )
        
        # Call Marlin GEMM for expert expert_id, treating it as a single-expert MoE
        # We need to create dummy topk_ids that map all tokens to expert 0 (since we're processing one expert at a time)
        dummy_topk_ids = torch.zeros((valid_tokens, 1), dtype=torch.int32, device=device)
        dummy_topk_weights = torch.ones((valid_tokens, 1), dtype=torch.float32, device=device)
        dummy_expert_ids = torch.zeros(valid_tokens, dtype=torch.int32, device=device)
        
        # Extract this expert's weights (treat as single expert)
        w1_expert = w1[expert_id:expert_id+1]  # [1, ...]
        w2_expert = w2[expert_id:expert_id+1]  # [1, ...]
        w1_scale_expert = w1_scale[expert_id:expert_id+1]
        w2_scale_expert = w2_scale[expert_id:expert_id+1]
        
        g_idx1_expert = g_idx1[expert_id:expert_id+1] if g_idx1 is not None else None
        g_idx2_expert = g_idx2[expert_id:expert_id+1] if g_idx2 is not None else None
        sort_indices1_expert = sort_indices1[expert_id:expert_id+1] if sort_indices1 is not None else None
        sort_indices2_expert = sort_indices2[expert_id:expert_id+1] if sort_indices2 is not None else None
        w1_zeros_expert = w1_zeros[expert_id:expert_id+1] if w1_zeros is not None else None
        w2_zeros_expert = w2_zeros[expert_id:expert_id+1] if w2_zeros is not None else None
        
        use_atomic_add = (dtype == torch.half or torch.cuda.get_device_capability(device)[0] >= 9)
        
        # First GEMM
        intermediate1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
            expert_input,
            intermediate1,
            w1_expert,
            w1_scale_expert,
            w1_zeros_expert,
            g_idx1_expert,
            sort_indices1_expert,
            workspace,
            dummy_topk_ids[:, 0],  # sorted_token_ids
            dummy_expert_ids,
            valid_tokens,
            dummy_topk_weights,
            moe_block_size=64,  # Use a reasonable block size
            top_k=1,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type_id=scalar_type1.id,
            size_m=valid_tokens,
            size_n=2 * N,
            size_k=hidden_size,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        
        # Apply SiLU and mul
        intermediate2 = torch.empty((valid_tokens, N), device=device, dtype=dtype)
        silu_and_mul(intermediate1, intermediate2)
        
        # Second GEMM: intermediate2 @ w2[expert_id] -> output
        intermediate3 = torch.empty((valid_tokens, 1, hidden_size), device=device, dtype=dtype)
        
        intermediate3 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
            intermediate2,
            intermediate3.view(valid_tokens, hidden_size),
            w2_expert,
            w2_scale_expert,
            w2_zeros_expert,
            g_idx2_expert,
            sort_indices2_expert,
            workspace,
            dummy_topk_ids[:, 0],
            dummy_expert_ids,
            valid_tokens,
            dummy_topk_weights,
            moe_block_size=64,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type_id=scalar_type2.id,
            size_m=valid_tokens,
            size_n=hidden_size,
            size_k=N,
            is_k_full=is_k_full,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        ).view(valid_tokens, 1, hidden_size)
        
        # Sum over topk dimension (which is 1 here) and store in output
        output[expert_id, :valid_tokens, :] = intermediate3.squeeze(1)
    
    if routed_scaling_factor is not None:
        output *= routed_scaling_factor
        
    return output
