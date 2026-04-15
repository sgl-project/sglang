"""
LoRA Virtual Experts Triton Ops.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    virtual_topk_ids_ptr,
    token_lora_mask_ptr,
    num_experts_for_weight: tl.constexpr,
    M,
    top_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses _get_virtual_topk_ids: comparison + clamp + arithmetic into one kernel.

    For each (m, k):
        lora_id = token_lora_mapping[m]
        mask[m] = (lora_id >= 0)
        safe_lora = max(lora_id, 0)
        if shared_outer:  (handled by num_experts_for_weight == 0 sentinel)
            virtual_topk_ids[m, k] = safe_lora * 1  (= safe_lora)
        else:
            virtual_topk_ids[m, k] = topk_ids[m, k] + safe_lora * num_experts_for_weight
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * top_k
    valid = offs < total

    m = offs // top_k
    # k = offs % top_k  # not needed directly

    lora_id = tl.load(token_lora_mapping_ptr + m, mask=valid, other=0)
    mask_val = lora_id >= 0
    safe_lora = tl.maximum(lora_id, 0)

    base = tl.load(topk_ids_ptr + offs, mask=valid, other=0)
    result = base + safe_lora * num_experts_for_weight
    tl.store(virtual_topk_ids_ptr + offs, result, mask=valid)

    # Write mask once per row (at first k position)
    k = offs % top_k
    is_first_k = k == 0
    tl.store(token_lora_mask_ptr + m, mask_val, mask=valid & is_first_k)


def _fused_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    shared_outer: bool,
    max_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns virtual topk_ids, token_lora_mask, and virtual_num_experts.
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device

    if shared_outer:
        num_experts_for_weight = 1
        # For shared_outer, we need topk_ids to be zeros
        zero_topk = torch.zeros_like(topk_ids)
        input_topk = zero_topk
    else:
        num_experts_for_weight = num_experts
        input_topk = topk_ids

    virtual_topk_ids = torch.empty_like(topk_ids)
    token_lora_mask = torch.empty(M, dtype=torch.bool, device=device)

    BLOCK_SIZE = 1024
    grid = ((M * top_k + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_virtual_topk_ids_kernel[grid](
        input_topk,
        token_lora_mapping,
        virtual_topk_ids,
        token_lora_mask,
        num_experts_for_weight,
        M,
        top_k,
        BLOCK_SIZE,
    )

    virtual_num_experts = num_experts_for_weight * max_loras
    return virtual_topk_ids, token_lora_mask, virtual_num_experts


@triton.jit
def _fused_sanitize_expert_ids_kernel(
    expert_ids_ptr,
    output_ptr,
    num_virtual_experts,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < N

    eid = tl.load(expert_ids_ptr + offs, mask=valid, other=0)
    result = tl.where(eid < num_virtual_experts, eid, -1)
    tl.store(output_ptr + offs, result, mask=valid)


def fused_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    num_virtual_experts: int,
) -> torch.Tensor:
    """
    Sanitize expert_ids by replacing values >= num_virtual_experts with -1.

    Returns a new tensor with expert_ids >= num_virtual_experts replaced by -1.
    """
    N = expert_ids.numel()
    output = torch.empty_like(expert_ids)

    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_sanitize_expert_ids_kernel[grid](
        expert_ids,
        output,
        num_virtual_experts,
        N,
        BLOCK_SIZE,
    )
    return output


@triton.jit
def _moe_lora_shrink_splitk_kernel(
    # Pointers
    a_ptr,  # type: ignore  # [num_tokens, K]
    b_ptr,  # type: ignore  # [num_virtual_experts, N, K]
    c_ptr,  # type: ignore  # [num_tokens * top_k, N]  (pre-zeroed when SPLIT_K > 1)
    sorted_token_ids_ptr,  # type: ignore
    expert_ids_ptr,  # type: ignore
    num_tokens_post_padded_ptr,  # type: ignore
    # Dimensions
    N,  # type: ignore
    K,  # type: ignore
    num_valid_tokens,  # type: ignore
    # Strides
    stride_am,  # type: ignore
    stride_ak,  # type: ignore
    stride_be,  # type: ignore
    stride_bn,  # type: ignore
    stride_bk,  # type: ignore
    stride_cm,  # type: ignore
    stride_cn,  # type: ignore
    # Constexprs
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Split-K grouped GEMM for the LoRA A (shrink) stage with few virtual experts."""
    pid = tl.program_id(0)
    pid_sk = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Token routing (same pattern as fused_moe_triton_kernels)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return

    # Pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Accumulate
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    for k in range(0, grid_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        k_mask = offs_k[:, None] < k_remaining
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # Write output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


def _invoke_moe_lora_shrink_splitk(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: dict[str, Any],
) -> None:
    """Launch split-K shrink kernel for LoRA A with few virtual experts."""
    N = weight.shape[1]
    K = weight.shape[2]
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = min(config.get("BLOCK_SIZE_N", 64), max(16, N))
    BLOCK_SIZE_K = config.get("BLOCK_SIZE_K", 64)
    GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)

    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M)
    num_n_blocks = triton.cdiv(N, BLOCK_SIZE_N)
    base_grid = num_m_blocks * num_n_blocks
    max_split_k = max(1, K // BLOCK_SIZE_K)
    SPLIT_K = min(max_split_k, max(1, 128 // base_grid)) if base_grid < 128 else 1

    grid = (SPLIT_K * base_grid,)

    _moe_lora_shrink_splitk_kernel[grid](
        hidden_states,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        topk_ids.numel(),
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=SPLIT_K,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 4),
    )


@torch.compile(dynamic=True)
def _align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch align_block_size for num_experts > 1024, compiled via torch.compile."""
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    num_valid_tokens = flat_topk_ids.numel()
    max_total_padded_tokens = (
        (num_valid_tokens + num_experts * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded_tokens // block_size

    sorted_token_ids = torch.full(
        (max_total_padded_tokens,),
        num_valid_tokens,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.full(
        (max_num_blocks,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_valid_tokens == 0:
        num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32, device=device)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    sorted_order = torch.argsort(flat_topk_ids)
    sorted_expert_ids = flat_topk_ids[sorted_order]
    expert_range = torch.arange(num_experts, device=device, dtype=torch.int64)
    counts_offsets = torch.searchsorted(sorted_expert_ids, expert_range, right=False)
    counts_end = torch.searchsorted(sorted_expert_ids, expert_range, right=True)
    counts = counts_end - counts_offsets
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded_tokens = padded_counts.sum().to(torch.int32).reshape(1)
    padded_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts

    token_ranks = (
        torch.arange(num_valid_tokens, device=device, dtype=torch.int64)
        - counts_offsets[sorted_expert_ids]
    )
    output_positions = padded_offsets[sorted_expert_ids] + token_ranks
    sorted_token_ids.scatter_(
        0,
        output_positions.to(torch.int64),
        sorted_order.to(torch.int32),
    )

    block_counts = padded_counts // block_size
    actual_num_blocks = block_counts.sum()

    if max_num_blocks <= 0:
        return sorted_token_ids, expert_ids, total_padded_tokens

    block_offsets = torch.cumsum(block_counts, dim=0)
    all_block_positions = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    assigned_experts = torch.searchsorted(
        block_offsets, all_block_positions, right=True
    ).to(torch.int32)
    expert_ids.copy_(
        torch.where(
            all_block_positions < actual_num_blocks,
            assigned_experts,
            torch.full_like(assigned_experts, -1),
        )
    )

    return sorted_token_ids, expert_ids, total_padded_tokens


_align_block_size_large = _align_block_size_torch


def _merged_experts_fused_moe_lora_add_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    return


def _merged_experts_fused_moe_lora_add_impl(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
) -> None:
    """
    1. Prepare virtual expert routing metadata from topk_ids + token_lora_mapping * num_experts.
    2. Flatten LoRA weights from [max_loras, num_experts, ...] to [max_loras * num_experts, ...].
    3. Run regular SGLang fused-MoE kernels for LoRA A and LoRA B.
    4. Mask out tokens with token_lora_mapping == -1 on the add path.
    """
    max_loras, _, max_lora_rank, _ = lora_a.shape
    input_top_k = 1 if hidden_states.shape[0] == topk_ids.numel() else topk_ids.shape[1]

    def _merge_lora_expert_weight(t: torch.Tensor) -> torch.Tensor:
        # [max_loras, num_experts, x, y] -> [max_loras * num_experts, x, y]
        return t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])

    def _get_stage_config(
        weight: torch.Tensor,
        stage_top_k: int,
    ) -> dict[str, Any]:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        config_dtype = get_config_dtype_str(dtype=hidden_states.dtype)
        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            weight.shape,
            weight.shape,
            stage_top_k,
            config_dtype,
        )
        try:
            cfg = get_config_func(token_lora_mapping.shape[0])
        except ValueError:
            K_dim = weight.shape[2]
            N_dim = weight.shape[1]
            if K_dim >= 1024:
                default_block_k = 256
            elif K_dim >= 64:
                default_block_k = 64
            else:
                default_block_k = max(16, K_dim)
            cfg = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": min(64, max(16, N_dim)),
                "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        return cfg

    def _align_block_size(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The native align kernel consumes num_experts + 1 internally for its
        # sentinel bucket, so the 1024-expert boundary must use the fallback path.
        if num_experts < 1024:
            from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
                moe_align_block_size as native_moe_align_block_size,
            )

            return native_moe_align_block_size(topk_ids, block_size, num_experts)
        return _align_block_size_large(topk_ids, block_size, num_experts)

    def _get_routing(
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor,
        num_experts: int,
        shared_outer: bool,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check routing_cache for cross-call reuse (gate_up and down share routing)
        cache_key = (num_experts, shared_outer, block_size)
        if routing_cache is not None:
            cached = routing_cache.get(cache_key)
            if cached is not None:
                return cached

        virtual_topk_ids, token_lora_mask, virtual_num_experts = (
            _fused_virtual_topk_ids(
                topk_ids, token_lora_mapping, num_experts, shared_outer, max_loras
            )
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(
            virtual_topk_ids,
            block_size=block_size,
            num_experts=virtual_num_experts,
        )
        # _align_block_size uses a worst-case padded allocation. Trim the routing buffers
        # to a tighter upper bound so we keep the real routed work but drop unused padding
        num_tokens = topk_ids.numel()
        max_nonempty = min(num_tokens, virtual_num_experts)
        tight_padded = (
            triton.cdiv(num_tokens + max_nonempty * (block_size - 1), block_size)
            * block_size
        )
        sorted_token_ids = sorted_token_ids[:tight_padded]
        expert_ids = expert_ids[: tight_padded // block_size]
        expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)
        result = (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
        )

        if routing_cache is not None:
            routing_cache[cache_key] = result

        return result

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )

    lora_a_virtual = _merge_lora_expert_weight(lora_a)
    lora_b_virtual = _merge_lora_expert_weight(lora_b)
    num_experts_a = lora_a.shape[1]
    num_experts_b = lora_b.shape[1]

    intermediate = torch.zeros(
        [token_lora_mapping.shape[0], topk_ids.shape[1], max_lora_rank],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    a_stage_config = _get_stage_config(lora_a_virtual, input_top_k)
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_a,
        experts_shared_outer_loras_a,
        a_stage_config["BLOCK_SIZE_M"],
    )

    _invoke_moe_lora_shrink_splitk(
        hidden_states,
        lora_a_virtual,
        intermediate.view(-1, max_lora_rank),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        input_top_k,
        a_stage_config,
    )

    b_stage_config = _get_stage_config(lora_b_virtual, 1)
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_b,
        experts_shared_outer_loras_b,
        b_stage_config["BLOCK_SIZE_M"],
    )

    invoke_fused_moe_kernel(
        intermediate.view(-1, max_lora_rank),
        lora_b_virtual,
        None,
        output,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        1,
        b_stage_config,
        tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
        False,
        False,
        False,
        False,
        False,
        None,
        fuse_add_to_output=True,
        add_output_mask=token_lora_mask,
        router_topk=topk_ids.shape[1],
    )


def _merged_experts_fused_moe_lora_add_op(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    _merged_experts_fused_moe_lora_add_impl(
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
    )


from sglang.srt.utils.common import direct_register_custom_op

direct_register_custom_op(
    op_name="merged_experts_fused_moe_lora_add",
    op_func=_merged_experts_fused_moe_lora_add_op,
    mutates_args=["output"],
    fake_impl=_merged_experts_fused_moe_lora_add_fake,
)


def merged_experts_fused_moe_lora_add(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
) -> None:
    """Public API: wraps the registered op with routing_cache support."""
    _merged_experts_fused_moe_lora_add_impl(
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
        routing_cache,
    )
