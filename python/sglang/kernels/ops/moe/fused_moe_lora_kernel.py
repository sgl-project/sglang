# Temporarily adapted from https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/fused_moe_lora_op.py, will optimize in future refactor

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.utils.common import is_blackwell_supported, is_sm90_supported

# Import SGLang's standard PDL support detection


_LORA_PTR_DICT: dict[tuple[int, ...], torch.Tensor] = {}


def _get_ptr(lora_weights: list[torch.Tensor], device: torch.device):
    """
    `_LORA_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)

    if (ptr_tensor := _LORA_PTR_DICT.get(key)) is not None:
        return ptr_tensor

    tensor_ptrs = []
    for lora_weight in lora_weights:
        tensor_ptrs.append(lora_weight.data_ptr())
    ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)

    _LORA_PTR_DICT[key] = ptr_tensor
    return _LORA_PTR_DICT.get(key)


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_c_size",
        "c_base_offset",
    ]
)
def _fused_moe_lora_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    lora_ids,
    adapter_enabled,
    page_table_ptr,
    lora_ranks_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bl,
    stride_bpage,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    stride_pt_lora,
    stride_pt_page,
    slice_a_size,
    slice_c_size,
    c_base_offset,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    PAGE_RANK_SIZE: tl.constexpr,
    IS_PAGED: tl.constexpr,
    RANK_ON_N: tl.constexpr,
    SMALL_RANK_BLOCK: tl.constexpr,
    ACCUMULATE_OUTPUT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)

    if lora_id == -1:
        # Early exit for the no-lora case.
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        # Early exit for the no moe lora case.
        return
    if IS_PAGED:
        actual_rank = tl.load(lora_ranks_ptr + lora_id)
        if actual_rank <= 0:
            return
    max_loras = tl.num_programs(axis=2)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m,pid_n
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    if IS_PAGED and RANK_ON_N:
        # Dense Paged LoRA launches one rank-page program and returns before
        # loading the page or executing tl.dot when it is beyond cur_rank.
        # Paged MoE uses a tensor-core-friendly rank block (normally 16 for
        # page_rank_size=8), but follows the same per-adapter rule.  This is
        # what lets r8 and r64 slots in the same launch do different work.
        rank_block_start = pid_n * BLOCK_SIZE_N
        if rank_block_start >= actual_rank:
            return

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    # get the expert_id to process curr shard
    ind = lora_id * stride_el + pid_m
    expert_id = tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)
    if expert_id == -1:
        return

    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size + c_base_offset

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    # ================================================================= secure

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    token_ind = stride_tl * lora_id + offs_token_id
    offs_token = tl.load(
        sorted_token_ids_ptr + token_ind, token_ind < max_loras * stride_tl, 0
    )
    token_mask = offs_token < num_valid_tokens

    # ================================================================= secure

    # get a_ptrs,b_ptrs
    a_ptrs = cur_a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if IS_PAGED and RANK_ON_N:
        if actual_rank <= SMALL_RANK_BLOCK:
            # Keep the normal large-rank tile in this launch, but compute an
            # r8/r16 adapter with one tensor-core-sized rank tile. Invalid
            # logical pages are rejected before the weight load and tl.dot.
            small_offs_n = tl.arange(0, SMALL_RANK_BLOCK).to(tl.int64)
            small_logical_page = small_offs_n // PAGE_RANK_SIZE
            small_rank_in_page = small_offs_n % PAGE_RANK_SIZE
            small_physical_page = tl.load(
                page_table_ptr
                + lora_id * stride_pt_lora
                + small_logical_page * stride_pt_page,
                mask=small_offs_n < N,
                other=-1,
            )
            small_safe_physical_page = tl.maximum(small_physical_page, 0)
            small_b_ptrs = (
                cur_b_ptr
                + small_safe_physical_page[None, :] * stride_bpage
                + expert_id * stride_be
                + offs_k[:, None] * stride_bk
                + small_rank_in_page[None, :] * stride_bn
            )

            if USE_GDC and IS_PRIMARY:
                tl.extra.cuda.gdc_launch_dependents()

            small_accumulator = tl.zeros(
                (BLOCK_SIZE_M, SMALL_RANK_BLOCK), dtype=tl.float32
            )

            if USE_GDC and not IS_PRIMARY:
                tl.extra.cuda.gdc_wait()

            small_a_ptrs = a_ptrs
            for k in range(0, grid_k):
                k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
                b = tl.load(
                    small_b_ptrs,
                    mask=(small_physical_page[None, :] >= 0)
                    & (small_offs_n[None, :] < actual_rank)
                    & (offs_k[:, None] < k_remaining),
                    other=0.0,
                )
                a = tl.load(
                    small_a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                small_accumulator += tl.dot(a, b.to(a.dtype))
                small_a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
                small_b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(
                    topk_weights_ptr + offs_token, mask=token_mask, other=0
                )
                small_accumulator = small_accumulator * moe_weight[:, None]
            small_accumulator = small_accumulator.to(c_ptr.dtype.element_ty)
            small_c_ptrs = (
                cur_c_ptr
                + stride_cm * offs_token[:, None]
                + stride_cn * small_offs_n[None, :]
            )
            small_c_mask = token_mask[:, None] & (small_offs_n[None, :] < actual_rank)
            if SPLIT_K == 1:
                tl.store(small_c_ptrs, small_accumulator, mask=small_c_mask)
            else:
                tl.atomic_add(
                    small_c_ptrs,
                    small_accumulator,
                    mask=small_c_mask,
                    sem="relaxed",
                )
            return

    if not IS_PAGED:
        b_ptrs = (
            cur_b_ptr
            + lora_id * stride_bl
            + expert_id * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
    elif RANK_ON_N:
        logical_page = offs_bn // PAGE_RANK_SIZE
        rank_in_page = offs_bn % PAGE_RANK_SIZE
        physical_page = tl.load(
            page_table_ptr + lora_id * stride_pt_lora + logical_page * stride_pt_page
        )
        safe_physical_page = tl.maximum(physical_page, 0)
        b_ptrs = (
            cur_b_ptr
            + safe_physical_page[None, :] * stride_bpage
            + expert_id * stride_be
            + offs_k[:, None] * stride_bk
            + rank_in_page[None, :] * stride_bn
        )

    if USE_GDC and IS_PRIMARY:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        tl.extra.cuda.gdc_launch_dependents()

    # ================================================================= secure

    # accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ================================================================= secure

    # GDC wait waits for ALL programs in the prior kernel to complete
    # before continuing.
    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    if IS_PAGED and not RANK_ON_N and actual_rank <= SMALL_RANK_BLOCK:
        # Expand must not execute a masked 64-wide dot for an r8 adapter.
        small_offs_k = tl.arange(0, SMALL_RANK_BLOCK)
        small_logical_page = small_offs_k // PAGE_RANK_SIZE
        small_rank_in_page = small_offs_k % PAGE_RANK_SIZE
        small_physical_page = tl.load(
            page_table_ptr
            + lora_id * stride_pt_lora
            + small_logical_page * stride_pt_page,
            mask=small_offs_k < K,
            other=-1,
        )
        small_safe_physical_page = tl.maximum(small_physical_page, 0)
        small_a_ptrs = cur_a_ptr + (
            offs_token[:, None] // top_k * stride_am + small_offs_k[None, :] * stride_ak
        )
        small_b_ptrs = (
            cur_b_ptr
            + small_safe_physical_page[:, None] * stride_bpage
            + expert_id * stride_be
            + small_rank_in_page[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        a = tl.load(
            small_a_ptrs,
            mask=token_mask[:, None] & (small_offs_k[None, :] < actual_rank),
            other=0.0,
        )
        b = tl.load(
            small_b_ptrs,
            mask=(small_physical_page[:, None] >= 0)
            & (small_offs_k[:, None] < actual_rank)
            & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b.to(a.dtype))
    else:
        for k in range(0, grid_k):
            k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
            if not IS_PAGED:
                b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                accumulator += tl.dot(a, b.to(a.dtype))
            elif RANK_ON_N:
                b = tl.load(
                    b_ptrs,
                    mask=(physical_page[None, :] >= 0)
                    & (offs_bn[None, :] < actual_rank)
                    & (offs_k[:, None] < k_remaining),
                    other=0.0,
                )
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                accumulator += tl.dot(a, b.to(a.dtype))
            else:
                rank_block_start = k * BLOCK_SIZE_K * SPLIT_K
                if rank_block_start < actual_rank:
                    rank_offset = rank_block_start + offs_k
                    logical_page = rank_offset // PAGE_RANK_SIZE
                    rank_in_page = rank_offset % PAGE_RANK_SIZE
                    physical_page = tl.load(
                        page_table_ptr
                        + lora_id * stride_pt_lora
                        + logical_page * stride_pt_page
                    )
                    safe_physical_page = tl.maximum(physical_page, 0)
                    paged_b_ptrs = (
                        cur_b_ptr
                        + safe_physical_page[:, None] * stride_bpage
                        + expert_id * stride_be
                        + rank_in_page[:, None] * stride_bk
                        + offs_bn[None, :] * stride_bn
                    )
                    b = tl.load(
                        paged_b_ptrs,
                        mask=(physical_page[:, None] >= 0)
                        & (rank_offset[:, None] < actual_rank)
                        & (offs_bn[None, :] < N),
                        other=0.0,
                    )
                    a = tl.load(
                        a_ptrs,
                        mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                        other=0.0,
                    )
                    accumulator += tl.dot(a, b.to(a.dtype))
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
            if not IS_PAGED or RANK_ON_N:
                b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if ACCUMULATE_OUTPUT:
        accumulator += tl.load(c_ptrs, mask=c_mask, other=0.0)

    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    # (num_slices, num_tokens, top_k_num, max_lora_rank)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    top_k_divisor: int = None,
    mul_routed_weight: bool = False,
    page_table: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    page_rank_size: int = 0,
) -> None:
    w1_lora_a_stacked = lora_a_stacked[0]
    is_paged = page_table is not None and page_rank_size > 0
    if is_paged:
        assert lora_ranks is not None
        N = page_table.shape[1] * page_rank_size
        page_table_arg = page_table
        lora_ranks_arg = lora_ranks
        num_slots = page_table.shape[0]
    else:
        page_table_arg = adapter_enabled.view(-1, 1)
        lora_ranks_arg = adapter_enabled
        num_slots = w1_lora_a_stacked.shape[0]

    use_gdc = is_sm90_supported() or is_blackwell_supported()
    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
    }

    b_ptr = _get_ptr(lora_a_stacked, device)

    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        num_slots,
    )
    _fused_moe_lora_kernel[grid](
        qcurr_hidden_states,
        b_ptr,
        a_intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        lora_ids,
        adapter_enabled,
        page_table_arg,
        lora_ranks_arg,
        qcurr_hidden_states.stride(0),
        qcurr_hidden_states.stride(1),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(0),
        w1_lora_a_stacked.stride(1),
        w1_lora_a_stacked.stride(3),
        w1_lora_a_stacked.stride(2),
        a_intermediate_cache1.stride(2),
        a_intermediate_cache1.stride(3),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        page_table_arg.stride(0),
        page_table_arg.stride(1),
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        c_base_offset=0,
        num_slice_a=1,
        num_slice_c=num_slices,
        top_k=(
            top_k_divisor
            if top_k_divisor is not None
            else (1 if mul_routed_weight else top_k_num)
        ),
        MUL_ROUTED_WEIGHT=False,
        IS_PRIMARY=True,
        PAGE_RANK_SIZE=page_rank_size if is_paged else 1,
        IS_PAGED=is_paged,
        RANK_ON_N=True,
        SMALL_RANK_BLOCK=16,
        ACCUMULATE_OUTPUT=False,
        **shrink_config,
    )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    page_table: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    page_rank_size: int = 0,
) -> None:

    b_ptr = _get_ptr(lora_b_stacked, device)
    is_paged = page_table is not None and page_rank_size > 0
    if is_paged:
        assert lora_ranks is not None
        K = page_table.shape[1] * page_rank_size
        page_table_arg = page_table
        lora_ranks_arg = lora_ranks
        num_slots = page_table.shape[0]
    else:
        K = max_lora_rank
        page_table_arg = adapter_enabled.view(-1, 1)
        lora_ranks_arg = adapter_enabled
        num_slots = lora_b_stacked[0].shape[0]
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[3]
    )

    use_gdc = is_sm90_supported() or is_blackwell_supported()
    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,  # Set split_k = 1 for expand calls
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
    }

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        num_slots,
    )
    _fused_moe_lora_kernel[grid](
        a_intermediate_cache1,
        b_ptr,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        num_experts,
        lora_ids,
        adapter_enabled,
        page_table_arg,
        lora_ranks_arg,
        a_intermediate_cache1.stride(0),
        a_intermediate_cache1.stride(1),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(0),
        w1_lora_b_stacked.stride(1),
        w1_lora_b_stacked.stride(3),
        w1_lora_b_stacked.stride(2),
        output.stride(1),
        output.stride(2),
        sorted_token_ids.stride(0),
        expert_ids.stride(0),
        page_table_arg.stride(0),
        page_table_arg.stride(1),
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=N,
        c_base_offset=offset,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        top_k=1,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        IS_PRIMARY=False,
        PAGE_RANK_SIZE=page_rank_size if is_paged else 1,
        IS_PAGED=is_paged,
        RANK_ON_N=False,
        SMALL_RANK_BLOCK=16,
        ACCUMULATE_OUTPUT=True,
        **expand_config,
    )


@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, N, max_lora_rank,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    page_table: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    page_rank_size: int = 0,
) -> None:
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert (
        sorted_token_ids.dim()
        == expert_ids.dim()
        == topk_weights.dim()
        == qcurr_hidden_states.dim()
        == 2
    )
    assert (
        sorted_token_ids.shape[0]
        == expert_ids.shape[0]
        == num_tokens_post_padded.shape[0]
    )
    assert output.shape[0] == topk_weights.shape[0]
    assert top_k_num == topk_weights.shape[1]
    device = qcurr_hidden_states.device
    is_paged = page_table is not None and page_rank_size > 0
    if is_paged:
        if fully_sharded:
            raise NotImplementedError(
                "Paged MoE LoRA does not support fully-sharded LoRA."
            )
        assert lora_ranks is not None
        assert page_table.dtype == torch.int32
        assert lora_ranks.dtype == torch.int32
        assert page_table.device == device and lora_ranks.device == device
        max_lora_rank = page_table.shape[1] * page_rank_size
    num_slices = len(lora_a_stacked)
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = lora_a_stacked[0].shape[1]
    N = max_lora_rank
    M = topk_weights.shape[0]
    EM = sorted_token_ids.shape[1]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]

    # Detect whether input is already expanded (down path: [M*top_k, dim])
    # or not (gate_up path: [M, dim]). Down path needs divisor=1.
    input_is_expanded = qcurr_hidden_states.shape[0] == M * top_k_num
    shrink_top_k_divisor = 1 if input_is_expanded else top_k_num

    a_intermediate_cache1 = torch.zeros(
        (num_slices, M, top_k_num, max_lora_rank),
        dtype=output.dtype,
        device=device,
    )

    _fused_moe_lora_shrink(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        shrink_block_size_m,
        shrink_block_size_n,
        shrink_block_size_k,
        shrink_group_size_m,
        shrink_num_warps,
        shrink_num_stages,
        shrink_split_k,
        top_k_divisor=shrink_top_k_divisor,
        mul_routed_weight=False,
        page_table=page_table,
        lora_ranks=lora_ranks,
        page_rank_size=page_rank_size,
    )

    if fully_sharded:
        if max_lora_rank == w1_lora_b_stacked.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                a_intermediate_cache1
            )
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(
                a_intermediate_cache1
            )

            # reset max_lora_rank to the full rank after allgather
            max_lora_rank = a_intermediate_cache1.shape[-1]

    _fused_moe_lora_expand(
        output,
        a_intermediate_cache1,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        max_lora_rank,
        w1_output_dim_size,
        expand_block_size_m,
        expand_block_size_n,
        expand_block_size_k,
        expand_group_size_m,
        expand_num_warps,
        expand_num_stages,
        expand_split_k,
        mul_routed_weight,
        offset,
        page_table,
        lora_ranks,
        page_rank_size,
    )


def _fused_moe_lora_fake(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    page_table: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    page_rank_size: int = 0,
) -> None:
    return


def _fused_moe_lora_shrink_fake(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    page_table: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    page_rank_size: int = 0,
) -> None:
    return


def _fused_moe_lora_expand_fake(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    page_table: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    page_rank_size: int = 0,
) -> None:
    return


# Register as SGLang custom ops following the same pattern as other ops
try:
    from sglang.srt.utils.common import direct_register_custom_op

    direct_register_custom_op(
        op_name="fused_moe_lora",
        op_func=_fused_moe_lora,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_shrink",
        op_func=_fused_moe_lora_shrink,
        mutates_args=["a_intermediate_cache1"],
        fake_impl=_fused_moe_lora_shrink_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_expand",
        op_func=_fused_moe_lora_expand,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_expand_fake,
    )

    # Export through torch.ops.sglang namespace
    fused_moe_lora = torch.ops.sglang.fused_moe_lora
    fused_moe_lora_shrink = torch.ops.sglang.fused_moe_lora_shrink
    fused_moe_lora_expand = torch.ops.sglang.fused_moe_lora_expand

except AttributeError:
    fused_moe_lora = _fused_moe_lora
    fused_moe_lora_shrink = _fused_moe_lora_shrink
    fused_moe_lora_expand = _fused_moe_lora_expand
