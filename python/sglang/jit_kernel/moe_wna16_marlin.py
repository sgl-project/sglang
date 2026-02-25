from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

# Constants matching device::marlin_moe:: in marlin.cuh
_MAX_THREAD_N = 256


@cache_once
def _jit_moe_wna16_marlin_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_wna16_marlin",
        *args,
        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],
        cuda_wrappers=[
            (
                "moe_wna16_marlin_gemm",
                f"moe_wna16_marlin_gemm<{args}>",
            )
        ],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device

    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx_or_none is not None
        and perm_or_none is not None
        and g_idx_or_none.numel() > 0
        and perm_or_none.numel() > 0
        and g_idx_or_none.size(-1) > 0
        and perm_or_none.size(-1) > 0
    )

    # Determine has_zp
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0

    # Determine has_bias
    has_bias = b_bias_or_none is not None

    # Derive num_groups and group_size from b_scales
    num_groups = b_scales.size(1)
    if has_act_order:
        if is_k_full:
            group_size = size_k // num_groups
        else:
            group_size = 0
    else:
        if num_groups > 1:
            group_size = size_k // num_groups
        else:
            group_size = -1

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m * top_k, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce and not use_atomic_add:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        # max num of threadblocks is sms * 4
        max_c_tmp_size = min(
            size_n * sorted_token_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Convert Optional tensors to empty tensors
    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
    perm_t = _or_empty(perm_or_none, device, torch.int32)
    b_zeros_t = _or_empty(b_zeros_or_none, device, a.dtype)
    b_bias_t = _or_empty(b_bias_or_none, device, a.dtype)
    global_scale_t = _or_empty(global_scale_or_none, device, a.dtype)

    module = _jit_moe_wna16_marlin_module(a.dtype)
    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias_t,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        a_tmp,
        c_tmp,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        has_act_order,
        has_bias,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c
