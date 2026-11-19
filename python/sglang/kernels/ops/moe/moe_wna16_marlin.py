from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import msgspec
import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

# Constants matching device::marlin_moe:: in marlin.cuh
_MAX_THREAD_N = 256


@cache_once
def _jit_moe_wna16_marlin_module(
    dtype: torch.dtype,
    is_ep: bool,
    has_bias: bool,
    expert_ids_are_sorted: bool = False,
    zero_inactive_output: bool = False,
    has_inactive_prefix_count: bool = False,
) -> Module:
    args = make_cpp_args(
        dtype,
        is_ep,
        has_bias,
        expert_ids_are_sorted,
        zero_inactive_output,
        has_inactive_prefix_count,
    )
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


class HostCopy(msgspec.Struct, frozen=True):
    """Paged experts, fused host copy: every expert of one GEMM's slot tables in pinned host
    memory, so that the kernel pages an expert in itself (marlin_template.h)."""

    weight: int  # device address of the host table of packed weights
    scales: int  # device address of the host table of block scales
    global_scale: (
        int  # device address of the host table of NVFP4 global scales; 0 without them
    )
    num_experts: int
    expert_slot: torch.Tensor  # [E] int32: the slot each expert is paged into
    count: torch.Tensor  # [1] int32: experts the current step pages in


# By the address of the slot table of packed weights; set by sglang.srt.layers.moe.paged_experts.
HOST_COPIES: Dict[int, HostCopy] = {}


def host_expert_ids(b_q_weight: torch.Tensor) -> int:
    """How many expert ids past its slots the routed entries of ``b_q_weight`` may use: with a host
    copy an id of K + e is expert e, read from host memory; 0 without one."""
    host = HOST_COPIES.get(b_q_weight.data_ptr()) if HOST_COPIES else None
    return 0 if host is None else host.num_experts


def _host_copy_args(
    host: Optional[HostCopy], b_q_weight: torch.Tensor, b_scales: torch.Tensor
) -> tuple:
    """The fused-host-copy kernel's extra arguments: the host tables as offsets from the slot
    tables in 16-byte units, then addresses and the slot count. None without a host copy."""
    if host is None:
        return ()
    return (
        (host.weight - b_q_weight.data_ptr()) // 16,
        (host.scales - b_scales.data_ptr()) // 16,
        host.global_scale,
        host.expert_slot.data_ptr(),
        b_q_weight.shape[0],
        host.count.data_ptr(),
    )


@cache_once
def _jit_moe_wna16_marlin_host_copy_module(
    dtype: torch.dtype,
    is_ep: bool,
    has_bias: bool,
    expert_ids_are_sorted: bool = False,
    zero_inactive_output: bool = False,
    has_inactive_prefix_count: bool = False,
) -> Module:
    """The kernel with the fused host copy: the same source built with SGL_MARLIN_MOE_HOST_COPY.
    It is compiled as a separate module, so that the module above, used without the host copy, is
    built only from the original code."""
    args = make_cpp_args(
        dtype,
        is_ep,
        has_bias,
        expert_ids_are_sorted,
        zero_inactive_output,
        has_inactive_prefix_count,
    )
    return load_jit(
        "moe_wna16_marlin_host_copy",
        *args,
        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],
        cuda_wrappers=[
            (
                "moe_wna16_marlin_gemm",
                f"moe_wna16_marlin_gemm<{args}>",
            )
        ],
        extra_cuda_cflags=["-DSGL_MARLIN_MOE_HOST_COPY"],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
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
    expert_ids_are_sorted: bool = False,
    zero_inactive_output: bool = False,
    has_inactive_prefix_count: bool = False,
) -> torch.Tensor:
    """Project routes, preserving inactive rows in a caller-provided output.

    Inactive rows are zero when this function allocates the output. With
    ``zero_inactive_output=True``, the GEMM also zeros caller-owned rows present
    in inactive routing blocks; omitted routes remain untouched.
    ``expert_ids_are_sorted`` certifies ascending block expert IDs within the
    padded token count, with any -1 blocks first. Unused tail entries are ignored.
    Leave it false after remapping that can produce scattered inactive blocks.
    The sorted specialization is only used without expert parallelism.
    ``has_inactive_prefix_count`` additionally certifies that a supporting
    producer wrote an aligned int32 pair ``[padded_tokens, inactive_blocks]``
    for these tables and this block size. It is ignored outside sorted non-EP.
    """
    device = a.device

    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.zeros((size_m * top_k, size_n), dtype=a.dtype, device=device)

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
        sms = current_arch().num_sms
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

    host = HOST_COPIES.get(b_q_weight.data_ptr()) if HOST_COPIES else None
    load_module = (
        _jit_moe_wna16_marlin_module
        if host is None
        else _jit_moe_wna16_marlin_host_copy_module
    )
    module = load_module(
        a.dtype,
        is_ep,
        has_bias,
        expert_ids_are_sorted and not is_ep,
        zero_inactive_output,
        has_inactive_prefix_count and expert_ids_are_sorted and not is_ep,
    )
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
        *_host_copy_args(host=host, b_q_weight=b_q_weight, b_scales=b_scales),
    )

    return c
