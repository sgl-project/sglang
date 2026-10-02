import ctypes
import functools
import inspect
import logging
import math
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.runtime import driver

logger = logging.getLogger(__name__)
__kernel_persistent_info = None
__valu_info = None
__valu_stream = None
__tfu_stream = None


def package_args(func: Callable) -> Callable:
    """
    pack the All arguments into a Python dictionary.
    """

    def wrapper(*args, **kwargs) -> Dict:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    return wrapper


@functools.lru_cache
def get_limit_blocks(n_regs: int, num_warps: int, smems: int, kernel_name: str) -> int:
    """
    calculate the limit blocks per multiproc for triton kernel
    """
    properties = driver.active.utils.get_device_properties(0)
    regs_per_multiproc = properties["max_num_regs"]  # 128*1024
    smem_per_multiproc = properties["max_shared_mem"]  # 256*1024
    thread_per_warp = properties["warpSize"]  # 32
    reg_alloc_unit_size = 64
    warp_alloc_granularity = 8
    smem_alloc_unit_size = 128
    max_threadblocks_per_multiproc = 64

    regs_per_warp = (
        math.ceil(n_regs * thread_per_warp / reg_alloc_unit_size) * reg_alloc_unit_size
    )
    regs_per_warp = max(regs_per_warp, reg_alloc_unit_size)
    warps_per_sm = (
        math.floor(regs_per_multiproc / regs_per_warp / warp_alloc_granularity)
        * warp_alloc_granularity
    )
    limit_blocks_due_to_regs = math.floor(warps_per_sm / num_warps)

    smem_per_block = math.ceil(smems / smem_alloc_unit_size) * smem_alloc_unit_size
    smem_per_block = max(smem_per_block, smem_alloc_unit_size)
    limit_blocks_due_to_smem = math.floor(smem_per_multiproc / smem_per_block)

    logger.info(
        f"kernel {kernel_name}: n_regs={n_regs}, num_warps={num_warps}, smems={smems}, limitBlocksDueToSMem={limit_blocks_due_to_smem}, limitBlocksDueToRegs={limit_blocks_due_to_regs}"
    )
    print(
        f"kernel {kernel_name}: n_regs={n_regs}, num_warps={num_warps}, smems={smems}, limitBlocksDueToSMem={limit_blocks_due_to_smem}, limitBlocksDueToRegs={limit_blocks_due_to_regs}"
    )

    return min(
        limit_blocks_due_to_regs,
        limit_blocks_due_to_smem,
        max_threadblocks_per_multiproc,
    )


def get_scheduled_grid(
    BLOCK_SIZE_N: int, N: int, num_cores: int, limit_blocks: int
) -> Tuple[int]:
    """
    columns wise Prior
    """
    n_tiles = triton.cdiv(N, BLOCK_SIZE_N)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    grid_x = gcd(n_tiles, num_cores * limit_blocks)
    if grid_x == 1:
        grid_x = min(n_tiles, num_cores * limit_blocks)
    grid_y = (num_cores * limit_blocks) // grid_x
    return (grid_x, grid_y)


def precompile_and_calculate(
    kernel_func: triton.JITFunction, all_args: Dict, num_sms: int = None
) -> Callable:
    """
    This function's purpose is to pre-compile the Triton kernel and calculate the grid parameters required for Persistence based on the compilation results.
    """
    compiled_kernel = kernel_func.warmup(
        *all_args["args"], grid=(1,), **all_args["kwargs"]
    )
    compiled_kernel._init_handles()
    num_regs = compiled_kernel.n_regs
    num_warps = compiled_kernel.metadata.num_warps
    shared_mem_size = compiled_kernel.metadata.shared
    kernel_name = compiled_kernel.metadata.name
    limit_blocks = get_limit_blocks(num_regs, num_warps, shared_mem_size, kernel_name)
    if num_sms:
        grid = lambda META: get_scheduled_grid(
            META["BLOCK_SIZE_N"], META["N"], num_sms, limit_blocks
        )
    else:
        num_sms = (
            20
            if "PPU-ZW810E" == torch.cuda.get_device_name()
            else torch.cuda.get_device_properties().multi_processor_count
        )
        grid = lambda META: (num_sms * math.ceil(limit_blocks * 1.0),)
    return grid


def set_valu_core(stream: torch.cuda.Stream, val: int):
    """
    val:
        0: set None
        1: set 32 valu CU # codespell:ignore
        2: set 24 valu CU # codespell:ignore
    """
    libcudart = ctypes.CDLL("libcuda.so")
    cudaStream_t = ctypes.c_void_p
    cuStreamSetAttributeAD = libcudart.cuStreamSetAttributeAD
    cuStreamSetAttributeAD.argtypes = [cudaStream_t, ctypes.c_int, ctypes.c_void_p]
    cuStreamSetAttributeAD.restype = ctypes.c_int
    stream_handle = cudaStream_t(stream.cuda_stream)
    yield_val = ctypes.c_int(val)
    cudaStreamLaunchAttrID_yield_multiprocessor = 2
    err = cuStreamSetAttributeAD(
        stream_handle,
        cudaStreamLaunchAttrID_yield_multiprocessor,
        ctypes.byref(yield_val),
    )
    return err


def get_work_streams() -> Tuple[torch.cuda.Stream]:
    """
    Return the two CUDA streams retained in the current context. If none exist, create them.
    """
    global __valu_stream, __tfu_stream
    if not __valu_stream:
        __valu_stream = torch.cuda.Stream()
    if not __tfu_stream:
        __tfu_stream = torch.cuda.Stream()
    return __tfu_stream, __valu_stream


@triton.jit
def bincount_kernel(topk_ids_ptr, count_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Input:
        topk_ids_ptr
        n_elements
    output:
        count_ptr
    """
    self_expert_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    expert_ids = tl.load(topk_ids_ptr + offsets, mask=(offsets < n_elements))
    key_mask = (self_expert_id == expert_ids) & (offsets < n_elements)
    count = tl.sum(tl.where(key_mask, 1, 0))
    tl.store(count_ptr + self_expert_id, count)


@triton.jit
def argsort_kernel(x_ptr, indices_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Input:
        x_ptr: a 1d Tensor
        n_elements: an integer
    output:
        indices_ptr: a 1d Tensor
    """

    pos = tl.program_id(0)
    if pos >= n_elements:
        return

    x_ptrs = x_ptr + tl.arange(0, BLOCK_SIZE)
    x_mask = tl.arange(0, BLOCK_SIZE) < n_elements
    x = tl.load(x_ptrs, mask=x_mask)
    val = tl.load(x_ptr + pos)

    cond_mask = (x < val) | ((x == val) & (tl.arange(0, BLOCK_SIZE) < pos))
    cond_mask = x_mask & cond_mask
    rank = tl.sum(tl.where(cond_mask, 1, 0))
    tl.store(indices_ptr + rank, pos)


@triton.jit
def find_critical_pos_kernel(
    token_counts_ptr,
    sorted_counts_ids_ptr,
    num_experts,
    valu_tokens,
    critical_pos_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Input:
        token_counts_ptr
        sorted_counts_ids_ptr
        num_experts
        valu_tokens
    output:
        critical_pos_ptr: The critical_pos refers to the position of the first expert in the sorted counts where its prefix sum exceeds the valu_tokens threshold.
    """
    pid = tl.program_id(0)
    if pid == 0:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_experts
        sorted_counts_ids = tl.load(
            sorted_counts_ids_ptr + offsets,
            mask=mask,
        )
        sorted_counts = tl.load(
            token_counts_ptr + sorted_counts_ids,
            mask=mask,
        )
        cumsum_counts = tl.cumsum(sorted_counts)
        cumsum_counts = tl.where(cumsum_counts < valu_tokens, cumsum_counts, -1)
        critical_pos = tl.argmax(cumsum_counts, axis=-1) + 1
        tl.store(critical_pos_ptr, critical_pos)


# for valu process # codespell:ignore
@triton.jit
def isin_and_gather_kernel(
    x_ptr,
    token_counts_ptr,
    sorted_counts_ids_ptr,
    critical_pos_ptr,
    sorted_token_ids,
    sorted_expert_ids,
    valid_tokens_ptr,
    EM,
    BLOCK_SIZE: tl.constexpr,
):
    """
    input:
        x_ptr: A Tensor
        token_counts_ptr: A Tensor
        sorted_counts_ids_ptr: A Tensor
        critical_pos_ptr: a (1, ) Tensor
        EM: an integer
    output:
        sorted_token_ids
        sorted_expert_ids
        valid_tokens_ptr
    """
    pos = tl.program_id(0)
    critical_pos = tl.load(critical_pos_ptr)

    if pos >= critical_pos:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = offsets < pos
    expert_ids = tl.load(
        sorted_counts_ids_ptr + offsets,
        mask=expert_mask,
    )
    token_counts = tl.load(token_counts_ptr + expert_ids, mask=expert_mask, other=0)
    out_offset = tl.sum(token_counts)

    self_expert_id = tl.load(sorted_counts_ids_ptr + pos)

    # gather
    x_mask = offsets < EM
    x = tl.load(x_ptr + offsets, mask=x_mask)
    key_mask = (x == self_expert_id) & x_mask
    ranks = tl.cumsum(tl.where(key_mask, 1, 0)) - 1

    tl.store(sorted_token_ids + out_offset + ranks, offsets, mask=key_mask)
    tl.store(sorted_expert_ids + out_offset + ranks, x, mask=key_mask)

    if pos == critical_pos - 1:
        valid_tokens = out_offset + tl.sum(tl.where(key_mask, 1, 0))
        tl.store(valid_tokens_ptr, valid_tokens)


# for tfu process
@triton.jit
def gather_and_align_block_size(
    x_ptr,
    token_counts_ptr,
    sorted_counts_ids_ptr,
    critical_pos_ptr,
    sorted_token_ids,
    sorted_expert_ids,
    num_tokens_post_padded_ptr,
    EM,
    num_experts,
    align_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    input:
        x_ptr: A Tensor
        token_counts_ptr: A Tensor
        sorted_counts_ids_ptr: A Tensor
        critical_pos_ptr: a (1, ) Tensor
        EM: an integer
        align_size: an integer
    output:
        sorted_token_ids
        sorted_expert_ids
        num_tokens_post_padded_ptr
    """
    pos = tl.program_id(0)
    critical_pos = tl.load(critical_pos_ptr)

    if pos < critical_pos:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = (offsets >= critical_pos) & (offsets < pos)
    expert_ids = tl.load(
        sorted_counts_ids_ptr + offsets,
        mask=expert_mask,
    )
    token_counts = tl.load(token_counts_ptr + expert_ids, mask=expert_mask, other=0)

    # padding
    token_counts_padding = tl.cdiv(token_counts, align_size) * align_size
    out_offset = tl.sum(token_counts_padding)

    self_expert_id = tl.load(sorted_counts_ids_ptr + pos)

    # gather tokens ids
    x_mask = offsets < EM
    x = tl.load(x_ptr + offsets, mask=x_mask)
    key_mask = (x == self_expert_id) & x_mask
    ranks = tl.cumsum(tl.where(key_mask, 1, 0)) - 1
    tl.store(sorted_token_ids + out_offset + ranks, offsets, mask=key_mask)

    # gather expert ids
    self_token_count = tl.sum(tl.where(key_mask, 1, 0))
    num_blocks = tl.cdiv(self_token_count, align_size)
    tl.store(
        sorted_expert_ids + tl.cdiv(out_offset, align_size) + offsets,
        self_expert_id,
        mask=(offsets < num_blocks),
    )

    # write padding
    padding_mask = (self_token_count <= offsets) & (offsets < num_blocks * align_size)
    tl.store(sorted_token_ids + out_offset + offsets, EM, mask=padding_mask)

    # write num_tokens after padding
    if pos == num_experts - 1:
        num_tokens_post_padded = out_offset + num_blocks * align_size
        tl.store(num_tokens_post_padded_ptr, num_tokens_post_padded)


def diverse_experts(
    topk_ids: torch.Tensor, num_experts: int, diverse_ratio: float, align_size: int
) -> Tuple[torch.Tensor]:
    """
    output:
        valu_expert_ids
        valu_sorted_token_ids
        valu_valid_tokens
        tfu_expert_ids
        tfu_sorted_token_ids
        num_tokens_post_padded
    input:
        topk_ids
        num_experts
        diverse_ratio
        align_size
    """
    m, topk = topk_ids.shape
    EM = m * topk
    valu_tokens = int(EM * diverse_ratio)

    # 1. do bincount
    bin_counts = torch.empty(
        (num_experts,), dtype=topk_ids.dtype, device=topk_ids.device
    )
    BLOCK_SIZE = triton.next_power_of_2(EM)
    grid = lambda META: (len(bin_counts),)
    bincount_kernel[grid](topk_ids.view(-1), bin_counts, EM, BLOCK_SIZE)

    # 2. argsort for bincount
    sorted_bin_ids = torch.empty_like(bin_counts)

    BLOCK_SIZE = triton.next_power_of_2(len(bin_counts))
    grid = lambda META: (len(bin_counts),)
    argsort_kernel[grid](bin_counts, sorted_bin_ids, num_experts, BLOCK_SIZE)

    # 3. find the critical position. the critical position can divide valu experts and tfu experts # codespell:ignore
    critical_pos = torch.empty((1,), dtype=topk_ids.dtype, device=topk_ids.device)
    BLOCK_SIZE = triton.next_power_of_2(len(bin_counts))
    grid = lambda META: (1,)
    find_critical_pos_kernel[grid](
        bin_counts, sorted_bin_ids, num_experts, valu_tokens, critical_pos, BLOCK_SIZE
    )

    tfu_strm, valu_strm = get_work_streams()
    # 4. valu prepare # codespell:ignore
    valu_sorted_token_ids = torch.empty(
        (valu_tokens,), dtype=topk_ids.dtype, device=topk_ids.device
    )
    valu_expert_ids = torch.empty(
        (valu_tokens,), dtype=topk_ids.dtype, device=topk_ids.device
    )
    valu_valid_tokens = torch.empty((1,), dtype=topk_ids.dtype, device=topk_ids.device)
    BLOCK_SIZE = triton.next_power_of_2(EM)
    grid = lambda META: (len(bin_counts),)

    set_valu_core(valu_strm, 0)
    valu_strm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(valu_strm):
        isin_and_gather_kernel[grid](
            topk_ids.view(-1),
            bin_counts,
            sorted_bin_ids,
            critical_pos,
            valu_sorted_token_ids,
            valu_expert_ids,
            valu_valid_tokens,
            EM,
            BLOCK_SIZE,
        )

    # 5. tfu prepare
    max_padding_size = EM + num_experts * (align_size - 1)
    tfu_sorted_token_ids = torch.empty(
        (max_padding_size,), dtype=topk_ids.dtype, device=topk_ids.device
    )
    tfu_expert_ids = torch.empty(
        triton.cdiv(max_padding_size, align_size),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        (1,), dtype=topk_ids.dtype, device=topk_ids.device
    )
    BLOCK_SIZE = triton.next_power_of_2(EM)
    grid = lambda META: (len(bin_counts),)

    set_valu_core(tfu_strm, 0)
    tfu_strm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(tfu_strm):
        gather_and_align_block_size[grid](
            topk_ids.view(-1),
            bin_counts,
            sorted_bin_ids,
            critical_pos,
            tfu_sorted_token_ids,
            tfu_expert_ids,
            num_tokens_post_padded,
            EM,
            num_experts,
            align_size,
            BLOCK_SIZE,
        )

    torch.cuda.current_stream().wait_stream(valu_strm)
    torch.cuda.current_stream().wait_stream(tfu_strm)

    return (
        valu_expert_ids,
        valu_sorted_token_ids,
        valu_valid_tokens,
        tfu_expert_ids,
        tfu_sorted_token_ids,
        num_tokens_post_padded,
    )


def align_size_for_gemv(topk_ids: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    When using the GEMV algorithm, no alignment operation is required.
    """
    m, topk = topk_ids.shape
    EM = m * topk
    valu_sorted_token_ids = torch.empty(
        (EM,), dtype=topk_ids.dtype, device=topk_ids.device
    )
    valu_expert_ids = topk_ids
    valu_valid_tokens = EM
    return valu_expert_ids, valu_sorted_token_ids, valu_valid_tokens, None, None, None


def align_size_for_groupedgemm(
    topk_ids: torch.Tensor, num_experts: int, align_size: int
) -> Tuple[torch.Tensor]:
    """
    Perform the alignment operation using the framework's default operator.
    """
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_align_block_size

    tfu_sorted_token_ids, tfu_expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, align_size, num_experts
    )
    return (
        None,
        None,
        None,
        tfu_expert_ids,
        tfu_sorted_token_ids,
        num_tokens_post_padded,
    )


def prepare_for_dispatch(
    topk_ids: torch.Tensor, config: Dict, num_experts: int
) -> Tuple[torch.Tensor]:
    """
    Select different preprocessing methods based on the JSON configuration.
    """
    assert "diverse_ratio" in config.keys(), (
        "A config json with version=2 need be offered when using FUSEDMOE_OPT=1"
    )
    assert 0.0 <= config["diverse_ratio"] <= 1.0
    diverse_ratio = float(
        os.environ.get("FUSEDMOE_DIVERSE_RATIO", config["diverse_ratio"])
    )
    config["diverse_ratio"] = diverse_ratio

    if config["diverse_ratio"] == 1.0:
        return align_size_for_gemv(topk_ids)
    else:
        assert (
            config["head"]["TensorCore"]["BLOCK_SIZE_M"]
            == config["tail"]["TensorCore"]["BLOCK_SIZE_M"]
        )
        block_size_m = config["head"]["TensorCore"]["BLOCK_SIZE_M"]
        if (
            config["diverse_ratio"] == 0.0
            or int(os.environ.get("FUSEDMOE_DISABLE_OVERLAP", 0)) == 1
        ):
            return align_size_for_groupedgemm(topk_ids, num_experts, block_size_m)
        else:
            return diverse_experts(topk_ids, num_experts, diverse_ratio, block_size_m)


def invoke_optimal_mixed_fused_moe_impl(
    tfu_args: Dict, valu_args: Dict, num_cuda_cores: int
) -> None:
    """
    The complete call flow for the parallel computation of the Valu and Tfu kernels. # codespell:ignore
    """
    valu_cores_dict = {24: 2, 32: 1, 64: 0}
    valu_cores = valu_cores_dict[num_cuda_cores]

    # tfu_grid = precompile_and_calculate(fused_moe_groupedgemm_persistent_kernel, tfu_args, 20)
    tfu_grid = precompile_and_calculate(fused_moe_persistence_kernel, tfu_args)
    valu_grid = precompile_and_calculate(
        fused_moe_gemv_persistent_kernel, valu_args, num_cuda_cores
    )

    tfu_strm, valu_strm = get_work_streams()
    valu_strm.wait_stream(torch.cuda.current_stream())
    tfu_strm.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(tfu_strm):
        # fused_moe_groupedgemm_persistent_kernel[tfu_grid](*tfu_args["args"], **tfu_args["kwargs"])
        fused_moe_persistence_kernel[tfu_grid](*tfu_args["args"], **tfu_args["kwargs"])

    set_valu_core(valu_strm, valu_cores)
    with torch.cuda.stream(valu_strm):
        fused_moe_gemv_persistent_kernel[valu_grid](
            *valu_args["args"], **valu_args["kwargs"]
        )

    torch.cuda.current_stream().wait_stream(valu_strm)
    torch.cuda.current_stream().wait_stream(tfu_strm)


def invoke_optimal_gemv_fused_moe_impl(valu_args: Dict) -> None:
    """
    run valu implementation # codespell:ignore
    """
    valu_grid = lambda META: (
        META["EM"],
        triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),
    )
    tfu_strm, valu_strm = get_work_streams()
    valu_strm.wait_stream(torch.cuda.current_stream())
    tfu_strm.wait_stream(torch.cuda.current_stream())
    set_valu_core(valu_strm, 0)
    with torch.cuda.stream(valu_strm):
        fused_moe_gemv_kernel[valu_grid](*valu_args["args"], **valu_args["kwargs"])
    torch.cuda.current_stream().wait_stream(valu_strm)
    torch.cuda.current_stream().wait_stream(tfu_strm)


def invoke_optimal_groupedgemm_fused_moe_impl(tfu_args: Dict) -> None:
    """
    run TensorCore implementation
    """
    if int(os.environ.get("TRITON_INTERPRET", 0)) == 1:
        tfu_grid = lambda META: (4 * 20,)
        fused_moe_persistence_kernel[tfu_grid](
            *tfu_args["args_dev"], **tfu_args["kwargs"]
        )
    else:
        tfu_grid = precompile_and_calculate(fused_moe_persistence_kernel, tfu_args)
        fused_moe_persistence_kernel[tfu_grid](*tfu_args["args"], **tfu_args["kwargs"])


def invoke_special_optimal_fused_moe_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    dispatch_tuple: Tuple[torch.Tensor],
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    ahead: bool,
) -> None:
    """
    Pack the parameters and select different implementation methods based on the dispatch ratio.
    """
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    M = A.shape[0]
    num_tokens = M * top_k
    (
        valu_expert_ids,
        valu_sorted_token_ids,
        valu_valid_tokens,
        tfu_expert_ids,
        tfu_sorted_token_ids,
        num_tokens_post_padded,
    ) = dispatch_tuple
    diverse_ratio = config["diverse_ratio"]
    config = config["head"] if ahead else config["tail"]
    tfu_config = config.get("TensorCore", dict())
    valu_config = config.get("CudaCore", dict())
    num_cuda_cores = config.get("num_cores", 0)

    valu_EM = 0 if valu_sorted_token_ids is None else valu_sorted_token_ids.shape[0]
    tfu_EM = 0 if tfu_sorted_token_ids is None else tfu_sorted_token_ids.shape[0]

    if tfu_EM > 0:
        tfu_config["SWIZZLE_SIZE"] = triton.cdiv(B.shape[1], tfu_config["BLOCK_SIZE_N"])
        tfu_config["num_sms"] = (
            20
            if "PPU-ZW810E" == torch.cuda.get_device_name()
            else torch.cuda.get_device_properties().multi_processor_count
        )
        if int(os.environ.get("FUSEDMOE_USE_AIU", "1")) == 1:
            tfu_config["use_aiu"] = B.shape[2] % tfu_config["BLOCK_SIZE_K"] == 0
        else:
            tfu_config["use_aiu"] = False

    # package for Convention
    tfu_args = package_args(fused_moe_persistence_kernel[1])(
        A,
        B,
        C,
        topk_weights,
        tfu_sorted_token_ids,
        tfu_expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        tfu_EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **tfu_config,
    )

    valu_args = package_args(fused_moe_gemv_persistent_kernel[1])(
        A,
        B,
        C,
        topk_weights,
        valu_sorted_token_ids,
        valu_expert_ids,
        valu_valid_tokens,
        B.shape[1],
        B.shape[2],
        valu_EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **valu_config,
    )

    if diverse_ratio == 1.0:
        invoke_optimal_gemv_fused_moe_impl(valu_args=valu_args)
    elif (
        diverse_ratio == 0.0 or int(os.environ.get("FUSEDMOE_DISABLE_OVERLAP", 0)) == 1
    ):
        invoke_optimal_groupedgemm_fused_moe_impl(tfu_args=tfu_args)
    else:
        invoke_optimal_mixed_fused_moe_impl(
            tfu_args=tfu_args, valu_args=valu_args, num_cuda_cores=num_cuda_cores
        )


def add_descriptor_to_name(prefix_name: str) -> Callable:
    """
    This function's purpose is to generate a specific name based on the constant parameters of the Triton kernel.

    """

    def descriptor_wrapper(proxy) -> str:
        constants = proxy.constants
        descriptors = list()
        descriptors.append(f"mul{int(constants['MUL_ROUTED_WEIGHT'])}")
        descriptors.append(f"{str(constants['compute_type'])}")
        descriptors.append(f"aiu{int(constants.get('use_aiu', 0))}")
        descriptors.append(f"warp{constants['num_warps']}")
        descriptors.append(f"stage{constants['num_stages']}")
        descriptors.append(f"m{constants['BLOCK_SIZE_M']}")
        descriptors.append(f"n{constants['BLOCK_SIZE_N']}")
        descriptors.append(f"k{constants['BLOCK_SIZE_K']}")
        descriptors.append(f"sw{constants.get('SWIZZLE_SIZE', 1)}")
        descriptors.append(f"ur{constants['loop_unroll_factor']}")
        descriptors.append(f"K{constants['K']}N{constants['N']}")
        return f"{prefix_name}_{'_'.join(descriptors)}"

    return descriptor_wrapper


@triton.jit
def valu_dot(a, b, Tile_M: tl.constexpr, Tile_N: tl.constexpr, Tile_K: tl.constexpr):
    b = b.trans()
    a = a.reshape(Tile_M, 1, Tile_K)
    b = b.reshape(1, Tile_N, Tile_K)
    a = a.to(tl.float32)
    b = b.to(tl.float32)
    ss = a * b
    return tl.sum(ss, 2)


@triton.jit
def fma_dot(a, b, Tile_M: tl.constexpr, Tile_N: tl.constexpr, Tile_K: tl.constexpr):
    acc = tl.zeros((Tile_M, Tile_N), dtype=tl.float32)
    for t in tl.static_range(Tile_K):
        a_val = a[:, t][:, None]
        b_val = b[t, :][None, :]
        acc = tl.math.fma(a_val, b_val, acc)
    return acc


@triton.jit(repr=add_descriptor_to_name("moe_ultragemv"))
def fused_moe_gemv_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr = 1,
    loop_unroll_factor: tl.constexpr = 4,
    use_id_map: tl.constexpr = False,
):
    """ """
    pid_m = tl.program_id(axis=0)  # M
    pid_n = tl.program_id(axis=1)  # N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if use_id_map:
        offs_token = tl.load(
            sorted_token_ids_ptr + pid_m,
        ).to(tl.int64)
    else:
        offs_token = pid_m + tl.arange(0, 1).to(tl.int64)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    off_experts = tl.load(
        expert_ids_ptr + pid_m,
    ).to(tl.int64)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(
        0,
        tl.cdiv(K, BLOCK_SIZE_K),
        num_stages=num_stages,
        loop_unroll_factor=loop_unroll_factor,
    ):
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            cache_modifier=".ca",
            eviction_policy="evict_last",
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, cache_modifier=".cg"
        )

        accumulator += valu_dot(a, b, 1, BLOCK_SIZE_N, BLOCK_SIZE_K)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = offs_cn[None, :] < N
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit(repr=add_descriptor_to_name("moe_gemv"))
def fused_moe_gemv_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr = 1,
    loop_unroll_factor: tl.constexpr = 4,
    use_id_map: tl.constexpr = True,
):

    pid_m = tl.program_id(axis=1)  # M
    pid_n = tl.program_id(axis=0)  # N (0 ~ 3)
    num_blk_along_M = tl.num_programs(axis=1)
    num_blk_along_N = tl.num_programs(axis=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    num_tokens_post_padded = tl.load(
        num_tokens_post_padded_ptr,
    )

    for tid_m in tl.range(
        pid_m,
        num_tokens_post_padded,
        num_blk_along_M,
        num_stages=1,
        loop_unroll_factor=1,
    ):
        for tid_n in tl.range(
            pid_n,
            tl.cdiv(N, BLOCK_SIZE_N),
            num_blk_along_N,
            num_stages=1,
            loop_unroll_factor=1,
        ):
            if use_id_map:
                offs_token = tl.load(
                    sorted_token_ids_ptr + tid_m,
                ).to(tl.int64)
            else:
                offs_token = tid_m + tl.arange(0, 1).to(tl.int64)

            a_ptrs = a_ptr + (
                offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
            )

            offs_bn = (
                tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            ) % N
            off_experts = tl.load(
                expert_ids_ptr + tid_m,
            ).to(tl.int64)
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )

            accumulator = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)

            for k in tl.range(
                0,
                tl.cdiv(K, BLOCK_SIZE_K),
                num_stages=num_stages,
                loop_unroll_factor=loop_unroll_factor,
            ):
                a = tl.load(
                    a_ptrs,
                    mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                    cache_modifier=".ca",
                    eviction_policy="evict_last",
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    cache_modifier=".cg",
                    eviction_policy="evict_first",
                )
                accumulator += valu_dot(a, b, 1, BLOCK_SIZE_N, BLOCK_SIZE_K)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token)
                accumulator = accumulator * moe_weight[:, None]
            accumulator = accumulator.to(compute_type)

            offs_cn = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
            c_mask = offs_cn[None, :] < N
            tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit(repr=add_descriptor_to_name("moe_groupedgemm"))
def fused_moe_groupedgemm_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr = 2,
    loop_unroll_factor: tl.constexpr = 4,
    num_sms: tl.constexpr = 20,
    SWIZZLE_SIZE: tl.constexpr = 1,
    use_aiu: tl.constexpr = True,
):

    pid_m = tl.program_id(axis=1)  # M
    pid_n = tl.program_id(axis=0)  # N (0 ~ 3)
    num_blk_along_M = tl.num_programs(axis=1)
    num_blk_along_N = tl.num_programs(axis=0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_tokens_post_padded = tl.load(
        num_tokens_post_padded_ptr,
    )

    for tid_m in tl.range(
        pid_m,
        tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M),
        num_blk_along_M,
        num_stages=1,
        loop_unroll_factor=1,
    ):
        for tid_n in tl.range(
            pid_n,
            tl.cdiv(N, BLOCK_SIZE_N),
            num_blk_along_N,
            num_stages=1,
            loop_unroll_factor=2,
        ):
            offs_token = tl.load(
                sorted_token_ids_ptr
                + tid_m * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M),
            ).to(tl.int64)
            token_mask = offs_token < num_valid_tokens
            a_ptrs = a_ptr + (
                offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
            )

            off_experts = tl.load(
                expert_ids_ptr + tid_m,
            ).to(tl.int64)
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr + off_experts * stride_be,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(0, tid_n * BLOCK_SIZE_N),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                order=(0, 1),
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in tl.range(
                0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=num_stages, loop_unroll_factor=1
            ):
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                    cache_modifier=".ca",
                    eviction_policy="evict_first",
                )
                b = tl.aiu_load(b_block_ptr)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(
                    topk_weights_ptr + offs_token, mask=token_mask, other=0
                )
                accumulator = accumulator * moe_weight[:, None]
            accumulator = accumulator.to(compute_type)

            offs_cn = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def swizzle1d(pid, M: tl.constexpr, N: tl.constexpr, group_size_m: tl.constexpr):
    group_size = N * group_size_m
    group_id = pid // group_size
    # group_id_m = (pid // N) // group_size_m
    base_in_group = group_id * group_size
    off_in_group = pid - base_in_group
    size_m = min(M - base_in_group // N, group_size_m)
    off_m_in_group = off_in_group // N
    off_n_in_group = off_in_group % N
    return base_in_group + off_n_in_group * size_m + off_m_in_group


@triton.jit(repr=add_descriptor_to_name("moe_persistence"))
def fused_moe_persistence_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr = 2,
    loop_unroll_factor: tl.constexpr = 4,
    num_sms: tl.constexpr = 20,
    SWIZZLE_SIZE: tl.constexpr = 1,
    use_aiu: tl.constexpr = True,
):

    pid = tl.program_id(axis=0)
    num_workers = tl.num_programs(axis=0)
    swizzle_pid = swizzle1d(pid, num_workers // num_sms, num_sms, SWIZZLE_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    grid_size_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    grid_size_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_tiles = grid_size_m * grid_size_n

    loop_k_iters: tl.constexpr = triton.cdiv(K, BLOCK_SIZE_K)
    faltten: tl.constexpr = loop_k_iters < 4
    unroll: tl.constexpr = loop_k_iters if faltten else 1

    for tid in tl.range(
        swizzle_pid,
        num_tiles,
        num_workers,
        num_stages=num_stages,
        loop_unroll_factor=unroll,
        flatten=faltten,
    ):
        tid_m = tid // grid_size_n
        tid_n = tid % grid_size_n

        offs_token = tl.load(
            sorted_token_ids_ptr + tid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M),
        ).to(tl.int64)
        token_mask = offs_token < num_valid_tokens
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        off_experts = tl.load(
            expert_ids_ptr + tid_m,
        ).to(tl.int64)

        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + off_experts * stride_be,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, (tid_n * BLOCK_SIZE_N).to(tl.int32)),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(0, 1),
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in tl.range(0, loop_k_iters):
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            )
            if use_aiu:
                b = tl.aiu_load(b_block_ptr)
            else:
                b = tl.load(b_block_ptr, boundary_check=(0, 1))

            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            accumulator = accumulator * moe_weight[:, None]
        accumulator = accumulator.to(compute_type)

        offs_cn = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
