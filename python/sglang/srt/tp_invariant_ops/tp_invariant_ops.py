import contextlib
import math
import os
import sys
from typing import Any, Callable, Dict

import torch
import torch.distributed as dist
import triton
import triton.language as tl

# Triton's constexpr tree unrolling causes deep AST recursion in the JIT
# compiler.  The two-level tree (v2) bounds compilation depth to
# max(log2(SUBTREE), log2(E/SUBTREE)) instead of log2(E), but we still
# need headroom for the per-level AST visitor overhead.
if sys.getrecursionlimit() < 16384:
    sys.setrecursionlimit(16384)


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def _get_tl_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16


# ---- kernel ----
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tp_persistent(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    LEVEL_K: tl.constexpr,
    TILE_K: tl.constexpr,
    FIRST_LEVEL_BLOCK: tl.constexpr,
    NEXT_POWER_OF_LEVEL: tl.constexpr,
    NEXT_POWER_OF_REMAIN_LEVEL: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    manual_acc = 3
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    S = tl.zeros((NEXT_POWER_OF_REMAIN_LEVEL, BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    S_mask = tl.arange(0, NEXT_POWER_OF_REMAIN_LEVEL)[:, None, None]
    level_ids = tl.arange(0, NEXT_POWER_OF_LEVEL)

    base_offs_m = tl.arange(0, BLOCK_M)
    base_offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )

        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N

        offs_am = start_m + base_offs_m
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)

        offs_bn = start_n + base_offs_n
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)

        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk

        count = tl.zeros((NEXT_POWER_OF_LEVEL,), dtype=tl.int32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        for s_tile_idx in range(0, TILE_K):
            k0 = s_tile_idx * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & ((k0 + offs_k)[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=((k0 + offs_k)[:, None] < K) & (offs_bn[None, :] < N),
                other=0.0,
            )
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

            acc = tl.dot(a, b).to(ACC_DTYPE)

            break_flag = 0
            for level in range(LEVEL_K):
                if break_flag == 0:
                    idx_mask = level_ids == level

                    count_value_added = tl.sum(count * idx_mask) + 1

                    table_value = FIRST_LEVEL_BLOCK if level == 0 else 2

                    carry_over = (table_value == count_value_added).to(tl.int1)

                    if count_value_added > 1:
                        if level == 0:
                            acc = acc1 + acc
                        elif level == 1:
                            acc = acc2 + acc
                        elif level == 2:
                            acc = acc3 + acc
                        else:
                            tmp_acc_mask = S_mask == (level - manual_acc)
                            acc = (
                                tl.sum(S * tmp_acc_mask, axis=0, dtype=ACC_DTYPE) + acc
                            )

                    count = tl.where(
                        idx_mask, count_value_added * (1 - carry_over), count
                    )
                    if not carry_over:
                        break_flag = 1
                        if level == 0:
                            acc1 = acc
                        elif level == 1:
                            acc2 = acc
                        elif level == 2:
                            acc3 = acc
                        else:
                            tmp_acc_mask = S_mask == (level - manual_acc)
                            S = tl.where(tmp_acc_mask, acc[None, :, :], S)

        c_ptr = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
        offs_cm = start_m + base_offs_m
        offs_cn = start_n + base_offs_n
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        offs_cm = tl.where(offs_cm < M, offs_cm, 0)
        offs_cn = tl.where(offs_cn < N, offs_cn, 0)
        offs_cm = tl.max_contiguous(tl.multiple_of(offs_cm, BLOCK_M), BLOCK_M)
        offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_N), BLOCK_N)
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptr, acc.to(OUT_DTYPE), mask=mask_c)


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tp_persistent_optim(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    LEVEL_K: tl.constexpr,
    TILE_K: tl.constexpr,
    FIRST_LEVEL_BLOCK: tl.constexpr,
    NEXT_POWER_OF_LEVEL: tl.constexpr,
    NEXT_POWER_OF_REMAIN_LEVEL: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Flat loop (same structure as original) with inlined level-0 handling.
    # Level-0 uses scalar counter; tree merge (levels 1+) only on carry.
    # Accumulators are conditionally allocated based on LEVEL_K (constexpr)
    # to minimize register pressure and maximize occupancy.
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    acc5 = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    S = tl.zeros((NEXT_POWER_OF_REMAIN_LEVEL, BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    S_mask = tl.arange(0, NEXT_POWER_OF_REMAIN_LEVEL)[:, None, None]
    level_ids = tl.arange(0, NEXT_POWER_OF_LEVEL)

    offs_k = tl.arange(0, BLOCK_K)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )

        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N

        offs_am = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_am < M
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        offs_am = tl.where(mask_m, offs_am, 0)

        offs_bn = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_bn < N
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_bn = tl.where(mask_n, offs_bn, 0)

        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
        mask_m_bc = mask_m[:, None]
        mask_n_bc = mask_n[None, :]

        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk

        c0 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0
        count = tl.zeros((NEXT_POWER_OF_LEVEL,), dtype=tl.int32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
        for _ in range(0, TILE_K):
            a = tl.load(a_ptrs, mask=mask_m_bc, other=0.0)
            b = tl.load(b_ptrs, mask=mask_n_bc, other=0.0)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

            acc = tl.dot(a, b).to(ACC_DTYPE)

            c0 += 1
            if c0 > 1:
                acc = acc1 + acc
            if c0 < FIRST_LEVEL_BLOCK:
                acc1 = acc
            else:
                c0 = 0
                break_flag = 0
                for level in range(1, LEVEL_K):
                    if break_flag == 0:
                        if level == 1:
                            c1 += 1
                            if c1 > 1:
                                acc = acc2 + acc
                            if c1 == 2:
                                c1 = 0
                            else:
                                acc2 = acc
                                break_flag = 1
                        elif level == 2:
                            c2 += 1
                            if c2 > 1:
                                acc = acc3 + acc
                            if c2 == 2:
                                c2 = 0
                            else:
                                acc3 = acc
                                break_flag = 1
                        elif level == 3:
                            c3 += 1
                            if c3 > 1:
                                acc = acc4 + acc
                            if c3 == 2:
                                c3 = 0
                            else:
                                acc4 = acc
                                break_flag = 1
                        elif level == 4:
                            c4 += 1
                            if c4 > 1:
                                acc = acc5 + acc
                            if c4 == 2:
                                c4 = 0
                            else:
                                acc5 = acc
                                break_flag = 1
                        else:
                            idx_mask = level_ids == level
                            count_value_added = tl.sum(count * idx_mask) + 1
                            carry_over = (2 == count_value_added).to(tl.int1)
                            if count_value_added > 1:
                                tmp_acc_mask = S_mask == (level - 5)
                                acc = (
                                    tl.sum(S * tmp_acc_mask, axis=0, dtype=ACC_DTYPE)
                                    + acc
                                )
                            count = tl.where(
                                idx_mask, count_value_added * (1 - carry_over), count
                            )
                            if not carry_over:
                                break_flag = 1
                                tmp_acc_mask = S_mask == (level - 5)
                                S = tl.where(tmp_acc_mask, acc[None, :, :], S)

        offs_cm = start_m + tl.arange(0, BLOCK_M)
        offs_cn = start_n + tl.arange(0, BLOCK_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        offs_cm = tl.where(mask_m, offs_cm, 0)
        offs_cn = tl.where(mask_n, offs_cn, 0)
        offs_cm = tl.max_contiguous(tl.multiple_of(offs_cm, BLOCK_M), BLOCK_M)
        offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_N), BLOCK_N)
        c_ptr = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        mask_c = mask_m_bc & mask_n_bc
        tl.store(c_ptr, acc.to(OUT_DTYPE), mask=mask_c)


def _matmul_tp_persistent_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor = None,
    fp32_accum: bool = False,
    use_optim_kernel: bool = False,
):
    assert A.shape[-1] == B.shape[-2], "Dim doesn't match"

    out_dtype = A.dtype
    acc_dtype = torch.float32 if fp32_accum else A.dtype

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            ),
        )

    base_configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 2,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 2,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 2,
            "num_warps": 8,
        },
    }

    configs = base_configs
    BLOCK_M = configs[out_dtype]["BLOCK_SIZE_M"]
    BLOCK_N = configs[out_dtype]["BLOCK_SIZE_N"]
    BLOCK_K = configs[out_dtype]["BLOCK_SIZE_K"]
    GROUP_SIZE_M = configs[out_dtype]["GROUP_SIZE_M"]
    num_stages = configs[out_dtype]["num_stages"]
    num_warps = configs[out_dtype]["num_warps"]

    M, K = A.shape
    _, N = B.shape
    assert (
        K % BLOCK_K == 0
    ), f"Dimension K should be divisible by BLOCK_K. Got K={K}, BLOCK_K={BLOCK_K}."
    T = K // BLOCK_K
    FIRST_LEVEL_BLOCK = T

    if use_optim_kernel:
        num_n_tiles = triton.cdiv(N, BLOCK_N)
        total_tiles = triton.cdiv(M, BLOCK_M) * num_n_tiles

        if total_tiles * 4 <= NUM_SMS:
            while BLOCK_M > 16 and triton.cdiv(M, BLOCK_M) * num_n_tiles < NUM_SMS:
                BLOCK_M //= 2
        elif total_tiles * 2 <= NUM_SMS:
            while BLOCK_M > 32 and triton.cdiv(M, BLOCK_M) * num_n_tiles < NUM_SMS:
                BLOCK_M //= 2

        if out_dtype in (torch.bfloat16, torch.float16):
            num_warps = 4 if BLOCK_M <= 16 else 8
            num_stages = 3 if K >= 1024 else 2

    LEVEL_K = 1
    while FIRST_LEVEL_BLOCK > 2 and FIRST_LEVEL_BLOCK % 2 == 0:
        FIRST_LEVEL_BLOCK //= 2
        LEVEL_K += 1

    C = torch.empty((M, N), device=A.device, dtype=out_dtype)

    # Original kernel manually handles levels 0-2 (3 accumulators);
    # optim kernel handles levels 0-4 (5 register accumulators),
    # so S tensor is smaller / unused for common LEVEL_K <= 5.
    manual_acc = 5 if use_optim_kernel else 3

    NEXT_POWER_OF_LEVEL = 2 ** math.ceil(math.log2(LEVEL_K))
    NEXT_POWER_OF_REMAIN_LEVEL = (
        2 ** math.ceil(math.log2(LEVEL_K - manual_acc)) if LEVEL_K > manual_acc else 1
    )

    kernel = (
        matmul_kernel_tp_persistent_optim
        if use_optim_kernel
        else matmul_kernel_tp_persistent
    )
    kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        NEXT_POWER_OF_REMAIN_LEVEL=NEXT_POWER_OF_REMAIN_LEVEL,
        LEVEL_K=LEVEL_K,
        TILE_K=T,
        FIRST_LEVEL_BLOCK=FIRST_LEVEL_BLOCK,
        NEXT_POWER_OF_LEVEL=NEXT_POWER_OF_LEVEL,
        ACC_DTYPE=_get_tl_dtype(acc_dtype),
        OUT_DTYPE=_get_tl_dtype(out_dtype),
        A_LARGE=A.numel() > 2**31,
        B_LARGE=B.numel() > 2**31,
        C_LARGE=C.numel() > 2**31,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if bias is not None:
        C += bias
    return C


def matmul_tp_persistent(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor = None,
    fp32_accum: bool = False,
):
    return _matmul_tp_persistent_impl(
        A=A,
        B=B,
        bias=bias,
        fp32_accum=fp32_accum,
        use_optim_kernel=False,
    )


def matmul_tp_inv(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor = None,
    fp32_accum: bool = False,
):
    return matmul_tp_persistent(A, B, bias=bias, fp32_accum=fp32_accum)


def matmul_tp_persistent_optim(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor = None,
    fp32_accum: bool = False,
):
    return _matmul_tp_persistent_impl(
        A=A,
        B=B,
        bias=bias,
        fp32_accum=fp32_accum,
        use_optim_kernel=True,
    )


def tree_all_reduce_sum(x: torch.Tensor, device_group=None) -> torch.Tensor:
    rank = dist.get_rank(device_group)
    world_size = dist.get_world_size(device_group)

    if world_size & (world_size - 1) != 0:
        raise ValueError(
            "world_size must be a power of 2 in order to use all_reduce_sum."
        )

    result = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(result, x, group=device_group)

    for level in range(1, world_size.bit_length()):
        for left in range(0, world_size, 1 << level):
            right = left + (1 << (level - 1))
            result[left] += result[right]

    return result[0]


def tree_all_reduce_sum_optim(x: torch.Tensor, device_group=None) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor.")
    if not x.is_contiguous():
        raise ValueError(
            "x must be contiguous. Call x = x.contiguous() OUTSIDE graph capture."
        )

    world_size = dist.get_world_size(device_group)
    if world_size & (world_size - 1) != 0:
        raise ValueError("world_size must be a power of 2.")

    # cache
    if not hasattr(tree_all_reduce_sum, "_cache"):
        tree_all_reduce_sum._cache = {}

    key = (id(device_group), x.device.index, tuple(x.shape), x.dtype, world_size)
    st = tree_all_reduce_sum._cache.get(key)
    if st is None:
        gather = torch.empty(
            (world_size,) + tuple(x.shape), device=x.device, dtype=x.dtype
        )
        out = torch.empty_like(x)
        st = tree_all_reduce_sum._cache[key] = (gather, out)

    gather, out = st

    # 1) all_gather into one contiguous buffer
    dist.all_gather_into_tensor(gather, x, group=device_group)

    # 2) deterministic tree pairing EXACTLY like your original:
    # for level in range(1, bit_length):
    #   for left in range(0, world_size, 1<<level):
    #       right = left + (1<<(level-1))
    #       gather[left] += gather[right]
    #
    # vectorized form: left indices are 0:step: , right are half:step:
    levels = world_size.bit_length()
    for level in range(1, levels):
        step = 1 << level
        half = step >> 1
        # Views only (no alloc); one add_ kernel per level
        gather[0:world_size:step].add_(gather[half:world_size:step])

    out.copy_(gather[0])
    tree_all_reduce_sum._cache.clear()
    return out


_tp_inv_MODE = False

try:
    def_lib = torch.library.Library("tp_inv_ops", "DEF")
    def_lib.define("matmul_tp_inv(Tensor a, Tensor b, Tensor? bias=None) -> Tensor")
except RuntimeError:
    pass

try:
    impl = torch.library.Library("tp_inv_ops", "IMPL")

    impl.impl("matmul_tp_inv", matmul_tp_persistent, "CUDA")
except RuntimeError:
    pass


def is_tp_invariant_mode_enabled():
    return _tp_inv_MODE


def enable_tp_invariant_mode():
    global _tp_inv_MODE

    if _tp_inv_MODE:
        return

    _tp_inv_MODE = True


def disable_tp_invariant_mode():
    global _tp_inv_MODE

    _tp_inv_MODE = False


@contextlib.contextmanager
def set_tp_invariant_mode(enabled=True):
    global _tp_inv_MODE

    old_state = _tp_inv_MODE

    if enabled:
        enable_tp_invariant_mode()
    else:
        disable_tp_invariant_mode()

    try:
        yield
    finally:
        _tp_inv_MODE = old_state


def scatter_input_by_local_expert(
    topk: torch.Tensor, input: torch.Tensor, E: int
) -> torch.Tensor:
    """
    Args:
        topk: [M, topk], long, -1 means remote expert
        input: [M, topk, hidden_size], float
        E: int, number of local experts (expert ids in [0, E))
    Returns:
        output: [M, E, hidden_size]
    """
    M, _, hidden_size = input.shape

    # Mask out remote experts in output
    valid = (topk != -1).unsqueeze(-1)  # [M, topk, 1]
    output_masked = input * valid.to(input.dtype)  # [M, topk, hidden_size]

    # Replace -1 with 0 for safe indexing (value doesn't matter because output is zero)
    topk_index = topk.clamp(min=0)  # turns -1 into 0, leaves others unchanged

    # Expand index to match output
    index = topk_index.unsqueeze(-1).expand(
        -1, -1, hidden_size
    )  # [M, topk, hidden_size]

    # Initialize result
    output = torch.zeros(M, E, hidden_size, device=input.device, dtype=input.dtype)

    # Scatter add
    output.scatter_add_(1, index, output_masked)

    return output


@triton.jit
def _load_expert_tile(
    input_ptr,
    input_base,
    input_stride_1,
    topk_ids_base,
    topk_ids_stride_1,
    zero_ptrs,
    offs_dim,
    mask,
    mask_token,
    e: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Load tile for expert e: search topk_ids for slot, redirect to zero_buf if remote."""
    found_slot = tl.full([BLOCK_M], -1, dtype=tl.int32)
    for k in range(TOPK):
        kid = tl.load(
            topk_ids_base + k * topk_ids_stride_1,
            mask=mask_token,
            other=-1,
        ).to(tl.int32)
        match = (kid == e) & (found_slot == -1)
        found_slot = tl.where(match, k, found_slot)

    is_valid = found_slot != -1
    slot_safe = tl.maximum(found_slot, 0)
    input_ptrs = input_base + slot_safe[:, None] * input_stride_1 + offs_dim[None, :]
    load_ptrs = tl.where(is_valid[:, None], input_ptrs, zero_ptrs)
    return tl.load(load_ptrs, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _tree_reduce_pair(
    input_ptr,
    input_base,
    input_stride_1,
    topk_ids_base,
    topk_ids_stride_1,
    zero_ptrs,
    offs_dim,
    mask,
    mask_token,
    start: tl.constexpr,
    size: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Recursively compute binary-tree sum over experts [start, start+size).
    Returns a fp32 tile [BLOCK_M, BLOCK_DIM].

    Tree structure (e.g. size=4, start=0):
      _tree_reduce_pair(0, 4)
        = _tree_reduce_pair(0, 2) + _tree_reduce_pair(2, 2)
        = (_tree_reduce_pair(0,1) + _tree_reduce_pair(1,1))
          + (_tree_reduce_pair(2,1) + _tree_reduce_pair(3,1))
        = (load(e0) + load(e1)) + (load(e2) + load(e3))

    Since size is constexpr and always a power of 2, Triton fully unrolls this
    into a fixed sequence of loads and adds with no dynamic branching.
    Max register depth = log2(E) tiles, e.g. E=64 -> 6 tiles.
    """
    if size == 1:
        return _load_expert_tile(
            input_ptr,
            input_base,
            input_stride_1,
            topk_ids_base,
            topk_ids_stride_1,
            zero_ptrs,
            offs_dim,
            mask,
            mask_token,
            start,
            TOPK,
            BLOCK_M,
        )
    else:
        half: tl.constexpr = size // 2
        left = _tree_reduce_pair(
            input_ptr,
            input_base,
            input_stride_1,
            topk_ids_base,
            topk_ids_stride_1,
            zero_ptrs,
            offs_dim,
            mask,
            mask_token,
            start,
            half,
            TOPK,
            BLOCK_M,
        )
        right = _tree_reduce_pair(
            input_ptr,
            input_base,
            input_stride_1,
            topk_ids_base,
            topk_ids_stride_1,
            zero_ptrs,
            offs_dim,
            mask,
            mask_token,
            start + half,
            half,
            TOPK,
            BLOCK_M,
        )
        return left + right


@triton.jit
def _fused_tree_reduce_kernel(
    # input: [M, topk, hidden_dim], contiguous
    input_ptr,
    input_stride_0,
    input_stride_1,
    # topk_ids: [M, topk], -1 means remote
    topk_ids_ptr,
    topk_ids_stride_0,
    topk_ids_stride_1,
    # zero_buf: [hidden_dim], all zeros
    zero_buf_ptr,
    # output: [M, hidden_dim]
    output_ptr,
    output_stride_0,
    # scalars
    token_num,
    hidden_dim,
    routed_scaling_factor,
    # constexpr
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Fused scatter + binary-tree reduce in a single kernel.
    Zero extra memory: all intermediate results live in registers.

    For each token block, loads expert tiles on-the-fly (with zero-buf redirect
    for remote experts), and reduces them in binary-tree order using recursive
    constexpr unrolling. The tree structure is:
      result = tree_sum(0, E)
      tree_sum(s, n) = tree_sum(s, n/2) + tree_sum(s+n/2, n/2)   if n > 1
      tree_sum(s, 1) = load_expert(s)                              base case

    Register pressure: log2(E) tiles of [BLOCK_M, BLOCK_DIM] fp32.
    E.g. E=64, BLOCK_M=1, BLOCK_DIM=2048 -> 6 * 8KB = 48KB, well within limits.
    """
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    topk_ids_stride_0 = tl.cast(topk_ids_stride_0, dtype=tl.int64)
    topk_ids_stride_1 = tl.cast(topk_ids_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim
    mask = mask_token[:, None] & mask_dim[None, :]

    zero_ptrs = zero_buf_ptr + offs_dim[None, :]
    input_base = input_ptr + offs_token[:, None] * input_stride_0
    topk_ids_base = topk_ids_ptr + offs_token * topk_ids_stride_0

    # Binary tree reduce over all E experts, entirely in registers.
    result = _tree_reduce_pair(
        input_ptr,
        input_base,
        input_stride_1,
        topk_ids_base,
        topk_ids_stride_1,
        zero_ptrs,
        offs_dim,
        mask,
        mask_token,
        0,
        E,
        TOPK,
        BLOCK_M,
    )

    result *= routed_scaling_factor

    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(store_ptrs, result.to(input_ptr.dtype.element_ty), mask=mask)


# Persistent zero buffer: only [hidden_dim] elements, negligible memory.
# Allocated once on first call, never freed.
_zero_buf_cache: torch.Tensor | None = None


def moe_sum_tree_reduce_v1(
    input: torch.Tensor,  # [M, topk, hidden_dim]
    output: torch.Tensor,  # [M, hidden_dim]
    curr_topk_ids: torch.Tensor,  # [M, topk], -1 means remote
    routed_scaling_factor: float,
    E: int,
):
    """
    Fused MoE tree reduce: zero extra memory, CUDA Graph safe.

    Single kernel: loads expert tiles on-the-fly with zero-buf pointer redirect
    for remote experts (L1 cache hit), reduces in binary-tree order entirely
    in registers. No scratch buffer needed.

    Invariant guarantee: binary tree reduce order is fixed by expert id,
    identical across all EP ranks regardless of which experts are local.

    Memory overhead: only a single [hidden_dim] zero buffer (~14KB for H=7168 bf16).
    """
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim
    assert (E & (E - 1)) == 0, f"E must be power of 2, got {E}"

    # Fast path for the dominant K=8 case: avoids generic expert-tree loads/scans.
    if topk == 8:
        _moe_sum_tree_reduce_k8_fast_path(
            input=input,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=routed_scaling_factor,
            E=E,
        )
        return

    # Zero buffer: [hidden_dim], allocated once, reused forever
    global _zero_buf_cache
    if (
        _zero_buf_cache is None
        or _zero_buf_cache.device != input.device
        or _zero_buf_cache.dtype != input.dtype
        or _zero_buf_cache.numel() < hidden_dim
    ):
        _zero_buf_cache = torch.zeros(
            hidden_dim, device=input.device, dtype=input.dtype
        )
    zero_buf = _zero_buf_cache

    BLOCK_M = 1
    BLOCK_DIM = 2048
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _fused_tree_reduce_kernel[grid](
        input,
        input.stride(0),
        input.stride(1),
        curr_topk_ids,
        curr_topk_ids.stride(0),
        curr_topk_ids.stride(1),
        zero_buf,
        output,
        output.stride(0),
        token_num=token_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        E=E,
        TOPK=topk,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        num_warps=num_warps,
    )
    return


import math

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_sum_tree_reduce_k8_fused_kernel_opt2d(
    x_ptr,
    ids_ptr,
    out_ptr,
    sx_m: tl.constexpr,
    sx_k: tl.constexpr,
    sx_h: tl.constexpr,  # x: [M,8,H]
    sid_m: tl.constexpr,
    sid_k: tl.constexpr,  # ids: [M,8]
    so_m: tl.constexpr,
    so_h: tl.constexpr,  # out: [M,H]
    M,
    H,  # runtime OK
    E_LEVEL,  # runtime int (log2(E))
    routed_scaling_factor,  # runtime scalar
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_m = m < M
    mask_h = h < H
    mask_mh = mask_m[:, None] & mask_h[None, :]

    # ---- load ids (int32), -1 means remote ----
    ids0 = tl.load(ids_ptr + m * sid_m + 0 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids1 = tl.load(ids_ptr + m * sid_m + 1 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids2 = tl.load(ids_ptr + m * sid_m + 2 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids3 = tl.load(ids_ptr + m * sid_m + 3 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids4 = tl.load(ids_ptr + m * sid_m + 4 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids5 = tl.load(ids_ptr + m * sid_m + 5 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids6 = tl.load(ids_ptr + m * sid_m + 6 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids7 = tl.load(ids_ptr + m * sid_m + 7 * sid_k, mask=mask_m, other=-1).to(tl.int32)

    # h comes from tl.arange and can be annotated for alignment.
    tl.multiple_of(h, 8)
    tl.max_contiguous(h, BLOCK_H)

    # m is also a tensor.
    tl.multiple_of(m, BLOCK_M)

    # ---- load values: remote handled by mask->other=0 (NO zero_tile / NO tl.where big tiles) ----
    m0 = mask_mh & (ids0 != -1)[:, None]
    m1 = mask_mh & (ids1 != -1)[:, None]
    m2 = mask_mh & (ids2 != -1)[:, None]
    m3 = mask_mh & (ids3 != -1)[:, None]
    m4 = mask_mh & (ids4 != -1)[:, None]
    m5 = mask_mh & (ids5 != -1)[:, None]
    m6 = mask_mh & (ids6 != -1)[:, None]
    m7 = mask_mh & (ids7 != -1)[:, None]

    v0 = tl.load(
        x_ptr + m[:, None] * sx_m + 0 * sx_k + h[None, :] * sx_h, mask=m0, other=0.0
    )
    v1 = tl.load(
        x_ptr + m[:, None] * sx_m + 1 * sx_k + h[None, :] * sx_h, mask=m1, other=0.0
    )
    v2 = tl.load(
        x_ptr + m[:, None] * sx_m + 2 * sx_k + h[None, :] * sx_h, mask=m2, other=0.0
    )
    v3 = tl.load(
        x_ptr + m[:, None] * sx_m + 3 * sx_k + h[None, :] * sx_h, mask=m3, other=0.0
    )
    v4 = tl.load(
        x_ptr + m[:, None] * sx_m + 4 * sx_k + h[None, :] * sx_h, mask=m4, other=0.0
    )
    v5 = tl.load(
        x_ptr + m[:, None] * sx_m + 5 * sx_k + h[None, :] * sx_h, mask=m5, other=0.0
    )
    v6 = tl.load(
        x_ptr + m[:, None] * sx_m + 6 * sx_k + h[None, :] * sx_h, mask=m6, other=0.0
    )
    v7 = tl.load(
        x_ptr + m[:, None] * sx_m + 7 * sx_k + h[None, :] * sx_h, mask=m7, other=0.0
    )

    x_dtype = x_ptr.dtype.element_ty

    # ---- deterministic dense-tree-equivalent reduce (same order concept as baseline) ----
    # Remote entries have already been masked to 0.0 at load time (m0..m7).

    for bit in tl.range(0, E_LEVEL):
        bitmask = 1 << bit

        # ========== lane0 as source ==========
        cond = (ids0 != -1) & ((ids0 & bitmask) != 0)
        target = ids0 ^ bitmask
        mm1 = cond & (ids1 == target)
        mm2 = cond & (ids2 == target)
        mm3 = cond & (ids3 == target)
        mm4 = cond & (ids4 == target)
        mm5 = cond & (ids5 == target)
        mm6 = cond & (ids6 == target)
        mm7 = cond & (ids7 == target)
        hit = mm1 | mm2 | mm3 | mm4 | mm5 | mm6 | mm7

        src = v0.to(tl.float32)
        v1 = tl.where(mm1[:, None], (v1.to(tl.float32) + src).to(x_dtype), v1)
        v2 = tl.where(mm2[:, None], (v2.to(tl.float32) + src).to(x_dtype), v2)
        v3 = tl.where(mm3[:, None], (v3.to(tl.float32) + src).to(x_dtype), v3)
        v4 = tl.where(mm4[:, None], (v4.to(tl.float32) + src).to(x_dtype), v4)
        v5 = tl.where(mm5[:, None], (v5.to(tl.float32) + src).to(x_dtype), v5)
        v6 = tl.where(mm6[:, None], (v6.to(tl.float32) + src).to(x_dtype), v6)
        v7 = tl.where(mm7[:, None], (v7.to(tl.float32) + src).to(x_dtype), v7)

        v0 = tl.where(hit[:, None], 0.0, v0)
        ids0 = tl.where(hit, -1, ids0)
        ids0 = tl.where(cond & (~hit), target, ids0)

        # ========== lane1 as source ==========
        cond = (ids1 != -1) & ((ids1 & bitmask) != 0)
        target = ids1 ^ bitmask
        mm0 = cond & (ids0 == target)
        mm2 = cond & (ids2 == target)
        mm3 = cond & (ids3 == target)
        mm4 = cond & (ids4 == target)
        mm5 = cond & (ids5 == target)
        mm6 = cond & (ids6 == target)
        mm7 = cond & (ids7 == target)
        hit = mm0 | mm2 | mm3 | mm4 | mm5 | mm6 | mm7

        src = v1.to(tl.float32)
        v0 = tl.where(mm0[:, None], (v0.to(tl.float32) + src).to(x_dtype), v0)
        v2 = tl.where(mm2[:, None], (v2.to(tl.float32) + src).to(x_dtype), v2)
        v3 = tl.where(mm3[:, None], (v3.to(tl.float32) + src).to(x_dtype), v3)
        v4 = tl.where(mm4[:, None], (v4.to(tl.float32) + src).to(x_dtype), v4)
        v5 = tl.where(mm5[:, None], (v5.to(tl.float32) + src).to(x_dtype), v5)
        v6 = tl.where(mm6[:, None], (v6.to(tl.float32) + src).to(x_dtype), v6)
        v7 = tl.where(mm7[:, None], (v7.to(tl.float32) + src).to(x_dtype), v7)

        v1 = tl.where(hit[:, None], 0.0, v1)
        ids1 = tl.where(hit, -1, ids1)
        ids1 = tl.where(cond & (~hit), target, ids1)

        # ========== lane2 as source ==========
        cond = (ids2 != -1) & ((ids2 & bitmask) != 0)
        target = ids2 ^ bitmask
        mm0 = cond & (ids0 == target)
        mm1 = cond & (ids1 == target)
        mm3 = cond & (ids3 == target)
        mm4 = cond & (ids4 == target)
        mm5 = cond & (ids5 == target)
        mm6 = cond & (ids6 == target)
        mm7 = cond & (ids7 == target)
        hit = mm0 | mm1 | mm3 | mm4 | mm5 | mm6 | mm7

        src = v2.to(tl.float32)
        v0 = tl.where(mm0[:, None], (v0.to(tl.float32) + src).to(x_dtype), v0)
        v1 = tl.where(mm1[:, None], (v1.to(tl.float32) + src).to(x_dtype), v1)
        v3 = tl.where(mm3[:, None], (v3.to(tl.float32) + src).to(x_dtype), v3)
        v4 = tl.where(mm4[:, None], (v4.to(tl.float32) + src).to(x_dtype), v4)
        v5 = tl.where(mm5[:, None], (v5.to(tl.float32) + src).to(x_dtype), v5)
        v6 = tl.where(mm6[:, None], (v6.to(tl.float32) + src).to(x_dtype), v6)
        v7 = tl.where(mm7[:, None], (v7.to(tl.float32) + src).to(x_dtype), v7)

        v2 = tl.where(hit[:, None], 0.0, v2)
        ids2 = tl.where(hit, -1, ids2)
        ids2 = tl.where(cond & (~hit), target, ids2)

        # ========== lane3 as source ==========
        cond = (ids3 != -1) & ((ids3 & bitmask) != 0)
        target = ids3 ^ bitmask
        mm0 = cond & (ids0 == target)
        mm1 = cond & (ids1 == target)
        mm2 = cond & (ids2 == target)
        mm4 = cond & (ids4 == target)
        mm5 = cond & (ids5 == target)
        mm6 = cond & (ids6 == target)
        mm7 = cond & (ids7 == target)
        hit = mm0 | mm1 | mm2 | mm4 | mm5 | mm6 | mm7

        src = v3.to(tl.float32)
        v0 = tl.where(mm0[:, None], (v0.to(tl.float32) + src).to(x_dtype), v0)
        v1 = tl.where(mm1[:, None], (v1.to(tl.float32) + src).to(x_dtype), v1)
        v2 = tl.where(mm2[:, None], (v2.to(tl.float32) + src).to(x_dtype), v2)
        v4 = tl.where(mm4[:, None], (v4.to(tl.float32) + src).to(x_dtype), v4)
        v5 = tl.where(mm5[:, None], (v5.to(tl.float32) + src).to(x_dtype), v5)
        v6 = tl.where(mm6[:, None], (v6.to(tl.float32) + src).to(x_dtype), v6)
        v7 = tl.where(mm7[:, None], (v7.to(tl.float32) + src).to(x_dtype), v7)

        v3 = tl.where(hit[:, None], 0.0, v3)
        ids3 = tl.where(hit, -1, ids3)
        ids3 = tl.where(cond & (~hit), target, ids3)

        # ========== lane4 as source ==========
        cond = (ids4 != -1) & ((ids4 & bitmask) != 0)
        target = ids4 ^ bitmask
        mm0 = cond & (ids0 == target)
        mm1 = cond & (ids1 == target)
        mm2 = cond & (ids2 == target)
        mm3 = cond & (ids3 == target)
        mm5 = cond & (ids5 == target)
        mm6 = cond & (ids6 == target)
        mm7 = cond & (ids7 == target)
        hit = mm0 | mm1 | mm2 | mm3 | mm5 | mm6 | mm7

        src = v4.to(tl.float32)
        v0 = tl.where(mm0[:, None], (v0.to(tl.float32) + src).to(x_dtype), v0)
        v1 = tl.where(mm1[:, None], (v1.to(tl.float32) + src).to(x_dtype), v1)
        v2 = tl.where(mm2[:, None], (v2.to(tl.float32) + src).to(x_dtype), v2)
        v3 = tl.where(mm3[:, None], (v3.to(tl.float32) + src).to(x_dtype), v3)
        v5 = tl.where(mm5[:, None], (v5.to(tl.float32) + src).to(x_dtype), v5)
        v6 = tl.where(mm6[:, None], (v6.to(tl.float32) + src).to(x_dtype), v6)
        v7 = tl.where(mm7[:, None], (v7.to(tl.float32) + src).to(x_dtype), v7)

        v4 = tl.where(hit[:, None], 0.0, v4)
        ids4 = tl.where(hit, -1, ids4)
        ids4 = tl.where(cond & (~hit), target, ids4)

        # ========== lane5 as source ==========
        cond = (ids5 != -1) & ((ids5 & bitmask) != 0)
        target = ids5 ^ bitmask
        mm0 = cond & (ids0 == target)
        mm1 = cond & (ids1 == target)
        mm2 = cond & (ids2 == target)
        mm3 = cond & (ids3 == target)
        mm4 = cond & (ids4 == target)
        mm6 = cond & (ids6 == target)
        mm7 = cond & (ids7 == target)
        hit = mm0 | mm1 | mm2 | mm3 | mm4 | mm6 | mm7

        src = v5.to(tl.float32)
        v0 = tl.where(mm0[:, None], (v0.to(tl.float32) + src).to(x_dtype), v0)
        v1 = tl.where(mm1[:, None], (v1.to(tl.float32) + src).to(x_dtype), v1)
        v2 = tl.where(mm2[:, None], (v2.to(tl.float32) + src).to(x_dtype), v2)
        v3 = tl.where(mm3[:, None], (v3.to(tl.float32) + src).to(x_dtype), v3)
        v4 = tl.where(mm4[:, None], (v4.to(tl.float32) + src).to(x_dtype), v4)
        v6 = tl.where(mm6[:, None], (v6.to(tl.float32) + src).to(x_dtype), v6)
        v7 = tl.where(mm7[:, None], (v7.to(tl.float32) + src).to(x_dtype), v7)

        v5 = tl.where(hit[:, None], 0.0, v5)
        ids5 = tl.where(hit, -1, ids5)
        ids5 = tl.where(cond & (~hit), target, ids5)

        # ========== lane6 as source ==========
        cond = (ids6 != -1) & ((ids6 & bitmask) != 0)
        target = ids6 ^ bitmask
        mm0 = cond & (ids0 == target)
        mm1 = cond & (ids1 == target)
        mm2 = cond & (ids2 == target)
        mm3 = cond & (ids3 == target)
        mm4 = cond & (ids4 == target)
        mm5 = cond & (ids5 == target)
        mm7 = cond & (ids7 == target)
        hit = mm0 | mm1 | mm2 | mm3 | mm4 | mm5 | mm7

        src = v6.to(tl.float32)
        v0 = tl.where(mm0[:, None], (v0.to(tl.float32) + src).to(x_dtype), v0)
        v1 = tl.where(mm1[:, None], (v1.to(tl.float32) + src).to(x_dtype), v1)
        v2 = tl.where(mm2[:, None], (v2.to(tl.float32) + src).to(x_dtype), v2)
        v3 = tl.where(mm3[:, None], (v3.to(tl.float32) + src).to(x_dtype), v3)
        v4 = tl.where(mm4[:, None], (v4.to(tl.float32) + src).to(x_dtype), v4)
        v5 = tl.where(mm5[:, None], (v5.to(tl.float32) + src).to(x_dtype), v5)
        v7 = tl.where(mm7[:, None], (v7.to(tl.float32) + src).to(x_dtype), v7)

        v6 = tl.where(hit[:, None], 0.0, v6)
        ids6 = tl.where(hit, -1, ids6)
        ids6 = tl.where(cond & (~hit), target, ids6)

        # ========== lane7 as source ==========
        cond = (ids7 != -1) & ((ids7 & bitmask) != 0)
        target = ids7 ^ bitmask
        mm0 = cond & (ids0 == target)
        mm1 = cond & (ids1 == target)
        mm2 = cond & (ids2 == target)
        mm3 = cond & (ids3 == target)
        mm4 = cond & (ids4 == target)
        mm5 = cond & (ids5 == target)
        mm6 = cond & (ids6 == target)
        hit = mm0 | mm1 | mm2 | mm3 | mm4 | mm5 | mm6

        src = v7.to(tl.float32)
        v0 = tl.where(mm0[:, None], (v0.to(tl.float32) + src).to(x_dtype), v0)
        v1 = tl.where(mm1[:, None], (v1.to(tl.float32) + src).to(x_dtype), v1)
        v2 = tl.where(mm2[:, None], (v2.to(tl.float32) + src).to(x_dtype), v2)
        v3 = tl.where(mm3[:, None], (v3.to(tl.float32) + src).to(x_dtype), v3)
        v4 = tl.where(mm4[:, None], (v4.to(tl.float32) + src).to(x_dtype), v4)
        v5 = tl.where(mm5[:, None], (v5.to(tl.float32) + src).to(x_dtype), v5)
        v6 = tl.where(mm6[:, None], (v6.to(tl.float32) + src).to(x_dtype), v6)

        v7 = tl.where(hit[:, None], 0.0, v7)
        ids7 = tl.where(hit, -1, ids7)
        ids7 = tl.where(cond & (~hit), target, ids7)

    # ---- final: since ids unique per token, bucket0 is unique; compute directly in fp32 (no out_tile chain) ----
    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc += tl.where((ids0 == 0)[:, None], v0.to(tl.float32), 0.0)
    acc += tl.where((ids1 == 0)[:, None], v1.to(tl.float32), 0.0)
    acc += tl.where((ids2 == 0)[:, None], v2.to(tl.float32), 0.0)
    acc += tl.where((ids3 == 0)[:, None], v3.to(tl.float32), 0.0)
    acc += tl.where((ids4 == 0)[:, None], v4.to(tl.float32), 0.0)
    acc += tl.where((ids5 == 0)[:, None], v5.to(tl.float32), 0.0)
    acc += tl.where((ids6 == 0)[:, None], v6.to(tl.float32), 0.0)
    acc += tl.where((ids7 == 0)[:, None], v7.to(tl.float32), 0.0)

    acc *= routed_scaling_factor

    out_ptrs = out_ptr + m[:, None] * so_m + h[None, :] * so_h
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_mh)


def _moe_sum_tree_reduce_k8_fast_path(
    input: torch.Tensor,  # [M, 8, H]
    output: torch.Tensor,  # [M, H]
    curr_topk_ids: torch.Tensor,  # [M, 8], -1 means remote
    routed_scaling_factor: float,
    E: int,
):
    assert input.is_contiguous()
    assert output.is_contiguous()
    assert curr_topk_ids.is_contiguous()
    M, K, H = input.shape
    assert K == 8
    assert output.shape == (M, H)
    assert (E & (E - 1)) == 0

    E_LEVEL = int(math.log2(E))

    # K=8 specialization: use wider H tiles for better memory throughput.
    if H >= 4096:
        BLOCK_M = 8
        BLOCK_H = 128
        num_warps = 4
    elif H >= 2048:
        BLOCK_M = 8
        BLOCK_H = 256
        num_warps = 8
    else:
        BLOCK_M = 8
        BLOCK_H = 128
        num_warps = 4

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_H))
    _moe_sum_tree_reduce_k8_fused_kernel_opt2d[grid](
        input,
        curr_topk_ids,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        curr_topk_ids.stride(0),
        curr_topk_ids.stride(1),
        output.stride(0),
        output.stride(1),
        M,
        H,
        E_LEVEL,
        routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return output


@triton.jit
def _moe_sum_tree_reduce_topk16_sparse_kernel(
    x_ptr,
    ids_ptr,
    out_ptr,
    sx_m,
    sx_k,
    sx_h,  # x: [M, K, H]
    sid_m,
    sid_k,  # ids: [M, K]
    so_m,
    so_h,  # out: [M, H]
    M,
    K,
    H,  # runtime
    E_LEVEL,  # log2(E)
    routed_scaling_factor,
    BLOCK_H: tl.constexpr,
    MAX_TOPK: tl.constexpr,  # fixed compile-time capacity, e.g. 16
):
    # One program handles one token + one hidden tile.
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    m = pid_m
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_m = m < M
    mask_h = h < H
    mask_mh = mask_m & mask_h

    slot = tl.arange(0, MAX_TOPK)
    in_k = slot < K

    ids = tl.load(
        ids_ptr + m * sid_m + slot * sid_k, mask=(mask_m & in_k), other=-1
    ).to(tl.int32)
    valid = (ids != -1) & in_k & mask_m

    vals = tl.zeros((MAX_TOPK, BLOCK_H), dtype=tl.float32)
    for s in range(MAX_TOPK):
        vmask = mask_h & valid[s]
        v = tl.load(
            x_ptr + m * sx_m + s * sx_k + h * sx_h,
            mask=vmask,
            other=0.0,
        ).to(tl.float32)
        vals = tl.where((slot == s)[:, None], v[None, :], vals)

    idx = tl.arange(0, MAX_TOPK)
    for bit in tl.range(0, E_LEVEL):
        bitmask = 1 << bit

        for s in range(MAX_TOPK):
            id_s = ids[s]
            cond = (id_s != -1) & ((id_s & bitmask) != 0)
            target = id_s ^ bitmask

            match = (ids == target) & (idx != s)
            # 1-based index; 0 means no match.
            match_pos1 = tl.max(tl.where(match, idx + 1, 0))
            has = match_pos1 > 0
            j = match_pos1 - 1

            src = vals[s, :]
            hit_mask = idx == j
            vals = vals + tl.where(hit_mask[:, None] & has, src[None, :], 0.0)
            vals = tl.where((idx == s)[:, None] & has, 0.0, vals)
            ids = tl.where((idx == s) & has, -1, ids)

            # No partner found at this level: update bucket id only.
            ids = tl.where((idx == s) & cond & (~has), target, ids)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for s in range(MAX_TOPK):
        acc += tl.where(ids[s] == 0, vals[s, :], 0.0)

    acc *= routed_scaling_factor
    tl.store(
        out_ptr + m * so_m + h * so_h, acc.to(out_ptr.dtype.element_ty), mask=mask_mh
    )


def moe_sum_tree_reduce_v1_topk_sparse16(
    input: torch.Tensor,  # [M, K, H]
    output: torch.Tensor,  # [M, H]
    curr_topk_ids: torch.Tensor,  # [M, K], -1 means remote
    routed_scaling_factor: float,
    E: int,
    max_topk: int = 16,
):
    """
    Supplemental kernel path for topk != 8:
    - optimized for small topk (<=16)
    - keeps deterministic tree-equivalent reduction semantics
    - does not replace existing v1 entrypoint automatically
    """
    assert input.is_contiguous()
    assert output.is_contiguous()
    assert curr_topk_ids.is_contiguous()
    M, K, H = input.shape
    assert output.shape == (M, H)
    assert K <= max_topk, f"K={K} > max_topk={max_topk}"
    assert (E & (E - 1)) == 0, "E must be power of 2"

    E_LEVEL = int(math.log2(E))
    BLOCK_H = 128 if H >= 4096 else 256
    num_warps = 4 if BLOCK_H == 128 else 8

    grid = (M, triton.cdiv(H, BLOCK_H))
    _moe_sum_tree_reduce_topk16_sparse_kernel[grid](
        input,
        curr_topk_ids,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        curr_topk_ids.stride(0),
        curr_topk_ids.stride(1),
        output.stride(0),
        output.stride(1),
        M,
        K,
        H,
        E_LEVEL,
        routed_scaling_factor,
        BLOCK_H=BLOCK_H,
        MAX_TOPK=max_topk,
        num_warps=num_warps,
    )
    return output


def moe_sum_tree_reduce_v0(
    input: torch.Tensor,  # [M, 8, H]
    output: torch.Tensor,  # [M, H]
    curr_topk_ids: torch.Tensor,  # [M, 8], -1 means remote
    routed_scaling_factor: float,
    E: int,
):
    # Backward-compatible alias.
    return _moe_sum_tree_reduce_k8_fast_path(
        input=input,
        output=output,
        curr_topk_ids=curr_topk_ids,
        routed_scaling_factor=routed_scaling_factor,
        E=E,
    )


@triton.jit
def _moe_tree_reduce_sparse_k8_kernel(
    x_ptr,
    ids_ptr,
    out_ptr,
    sx_m: tl.constexpr,
    sx_k: tl.constexpr,
    sx_h: tl.constexpr,
    sid_m: tl.constexpr,
    sid_k: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    M,
    H,
    routed_scaling_factor,
    LOGE,  # runtime: tl.range does not unroll
    CAST_MODE: tl.constexpr,  # 0: per-level bf16 round, 1: no intermediate cast
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    tl.multiple_of(m, BLOCK_M)
    tl.multiple_of(h, 8)
    tl.max_contiguous(h, BLOCK_H)

    mask_m = m < M
    mask_h = h < H
    mask_mh = mask_m[:, None] & mask_h[None, :]

    x_ty = x_ptr.dtype.element_ty
    NEG2 = -2  # sentinel: topk_ids cannot be -2

    # ---- ids ----
    ids0 = tl.load(ids_ptr + m * sid_m + 0 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids1 = tl.load(ids_ptr + m * sid_m + 1 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids2 = tl.load(ids_ptr + m * sid_m + 2 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids3 = tl.load(ids_ptr + m * sid_m + 3 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids4 = tl.load(ids_ptr + m * sid_m + 4 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids5 = tl.load(ids_ptr + m * sid_m + 5 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids6 = tl.load(ids_ptr + m * sid_m + 6 * sid_k, mask=mask_m, other=-1).to(tl.int32)
    ids7 = tl.load(ids_ptr + m * sid_m + 7 * sid_k, mask=mask_m, other=-1).to(tl.int32)

    # ---- vals: load -> fp32 ----
    f0 = tl.load(
        x_ptr + m[:, None] * sx_m + 0 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids0 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)
    f1 = tl.load(
        x_ptr + m[:, None] * sx_m + 1 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids1 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)
    f2 = tl.load(
        x_ptr + m[:, None] * sx_m + 2 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids2 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)
    f3 = tl.load(
        x_ptr + m[:, None] * sx_m + 3 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids3 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)
    f4 = tl.load(
        x_ptr + m[:, None] * sx_m + 4 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids4 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)
    f5 = tl.load(
        x_ptr + m[:, None] * sx_m + 5 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids5 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)
    f6 = tl.load(
        x_ptr + m[:, None] * sx_m + 6 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids6 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)
    f7 = tl.load(
        x_ptr + m[:, None] * sx_m + 7 * sx_k + h[None, :] * sx_h,
        mask=mask_mh & (ids7 != -1)[:, None],
        other=0.0,
    ).to(tl.float32)

    # ---- main: runtime loop ----
    for bit in tl.range(0, LOGE):
        bitmask = 1 << bit

        # snapshot ids for matching (critical!)
        a0, a1, a2, a3, a4, a5, a6, a7 = ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids7

        # ---------------- s0 ----------------
        c0 = (a0 != -1) & ((a0 & bitmask) != 0)
        t0 = tl.where(c0, a0 ^ bitmask, NEG2)
        m01 = a1 == t0
        m02 = a2 == t0
        m03 = a3 == t0
        m04 = a4 == t0
        m05 = a5 == t0
        m06 = a6 == t0
        m07 = a7 == t0
        hit0 = m01 | m02 | m03 | m04 | m05 | m06 | m07
        src = f0
        f1 = tl.where(m01[:, None], f1 + src, f1)
        f2 = tl.where(m02[:, None], f2 + src, f2)
        f3 = tl.where(m03[:, None], f3 + src, f3)
        f4 = tl.where(m04[:, None], f4 + src, f4)
        f5 = tl.where(m05[:, None], f5 + src, f5)
        f6 = tl.where(m06[:, None], f6 + src, f6)
        f7 = tl.where(m07[:, None], f7 + src, f7)
        f0 = tl.where(hit0[:, None], 0.0, f0)
        ids0 = tl.where(c0, tl.where(hit0, -1, t0), ids0)

        # ---------------- s1 ----------------
        c1 = (a1 != -1) & ((a1 & bitmask) != 0)
        t1 = tl.where(c1, a1 ^ bitmask, NEG2)
        m10 = a0 == t1
        m12 = a2 == t1
        m13 = a3 == t1
        m14 = a4 == t1
        m15 = a5 == t1
        m16 = a6 == t1
        m17 = a7 == t1
        hit1 = m10 | m12 | m13 | m14 | m15 | m16 | m17
        src = f1
        f0 = tl.where(m10[:, None], f0 + src, f0)
        f2 = tl.where(m12[:, None], f2 + src, f2)
        f3 = tl.where(m13[:, None], f3 + src, f3)
        f4 = tl.where(m14[:, None], f4 + src, f4)
        f5 = tl.where(m15[:, None], f5 + src, f5)
        f6 = tl.where(m16[:, None], f6 + src, f6)
        f7 = tl.where(m17[:, None], f7 + src, f7)
        f1 = tl.where(hit1[:, None], 0.0, f1)
        ids1 = tl.where(c1, tl.where(hit1, -1, t1), ids1)

        # ---------------- s2 ----------------
        c2 = (a2 != -1) & ((a2 & bitmask) != 0)
        t2 = tl.where(c2, a2 ^ bitmask, NEG2)
        m20 = a0 == t2
        m21 = a1 == t2
        m23 = a3 == t2
        m24 = a4 == t2
        m25 = a5 == t2
        m26 = a6 == t2
        m27 = a7 == t2
        hit2 = m20 | m21 | m23 | m24 | m25 | m26 | m27
        src = f2
        f0 = tl.where(m20[:, None], f0 + src, f0)
        f1 = tl.where(m21[:, None], f1 + src, f1)
        f3 = tl.where(m23[:, None], f3 + src, f3)
        f4 = tl.where(m24[:, None], f4 + src, f4)
        f5 = tl.where(m25[:, None], f5 + src, f5)
        f6 = tl.where(m26[:, None], f6 + src, f6)
        f7 = tl.where(m27[:, None], f7 + src, f7)
        f2 = tl.where(hit2[:, None], 0.0, f2)
        ids2 = tl.where(c2, tl.where(hit2, -1, t2), ids2)

        # ---------------- s3 ----------------
        c3 = (a3 != -1) & ((a3 & bitmask) != 0)
        t3 = tl.where(c3, a3 ^ bitmask, NEG2)
        m30 = a0 == t3
        m31 = a1 == t3
        m32 = a2 == t3
        m34 = a4 == t3
        m35 = a5 == t3
        m36 = a6 == t3
        m37 = a7 == t3
        hit3 = m30 | m31 | m32 | m34 | m35 | m36 | m37
        src = f3
        f0 = tl.where(m30[:, None], f0 + src, f0)
        f1 = tl.where(m31[:, None], f1 + src, f1)
        f2 = tl.where(m32[:, None], f2 + src, f2)
        f4 = tl.where(m34[:, None], f4 + src, f4)
        f5 = tl.where(m35[:, None], f5 + src, f5)
        f6 = tl.where(m36[:, None], f6 + src, f6)
        f7 = tl.where(m37[:, None], f7 + src, f7)
        f3 = tl.where(hit3[:, None], 0.0, f3)
        ids3 = tl.where(c3, tl.where(hit3, -1, t3), ids3)

        # ---------------- s4 ----------------
        c4 = (a4 != -1) & ((a4 & bitmask) != 0)
        t4 = tl.where(c4, a4 ^ bitmask, NEG2)
        m40 = a0 == t4
        m41 = a1 == t4
        m42 = a2 == t4
        m43 = a3 == t4
        m45 = a5 == t4
        m46 = a6 == t4
        m47 = a7 == t4
        hit4 = m40 | m41 | m42 | m43 | m45 | m46 | m47
        src = f4
        f0 = tl.where(m40[:, None], f0 + src, f0)
        f1 = tl.where(m41[:, None], f1 + src, f1)
        f2 = tl.where(m42[:, None], f2 + src, f2)
        f3 = tl.where(m43[:, None], f3 + src, f3)
        f5 = tl.where(m45[:, None], f5 + src, f5)
        f6 = tl.where(m46[:, None], f6 + src, f6)
        f7 = tl.where(m47[:, None], f7 + src, f7)
        f4 = tl.where(hit4[:, None], 0.0, f4)
        ids4 = tl.where(c4, tl.where(hit4, -1, t4), ids4)

        # ---------------- s5 ----------------
        c5 = (a5 != -1) & ((a5 & bitmask) != 0)
        t5 = tl.where(c5, a5 ^ bitmask, NEG2)
        m50 = a0 == t5
        m51 = a1 == t5
        m52 = a2 == t5
        m53 = a3 == t5
        m54 = a4 == t5
        m56 = a6 == t5
        m57 = a7 == t5
        hit5 = m50 | m51 | m52 | m53 | m54 | m56 | m57
        src = f5
        f0 = tl.where(m50[:, None], f0 + src, f0)
        f1 = tl.where(m51[:, None], f1 + src, f1)
        f2 = tl.where(m52[:, None], f2 + src, f2)
        f3 = tl.where(m53[:, None], f3 + src, f3)
        f4 = tl.where(m54[:, None], f4 + src, f4)
        f6 = tl.where(m56[:, None], f6 + src, f6)
        f7 = tl.where(m57[:, None], f7 + src, f7)
        f5 = tl.where(hit5[:, None], 0.0, f5)
        ids5 = tl.where(c5, tl.where(hit5, -1, t5), ids5)

        # ---------------- s6 ----------------
        c6 = (a6 != -1) & ((a6 & bitmask) != 0)
        t6 = tl.where(c6, a6 ^ bitmask, NEG2)
        m60 = a0 == t6
        m61 = a1 == t6
        m62 = a2 == t6
        m63 = a3 == t6
        m64 = a4 == t6
        m65 = a5 == t6
        m67 = a7 == t6
        hit6 = m60 | m61 | m62 | m63 | m64 | m65 | m67
        src = f6
        f0 = tl.where(m60[:, None], f0 + src, f0)
        f1 = tl.where(m61[:, None], f1 + src, f1)
        f2 = tl.where(m62[:, None], f2 + src, f2)
        f3 = tl.where(m63[:, None], f3 + src, f3)
        f4 = tl.where(m64[:, None], f4 + src, f4)
        f5 = tl.where(m65[:, None], f5 + src, f5)
        f7 = tl.where(m67[:, None], f7 + src, f7)
        f6 = tl.where(hit6[:, None], 0.0, f6)
        ids6 = tl.where(c6, tl.where(hit6, -1, t6), ids6)

        # ---------------- s7 ----------------
        c7 = (a7 != -1) & ((a7 & bitmask) != 0)
        t7 = tl.where(c7, a7 ^ bitmask, NEG2)
        m70 = a0 == t7
        m71 = a1 == t7
        m72 = a2 == t7
        m73 = a3 == t7
        m74 = a4 == t7
        m75 = a5 == t7
        m76 = a6 == t7
        hit7 = m70 | m71 | m72 | m73 | m74 | m75 | m76
        src = f7
        f0 = tl.where(m70[:, None], f0 + src, f0)
        f1 = tl.where(m71[:, None], f1 + src, f1)
        f2 = tl.where(m72[:, None], f2 + src, f2)
        f3 = tl.where(m73[:, None], f3 + src, f3)
        f4 = tl.where(m74[:, None], f4 + src, f4)
        f5 = tl.where(m75[:, None], f5 + src, f5)
        f6 = tl.where(m76[:, None], f6 + src, f6)
        f7 = tl.where(hit7[:, None], 0.0, f7)
        ids7 = tl.where(c7, tl.where(hit7, -1, t7), ids7)

        # ---- per-level bf16 rounding (optional) ----
        if CAST_MODE == 0:
            # emulate "store bf16 per level then reload": round once per level
            f0 = f0.to(x_ty).to(tl.float32)
            f1 = f1.to(x_ty).to(tl.float32)
            f2 = f2.to(x_ty).to(tl.float32)
            f3 = f3.to(x_ty).to(tl.float32)
            f4 = f4.to(x_ty).to(tl.float32)
            f5 = f5.to(x_ty).to(tl.float32)
            f6 = f6.to(x_ty).to(tl.float32)
            f7 = f7.to(x_ty).to(tl.float32)

    # ---- gather root (id==0) ----
    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc += tl.where((ids0 == 0)[:, None], f0, 0.0)
    acc += tl.where((ids1 == 0)[:, None], f1, 0.0)
    acc += tl.where((ids2 == 0)[:, None], f2, 0.0)
    acc += tl.where((ids3 == 0)[:, None], f3, 0.0)
    acc += tl.where((ids4 == 0)[:, None], f4, 0.0)
    acc += tl.where((ids5 == 0)[:, None], f5, 0.0)
    acc += tl.where((ids6 == 0)[:, None], f6, 0.0)
    acc += tl.where((ids7 == 0)[:, None], f7, 0.0)

    acc *= routed_scaling_factor

    out_ptrs = out_ptr + m[:, None] * so_m + h[None, :] * so_h
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_mh)


def _launch_sparse_tree_k8_k10(
    input: torch.Tensor,  # [M, K, H], bf16 contiguous
    output: torch.Tensor,  # [M, H], bf16 contiguous
    curr_topk_ids: torch.Tensor,  # [M, K], int32/int64 contiguous, -1 remote
    routed_scaling_factor: float,
    E: int,
    cast_mode: int = 1,  # <-- you asked for "last cast" version, so default = 1
):
    assert (
        input.is_contiguous()
        and output.is_contiguous()
        and curr_topk_ids.is_contiguous()
    )
    M, K, H = input.shape
    assert output.shape == (M, H)
    assert K in (8, 10)
    assert (E & (E - 1)) == 0
    LOGE = int(math.log2(E))

    # stable params for CUDA Graph
    BLOCK_M = 8
    BLOCK_H = 256
    num_warps = 8
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_H))

    # NOTE: for strict CUDA Graph safety, avoid dtype conversion here.
    # Please ensure curr_topk_ids is int32 upstream.
    assert (
        curr_topk_ids.dtype == torch.int32
    ), "make ids int32 upstream for CUDA Graph stability"

    if K == 8:
        _moe_tree_reduce_sparse_k8_kernel[grid](
            input,
            curr_topk_ids,
            output,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            curr_topk_ids.stride(0),
            curr_topk_ids.stride(1),
            output.stride(0),
            output.stride(1),
            M,
            H,
            routed_scaling_factor,
            LOGE=LOGE,
            CAST_MODE=cast_mode,
            BLOCK_M=BLOCK_M,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
        )


def moe_sum_tree_reduce_v2(
    input: torch.Tensor,
    output: torch.Tensor,
    curr_topk_ids: torch.Tensor,
    routed_scaling_factor: float,
    E: int,
    *,
    cast_mode: int = 0,  # default to "last cast" as requested
):
    assert (
        input.is_contiguous()
        and output.is_contiguous()
        and curr_topk_ids.is_contiguous()
    )
    M, K, H = input.shape
    if M == 0:
        return output
    if K in (8, 10):
        _launch_sparse_tree_k8_k10(
            input=input,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=routed_scaling_factor,
            E=E,
            cast_mode=cast_mode,
        )
        return output

    # fallback: keep your existing generic path here
    return output


def moe_sum_tree_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    curr_topk_ids: torch.Tensor,
    routed_scaling_factor: float,
    E: int,
):
    curr_topk_ids = curr_topk_ids.to(torch.int32)
    if os.environ.get("SGLANG_MOE_TREE_REDUCE_USE_V2", "0") == "1":
        return moe_sum_tree_reduce_v2(
            input=input,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=routed_scaling_factor,
            E=E,
        )

    if input.shape[1] == 8:
        return moe_sum_tree_reduce_v0(
            input=input,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=routed_scaling_factor,
            E=E,
        )
    return moe_sum_tree_reduce_v1(
        input=input,
        output=output,
        curr_topk_ids=curr_topk_ids,
        routed_scaling_factor=routed_scaling_factor,
        E=E,
    )


# if os.getenv("MOE_SUM_OPTIM", "0") == "1":
#     moe_sum_tree_reduce = moe_sum_tree_reduce_optim
#     print("using optimized moe_sum_tree_reduce_optim")
# else:
#     print("using original moe_sum_tree_reduce_original")

if os.getenv("TREE_ALL_REDUCE_OPTIM", "0") == "1":
    tree_all_reduce_sum = tree_all_reduce_sum_optim
