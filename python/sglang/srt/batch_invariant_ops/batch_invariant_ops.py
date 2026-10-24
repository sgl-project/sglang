# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py

import contextlib
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.utils import is_gfx95_supported, is_npu
from sglang.srt.utils.common import (
    calc_diff,
    get_bool_env_var,
    get_device_core_count,
    get_dispatch_device_backend,
)

_is_npu = is_npu()
_is_gfx95 = is_gfx95_supported()
if _is_npu:
    import torch_npu

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

_ENABLE_MM_DEEPGEMM = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "1"
)
# If true, allows to fallback to batch variant gemm when the shape cannot be run in DeepGEMM
_ENABLE_MM_FALLBACK_VARIANT = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT", "0"
)
_ENABLE_MM_COMPARISON_TEST = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST"
)

if not _ENABLE_MM_DEEPGEMM:
    print("Disable DeepGEMM in batch invariant ops. Performance may be suboptimal.")

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]


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


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def _matmul_persistent_configs() -> dict:
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    if _is_gfx95:
        # Measured on MI350X: faster than the default on average over M = 1-4096
        # on dense and MoE linear shapes up to N ~ 8k, neutral on wider N; needs
        # gfx950's 160 KB of LDS.
        configs[torch.bfloat16] = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
            "matrix_instr_nonkdim": 16,
        }
    return configs


def _matmul_persistent_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, (
        "Currently assuming bias is 1D, let Horace know if you run into this"
    )
    NUM_SMS = get_device_core_count()
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = _matmul_persistent_configs()
    # print(a.device, b.device, c.device)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[a.dtype],
    )
    return c


# The segmented matmul splits K into _SEGMENT_COUNT fixed segments and adds the
# segment sums in order, at every M. Small M runs the segments in separate programs
# (fp32 partials, then a fixed-order reduce) and large M runs them in one program;
# both give the same bits, so tiles and the split may be chosen by M and N. Outputs
# with the same BLOCK_K and segment count were bitwise equal across tiles and warps
# on MI350X; changing either changes the numerics.
_SEGMENT_COUNT = 4
_SEGMENT_BLOCK_K = 64


@triton.jit
def _segmented_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    partial_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    SEGMENT_K: tl.constexpr,
    SEGMENT_COUNT: tl.constexpr,
    SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    segment = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    first = segment if SPLIT else 0
    last = segment + 1 if SPLIT else SEGMENT_COUNT
    for s in range(first, last):
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k0 in range(0, SEGMENT_K, BLOCK_SIZE_K):
            k = s * SEGMENT_K + k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
        total += acc
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT:
        tl.store(
            partial_ptr
            + segment.to(tl.int64) * M * N
            + offs_m[:, None] * N
            + offs_n[None, :],
            total,
            mask=mask,
        )
    else:
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            total += bias.to(tl.float32)[None, :]
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            total.to(c_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _segment_reduce_kernel(
    partial_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    SEGMENT_COUNT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SEGMENT_COUNT):
        total += tl.load(
            partial_ptr + s * (M * N).to(tl.int64) + offs, mask=mask, other=0.0
        )
    if HAS_BIAS:
        total += tl.load(bias_ptr + offs % N, mask=mask, other=0.0).to(tl.float32)
    tl.store(c_ptr + offs, total.to(c_ptr.dtype.element_ty), mask=mask)


def _segmented_matmul_launch(M: int, N: int) -> tuple:
    """(BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps, num_stages, split) by M and N.

    None of these change the bits. Measured on MI350X over the bf16 linear shapes of
    Qwen3-8B, Qwen3.8-27B, Qwen3-30B-A3B and gpt-oss-20b, M = 1-8192.

    Without the split, one program loops over all segments, and it is never
    software-pipelined (num_stages 1): with pipelining across that loop the output
    was not run-to-run deterministic for some K on MI350X.
    """
    if N < 1024:
        return 32, 64, 4, 2, True
    if N <= 8192:
        if M <= 32:
            return 32, 64, 4, 2, True
        if M <= 64:
            return 64, 64, 4, 2, True
        if M <= 256:
            return 128, 128, 8, 3, True
        if M <= 1024:
            return 128, 128, 8, 2, True
        if M <= 2048:
            return 128, 128, 8, 1, False
        return 128, 256, 8, 1, False
    if M <= 32:
        return 32, 64, 4, 2, True
    if M <= 64:
        return 64, 64, 4, 2, True
    if M <= 128:
        return 128, 128, 8, 2, True
    return 256, 128, 8, 1, False


def _matmul_segmented_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
):
    M, K = a.shape
    N = b.shape[1]
    block_m, block_n, num_warps, num_stages, split = _segmented_matmul_launch(M, N)
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    segment_k = (
        triton.cdiv(triton.cdiv(K, _SEGMENT_COUNT), _SEGMENT_BLOCK_K) * _SEGMENT_BLOCK_K
    )
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    partial = (
        torch.empty((_SEGMENT_COUNT, M, N), device=a.device, dtype=torch.float32)
        if split
        else c
    )
    _segmented_matmul_kernel[(num_tiles, _SEGMENT_COUNT if split else 1)](
        a,
        b,
        c,
        partial,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        SEGMENT_K=segment_k,
        SEGMENT_COUNT=_SEGMENT_COUNT,
        SPLIT=split,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=_SEGMENT_BLOCK_K,
        GROUP_SIZE_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=16,
    )
    if split:
        block = 4096
        _segment_reduce_kernel[(triton.cdiv(M * N, block),)](
            partial,
            c,
            bias,
            M,
            N,
            SEGMENT_COUNT=_SEGMENT_COUNT,
            HAS_BIAS=bias is not None,
            BLOCK=block,
        )
    return c


def _matmul_persistent_deepgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
):
    M, K = a.shape
    K, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # DeepGEMM 0.2 defaults BF16 GEMMs to cuBLASLt, whose reduction can
    # depend on the batch size. Older wheels always use the invariant kernel.
    get_deterministic = getattr(deep_gemm, "get_deterministic_algorithms", None)
    deterministic = get_deterministic() if get_deterministic else True
    if not deterministic:
        deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.bf16_gemm_nn(a, b, out)
    except RuntimeError as e:
        raise RuntimeError(
            f"DeepGEMM failed for matrix shapes M={M}, N={N}, K={K}. "
            f"This typically occurs when dimensions are too small for DeepGEMM's TMA descriptors. "
            f"Consider increasing MIN_DEEPGEMM_DIM in matmul_persistent() or disabling DeepGEMM "
            f"for small matrices. Original error: {e}"
        ) from e
    finally:
        if not deterministic:
            deep_gemm.use_deterministic_algorithms(False)

    # TODO can this be put in DeepGEMM's `c`?
    if bias is not None:
        out += bias

    return out


def matmul_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
):
    K, N = b.shape
    out_dtype = out_dtype or a.dtype

    # DeepGEMM has minimum dimension requirements for TMA descriptors
    MIN_DEEPGEMM_DIM = 16
    element_size = a.element_size()
    deepgemm_tma_aligned = (N * element_size) % 16 == 0 and (K * element_size) % 16 == 0

    if (
        _ENABLE_MM_DEEPGEMM
        and ENABLE_JIT_DEEPGEMM
        and (a.dtype == torch.bfloat16)
        and (b.dtype == torch.bfloat16)
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
        and N >= MIN_DEEPGEMM_DIM
        and deepgemm_tma_aligned
    ):
        if _ENABLE_MM_COMPARISON_TEST:
            out_triton = _matmul_persistent_triton(
                a=a, b=b, bias=bias, out_dtype=out_dtype
            )
            out_deepgemm = _matmul_persistent_deepgemm(
                a=a, b=b, bias=bias, out_dtype=out_dtype
            )
            diff = calc_diff(out_triton, out_deepgemm)
            assert diff < 0.0001, f"{diff=} {out_triton=} {out_deepgemm=}"
            # can be enabled for debugging
            # print(
            #     f"{diff=} "
            #     f"{(out_triton - out_deepgemm).abs().mean()=} "
            #     f"{(out_triton - out_deepgemm).abs().sum()=} "
            #     f"{torch.sum(out_triton != out_deepgemm)=} "
            # )
            # print(f"{a=} {b=} {bias=} {out_triton=} {out_deepgemm=}")
            return out_deepgemm

        return _matmul_persistent_deepgemm(a=a, b=b, bias=bias, out_dtype=out_dtype)

    if _ENABLE_MM_FALLBACK_VARIANT and out_dtype == a.dtype:
        out = torch.einsum("ik,kj->ij", a, b)
        if bias is not None:
            out += bias
        return out

    if _is_gfx95 and a.dtype == torch.bfloat16 and out_dtype == torch.bfloat16:
        return _matmul_segmented_triton(a=a, b=b, bias=bias, out_dtype=out_dtype)

    return _matmul_persistent_triton(a=a, b=b, bias=bias, out_dtype=out_dtype)


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    # Load first block to infer dtype and initialize max_val with correct type
    col_idx_init = tl.arange(0, BLOCK_SIZE)
    mask_init = col_idx_init < n_cols
    vals_init = tl.load(
        row_start_ptr + col_idx_init, mask=mask_init, other=-float("inf")
    )
    max_val = tl.max(vals_init)

    # Continue with remaining blocks
    for col_offset in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    # Initialize sum_exp with correct dtype by using tl.sum on a zero vector
    sum_exp = tl.sum(tl.zeros([1], dtype=max_val.dtype))

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)
    >> Stashed changes
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda or input.is_xpu, "Input must be a CUDA or XPU tensor"
    assert -input.ndim <= dim < input.ndim, (
        f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"
    )

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def mm_out_batch_invariant(
    a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    return out.copy_(matmul_persistent(a, b))


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems


@triton.jit
def bmm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    B,
    M,
    N,
    K,  #
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """
    Batched matrix multiplication kernel that processes batches in parallel.
    Each tile processes a (BLOCK_SIZE_M, BLOCK_SIZE_N) output block for a specific batch.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_batch = num_pid_m * num_pid_n
    num_tiles_total = B * num_tiles_per_batch

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Process tiles in a deterministic order: batch-major ordering
    for tile_id in tl.range(start_pid, num_tiles_total, NUM_SMS, flatten=True):
        # Decompose tile_id into batch and within-batch tile
        batch_idx = tile_id // num_tiles_per_batch
        tile_in_batch = tile_id % num_tiles_per_batch

        pid_m, pid_n = _compute_pid(
            tile_in_batch, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Add batch offset
        if A_LARGE or B_LARGE:
            batch_idx_typed = batch_idx.to(tl.int64)
        else:
            batch_idx_typed = batch_idx

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + (
                batch_idx_typed * stride_ab
                + offs_am[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                batch_idx_typed * stride_bb
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = (
            c_ptr
            + batch_idx_typed * stride_cb
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    # Process batches in parallel with our persistent kernel
    if a.ndim == 3 and b.ndim == 3:
        # Check constraints
        assert a.shape[0] == b.shape[0], "Batch sizes must match"
        assert a.shape[2] == b.shape[1], "Incompatible dimensions"
        assert a.dtype == b.dtype, "Incompatible dtypes"

        B = a.shape[0]
        M = a.shape[1]
        K = a.shape[2]
        N = b.shape[2]
        dtype = a.dtype

        # Allocate output
        if out is None:
            c = torch.empty((B, M, N), device=a.device, dtype=dtype)
        else:
            c = out

        NUM_SMS = get_device_core_count()

        # Use fixed kernel configuration for determinism
        configs = {
            torch.bfloat16: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
            torch.float16: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
            torch.float32: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
        }

        config = configs.get(dtype)
        if config is None:
            raise ValueError(
                f"Unsupported dtype {dtype} for bmm_batch_invariant. "
                f"Supported dtypes are: {list(configs.keys())}"
            )

        # Grid: limit by NUM_SMS for persistent kernel approach
        num_tiles_per_batch = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(
            N, config["BLOCK_SIZE_N"]
        )
        num_tiles_total = B * num_tiles_per_batch
        grid = (min(NUM_SMS, num_tiles_total),)

        bmm_kernel_persistent[grid](
            a,
            b,
            c,  #
            B,
            M,
            N,
            K,  #
            a.stride(0),
            a.stride(1),
            a.stride(2),  #
            b.stride(0),
            b.stride(1),
            b.stride(2),  #
            c.stride(0),
            c.stride(1),
            c.stride(2),  #
            NUM_SMS=NUM_SMS,  #
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            C_LARGE=c.numel() > 2**31,
            **config,
        )

        return c
    else:
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch-invariant wrapper for RMS normalization.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.

    Adapted from @https://github.com/vllm-project/vllm/blob/66a168a197ba214a5b70a74fa2e713c9eeb3251a/vllm/model_executor/layers/batch_invariant.py#L649

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input, weight, eps=eps)


_ONES_CACHE: dict[Tuple, torch.Tensor] = {}


def _get_or_make_ones(shape, device, dtype) -> torch.Tensor:
    key = (tuple(shape), device, dtype)
    t = _ONES_CACHE.get(key)
    if t is None:
        t = torch.ones(shape, device=device, dtype=dtype)
        _ONES_CACHE[key] = t
    return t


def _rms_norm_aten_compat(input, normalized_shape, weight=None, eps=None):
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    if weight is None:
        weight = _get_or_make_ones(normalized_shape, input.device, input.dtype)
    assert tuple(normalized_shape) == (input.shape[-1],), (
        "rms_norm_batch_invariant only supports last-dim normalization "
        f"(got normalized_shape={tuple(normalized_shape)}, "
        f"input.shape={tuple(input.shape)})"
    )
    return rms_norm_batch_invariant(input, weight, eps=eps)


def _mm_dtype_compat(self, mat2, out_dtype):
    return matmul_persistent(self.contiguous(), mat2.contiguous(), out_dtype=out_dtype)


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_original_torch_bmm = None


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def enable_batch_invariant_mode(enable_bmm: bool = True):
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    if _batch_invariant_MODE:
        return

    dispatch_key = get_dispatch_device_backend()

    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    if not _is_npu:
        # Register for detected device
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)
        _batch_invariant_LIB.impl("aten::mm.out", mm_out_batch_invariant, dispatch_key)
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, dispatch_key)
        _batch_invariant_LIB.impl(
            "aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key
        )
        _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, dispatch_key)
        _batch_invariant_LIB.impl("aten::rms_norm", _rms_norm_aten_compat, dispatch_key)
        _batch_invariant_LIB.impl("aten::mm.dtype", _mm_dtype_compat, dispatch_key)

        if enable_bmm:
            _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, dispatch_key)
            # Also monkeypatch torch.bmm directly as a fallback
            _original_torch_bmm = torch.bmm
            torch.bmm = bmm_batch_invariant
    else:
        from sglang.srt.hardware_backend.npu.batch_invariant_ops.npu_batch_invariant_ops import (
            npu_add_rms_norm_batch_invariant,
            npu_fia_batch_invariant_get_max_workspace,
            npu_fused_infer_attention_score_batch_invariant,
            npu_log_softmax_batch_invariant,
            npu_matmul_batch_invariant,
            npu_mean_batch_invariant,
            npu_mm_batch_invariant,
        )

        _batch_invariant_LIB.impl("aten::mm", npu_mm_batch_invariant, dispatch_key)
        _batch_invariant_LIB.impl(
            "aten::matmul", npu_matmul_batch_invariant, dispatch_key
        )
        _batch_invariant_LIB.impl(
            "aten::mean.dim", npu_mean_batch_invariant, dispatch_key
        )
        _batch_invariant_LIB.impl(
            "aten::_log_softmax", npu_log_softmax_batch_invariant, dispatch_key
        )
        torch.ops.npu.npu_fused_infer_attention_score = (
            npu_fused_infer_attention_score_batch_invariant
        )
        torch_npu.npu_fused_infer_attention_score = (
            npu_fused_infer_attention_score_batch_invariant
        )
        torch_npu._npu_fused_infer_attention_score_get_max_workspace = (
            npu_fia_batch_invariant_get_max_workspace
        )
        torch_npu.npu_add_rms_norm = npu_add_rms_norm_batch_invariant


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    if _original_torch_bmm is not None:
        torch.bmm = _original_torch_bmm
        _original_torch_bmm = None
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    global _batch_invariant_MODE, _batch_invariant_LIB
    old_data = (_batch_invariant_MODE, _batch_invariant_LIB)
    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    yield
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE, _batch_invariant_LIB = old_data


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)
