import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import (
    get_device_core_count,
    get_device_name,
    is_cuda,
    is_gfx1250_supported,
    is_hip,
    log_info_on_rank0,
)
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_gfx1250 = is_gfx1250_supported()
logger = logging.getLogger(__name__)


@register_custom_op(mutates_args=["C"])
def deep_gemm_fp8_fp8_bf16_nt(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
) -> None:
    deep_gemm_wrapper.gemm_nt_f8f8bf16((A, As), (B, Bs), C)


@register_custom_op(mutates_args=["C"])
def deep_gemm_mxfp8_fp8_bf16_nt(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
) -> None:
    deep_gemm_wrapper.gemm_nt_mxfp8_f8f8bf16((A, As), (B, Bs), C)


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _w8a8_block_fp8_matmul_hopper(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    split = tl.program_id(axis=1)
    tiles_per_split = tl.cdiv(tl.cdiv(K, BLOCK_SIZE_K), SPLIT_K)
    first_tile = split * tiles_per_split
    C += split * M * N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    a_ptrs += first_tile * BLOCK_SIZE_K * stride_ak
    b_ptrs += first_tile * BLOCK_SIZE_K * stride_bk
    As_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_As_k
    Bs_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_Bs_k

    # Small-M Hopper configs transpose the MMA so the weight tile occupies M.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(
        first_tile, tl.minimum(first_tile + tiles_per_split, tl.cdiv(K, BLOCK_SIZE_K))
    ):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        if SWAP_AB:
            accumulator += (
                tl.dot(tl.trans(b), tl.trans(a)) * b_s[:, None] * a_s[None, :]
            )
        else:
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if SWAP_AB:
        accumulator = tl.trans(accumulator)

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _reduce_block_fp8_split_k(
    Parts, Out, ELEMENTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        Parts + splits[:, None] * ELEMENTS + offsets[None, :],
        offsets[None, :] < ELEMENTS,
        0.0,
    )
    tl.store(Out + offsets, tl.sum(values, axis=0), offsets < ELEMENTS)


@triton.jit
def _w8a8_block_fp8_matmul_gfx1250(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """
    gfx1250 (RDNA4) block-fp8 matmul.
    The shared ``_w8a8_block_fp8_matmul`` is unusable on gfx1250.
      1. fp8 ``tl.dot`` faults at runtime
      2. software pipelining (``num_stages`` > 1) miscompiles and yields NaN.
      3. the ``offs % M`` / ``offs % N`` modulo-wrap index trick is
         intermittently miscompiled into out-of-bounds addresses.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # No modulo-wrap on gfx1250; use explicit masks instead.
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = offs_am < M
    n_mask = offs_bn < N

    offs_am_c = tl.where(m_mask, offs_am, 0)
    offs_bn_c = tl.where(n_mask, offs_bn, 0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am_c[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn_c[None, :] * stride_bn)

    As_ptrs = As + offs_am_c * stride_As_m
    offs_bsn = offs_bn_c // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        a_s = tl.load(As_ptrs, mask=m_mask, other=0.0)
        b_s = tl.load(Bs_ptrs, mask=n_mask, other=0.0)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)

        # Upcast fp8 to bf16 in-register
        accumulator += (
            tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16)) * a_s[:, None] * b_s[None, :]
        )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _w8a8_block_fp8_matmul_unrolledx4(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    scale_step_k = BLOCK_SIZE_K // group_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # manually unroll to 4 iterations
    UNROLL_FACTOR = 4
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * UNROLL_FACTOR)):
        # 1st iteration
        if needs_masking:
            a = tl.load(
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

        # 2nd iteration
        if needs_masking:
            a = tl.load(
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 1) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 1) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

        # 3rd iteration
        if needs_masking:
            a = tl.load(
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 2) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 2) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

        # 4th iteration
        if needs_masking:
            a = tl.load(
                a_ptrs,
                mask=offs_k[None, :] < K - (k * UNROLL_FACTOR + 3) * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - (k * UNROLL_FACTOR + 3) * BLOCK_SIZE_K,
                other=0.0,
            )
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # Skip config lookup during torch.compile to avoid non-Tensor ops (e.g., device name).
    # Returning None forces the caller to use the default config path during compile.
    if torch._dynamo.is_compiling():
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json"

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            log_info_on_rank0(
                logger,
                f"Using configuration from {config_file_path} for W8A8 Block FP8 kernel.",
            )
            raw = {int(key): val for key, val in json.load(f).items()}

        sanitized = {}
        clamped_ms = []
        for m_key, cfg in raw.items():
            if cfg["BLOCK_SIZE_K"] < block_k and (
                not _is_cuda or block_k % cfg["BLOCK_SIZE_K"] != 0
            ):
                clamped_ms.append((m_key, cfg["BLOCK_SIZE_K"]))
                cfg = {**cfg, "BLOCK_SIZE_K": block_k}
            sanitized[m_key] = cfg
        if clamped_ms:
            logger.warning(
                "Clamped BLOCK_SIZE_K up to %d in tuned config %s for entries %s "
                "(scale stepping requires BLOCK_SIZE_K >= block_k).",
                block_k,
                json_file_name,
                clamped_ms,
            )

        return sanitized

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! "
            "Config file not found at %s"
        ),
        config_file_path,
    )
    return None


@functools.lru_cache
def get_w8a8_channelwise_fp8_configs(N: int, K: int) -> Optional[Dict[int, Any]]:
    """
    Return tuned Triton configurations for the per-token/per-channel w8a8 fp8
    GEMM (`triton_scaled_mm`), for one weight shape on the current device.

    The return value maps an irregular grid of token counts M to
    `triton_scaled_mm` tile configs; the closest M in the grid should be picked
    for a given batch. A null entry means CUTLASS was faster at that M, so the
    grid keeps the point (to stop a neighbouring M snapping onto it) but the
    caller must fall back. A missing file / shape likewise means "keep the
    default".
    """
    # Intentional: the tuned lookup (host-side device name + file I/O) isn't
    # traceable, so under torch.compile return None and fall back to CUTLASS --
    # same as mainline's block-FP8 get_w8a8_block_fp8_configs.
    if torch._dynamo.is_compiling():
        return None

    device_name = get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8_channelwise.json"
    )
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if not os.path.exists(config_file_path):
        return None

    with open(config_file_path) as f:
        log_info_on_rank0(
            logger,
            f"Using configuration from {config_file_path} for W8A8 channelwise FP8 GEMM.",
        )
        return {int(key): val for key, val in json.load(f).items()}


def get_w8a8_channelwise_fp8_config(N: int, K: int, M: int) -> Optional[Dict[str, int]]:
    """Tuned `triton_scaled_mm` config for this GEMM, or None to keep the default.

    N / K are the weight's output / contraction dims and M the token count. None
    means no tuned config applies -- the caller keeps its existing kernel choice
    -- either because this device / shape was never tuned, or the nearest tuned
    M is a point where CUTLASS won. The caller gates on the feature flag.
    """
    configs = get_w8a8_channelwise_fp8_configs(N, K)
    if not configs:
        return None
    return configs[min(configs.keys(), key=lambda x: abs(x - M))]


def select_w8a8_block_fp8_matmul_kernel(M, N, META):
    return _w8a8_block_fp8_matmul


if _is_hip:

    def use_w8a8_block_fp8_matmul_unrolledx4(M, N, META):
        # Use manually unrolledx4 kernel on AMD GPU when the grid size is small.
        # Empirical testing shows the sweet spot lies when it's less than the # of
        # compute units available on the device.
        num_workgroups = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]
        )
        return num_workgroups <= get_device_core_count()

    def select_w8a8_block_fp8_matmul_kernel(M, N, META):
        if use_w8a8_block_fp8_matmul_unrolledx4(M, N, META):
            return _w8a8_block_fp8_matmul_unrolledx4
        else:
            return _w8a8_block_fp8_matmul


def prepare_block_fp8_matmul_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[int, int, int]:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]
    assert A.is_contiguous()

    if As.dtype == torch.float:
        assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    elif As.dtype == torch.int:
        assert triton.cdiv(triton.cdiv(A.shape[-1], block_k), 4) == As.shape[-1], (
            f"{A.shape=} {As.shape=} {block_size=}"
        )
    else:
        raise NotImplementedError

    M = A.numel() // A.shape[-1]

    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape

    if Bs.dtype == torch.float:
        assert triton.cdiv(N, block_n) == Bs.shape[0]
        assert triton.cdiv(K, block_k) == Bs.shape[1]
    elif Bs.dtype == torch.int:
        assert N == Bs.shape[0], f"{B.shape=} {Bs.shape=} {block_size=}"
        assert triton.cdiv(triton.cdiv(K, block_k), 4) == Bs.shape[1], (
            f"{B.shape=} {Bs.shape=} {block_size=}"
        )
    else:
        raise NotImplementedError

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    return M, N, K, C


def w8a8_block_fp8_matmul_deepgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)

    # Deepgemm only supports output tensor type as bfloat16
    assert C.dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM

    deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, C)

    return C


def w8a8_mxfp8_matmul_deepgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    assert A.is_contiguous() and B.is_contiguous()
    assert output_dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    deep_gemm_mxfp8_fp8_bf16_nt(A, As, B, Bs, C)

    return C


def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.

    Returns:
        torch.Tensor: The result of matmul.
    """

    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)

    block_n, block_k = block_size

    configs = get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])
    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        # Block-wise quant: BLOCK_SIZE_K must be divisible by block_size[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }

    # Split-K accumulates K in SPLIT_K separate fp32 partials, so its results
    # do not match the single-accumulator kernels bit-for-bit.
    hopper_tuned = get_platform().is_sm90 and (
        config.get("SWAP_AB", False) or config.get("SPLIT_K", 1) > 1
    )
    if hopper_tuned:
        kernel = _w8a8_block_fp8_matmul_hopper
    elif _is_gfx1250:
        config = {**config, "num_stages": 1}
        kernel = _w8a8_block_fp8_matmul_gfx1250
    else:
        kernel = select_w8a8_block_fp8_matmul_kernel(M, N, config)

    split_k = config.get("SPLIT_K", 1) if hopper_tuned else 1
    if split_k > 1:
        assert split_k & (split_k - 1) == 0
        partials = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)
    else:
        partials = C

    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)

    def grid(META):
        blocks = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]
        )
        return (blocks, split_k) if hopper_tuned else (blocks,)

    kernel[grid](
        A,
        B,
        partials,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=needs_masking,
    )

    if split_k > 1:
        _reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](
            partials, C, M * N, split_k, 256
        )

    return C


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if output_dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return w8a8_block_fp8_matmul_deepgemm(
            A, B, As, Bs, block_size, output_dtype=output_dtype
        )

    return w8a8_block_fp8_matmul_triton(
        A, B, As, Bs, block_size, output_dtype=output_dtype
    )


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


def _as_column_scale(scale: torch.Tensor, expected_len: int) -> torch.Tensor:
    if scale.dim() <= 1:
        return scale.reshape(-1, 1)
    if scale.dim() == 2:
        if scale.shape[1] == 1:
            return scale
        if scale.shape[0] == 1 and scale.shape[1] == expected_len:
            return scale.t()
    return scale


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
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
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = _as_column_scale(scale_a, M)
    scale_b = _as_column_scale(scale_b, N)

    assert scale_a.dim() == 2 and scale_b.dim() == 2
    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    if use_heuristic:
        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
        elif next_power_of_2_M <= 64:
            tile_shape = (64, 64, 256)
        elif next_power_of_2_M <= 128:
            tile_shape = (64, 128, 128)
        else:
            tile_shape = (128, 128, 128)
        block_size_m, block_size_n, block_size_k = tile_shape
    # else: use the block_size_{m,n,k} the caller passed (e.g. a tuned config).

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # num_warps / num_stages are triton.jit launch kwargs, and triton rejects an
    # explicit None -- pass them only when the caller set them, so the default
    # path keeps triton's own defaults.
    launch_kwargs = {}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        **launch_kwargs,
    )

    return result.to(out_dtype)
