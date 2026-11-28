"""Router logits GEMM with FP32 weights: logits[M, N] = fp32(hidden_bf16[M, K]) @ W_fp32[K, N].

The eager path is ``hidden.float()`` (a copy kernel) + a hipBLASLt FP32 GEMM (split-K + reduce kernel). Both
kernels here run IEEE FP32 MFMA dots on split-K partials (bf16 -> fp32 is exact) and sum the splits in a fixed
order, so they differ from the eager path only in the K summation order.

* ``router_gemv_fp32``: decode sizes, M <= ROUTER_GEMV_MAX_M, one MI355X-tuned config (K=8192, N=256:
  12.4 us vs 17.7 us torch at M=1); 3 launches -> 2.
* ``router_gemm_tuned``: verify widths, tile configs per M from
  ``configs/N={N},K={K},device_name={device},dtype=bf16_fp32.json`` (the gemm-group config naming; the
  MI355X file holds the Muse Spark TP8 router, HIP-graph timed). The grid is the exact verify widths that
  were tuned: any other M (or a device / shape without a file) returns None and keeps the caller's path.
  Prefill widths (M > PREFILL_BF16X3_MIN_M) run ``_router_partial_bf16x3`` instead: the FP32 weight split into
  three bf16 terms (``_router_split_bf16x3``, once per call), three bf16 dots per tile (top-8 ids identical to the
  IEEE kernel). On gfx1250 the three terms are N-major rows of one bf16 weight [3N, K]
  (``_router_split_bf16x3_nk``) under aiter's Gluon a16w16 GEMM with an fp32 output, and the three fp32 partial
  logits are summed (``_router_sum3``).
* ``router_gemm_tlx``: any M, upstream ``tlx.ops.router_gemm`` (vendored as ``tlx_router_gemm``, plain
  Triton) on an ``[N, K]`` contiguous weight copy; BF16x3-split FP32 weight MFMA, deterministic split order.
  MI355X, K=8192, N=256, HIP graphs: 2.3 us vs 6.1 us gemv at M=1, 125 us vs 337 us torch at M=8192.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Dict, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.tlx_router_gemm import router_gemm as tlx_router_gemm
from sglang.srt.utils import get_device_name, log_info_on_rank0

logger = logging.getLogger(__name__)

ROUTER_GEMV_MAX_M = 64
# Prefill widths run the bf16x3 kernel (FP32 VALU dots cap the IEEE kernel's throughput on gfx1250).
PREFILL_BF16X3_MIN_M = 512
# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages); BLOCK_K / SPLIT_K fix the K summation order. With the
# weight split once per call, 128 x 64 / 8 warps / 3 stages is faster at M=1024 on gfx1250; logits are
# bit-identical to the in-tile split at 4 warps / 2 stages (every tile tried was identical).
PREFILL_BF16X3_CONFIG = (128, 64, 64, 8, 8, 3)
# gfx1250 Gluon a16w16 configs (max M, kernel config) for the [M, 3N] fp32-out GEMM, tuned at K=8192 N=256 with the
# split + sum (faster than the Triton in-tile split from M=1024 up).
_GLUON_BF16X3_CONFIGS = (
    (
        1024,
        dict(
            BLOCK_M=64,
            BLOCK_N=128,
            BLOCK_K=128,
            NUM_BUFFERS=4,
            num_warps=4,
            kernel_type="bandwidth_bound",
            waves_per_eu=0,
        ),
    ),
    (
        4096,
        dict(
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=64,
            NUM_BUFFERS=4,
            num_warps=4,
            kernel_type="compute_bound",
            waves_per_eu=0,
        ),
    ),
    (
        None,
        dict(
            BLOCK_M=256,
            BLOCK_N=128,
            BLOCK_K=64,
            NUM_BUFFERS=4,
            num_warps=8,
            kernel_type="compute_bound",
            waves_per_eu=0,
        ),
    ),
)


@triton.jit
def _router_partial_kernel(
    X,
    W,
    P,
    M,
    N,
    K,
    stride_xm,
    stride_wk,
    stride_pk,
    stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k0 = pid_k * (K // SPLIT_K)
    x_ptrs = X + offs_m[:, None].to(tl.int64) * stride_xm + (k0 + offs_k)[None, :]
    w_ptrs = W + (k0 + offs_k)[:, None].to(tl.int64) * stride_wk + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // SPLIT_K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc, input_precision="ieee")
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    tl.store(
        P
        + pid_k * stride_pk
        + offs_m[:, None].to(tl.int64) * stride_pm
        + offs_n[None, :],
        acc,
        mask=m_mask[:, None],
    )


@triton.jit
def _router_reduce_kernel(
    P, OUT, M, N, stride_pk, stride_pm, SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    m = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(P + k * stride_pk + m * stride_pm + offs_n, mask=mask, other=0.0)
    tl.store(OUT + m * N + offs_n, acc, mask=mask)


def router_gemv_fp32_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[0] <= ROUTER_GEMV_MAX_M
        and x.stride(1) == 1
        and w.dtype == torch.float32
        and w.dim() == 2
        and w.stride(1) == 1
        and w.shape[0] == x.shape[1]
        and 32 <= w.shape[1] <= 512
        and (w.shape[1] & (w.shape[1] - 1)) == 0
        and w.shape[0] % 8192 == 0
    )


def router_gemm_tlx_supported(x: torch.Tensor, w_nk: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and w_nk.dtype == torch.float32
        and w_nk.dim() == 2
        and w_nk.is_contiguous()
        and w_nk.shape[1] == x.shape[1]
        # The kernels' smallest N tile and K step.
        and w_nk.shape[0] % 16 == 0
        and x.shape[1] % 1024 == 0
    )


def router_gemm_tlx(x: torch.Tensor, w_nk: torch.Tensor) -> torch.Tensor:
    """``x.float() @ w_nk.T`` for an ``[N, K]`` FP32 weight; FP32 ``[M, N]``."""
    return tlx_router_gemm(x, w_nk, None)


def router_gemv_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [M, K] bf16 (row stride free), w: [K, N] fp32 contiguous rows. Returns fp32 [M, N]."""
    M, K = x.shape
    N = w.shape[1]
    # full-width N tiles (contiguous 1 KB rows of W, vectorized loads) and a deep K split so
    # ~128 programs each stream one 64-row slab of W; tuned on MI355X for K=8192, N=256
    BLOCK_M, BLOCK_K = 16, 64
    BLOCK_N = min(N, 128)
    SPLIT_K = max(
        1, min(64, K // BLOCK_K)
    )  # MI355X sweep (K=8192, N=256): 12.4 us vs 17.7 us torch at M=1
    part = torch.empty((SPLIT_K, M, N), dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    if M == 0:
        return out
    grid = (N // BLOCK_N, SPLIT_K, triton.cdiv(M, BLOCK_M))
    _router_partial_kernel[grid](
        x,
        w,
        part,
        M,
        N,
        K,
        x.stride(0),
        w.stride(0),
        part.stride(0),
        part.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        num_warps=4,
    )
    _router_reduce_kernel[(M,)](
        part,
        out,
        M,
        N,
        part.stride(0),
        part.stride(1),
        SPLIT_K=SPLIT_K,
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=4,
    )
    return out


@triton.jit
def _router_partial(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k0 = pid_k * (K // SPLIT_K)
    x_ptrs = X + offs_m[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    w_ptrs = W + (k0 + offs_k)[:, None].to(tl.int64) * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // SPLIT_K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc, input_precision="ieee")
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * N
    tl.store(
        P
        + pid_k.to(tl.int64) * M * N
        + offs_m[:, None].to(tl.int64) * N
        + offs_n[None, :],
        acc,
        mask=m_mask[:, None],
    )


@triton.jit
def _router_split_bf16x3(W, W3, n_el, BLOCK: tl.constexpr):
    # W = hi + mid + lo, each bf16 (8 + 8 + 8 mantissa bits cover fp32's 24): W3 = [hi, mid, lo], each W's shape.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n_el
    w = tl.load(W + i, mask=mask)
    hi = w.to(tl.bfloat16)
    r = w - hi.to(tl.float32)
    mid = r.to(tl.bfloat16)
    lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
    tl.store(W3 + i, hi, mask=mask)
    tl.store(W3 + n_el + i, mid, mask=mask)
    tl.store(W3 + 2 * n_el + i, lo, mask=mask)


@triton.jit
def _router_split_bf16x3_nk(
    W, W3, K, N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    # _router_split_bf16x3 written N-major: W [K, N] -> W3 [3N, K] = [hi^T; mid^T; lo^T].
    offs_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    w = tl.load(W + offs_k[:, None] * N + offs_n[None, :])
    hi = w.to(tl.bfloat16)
    r = w - hi.to(tl.float32)
    mid = r.to(tl.bfloat16)
    lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
    out = W3 + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    tl.store(out, hi)
    tl.store(out + N * K, mid)
    tl.store(out + 2 * N * K, lo)


@triton.jit
def _router_sum3(P, OUT, M, N: tl.constexpr, BLOCK: tl.constexpr):
    # OUT[m, n] = (hi + mid) + lo of the [M, 3N] partial logits.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < M * N
    m = i // N
    p = P + m.to(tl.int64) * 2 * N + i
    acc = tl.load(p, mask=mask) + tl.load(p + N, mask=mask)
    tl.store(OUT + i, acc + tl.load(p + 2 * N, mask=mask), mask=mask)


@triton.jit
def _router_partial_bf16x3(
    X,
    W3,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Three bf16 dots with fp32 accumulation over W3 = [hi, mid, lo] (_router_split_bf16x3) reproduce the IEEE fp32
    # product of the bf16 rows up to the summation order.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k0 = pid_k * (K // SPLIT_K)
    x_ptrs = X + offs_m[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    w_ptrs = W3 + (k0 + offs_k)[:, None].to(tl.int64) * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // SPLIT_K, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        acc = tl.dot(x, tl.load(w_ptrs), acc)
        acc = tl.dot(x, tl.load(w_ptrs + K * N), acc)
        acc = tl.dot(x, tl.load(w_ptrs + 2 * K * N), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * N
    tl.store(
        P
        + pid_k.to(tl.int64) * M * N
        + offs_m[:, None].to(tl.int64) * N
        + offs_n[None, :],
        acc,
        mask=m_mask[:, None],
    )


@triton.jit
def _router_reduce(
    P, OUT, M, N: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(P + k * M * N + i, mask=mask, other=0.0)
    tl.store(OUT + i, acc, mask=mask)


@lru_cache
def get_router_gemm_fp32_configs(N: int, K: int) -> Optional[Dict[int, Dict[str, int]]]:
    """Tuned ``router_gemm_tuned`` configs for a [K, N] FP32 router weight on this device, keyed by M."""
    device_name = get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=bf16_fp32.json"
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if not os.path.exists(config_file_path):
        return None
    with open(config_file_path) as f:
        log_info_on_rank0(
            logger,
            f"Using configuration from {config_file_path} for the FP32 router GEMM.",
        )
        return {int(key): val for key, val in json.load(f).items()}


def router_gemm_tuned(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    """Tuned split-K router GEMM at a tuned M; None (the caller keeps its path) otherwise."""
    if x.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.float32:
        return None
    m, k = x.shape
    if w.shape[0] != k or not x.is_contiguous() or not w.is_contiguous():
        return None
    n = w.shape[1]
    if m > PREFILL_BF16X3_MIN_M:
        return _router_gemm_bf16x3(x, w)
    cfgs = get_router_gemm_fp32_configs(n, k) or {}
    cfg = cfgs.get(m)
    if cfg is None and m > 512:
        # prefill widths: the smallest tuned M at or above m (exact-M tables cover the verify widths only)
        above = [key for key in cfgs if key >= m and key >= 1024]
        cfg = cfgs[min(above)] if above else None
    if cfg is None:
        return None
    bm, bn, bk, sk = (
        cfg["BLOCK_SIZE_M"],
        cfg["BLOCK_SIZE_N"],
        cfg["BLOCK_SIZE_K"],
        cfg["SPLIT_K"],
    )
    warps, stages = cfg["num_warps"], cfg["num_stages"]
    if n % bn or k % (sk * bk):
        return None
    part = torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _router_partial[(n // bn, sk, triton.cdiv(m, bm))](
        x, w, part, m, n, k, bm, bn, bk, sk, num_warps=warps, num_stages=stages
    )
    _router_reduce[(triton.cdiv(m * n, 1024),)](part, out, m, n, sk, 1024, num_warps=4)
    return out


def _router_gemm_bf16x3(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    m, k = x.shape
    n = w.shape[1]
    if _gluon_bf16x3_supported():
        return _router_gemm_bf16x3_gluon(x, w)
    bm, bn, bk, sk, warps, stages = PREFILL_BF16X3_CONFIG
    if n % bn or k % (sk * bk):
        return None
    # Split per call, not cached: expert placement permutes the router columns in place.
    w3 = torch.empty((3, k, n), dtype=torch.bfloat16, device=x.device)
    _router_split_bf16x3[(triton.cdiv(k * n, 4096),)](w, w3, k * n, 4096, num_warps=8)
    part = torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _router_partial_bf16x3[(n // bn, sk, triton.cdiv(m, bm))](
        x, w3, part, m, n, k, bm, bn, bk, sk, num_warps=warps, num_stages=stages
    )
    _router_reduce[(triton.cdiv(m * n, 1024),)](part, out, m, n, sk, 1024, num_warps=4)
    return out


@lru_cache(maxsize=1)
def _gluon_bf16x3_supported() -> bool:
    if (
        "gfx1250"
        not in torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    ):
        return False
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16_  # noqa: F401
    except ImportError:
        return False
    return True


def _router_gemm_bf16x3_gluon(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16_

    m, k = x.shape
    n = w.shape[1]
    if n % 64 or k % 64:
        return None
    # Split per call, not cached: expert placement permutes the router columns in place.
    w3 = torch.empty((3 * n, k), dtype=torch.bfloat16, device=x.device)
    _router_split_bf16x3_nk[(k // 64, n // 64)](w, w3, k, n, 64, 64, num_warps=4)
    cfg = next(c for max_m, c in _GLUON_BF16X3_CONFIGS if max_m is None or m <= max_m)
    part = gemm_a16w16_(
        x, w3, dtype=torch.float32, config=json.dumps(cfg), backend="gluon"
    )
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _router_sum3[(triton.cdiv(m * n, 1024),)](part, out, m, n, 1024, num_warps=4)
    return out
