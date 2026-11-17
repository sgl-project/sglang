"""FP32 MoE router logits ``x.float() @ weight.T`` for a BF16 ``x`` and an FP32 ``weight``, on gfx950.

gfx950 has no TF32 matrix instruction, and FP32 MFMA runs at a sixteenth of the BF16 rate. Every product here is
exact instead: ``x`` is BF16, and each FP32 weight element is the exact sum of three BF16 values (``hi`` rounds the
weight, ``mid`` the remainder, ``lo`` what is left), so ``x * hi + x * mid + x * lo`` are three exact BF16 products
accumulated in FP32. The weight is split in registers as it is loaded; it stays FP32 in memory, so the op reads the
weight once per call at 4 bytes per element, and an update of the weight needs no refresh.

Two kernels cover the token range:

- Up to ``GEMV_MAX_TOKENS`` tokens the call is a read of the weight. One program owns a few experts and one token,
  loads the whole FP32 rows in one go and reduces them with FP32 FMAs. Every expert row is read by one program per
  token, so all CUs stream the weight and there is no cross-program reduction.
- Above that, BF16 MFMA tiles split K over programs; a second launch adds the per-split partials in split order, so
  the result is deterministic. Small token counts use many splits to keep every CU busy; large ones use few.

The accumulation order depends on the token count's bucket, so the logits of a token may differ in the last bits
between batch sizes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GEMV_MAX_TOKENS = 9
# Element offsets at or past this need 64-bit row offsets.
INT32_OFFSETS = 1 << 31

# Up to this many tokens: (experts per program, K per load, num_warps). Measured on MI350X at E=256, K=8192.
_GEMV_CONFIGS = (
    (1, 1, 8192, 4),
    (4, 2, 4096, 4),
    (9, 2, 4096, 2),
)
# Up to this many tokens (None: any): (BLOCK_M, BLOCK_N, BLOCK_K, splits, num_warps, num_stages).
_MFMA_CONFIGS = (
    (16, 16, 32, 256, 32, 2, 1),
    (32, 32, 16, 256, 16, 2, 1),
    (64, 64, 16, 128, 16, 4, 2),
    (128, 128, 16, 128, 16, 4, 2),
    (256, 128, 16, 64, 16, 4, 2),
    (512, 128, 64, 64, 16, 4, 2),
    (1024, 128, 64, 128, 8, 4, 2),
    (2048, 128, 128, 64, 8, 8, 2),
    (4096, 256, 128, 64, 8, 4, 2),
    (8192, 256, 128, 64, 4, 4, 2),
    (None, 256, 128, 64, 2, 4, 2),
)


def _largest_divisor(value: int, cap: int, floor: int) -> int:
    """Largest power of two in ``[floor, cap]`` dividing ``value``."""
    d = cap
    while d > floor and value % d:
        d //= 2
    return d


def gemv_config(T: int, E: int, K: int):
    for max_tokens, block_n, block_k, num_warps in _GEMV_CONFIGS:
        if T <= max_tokens:
            return _largest_divisor(E, block_n, 1), _largest_divisor(K, block_k, 1024), num_warps
    raise ValueError(f"{T} tokens take the MFMA kernel")


def mfma_config(T: int, E: int, K: int):
    for max_tokens, block_m, block_n, block_k, splits, num_warps, num_stages in _MFMA_CONFIGS:
        if max_tokens is None or T <= max_tokens:
            block_n = _largest_divisor(E, block_n, 16)
            while splits > 1 and K % (splits * block_k):
                splits //= 2
            return block_m, block_n, block_k, splits, num_warps, num_stages
    raise AssertionError("unreachable")


@triton.jit
def _gemv_kernel(X, W, OUT, stride_x, N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 ROW64: tl.constexpr):
    pid_n = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64) if ROW64 else tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(W + offs_n[:, None] * K + offs_k[None, :])
        x = tl.load(X + row * stride_x + offs_k).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + row * N + offs_n, acc)


@triton.jit
def _mfma_kernel(X, W, P, M, stride_x, N: tl.constexpr, K: tl.constexpr, K_PER_SPLIT: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROW64: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    rows = offs_m.to(tl.int64) if ROW64 else offs_m
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_k * K_PER_SPLIT + kk + tl.arange(0, BLOCK_K)
        a = tl.load(X + rows[:, None] * stride_x + offs_k[None, :], mask=m_mask[:, None], other=0.0)
        w = tl.load(W + offs_n[None, :] * K + offs_k[:, None])
        hi = w.to(tl.bfloat16)
        rest = w - hi.to(tl.float32)
        mid = rest.to(tl.bfloat16)
        lo = (rest - mid.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a, hi, acc)
        acc = tl.dot(a, mid, acc)
        acc = tl.dot(a, lo, acc)
    out_rows = (pid_k * M + rows).to(tl.int64) if ROW64 else pid_k * M + rows
    tl.store(P + out_rows[:, None] * N + offs_n[None, :], acc, mask=m_mask[:, None])


@triton.jit
def _reduce_splits_kernel(P, OUT, MN, SPLITS: tl.constexpr, BLOCK: tl.constexpr, ROW64: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64) if ROW64 else tl.program_id(0)
    mn = MN.to(tl.int64) if ROW64 else MN
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < mn
    acc = tl.load(P + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLITS):
        acc += tl.load(P + s * mn + offs, mask=mask, other=0.0)
    tl.store(OUT + offs, acc, mask=mask)


def router_gemm(x, weight, out):
    T, K = x.shape
    E = weight.shape[0]
    if out is None:
        out = torch.empty(T, E, device=x.device, dtype=torch.float32)
    if T == 0:
        return out
    row64 = (T - 1) * x.stride(0) + K >= INT32_OFFSETS
    if T <= GEMV_MAX_TOKENS:
        block_n, block_k, num_warps = gemv_config(T, E, K)
        _gemv_kernel[(E // block_n, T)](x, weight, out, x.stride(0), E, K, block_n, block_k, row64, num_warps=num_warps)
        return out
    block_m, block_n, block_k, splits, num_warps, num_stages = mfma_config(T, E, K)
    grid = (E // block_n, splits, triton.cdiv(T, block_m))
    if splits == 1:
        _mfma_kernel[grid](x, weight, out, T, x.stride(0), E, K, K, block_m, block_n, block_k, row64,
                           num_warps=num_warps, num_stages=num_stages)
        return out
    partials = torch.empty(splits, T, E, device=x.device, dtype=torch.float32)
    row64 = row64 or splits * T * E >= INT32_OFFSETS
    _mfma_kernel[grid](x, weight, partials, T, x.stride(0), E, K, K // splits, block_m, block_n, block_k, row64,
                       num_warps=num_warps, num_stages=num_stages)
    block = 1024
    _reduce_splits_kernel[(triton.cdiv(T * E, block), )](partials, out, T * E, splits, block, row64, num_warps=4)
    return out
