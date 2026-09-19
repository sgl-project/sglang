# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 SwiGLU with BF16 rounding and DeepGEMM MXFP8 scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_quant(
    V, Q, S, ROW, COL, K: tl.constexpr, AM: tl.constexpr, BLOCK: tl.constexpr
):
    groups: tl.constexpr = BLOCK // 32
    v = tl.reshape(V, (groups, 32))
    amax = tl.maximum(tl.max(tl.abs(v), axis=1), 1e-10)
    exponent = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    exponent = tl.minimum(tl.maximum(exponent, 0.0), 254.0)
    inv = tl.exp2(127.0 - exponent)
    q = tl.reshape(tl.clamp(v * inv[:, None], -448.0, 448.0), (BLOCK,))
    tl.store(Q + ROW * K + COL, q, COL < K)
    group = tl.min(COL, 0) // 32 + tl.arange(0, groups)
    # Four UE8M0 bytes per int32, with the M dimension padded to four rows.
    offset = (group // 4) * AM * 4 + ROW * 4 + group % 4
    tl.store(S + offset, exponent.to(tl.uint8), group < K // 32)


@triton.jit
def _swiglu_quant(
    X,
    Q,
    S,
    K: tl.constexpr,
    AM: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * (2 * K) + col, col < K, 0).to(tl.float32)
    up = tl.load(X + row * (2 * K) + K + col, col < K, 0).to(tl.float32)
    gate = tl.minimum(gate, LIMIT)
    up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
    v = gate * tl.sigmoid(ALPHA * gate) * (up + 1.0)
    # The unfused activation materializes BF16 before group quantization.
    v = v.to(tl.bfloat16).to(tl.float32)
    _store_quant(v, Q, S, row, col, K, AM, BLOCK)


def _alloc_quant(m: int, k: int, device: torch.device):
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=device)
    aligned_m = triton.cdiv(m, 4) * 4
    scale_base = torch.empty((k // 128, aligned_m), dtype=torch.int32, device=device)
    return q, scale_base


def swiglu_quant(
    x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize non-interleaved SwiGLU to group-32 FP8 and packed MN-major scales."""
    assert x.ndim == 2 and x.is_contiguous() and x.dtype == torch.bfloat16
    assert x.is_cuda
    m, two_k = x.shape
    assert two_k > 0 and two_k % 256 == 0
    k = two_k // 2
    assert k % 128 == 0
    q, base = _alloc_quant(m, k, x.device)
    if m:
        _swiglu_quant[(m, triton.cdiv(k, 512))](
            x,
            q,
            base.view(torch.uint8),
            k,
            base.shape[1],
            alpha,
            limit,
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, base.T[:m]
