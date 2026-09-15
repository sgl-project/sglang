"""Compensated mHC prefill projection with a shared activation load.

Keep three BF16 components of the FP32 weights and accumulate their products
separately. The 16 fixed K slices bound FP32 accumulation error, as in the
compensated DeepGEMM path, while avoiding its second activation read/reduction.
"""

import torch
import triton
import triton.language as tl


def split_bf16_hc_weight(weight: torch.Tensor):
    assert weight.dtype == torch.float32 and weight.is_contiguous()
    high = weight.bfloat16()
    residual = weight - high.float()
    middle = residual.bfloat16()
    low = (residual - middle.float()).bfloat16()
    return high, middle, low


@triton.jit
def _hc_mix_stats_bf16x3(X, W_HI, W_MID, W_LO, MIX, SQ, M, BLOCK_M: tl.constexpr):
    # M stays runtime-valued so variable prefill lengths reuse the same binary.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 32)
    # 20480 input features / 16 independent slices.
    start = tl.program_id(1) * 1280
    ks = start + tl.arange(0, 64)
    hi = tl.zeros((BLOCK_M, 32), tl.float32)
    mid = tl.zeros((BLOCK_M, 32), tl.float32)
    lo = tl.zeros((BLOCK_M, 32), tl.float32)
    sq = tl.zeros((BLOCK_M,), tl.float32)
    for block in range(20):
        k = ks + block * 64
        x = tl.load(
            X + rows[:, None].to(tl.int64) * 20480 + k[None, :],
            rows[:, None] < M,
            0,
        )
        offsets = cols[None, :] * 20480 + k[:, None]
        w_hi = tl.load(W_HI + offsets, cols[None, :] < 24, 0)
        w_mid = tl.load(W_MID + offsets, cols[None, :] < 24, 0)
        w_lo = tl.load(W_LO + offsets, cols[None, :] < 24, 0)
        hi = tl.dot(x, w_hi, hi)
        mid = tl.dot(x, w_mid, mid)
        lo = tl.dot(x, w_lo, lo)
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, 1)
    offsets = (tl.program_id(1) * M + rows[:, None]) * 24 + cols[None, :]
    tl.store(MIX + offsets, (hi + mid) + lo, (rows[:, None] < M) & (cols[None, :] < 24))
    tl.store(SQ + tl.program_id(1) * M + rows, sq, rows < M)


def hc_mix_stats_sinkhorn_bf16x3(
    x: torch.Tensor,
    weight_parts,
    scale: torch.Tensor,
    base: torch.Tensor,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
):
    from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel

    m = x.shape[0]
    assert x.shape == (m, 20480) and x.is_contiguous()
    assert x.dtype == torch.bfloat16 and 4096 <= m <= 65536
    assert len(weight_parts) == 3
    assert all(
        w.shape == (24, 20480) and w.dtype == torch.bfloat16 and w.is_contiguous()
        for w in weight_parts
    )
    mix = torch.empty((16, m, 24), device=x.device, dtype=torch.float32)
    sq = torch.empty((16, m), device=x.device, dtype=torch.float32)
    pre = torch.empty((m, 4), device=x.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((m, 4, 4), device=x.device, dtype=torch.float32)
    _hc_mix_stats_bf16x3[(triton.cdiv(m, 128), 16)](
        x, *weight_parts, mix, sq, m, 128, num_warps=4, num_stages=3
    )
    _hc_mix_reduce_sinkhorn_kernel[(m,)](
        mix,
        sq,
        scale,
        base,
        pre,
        post,
        comb,
        m,
        1.0 / 20480,
        rms_eps,
        MIX=24,
        HC=4,
        NUM_SLICES=16,
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        num_warps=1,
    )
    return pre, post, comb
