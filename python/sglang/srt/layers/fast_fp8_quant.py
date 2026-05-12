"""Fast per-token-group FP8 (E4M3) quantization for the W_o GEMM input.

Replaces sglang_per_token_group_quant_fp8 specifically for the DSV4-Pro W_o
path where shapes are large enough that the current sgl_kernel implementation
runs at only ~42% peak HBM bandwidth (180 us vs ~80 us theoretical at GB300).

The kernel uses one program per (token, group) pair, BLOCK = group_size = 128.
Each program loads 128 bf16 elements, computes |max|/FP8_MAX as the scale,
quantizes to E4M3, and writes one fp8 chunk + one fp32 scale.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0


@triton.jit
def _per_token_group_quant_fp8_kernel(
    x_ptr,  # (M, K) bf16
    x_q_ptr,  # (M, K) fp8 e4m3
    x_s_ptr,  # (M, K/GROUP) fp32 row-major
    M,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,  # = K // GROUP
    GROUPS_PER_ITER: tl.constexpr,  # process this many groups per loop iter
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One CTA per token; processes GROUPS_PER_ITER groups per iter using a
    2D tile of shape (GROUPS_PER_ITER, GROUP). The per-group max reduces
    along the inner GROUP axis, so each iter pays one big load instead of
    GROUPS_PER_ITER small ones."""
    pid = tl.program_id(0).to(tl.int64)
    row_q = x_q_ptr + pid * K
    row_x = x_ptr + pid * K
    row_s = x_s_ptr + pid * NUM_GROUPS

    inner = tl.arange(0, GROUP)  # (GROUP,)
    g_lane = tl.arange(0, GROUPS_PER_ITER)  # (GROUPS_PER_ITER,)

    for g0 in tl.static_range(0, NUM_GROUPS, GROUPS_PER_ITER):
        # 2D tile: (GROUPS_PER_ITER, GROUP), absolute offset within row.
        offs = (g0 + g_lane)[:, None] * GROUP + inner[None, :]
        x = tl.load(row_x + offs).to(tl.float32)

        # Per-group max along the inner axis.
        abs_max = tl.max(tl.abs(x), axis=1)  # (GROUPS_PER_ITER,)
        scale = abs_max / FP8_MAX  # (GROUPS_PER_ITER,)
        inv_scale = 1.0 / (scale + EPS)  # (GROUPS_PER_ITER,)

        x_q = x * inv_scale[:, None]
        x_q = tl.minimum(tl.maximum(x_q, -FP8_MAX), FP8_MAX)

        tl.store(row_q + offs, x_q.to(row_q.dtype.element_ty))
        tl.store(row_s + g0 + g_lane, scale)


def fast_per_token_group_quant_fp8_128(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group FP8 quant with group_size=128, row-major scales."""
    assert x.is_contiguous()
    assert x.dtype == torch.bfloat16
    GROUP = 128
    M, K = x.shape
    assert K % GROUP == 0
    NUM_GROUPS = K // GROUP

    x_q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    x_s = torch.empty((M, NUM_GROUPS), dtype=torch.float32, device=x.device)

    # 8 groups x 128 = 1024 fp32 elements per iter ~ 4KB working set, fits
    # comfortably and lets each warp drive a 1024-element vectorized load.
    GROUPS_PER_ITER = min(8, NUM_GROUPS)
    while NUM_GROUPS % GROUPS_PER_ITER != 0:
        GROUPS_PER_ITER //= 2
    _per_token_group_quant_fp8_kernel[(M,)](
        x,
        x_q,
        x_s,
        M,
        K=K,
        GROUP=GROUP,
        NUM_GROUPS=NUM_GROUPS,
        GROUPS_PER_ITER=GROUPS_PER_ITER,
        EPS=1e-10,
        FP8_MAX=_FP8_MAX,
        num_warps=4,
    )
    return x_q, x_s
