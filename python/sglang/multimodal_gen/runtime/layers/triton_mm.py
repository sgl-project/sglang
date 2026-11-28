# SPDX-License-Identifier: Apache-2.0
"""Tuned Triton matmul for large unquantized linear layers on gfx1151.

rocBLAS/Tensile lands at ~25 TFLOPS on the 2048x2048 DiT projections of
MiniMax-H3 (M = 16448 tokens, K in {5376, 7168}, N in {5376, 21504, 28672})
where this kernel reaches ~29, i.e. 1.13-1.21x per projection and ~25% of the
end-to-end denoise time.  aiter's tgemm is no alternative here: gfx1151 has no
bf16 tuned config, so it falls back to the same rocBLAS solution.

Activated by ``SGLANG_USE_ROCM_TRITON_MM=1``; see ``apply_unquantized_linear``
in ``linear.py`` for the dispatch and the shape gate.

The result is *not* bit-exact with rocBLAS: both accumulate in fp32, but the
reduction order differs, so ~0.4% of the elements round to a neighbouring bf16
value (mean |delta| = 0.0003 on outputs of magnitude ~58).  Measured against an
fp64 product, the mean error of the two backends is identical to six decimals,
so this is a different rounding rather than a loss of accuracy.  A distilled
6-step sampler still amplifies it into a different image (PSNR ~34 dB), which
is why the path is opt-in.
"""

import torch
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.layers.utils import register_custom_op

# Tuned on gfx1151 with the MiniMax-H3 DiT shapes; BM=64 keeps the K pipeline
# at 3 stages without spilling.
_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 64
_GROUP_M = 8
_NUM_WARPS = 4
_NUM_STAGES = 3

# Below this the launch overhead and the missing tile parallelism cost more than
# the GEMM saves; the modulation projections of a DiT live here.
_MIN_M = 256
_MIN_K = 256


@triton.jit
def _mm_nt(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    MASK_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """C = A @ B.T with A [M, K], B [N, K], all row-major.

    N and K must be multiples of BN and BK so no column or K masking is needed;
    the caller guarantees that in ``can_use_triton_mm``.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_m = tl.cdiv(M, BM)
    # Group along M so consecutive programs reuse the same B tiles.
    group_size = GM * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GM
    group_rows = tl.minimum(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % group_size) % group_rows)
    pid_n = (pid % group_size) // group_rows

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mask_m = rm < M

    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rn[:, None] * K + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        if MASK_M:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs)
        acc += tl.dot(a, tl.trans(tl.load(b_ptrs)))
        a_ptrs += BK
        b_ptrs += BK
    # Same order as the GEMM epilogues: widen in fp32, add bias, round once.
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn).to(tl.float32)[None, :]
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    if MASK_M:
        tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=mask_m[:, None])
    else:
        tl.store(c_ptrs, acc.to(OUT_DTYPE))


def can_use_triton_mm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Whether ``triton_mm`` reproduces ``F.linear(x, weight, bias)`` here."""
    if x.dim() < 2 or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if weight.dtype != x.dtype or weight.dim() != 2:
        return False
    if bias is not None and (
        bias.dtype != x.dtype or bias.dim() != 1 or bias.shape[0] != weight.shape[0]
    ):
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    m, k = x.numel() // x.shape[-1], x.shape[-1]
    return (
        m >= _MIN_M
        and k >= _MIN_K
        and k % _BLOCK_K == 0
        and weight.shape[0] % _BLOCK_N == 0
    )


@register_custom_op(
    op_name="triton_mm_gfx1151",
    fake_impl=lambda x, weight, bias=None: x.new_empty(*x.shape[:-1], weight.shape[0]),
)
def triton_mm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T + bias`` for 2D contiguous operands."""
    k = x.shape[-1]
    x_2d = x.reshape(-1, k)
    m, n = x_2d.shape[0], weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, _BLOCK_N),)
    _mm_nt[grid](
        x_2d,
        weight,
        out,
        bias if bias is not None else x_2d,
        m,
        n,
        k,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        _GROUP_M,
        (m % _BLOCK_M) != 0,
        bias is not None,
        tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out.view(*x.shape[:-1], n)
