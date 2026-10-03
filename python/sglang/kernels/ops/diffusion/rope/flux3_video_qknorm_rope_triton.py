# SPDX-License-Identifier: Apache-2.0
"""BF16 packed QKV -> affine-free RMSNorm + rotate-half RoPE + contiguous V.

Matches ATen's vectorized RMSNorm for head dimension 64 (the reduction order in
``aten/src/ATen/native/cuda/layer_norm_kernel.cu``) and the eager rotate-half
products, including the BF16 rounding boundary between the norm and the RoPE.
``eps`` is the ``nn.RMSNorm`` epsilon; FLUX 3 video VAE uses ``1e-5``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.common.numerics import cuda_rsqrtf, round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _fold_sum(x, rows: tl.constexpr, width: tl.constexpr):
    a, b = tl.split(tl.permute(tl.reshape(x, (rows, 2, width)), (0, 2, 1)))
    return a + b


@triton.jit
def _rms_norm64(x, rows: tl.constexpr, eps):
    # ATen vectorized RMSNorm: four consecutive elements per thread,
    # serial FMA accumulation, then shuffle-down reduction. Keep this order
    # and the BF16 normalization boundary before the RoPE products.
    vec = tl.reshape(x, (rows, 16, 2, 2))
    a02, a13 = tl.split(vec)
    a0, a2 = tl.split(a02)
    a1, a3 = tl.split(a13)
    s = a0 * a0
    s = tl.fma(a1, a1, s)
    s = tl.fma(a2, a2, s)
    s = tl.fma(a3, a3, s)
    s = _fold_sum(s, rows, 8)
    s = _fold_sum(s, rows, 4)
    s = _fold_sum(s, rows, 2)
    s = _fold_sum(s, rows, 1)
    rstd = cuda_rsqrtf(s * (1.0 / 64) + eps)
    return round_bf16_to_fp32(x * rstd)


@triton.jit
def _qknorm_rope_kernel(
    X,
    C,
    S,
    Q,
    K,
    V,
    eps,
    N: tl.constexpr,
    H: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Packed video QKV can exceed 2**31 elements even when each output does not.
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, 64)
    offs = (row // H)[:, None] * (3 * H * 64) + (row % H)[:, None] * 64 + col[None, :]
    mask = row[:, None] < N
    q = tl.load(X + offs, mask, 0).to(tl.float32)
    k = tl.load(X + offs + H * 64, mask, 0).to(tl.float32)
    v = tl.load(X + offs + 2 * H * 64, mask, 0)
    q = _rms_norm64(q, ROWS, eps)
    k = _rms_norm64(k, ROWS, eps)
    q1, q2 = tl.split(tl.permute(tl.reshape(q, (ROWS, 2, 32)), (0, 2, 1)))
    k1, k2 = tl.split(tl.permute(tl.reshape(k, (ROWS, 2, 32)), (0, 2, 1)))
    halfcol = tl.arange(0, 32)
    cache = (row // H)[:, None] * 64 + halfcol[None, :]
    c1 = tl.load(C + cache, mask, 0).to(tl.float32)
    c2 = tl.load(C + cache + 32, mask, 0).to(tl.float32)
    s1 = tl.load(S + cache, mask, 0).to(tl.float32)
    s2 = tl.load(S + cache + 32, mask, 0).to(tl.float32)
    qo1 = round_bf16_to_fp32(q1 * c1) + round_bf16_to_fp32(-q2 * s1)
    qo2 = round_bf16_to_fp32(q2 * c2) + round_bf16_to_fp32(q1 * s2)
    ko1 = round_bf16_to_fp32(k1 * c1) + round_bf16_to_fp32(-k2 * s1)
    ko2 = round_bf16_to_fp32(k2 * c2) + round_bf16_to_fp32(k1 * s2)
    out = row[:, None] * 64 + halfcol[None, :]
    tl.store(Q + out, qo1, mask)
    tl.store(Q + out + 32, qo2, mask)
    tl.store(K + out, ko1, mask)
    tl.store(K + out + 32, ko2, mask)
    tl.store(V + row[:, None] * 64 + col[None, :], v, mask)


def can_use_flux3_video_qknorm_rope(qkv, cos, sin) -> bool:
    if (
        qkv.ndim != 7
        or qkv.shape[0] != 1
        or qkv.shape[4] != 3
        or qkv.shape[-1] != 64
        or qkv.numel() == 0
    ):
        return False
    cache_shape = (*qkv.shape[:4], 1, 64)
    return (
        qkv.is_cuda
        and torch.version.hip is None
        and qkv.dtype == torch.bfloat16
        and cos.shape == cache_shape
        and sin.shape == cache_shape
        and all(
            t.device == qkv.device
            and t.dtype == qkv.dtype
            and t.is_contiguous()
            and not (torch.is_grad_enabled() and t.requires_grad)
            for t in (qkv, cos, sin)
        )
    )


def _fake(qkv, cos, sin, eps: float = 1e-5):
    del cos, sin, eps
    shape = (*qkv.shape[:4], qkv.shape[-2], 64)
    return tuple(
        torch.empty(shape, device=qkv.device, dtype=qkv.dtype) for _ in range(3)
    )


@register_custom_op(
    op_name="diffusion_flux3_video_qknorm_rope", mutates_args=[], fake_impl=_fake
)
def flux3_video_qknorm_rope(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not can_use_flux3_video_qknorm_rope(qkv, cos, sin):
        raise ValueError("unsupported FLUX 3 video QK-norm + RoPE input")
    q, k, v = _fake(qkv, cos, sin)
    rows = q.numel() // 64
    with torch.cuda.device(qkv.device):
        _qknorm_rope_kernel[(triton.cdiv(rows, 32),)](
            qkv,
            cos,
            sin,
            q,
            k,
            v,
            eps,
            rows,
            qkv.shape[-2],
            32,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, k, v
