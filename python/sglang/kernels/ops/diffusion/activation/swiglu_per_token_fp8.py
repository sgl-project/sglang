# SPDX-License-Identifier: Apache-2.0
"""Packed BF16 SwiGLU + native per-token E4M3 quantization on SM90."""

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from sglang.kernels.numerics import round_bf16_to_fp32


def can_use_fused_packed_swiglu_per_token_fp8(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.ndim in (2, 3)
        and x.is_contiguous()
        and x.shape[-1] == 28672
        and x.numel() // x.shape[-1] >= 4096
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    )


@triton.jit
def _swiglu_per_token_fp8_kernel(X, Q, S, K: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, T)
    vals = ()
    mx = tl.full((T,), 0, tl.float32)
    for i in tl.static_range(tl.cdiv(K, T)):
        cc = c + i * T
        gate = tl.load(X + row * (2 * K) + cc, cc < K, 0).to(tl.float32)
        up = tl.load(X + row * (2 * K) + K + cc, cc < K, 0).to(tl.float32)
        a = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        a = round_bf16_to_fp32(a)
        v = round_bf16_to_fp32(a * up)
        vals += (v,)
        mx = tl.maximum(mx, tl.abs(v))
    s = tl.max(mx, 0) * (1.0 / 448.0)
    inv = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [s],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    inv = tl.where(s == 0, 0.0, inv)
    for i in tl.static_range(tl.cdiv(K, T)):
        cc = c + i * T
        q = tl.minimum(tl.maximum(vals[i] * inv, -448.0), 448.0)
        tl.store(Q + row * K + cc, q, cc < K)
    tl.store(S + row, s)


def fused_packed_swiglu_per_token_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precise SiLU + multiply, both BF16 rounds, then per-token E4M3.

    Matches silu_and_mul_with_activation_rounding + sglang_per_token_quant_fp8:
    amax/448 scales (zero for zero rows), approximate-reciprocal multiplication.
    Accepts contiguous SM90 BF16 [rows,28672] or [B,T,28672], rows>=4096.
    Returns unpadded FP8 [rows,14336] and FP32 scales [rows].
    """
    if not can_use_fused_packed_swiglu_per_token_fp8(x):
        raise ValueError("unsupported packed SwiGLU per-token FP8 inputs")
    dim = x.shape[-1] // 2
    rows = x.numel() // x.shape[-1]
    q = torch.empty((rows, dim), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows,), device=x.device, dtype=torch.float32)
    with torch.cuda.device(x.device):
        # Seven 2048-wide tiles avoid spilling the precise SiLU intermediates.
        _swiglu_per_token_fp8_kernel[(rows,)](
            x, q, scales, dim, 2048, num_warps=16, enable_fp_fusion=False
        )
    return q, scales
