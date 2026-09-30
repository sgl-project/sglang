# SPDX-License-Identifier: Apache-2.0
"""Paired RoPE preserving PyTorch CUDA complex64 multiplication rounding."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _complex_rope_kernel(
    x_ptr,
    rope_ptr,
    out_ptr,
    pairs,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair < pairs
    token = pair // (HEADS * (DIM // 2)) % SEQ
    column = pair % (DIM // 2)
    real = tl.load(x_ptr + 2 * pair, mask, 0).to(tl.float32)
    imag = tl.load(x_ptr + 2 * pair + 1, mask, 0).to(tl.float32)
    cos = tl.load(rope_ptr + token * DIM + 2 * column, mask, 0)
    sin = tl.load(rope_ptr + token * DIM + 2 * column + 1, mask, 0)
    # CUDA builds differ in which imaginary product is contracted into the FMA
    out_real = tl.fma(real, cos, -imag * sin)
    if FUSE_REAL_SIN:
        out_imag = tl.fma(real, sin, imag * cos)
    else:
        out_imag = tl.fma(imag, cos, real * sin)
    tl.store(out_ptr + 2 * pair, out_real, mask)
    tl.store(out_ptr + 2 * pair + 1, out_imag, mask)


@lru_cache
def _fuse_real_sin(device: torch.device) -> bool:
    # cancellation distinguishes the two orders without depending on the GPU name
    values = torch.tensor(
        [[1 + 2**-23, -1], [1, 1 - 2**-24]], device=device, dtype=torch.float32
    )
    z = torch.view_as_complex(values)
    return (z[0] * z[1]).imag.item() != 0


def can_use_fused_complex_rope(x: torch.Tensor, rope: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.ndim == 4
        and x.numel() > 0
        and x.shape[-1] % 2 == 0
        and x.is_contiguous()
        and rope.dtype == torch.complex64
        and rope.device == x.device
        and rope.shape == (x.shape[1], x.shape[-1] // 2)
        and rope.is_contiguous()
    )


def _fake_complex_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="fused_complex_rope",
    mutates_args=[],
    fake_impl=_fake_complex_rope,
)
def fused_complex_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Rotate contiguous BSHD activations with a shared S×(D/2) complex cache."""
    assert can_use_fused_complex_rope(x, rope)
    out = torch.empty_like(x)
    pairs = x.numel() // 2
    with torch.cuda.device(x.device):
        _complex_rope_kernel[(triton.cdiv(pairs, 256),)](
            x,
            torch.view_as_real(rope),
            out,
            pairs,
            x.shape[1],
            x.shape[2],
            x.shape[3],
            _fuse_real_sin(x.device),
            256,
            enable_fp_fusion=False,
        )
    return out
