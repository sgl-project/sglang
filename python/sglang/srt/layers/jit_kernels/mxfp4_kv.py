"""MXFP4 KV cache quantize/dequantize ops (CUDA, sm86).

Kernels live in ``cuda_kernels/mxfp4_kv.cu`` and are JIT-compiled on first
use via ``torch.utils.cpp_extension.load``. Layout (block_size = 32):

    data  [S, H, D/2]  uint8   <- packed E2M1 (2 per byte, lo = even idx)
    scale [S, H, D/32] uint8   <- E8M0 exponent-only scale

Only head_dim = 128 is supported (Qwen3).
"""

import os

import torch
from torch.utils import cpp_extension

_HERE = os.path.dirname(os.path.abspath(__file__))

_ext = None


_CU_SRC = os.path.join(_HERE, "cuda_kernels", "mxfp4_kv.cu")

# Declarations only; definitions live in cuda_sources (kernel launch syntax).
_CPP_SRC = r"""
#include <torch/extension.h>
void mxfp4_quantize_store(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, int64_t, int64_t);
void mxfp4_dequantize(torch::Tensor, torch::Tensor, torch::Tensor,
                      int64_t, int64_t);
void mxfp4_dequantize_indices(torch::Tensor, torch::Tensor, torch::Tensor,
                              torch::Tensor, int64_t, int64_t);
"""

# Launch wrappers: 256-thread CTA, 16 (token, head) rows per CTA.
_CUDA_WRAPPER_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

void mxfp4_quantize_store(torch::Tensor cache_kv, torch::Tensor loc,
                          torch::Tensor data, torch::Tensor scale,
                          int64_t t, int64_t h) {
  const int rows = (int)(t * h);
  const int grid = (rows + 15) / 16;
  mxfp4_quantize_store_kernel<<<grid, 256>>>(
      (const __nv_bfloat16*)cache_kv.data_ptr(),
      (const int*)loc.data_ptr(), (uint8_t*)data.data_ptr(),
      (uint8_t*)scale.data_ptr(), (int)t, (int)h);
}

void mxfp4_dequantize(torch::Tensor data, torch::Tensor scale,
                      torch::Tensor out, int64_t t, int64_t h) {
  const int rows = (int)(t * h);
  const int grid = (rows + 15) / 16;
  mxfp4_dequantize_kernel<<<grid, 256>>>(
      (const uint8_t*)data.data_ptr(), (const uint8_t*)scale.data_ptr(),
      (__nv_bfloat16*)out.data_ptr(), (int)t, (int)h);
}

void mxfp4_dequantize_indices(torch::Tensor data, torch::Tensor scale,
                              torch::Tensor indices, torch::Tensor out,
                              int64_t i, int64_t h) {
  const int rows = (int)(i * h);
  const int grid = (rows + 15) / 16;
  mxfp4_dequantize_indices_kernel<<<grid, 256>>>(
      (const uint8_t*)data.data_ptr(), (const uint8_t*)scale.data_ptr(),
      (const int*)indices.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
      (int)i, (int)h);
}
"""


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    with open(_CU_SRC) as f:
        cuda_src = f.read()
    _ext = cpp_extension.load_inline(
        name="mxfp4_kv",
        cpp_sources=_CPP_SRC,
        cuda_sources=[cuda_src, _CUDA_WRAPPER_SRC],
        functions=[
            "mxfp4_quantize_store",
            "mxfp4_dequantize",
            "mxfp4_dequantize_indices",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_86,code=sm_86",
            "-std=c++17",
        ],
        verbose=False,
    )
    return _ext


def quantize_and_store(
    cache_kv: torch.Tensor,  # [T, H, 128] bf16
    loc: torch.Tensor,       # [T] int32 slot per token
    data: torch.Tensor,      # [S, H, 64] uint8 out
    scale: torch.Tensor,     # [S, H, 4] uint8 out
) -> None:
    """Quantize bf16 KV rows and scatter them into pool slots."""
    ext = _load_ext()
    t, h, d = cache_kv.shape
    assert d == 128, "MXFP4 KV kernel only supports head_dim=128"
    assert cache_kv.is_contiguous() and loc.is_contiguous()
    assert loc.dtype == torch.int32
    ext.mxfp4_quantize_store(cache_kv.contiguous(), loc, data, scale, t, h)


def dequantize(
    data: torch.Tensor,  # [T, H, 64] uint8
    scale: torch.Tensor, # [T, H, 4] uint8
    out: torch.Tensor,   # [T, H, 128] bf16
) -> None:
    """Dequantize contiguous rows."""
    ext = _load_ext()
    t, h, _ = data.shape
    ext.mxfp4_dequantize(data, scale, out, t, h)


def dequantize_indices(
    data: torch.Tensor,   # [S, H, 64] uint8
    scale: torch.Tensor,  # [S, H, 4] uint8
    indices: torch.Tensor,  # [I] int32 slot list
    out: torch.Tensor,    # [I, H, 128] bf16
) -> None:
    """Dequantize rows gathered in ``indices`` order (flashinfer kv_indices)."""
    ext = _load_ext()
    i = indices.numel()
    h = data.shape[1]
    ext.mxfp4_dequantize_indices(data, scale, indices, out, i, h)


def reference_quantize(x: torch.Tensor) -> tuple:
    """Torch reference (block32 MXFP4), for kernel validation."""
    b, m, n = x.shape
    reshaped = x.view(b, m * n // 32, 32)
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    # exp = ceil(log2(block_max / 6)); block_max == 0 -> -127
    safe = torch.clamp(block_max / 6.0, min=1e-10)
    scale_exp = torch.ceil(torch.log2(safe))
    scale_exp = torch.where(block_max == 0, torch.full_like(scale_exp, -127.0), scale_exp)
    scale_exp = torch.clamp(scale_exp, -127, 127)
    scale_bits = (scale_exp + 127).to(torch.uint8)
    scaled = reshaped / torch.exp2(scale_exp)
    sign_bits = (scaled < 0).to(torch.uint8) << 3
    abs_vals = scaled.abs()
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= bounds, dim=-1).to(torch.uint8)
    fp4 = (sign_bits + magnitude_bits).view(b, m, n)
    packed = (fp4[..., 1::2] << 4) + fp4[..., 0::2]
    return packed, scale_bits.view(b, m, n // 32)
