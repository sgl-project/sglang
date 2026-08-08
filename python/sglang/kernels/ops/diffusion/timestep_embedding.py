from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils import is_xpu, print_warning_once

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# For XPU, try to import JIT infrastructure from sgl_kernel
# Always try import regardless of is_xpu() at module load time
try:
    from sgl_kernel.jit import timestep_embedding as _xpu_timestep_embedding

    _HAS_SGL_KERNEL_JIT = True
except ImportError:
    _HAS_SGL_KERNEL_JIT = False


logger = logging.getLogger(__name__)


def _native_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float,
    max_period: int,
) -> torch.Tensor:
    """Generic PyTorch sinusoidal timestep embedding (XPU fallback)."""
    half_dim = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        0, half_dim, dtype=torch.float32, device=t.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = t.float()[:, None] * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


@cache_once
def _jit_timestep_embedding_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "timestep_embedding",
        *args,
        cuda_files=["diffusion/timestep_embedding.cuh"],
        cuda_wrappers=[
            (
                "timestep_embedding",
                f"sglang_timestep_embedding::timestep_embedding<{args}>",
            )
        ],
    )


@debug_kernel_api
def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0.0,
    scale: float = 1,
    max_period: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        t = t.to(dtype)

    # XPU path - delegate to sgl_kernel.jit with fallback
    if is_xpu() and t.device.type == "xpu":
        if _HAS_SGL_KERNEL_JIT:
            try:
                return _xpu_timestep_embedding(
                    t,
                    dim,
                    flip_sin_to_cos,
                    downscale_freq_shift,
                    scale,
                    max_period,
                    dtype,
                )
            except (ValueError, RuntimeError) as e:
                print_warning_once(
                    f"XPU JIT timestep_embedding kernel failed ({e}), "
                    "falling back to native implementation"
                )
        else:
            logger.debug("sgl-kernel-xpu not installed, using native implementation")

        # Generic PyTorch fallback for XPU
        return _native_timestep_embedding(
            t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
        )
    else:
        # Original CUDA path
        output = torch.empty((t.shape[0], dim), dtype=torch.float32, device=t.device)
        module = _jit_timestep_embedding_module(t.dtype)
        module.timestep_embedding(
            t,
            output,
            dim,
            flip_sin_to_cos,
            float(downscale_freq_shift),
            float(scale),
            int(max_period),
        )
        return output
