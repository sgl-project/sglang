from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils import is_xpu

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# For XPU, try to import JIT infrastructure from sgl_kernel
# Always try import regardless of is_xpu() at module load time
try:
    from sgl_kernel.jit import timestep_embedding as _xpu_timestep_embedding

    _HAS_SGL_KERNEL_JIT = True
except ImportError:
    _HAS_SGL_KERNEL_JIT = False


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
    
    # XPU path - delegate to sgl_kernel.jit
    if is_xpu() and t.device.type == "xpu":
        if _HAS_SGL_KERNEL_JIT:
            return _xpu_timestep_embedding(
                t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period, dtype
            )
        else:
            raise RuntimeError(
                "XPU JIT kernels require sgl-kernel-xpu to be installed.\n"
                "Install it with: pip install sgl-kernel-xpu"
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
