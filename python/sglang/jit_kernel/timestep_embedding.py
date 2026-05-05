from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# XPU support
_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()

logger = logging.getLogger(__name__)

if _HAS_XPU:
    from sglang.jit_kernel.utils_xpu import load_jit_sycl


@cache_once
def _jit_timestep_embedding_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "timestep_embedding",
        *args,
        cuda_files=["diffusion/timestep_embedding.cuh"],
        cuda_wrappers=[("timestep_embedding", f"timestep_embedding<{args}>")],
    )


if _HAS_XPU:
    @cache_once
    def _jit_timestep_embedding_module_xpu(dtype: torch.dtype):
        """XPU/SYCL version of timestep_embedding JIT compilation"""
        # Map dtype to function suffix
        dtype_map = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype for XPU timestep_embedding: {dtype}")
        
        dtype_str = dtype_map[dtype]
        
        # Load the SYCL module
        module = load_jit_sycl(
            "timestep_embedding",
            dtype_str,
            sycl_files=["diffusion/timestep_embedding.hpp"],
        )
        
        # Return a wrapper that matches the CUDA API
        class XPUTimestepEmbeddingWrapper:
            def __init__(self, module, dtype_str):
                import ctypes
                self._module = module
                self._func_name = f"timestep_embedding_forward_{dtype_str}"
                self._argtypes = [
                    ctypes.c_void_p,   # queue
                    ctypes.c_void_p,   # t
                    ctypes.c_void_p,   # output
                    ctypes.c_int,      # dim
                    ctypes.c_bool,     # flip_sin_to_cos
                    ctypes.c_float,    # downscale_freq_shift
                    ctypes.c_float,    # scale
                    ctypes.c_int,      # max_period
                    ctypes.c_int,      # batch_size
                ]
                
            def timestep_embedding(
                self,
                t,
                output,
                dim,
                flip_sin_to_cos,
                downscale_freq_shift,
                scale,
                max_period
            ):
                # Validate layout assumptions before calling SYCL kernel
                if not t.is_contiguous():
                    raise ValueError("XPU timestep_embedding requires contiguous input tensor")
                if t.storage_offset() != 0:
                    raise ValueError("XPU timestep_embedding requires zero storage offset for input")
                if not output.is_contiguous():
                    raise ValueError("XPU timestep_embedding requires contiguous output tensor")
                if output.storage_offset() != 0:
                    raise ValueError("XPU timestep_embedding requires zero storage offset for output")
                
                # Get XPU queue
                queue = torch.xpu.current_stream().sycl_queue
                
                # Get batch size
                batch_size = t.shape[0]
                
                # Call the SYCL kernel using the module's helper method
                func = self._module.get_function(self._func_name, self._argtypes)
                
                func(
                    queue,
                    t.data_ptr(),
                    output.data_ptr(),
                    dim,
                    flip_sin_to_cos,
                    downscale_freq_shift,
                    scale,
                    max_period,
                    batch_size,
                )
        
        return XPUTimestepEmbeddingWrapper(module, dtype_str)


def _timestep_embedding_pytorch_fallback(
    t: torch.Tensor,
    dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float,
    max_period: int,
) -> torch.Tensor:
    """Pure PyTorch fallback implementation of timestep_embedding"""
    batch_size = t.shape[0]
    half_dim = dim // 2
    
    # Compute frequency schedule
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32, device=t.device))
        * torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
        / (half_dim - downscale_freq_shift)
    )
    
    # Compute angles
    t_float = t.float().view(-1, 1)
    args = scale * t_float * freqs.view(1, -1)
    
    # Compute embeddings
    cos_emb = torch.cos(args)
    sin_emb = torch.sin(args)
    
    if flip_sin_to_cos:
        output = torch.cat([cos_emb, sin_emb], dim=-1)
    else:
        output = torch.cat([sin_emb, cos_emb], dim=-1)
    
    return output


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
    """Compute sinusoidal timestep embeddings for diffusion models.
    
    Args:
        t: Input timesteps [batch_size]
        dim: Embedding dimension (must be divisible by 8)
        flip_sin_to_cos: If True, output is [cos, sin], else [sin, cos]
        downscale_freq_shift: Frequency downscaling shift
        scale: Scaling factor for timesteps
        max_period: Maximum period for sinusoidal encoding
        dtype: Target dtype for input (if conversion needed)
    
    Returns:
        Embedding tensor [batch_size, dim] in float32
    """
    
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        t = t.to(dtype)
    
    output = torch.empty((t.shape[0], dim), dtype=torch.float32, device=t.device)
    
    # XPU-specific path
    if _HAS_XPU and t.device.type == "xpu":
        if dim % 8 != 0:
            raise ValueError(f"XPU timestep_embedding requires dim divisible by 8, got {dim}")
        
        module = _jit_timestep_embedding_module_xpu(t.dtype)
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
    
    # Original CUDA path
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
