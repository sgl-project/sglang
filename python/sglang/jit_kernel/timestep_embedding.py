from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import is_xpu

if TYPE_CHECKING:
    from tvm_ffi.module import Module


if is_xpu():
    from sglang.jit_kernel.utils_xpu import load_jit_sycl


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


if is_xpu():

    @cache_once
    def _jit_timestep_embedding_module_xpu(dtype: torch.dtype):
        """XPU/SYCL version of timestep_embedding JIT compilation"""
        dtype_map = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype for XPU timestep_embedding: {dtype}")

        dtype_str = dtype_map[dtype]

        module = load_jit_sycl(
            "timestep_embedding",
            dtype_str,
            sycl_files=["diffusion/timestep_embedding.hpp"],
        )

        class XPUTimestepEmbeddingWrapper:
            def __init__(self, module, dtype_str):
                import ctypes

                self._module = module
                self._func_name = f"timestep_embedding_forward_{dtype_str}"
                self._argtypes = [
                    ctypes.c_void_p,  # queue
                    ctypes.c_void_p,  # t
                    ctypes.c_void_p,  # output
                    ctypes.c_int,  # dim
                    ctypes.c_bool,  # flip_sin_to_cos
                    ctypes.c_float,  # downscale_freq_shift
                    ctypes.c_float,  # scale
                    ctypes.c_int,  # max_period
                    ctypes.c_int,  # batch_size
                ]

            def timestep_embedding(
                self,
                t,
                output,
                dim,
                flip_sin_to_cos,
                downscale_freq_shift,
                scale,
                max_period,
            ):
                if not t.is_contiguous():
                    raise ValueError(
                        "XPU timestep_embedding requires contiguous input tensor"
                    )
                if t.storage_offset() != 0:
                    raise ValueError(
                        "XPU timestep_embedding requires zero storage offset for input"
                    )
                if not output.is_contiguous():
                    raise ValueError(
                        "XPU timestep_embedding requires contiguous output tensor"
                    )
                if output.storage_offset() != 0:
                    raise ValueError(
                        "XPU timestep_embedding requires zero storage offset for output"
                    )

                queue = torch.xpu.current_stream().sycl_queue
                batch_size = t.shape[0]

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
    output = torch.empty((t.shape[0], dim), dtype=torch.float32, device=t.device)

    # XPU path
    if is_xpu() and t.device.type == "xpu":
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
    else:
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
