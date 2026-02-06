from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_timestep_embedding_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "timestep_embedding",
        *args,
        cuda_files=["diffusion/timestep_embedding.cuh"],
        cuda_wrappers=[("timestep_embedding", f"timestep_embedding<{args}>")],
    )


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
