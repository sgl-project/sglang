from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_timestep_embedding_module() -> Module:
    return load_jit(
        "timestep_embedding",
        cuda_files=["diffusion/timestep_embedding.cuh"],
        cuda_wrappers=[("timestep_embedding", "timestep_embedding")],
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
    dtype = torch.float32
    output = torch.empty((t.shape[0], dim), dtype=dtype, device=t.device)
    module = _jit_timestep_embedding_module()
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
