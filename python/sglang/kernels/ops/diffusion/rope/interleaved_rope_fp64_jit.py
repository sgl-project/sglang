from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_interleaved_rope_fp64_module(dtype: torch.dtype) -> Module:
    if dtype is not torch.bfloat16:
        raise RuntimeError(f"Unsupported interleaved_rope_fp64 dtype: {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_interleaved_rope_fp64",
        *args,
        cuda_files=["diffusion/interleaved_rope_fp64.cuh"],
        cuda_wrappers=[
            (
                "interleaved_rope_fp64",
                f"interleaved_rope_fp64::InterleavedRopeFP64Kernel<{args}>::run",
            ),
        ],
    )


def _fake_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cos, sin
    return torch.empty_like(q), torch.empty_like(k)


@register_custom_op(
    op_name="diffusion_interleaved_rope_fp64",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def fused_interleaved_rope_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired interleaved RoPE with fp64 Diffusers semantics."""
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    module = _jit_interleaved_rope_fp64_module(q.dtype)
    module.interleaved_rope_fp64(q_out, k_out, q, k, cos, sin)
    return q_out, k_out


__all__ = [
    "fused_interleaved_rope_fp64",
]
