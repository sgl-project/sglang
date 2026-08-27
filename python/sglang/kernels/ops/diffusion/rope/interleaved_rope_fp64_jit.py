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
    module.interleaved_rope_fp64(
        q_out.view(-1),
        k_out.view(-1),
        q.view(-1),
        k.view(-1),
        cos.view(-1),
        sin.view(-1),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        q.shape[3],
    )
    return q_out, k_out


def can_use_interleaved_rope_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    if q.dim() != 4:
        return False
    expected_table_shape = (1, q.shape[1], 1, q.shape[3])
    return (
        q.dtype is torch.bfloat16
        and k.dtype is q.dtype
        and q.is_cuda
        and k.is_cuda
        and q.device == k.device == cos.device == sin.device
        and k.shape == q.shape
        and q.shape[-1] % 2 == 0
        and q.is_contiguous()
        and k.is_contiguous()
        and q.data_ptr() % 4 == 0
        and k.data_ptr() % 4 == 0
        and cos.dtype is torch.float64
        and sin.dtype is torch.float64
        and cos.shape == expected_table_shape
        and sin.shape == expected_table_shape
        and cos.is_contiguous()
        and sin.is_contiguous()
    )


__all__ = [
    "can_use_interleaved_rope_fp64",
    "fused_interleaved_rope_fp64",
]
