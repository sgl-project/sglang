from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@cache_once
def _jit_usp_relayout_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_usp_relayout",
        *args,
        cuda_files=["diffusion/usp_relayout.cuh"],
        cuda_wrappers=[
            (
                "usp_merge_heads",
                "sglang_usp_relayout::" f"UspMergeHeadsKernel<{args}>::run",
            ),
        ],
    )


def _fake_merge_heads(x: torch.Tensor) -> torch.Tensor:
    world, seq, batch, h_local, head_dim = x.shape
    return x.new_empty((batch, seq, world, h_local, head_dim))


@register_custom_op(
    op_name="diffusion_usp_merge_heads",
    mutates_args=[],
    fake_impl=_fake_merge_heads,
)
def _usp_merge_heads_custom_op(x: torch.Tensor) -> torch.Tensor:
    world, seq, batch, h_local, head_dim = x.shape
    out = x.new_empty((batch, seq, world, h_local, head_dim))
    module = _jit_usp_relayout_module(x.dtype)
    module.usp_merge_heads(out, x)
    return out


def can_use_usp_merge_heads(x: torch.Tensor) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and torch.version.hip is None
        and x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and x.dim() == 5
        and x.numel() > 0
        and x.is_contiguous()
    )


def _usp_merge_heads_cuda(x: torch.Tensor) -> torch.Tensor:
    """[W, S, B, h_local, D] -> [B, S, W, h_local, D] contiguous.

    Bit-exact single-pass replacement for
    ``x.permute(2, 1, 0, 3, 4).contiguous()`` on the Ulysses output path.
    """
    if not can_use_usp_merge_heads(x):
        raise RuntimeError("unsupported input for usp_merge_heads CUDA")
    return _usp_merge_heads_custom_op(x)


def usp_merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Merge Ulysses output heads with an exact eager fallback.

    The backend selection lives here so callers only express the layout
    transformation. Unsupported devices, layouts, and compiled regions retain
    the original PyTorch operation.
    """
    if not torch.compiler.is_compiling() and can_use_usp_merge_heads(x):
        return _usp_merge_heads_cuda(x)
    return x.permute(2, 1, 0, 3, 4).contiguous()
