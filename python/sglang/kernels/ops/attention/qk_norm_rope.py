from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(dtype: torch.dtype, head_dim: int) -> Module:
    if dtype not in (torch.float16, torch.bfloat16) or head_dim not in (64, 128, 256):
        raise ValueError(
            "qk_norm_rope requires FP16/BF16 and head dimension 64/128/256"
        )
    args = make_cpp_args(dtype, head_dim, is_arch_support_pdl())
    return load_jit(
        "qk_norm_rope",
        *args,
        cuda_files=["attention/qk_norm_rope.cuh"],
        cuda_wrappers=[("run", f"qk_norm_rope<{args}>")],
    )


@register_custom_op(mutates_args=["k", "out"])
def _qk_norm_rope_out(
    q: torch.Tensor,
    k: torch.Tensor,
    out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    eps: float,
) -> None:
    _module(q.dtype, head_dim).run(
        q.view(q.shape[0], q.shape[1] // head_dim, head_dim),
        k.view(k.shape[0], k.shape[1] // head_dim, head_dim),
        out.view(out.shape[0], out.shape[1] // head_dim, head_dim),
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        eps,
    )


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    """Cached full-head NeoX RoPE after QK RMSNorm, for FP16/BF16 CUDA tensors.

    Returns a separate contiguous Q; modifies K in its original storage and
    leaves Q unchanged. Q/K are [tokens, heads * head_dim] with aligned row
    strides and contiguous columns. Weights have the activation dtype, the
    table is contiguous FP32, and int32/int64 positions index valid table rows.
    Q/K may share packed QKV storage, but their elements must not overlap.
    Normalized values are rounded before rotation, as in separate CUDA kernels.
    """
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _qk_norm_rope_out(
        q, k, out, q_weight, k_weight, cos_sin_cache, positions, head_dim, eps
    )
    return out
