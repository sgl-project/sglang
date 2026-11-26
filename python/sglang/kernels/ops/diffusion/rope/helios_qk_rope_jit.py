"""Helios paired RoPE, preserving separate FP32 multiply/add rounding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_helios_qk_rope_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"Unsupported Helios QK RoPE dtype: {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "helios_qk_rope",
        *args,
        cuda_files=["diffusion/helios_qk_rope.cuh"],
        cuda_wrappers=[("helios_qk_rope", f"HeliosQKRoPEKernel<{args}>::run")],
    )


@register_custom_op(mutates_args=["q", "k"])
def fused_inplace_helios_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> None:
    """Rotate contiguous Q/K shaped [N, H, D] or [B, S, H, D] in place.

    Frequencies have shape [N, 2D] or [B, S, 2D]. The C++ launcher
    validates these shapes, dtypes, devices and pair alignment.
    """
    module = _jit_helios_qk_rope_module(q.dtype)
    module.helios_qk_rope(q, k, freqs)


__all__ = ["fused_inplace_helios_qk_rope"]
