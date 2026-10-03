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
    if dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"Unsupported Helios QK RoPE dtype {dtype}; expected float16 or bfloat16"
        )
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


def can_use_helios_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> bool:
    """Select the packed kernel's dtype and layout; the launcher validates inputs."""
    # Dynamo cannot trace pointer or storage-offset queries. Compiled Helios Q/K
    # come directly from aligned linear outputs; eager callers retain the guard.
    pair_aligned = True
    if not torch.compiler.is_compiling():
        pair_aligned = q.storage_offset() % 2 == 0 and k.storage_offset() % 2 == 0
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and freqs.dtype is torch.float32
        # The eager path also supports broadcast frequencies and unpaired Q/K.
        and k.shape == q.shape
        and freqs.shape == (*q.shape[:-2], 2 * q.shape[-1])
        and q.numel() > 0
        and q.is_contiguous()
        and k.is_contiguous()
        and freqs.is_contiguous()
        and pair_aligned
    )


__all__ = ["can_use_helios_qk_rope", "fused_inplace_helios_qk_rope"]
