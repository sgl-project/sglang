"""Bit-exact paired RoPE for Helios' transposed frequency layout.

The JIT kernel preserves the eager path's separate fp32 multiply and add/sub
rounding boundaries, then rounds each adjacent pair back to fp16/bf16 in
place. It is verified on head dimensions 64, 128, and 256, including Helios'
production ``[8640, 40, 128]`` Q/K shape. Unsupported layouts retain the eager
model path through :func:`can_use_helios_qk_rope`.
"""

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
    """Apply Helios' transposed RoPE to contiguous normalized Q/K in place."""
    module = _jit_helios_qk_rope_module(q.dtype)
    module.helios_qk_rope(q, k, freqs)


def can_use_helios_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> bool:
    """Return whether tensors match the native Helios paired-RoPE contract."""
    if q.dim() != 4 or freqs.dim() != 3:
        return False
    # Dynamo cannot trace pointer or storage-offset queries. Compiled Helios Q/K
    # come directly from aligned linear outputs; eager callers retain the guard.
    pair_aligned = True
    if not torch.compiler.is_compiling():
        pair_aligned = q.storage_offset() % 2 == 0 and k.storage_offset() % 2 == 0
    return (
        q.is_cuda
        and k.is_cuda
        and freqs.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and freqs.dtype is torch.float32
        and q.device == k.device == freqs.device
        and k.shape == q.shape
        and all(size > 0 for size in q.shape)
        and freqs.shape == (*q.shape[:2], 2 * q.shape[-1])
        and q.shape[-1] % 2 == 0
        and q.is_contiguous()
        and k.is_contiguous()
        and freqs.is_contiguous()
        and pair_aligned
    )


__all__ = ["can_use_helios_qk_rope", "fused_inplace_helios_qk_rope"]
