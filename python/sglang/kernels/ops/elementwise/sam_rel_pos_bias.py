"""SAM relative-position bias kernel wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_sam_rel_pos_bias_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "sam_rel_pos_bias",
        *args,
        cuda_files=["elementwise/sam_rel_pos_bias.cuh"],
        cuda_wrappers=[("sam_rel_pos_bias", f"sam_rel_pos_bias<{args}>")],
    )


@cache_once
def _platform_supported() -> bool:
    from sglang.kernels.selector import select_kernel
    from sglang.kernels.spec import PlatformInfo

    spec = select_kernel("elementwise.sam_rel_pos_bias")
    return spec.is_available(PlatformInfo.detect())


def can_use_sam_rel_pos_bias(rel_h: torch.Tensor, rel_w: torch.Tensor) -> bool:
    """Whether ``sam_rel_pos_bias`` accepts these inputs; otherwise sum eagerly."""
    if not (_platform_supported() and rel_h.is_cuda and rel_w.is_cuda):
        return False
    if rel_h.dtype != torch.bfloat16 or rel_w.dtype != torch.bfloat16:
        return False
    if rel_h.dim() != 5 or rel_w.dim() != 5:
        return False
    if rel_h.shape[-1] != 1 or rel_w.shape[-2] != 1:
        return False
    if rel_h.shape[:3] != rel_w.shape[:3]:
        return False
    key_width = rel_w.shape[-1]
    # Eight-wide 16-byte loads along key width; flat indices are uint32.
    return (
        key_width % 8 == 0
        and rel_w.stride(-1) == 1
        and rel_w.data_ptr() % 16 == 0
        and all(rel_w.stride(i) % 8 == 0 for i in range(3) if rel_w.shape[i] > 1)
        and rel_h.numel() * key_width < 2**32
    )


def sam_rel_pos_bias(rel_h: torch.Tensor, rel_w: torch.Tensor) -> torch.Tensor:
    """Materialize decomposed SAM relative-position bias.

    ``rel_h`` is ``[B, H, Q, KH, 1]`` and ``rel_w`` ``[B, H, Q, 1, KW]``, both
    bf16 on CUDA with ``KW % 8 == 0`` and ``rel_w`` contiguous along ``KW``;
    returns ``[B, H, Q, KH * KW]``.
    """
    batch, heads, queries, key_height, _ = rel_h.shape
    key_width = rel_w.shape[-1]
    output = torch.empty(
        batch,
        heads,
        queries,
        key_height * key_width,
        dtype=rel_h.dtype,
        device=rel_h.device,
    )
    _jit_sam_rel_pos_bias_module().sam_rel_pos_bias(output, rel_h, rel_w)
    return output
