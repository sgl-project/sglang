from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ltx25_decoder_rope_module(dtype: torch.dtype) -> Module:
    if dtype is not torch.bfloat16:
        raise RuntimeError(f"Unsupported ltx25_decoder_rope dtype: {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_ltx25_decoder_rope",
        *args,
        cuda_files=["diffusion/ltx25_decoder_rope.cuh"],
        cuda_wrappers=[
            (
                "ltx25_decoder_rope",
                f"ltx25_decoder_rope::LTX25DecoderRopeKernel<{args}>::run",
            ),
        ],
    )


def _fake_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    dim_t: int,
    dim_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, dim_t, dim_h
    return torch.empty_like(q), torch.empty_like(k)


@register_custom_op(
    op_name="diffusion_ltx25_decoder_rope",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def fused_ltx25_decoder_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    dim_t: int,
    dim_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired LTX-2.5 decoder RoPE from compact 3D tables."""
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    module = _jit_ltx25_decoder_rope_module(q.dtype)
    module.ltx25_decoder_rope(
        q_out, k_out, q, k, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, dim_t, dim_h
    )
    return q_out, k_out


__all__ = [
    "fused_ltx25_decoder_rope",
]
