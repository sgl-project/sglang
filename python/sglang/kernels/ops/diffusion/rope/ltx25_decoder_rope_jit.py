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
        q_out.view(-1),
        k_out.view(-1),
        q.view(-1),
        k.view(-1),
        cos_t.view(-1),
        sin_t.view(-1),
        cos_h.view(-1),
        sin_h.view(-1),
        cos_w.view(-1),
        sin_w.view(-1),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        q.shape[3],
        q.shape[4],
        q.shape[5],
        dim_t,
        dim_h,
    )
    return q_out, k_out


def can_use_ltx25_decoder_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    dim_split: tuple[int, int, int],
) -> bool:
    if (
        q.dim() != 6
        or len(tables) != 3
        or any(len(pair) != 2 for pair in tables)
        or len(dim_split) != 3
    ):
        return False
    dim_t, dim_h, dim_w = dim_split
    expected_shapes = (
        (q.shape[1], dim_t // 2),
        (q.shape[2], dim_h // 2),
        (q.shape[3], dim_w // 2),
    )
    flat_tables = tuple(table for pair in tables for table in pair)
    return (
        q.dtype is torch.bfloat16
        and k.dtype is q.dtype
        and q.is_cuda
        and k.is_cuda
        and q.device == k.device
        and k.shape == q.shape
        and all(size > 0 for size in q.shape)
        and q.shape[-1] == sum(dim_split)
        and all(dim > 0 and dim % 2 == 0 for dim in dim_split)
        and q.is_contiguous()
        and k.is_contiguous()
        and q.data_ptr() % 4 == 0
        and k.data_ptr() % 4 == 0
        and all(table.device == q.device for table in flat_tables)
        and all(table.dtype is torch.float32 for table in flat_tables)
        and all(
            cos.shape == sin.shape == expected_shape
            and cos.is_contiguous()
            and sin.is_contiguous()
            for (cos, sin), expected_shape in zip(tables, expected_shapes, strict=True)
        )
    )


__all__ = [
    "can_use_ltx25_decoder_rope",
    "fused_ltx25_decoder_rope",
]
