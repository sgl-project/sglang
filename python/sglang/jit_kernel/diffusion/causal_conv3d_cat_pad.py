from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels._jit import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@cache_once
def _jit_causal_conv3d_cat_pad_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_causal_conv3d_cat_pad",
        *args,
        cuda_files=["diffusion/causal_conv3d_cat_pad.cuh"],
        cuda_wrappers=[
            (
                "causal_conv3d_cat_pad",
                "sglang_causal_conv3d_cat_pad::"
                f"CausalConv3dCatPadKernel<{args}>::run",
            )
        ],
    )


def _causal_conv3d_cat_pad_fake_impl(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    pad_w_left: int,
    pad_w_right: int,
    pad_h_top: int,
    pad_h_bottom: int,
    pad_d_left: int,
    pad_d_right: int,
) -> torch.Tensor:
    cache_t = cache_x.shape[2]
    depth_left = pad_d_left - cache_t
    return torch.empty(
        (
            x.shape[0],
            x.shape[1],
            x.shape[2] + cache_t + depth_left + pad_d_right,
            x.shape[3] + pad_h_top + pad_h_bottom,
            x.shape[4] + pad_w_left + pad_w_right,
        ),
        device=x.device,
        dtype=x.dtype,
    )


@register_custom_op(
    op_name="diffusion_causal_conv3d_cat_pad",
    mutates_args=[],
    fake_impl=_causal_conv3d_cat_pad_fake_impl,
)
def _causal_conv3d_cat_pad_custom_op(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    pad_w_left: int,
    pad_w_right: int,
    pad_h_top: int,
    pad_h_bottom: int,
    pad_d_left: int,
    pad_d_right: int,
) -> torch.Tensor:
    out = _causal_conv3d_cat_pad_fake_impl(
        x,
        cache_x,
        pad_w_left,
        pad_w_right,
        pad_h_top,
        pad_h_bottom,
        pad_d_left,
        pad_d_right,
    )
    module = _jit_causal_conv3d_cat_pad_module(x.dtype)
    module.causal_conv3d_cat_pad(
        out,
        x,
        cache_x,
        pad_w_left,
        pad_w_right,
        pad_h_top,
        pad_h_bottom,
        pad_d_left,
        pad_d_right,
    )
    return out


def fused_causal_conv3d_cat_pad_cuda(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    padding: list[int] | tuple[int, ...],
) -> torch.Tensor:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"unsupported dtype for causal Conv3D cat/pad: {x.dtype}")
    if not torch.compiler.is_compiling():
        if (
            not x.is_cuda
            or not cache_x.is_cuda
            or x.dim() != 5
            or cache_x.dim() != 5
            or not x.is_contiguous()
            or not cache_x.is_contiguous()
            or not can_use_fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
        ):
            raise RuntimeError("unsupported input for causal Conv3D cat/pad CUDA")
    return _causal_conv3d_cat_pad_custom_op(x, cache_x, *padding)


def can_use_fused_causal_conv3d_cat_pad_cuda(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    padding: list[int] | tuple[int, ...],
) -> bool:
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_left, pad_d_right = padding
    cache_t = cache_x.shape[2]
    depth_left = pad_d_left - cache_t
    if depth_left < 0 or pad_d_right != 0:
        return False
    out_numel = (
        x.shape[0]
        * x.shape[1]
        * (x.shape[2] + cache_t + depth_left + pad_d_right)
        * (x.shape[3] + pad_h_top + pad_h_bottom)
        * (x.shape[4] + pad_w_left + pad_w_right)
    )
    elem_size = 4 if x.dtype == torch.float32 else 2
    vec_elems = 16 // elem_size
    return out_numel % vec_elems == 0
