from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

_H200_RMS_DIM = 128
_H200_LN_N = 5120
_USE_PDL = False

_B200_RMS_LARGE_S = 100000
_B200_RMS_LARGE_UNROLL = 4
_B200_LN_ALIGN = 16
_B200_RMS_ALIGN = 8
_B200_SUPPORTED_LN = frozenset(
    {
        (8640, 5120),
        (6, 512),
        (6, 3072),
        (12, 512),
        (12, 3072),
        (128, 512),
        (128, 3072),
        (256, 512),
        (256, 3072),
        (64, 1024),
        (256, 1024),
    }
)
_B200_SUPPORTED_RMS = frozenset(
    {
        (1320, 128),
        (16384, 128),
        (4096, 128),
        (6, 128),
        (128, 128),
        (768, 128),
        (64, 128),
    }
)


def _device_capability(t: torch.Tensor) -> tuple[int, int]:
    return torch.cuda.get_device_capability(t.device)


def _is_h200(t: torch.Tensor) -> bool:
    return _device_capability(t) == (9, 0)


def _is_blackwell(t: torch.Tensor) -> bool:
    return _device_capability(t)[0] >= 10


def _same_device_dtype_1d(t: Optional[torch.Tensor], x: torch.Tensor, n: int) -> bool:
    return (
        t is not None
        and t.device == x.device
        and t.dtype == x.dtype
        and t.dim() == 1
        and t.numel() == n
        and t.is_contiguous()
    )


def _aligned(t: Optional[torch.Tensor], nbytes: int) -> bool:
    del nbytes
    return t is None or t.storage_offset() == 0


@cache_once
def _h200_rms_module(dtype: torch.dtype):
    args = make_cpp_args(_H200_RMS_DIM, _USE_PDL, dtype)
    return load_jit(
        "diffusion_native_norm_h200_rms",
        *args,
        cuda_files=["diffusion/norm_infer_h200_rms_norm.cuh"],
        cuda_wrappers=[("rms_norm", f"RmsNormKernel<{args}>::run")],
    )


@cache_once
def _h200_ln_module(dtype: torch.dtype, has_bias: bool):
    args = make_cpp_args(_H200_LN_N, has_bias, _USE_PDL, dtype)
    return load_jit(
        "diffusion_native_norm_h200_ln",
        *args,
        cuda_files=["diffusion/norm_infer_h200_layer_norm.cuh"],
        cuda_wrappers=[("layer_norm", f"LayerNormKernel<{args}>::run")],
    )


@cache_once
def _b200_ln_module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_native_norm_b200_ln",
        "v1",
        *args,
        cuda_files=["diffusion/norm_infer_b200.cuh"],
        cuda_wrappers=[("norm_infer_ln", f"LayerNormInferKernel<{args}>::run")],
    )


@cache_once
def _b200_rms_module(dim: int, unroll: int, dtype: torch.dtype):
    args = make_cpp_args(dim, unroll, dtype)
    return load_jit(
        "diffusion_native_norm_b200_rms",
        "v1",
        *args,
        cuda_files=["diffusion/norm_infer_b200.cuh"],
        cuda_wrappers=[("rms_onepass", f"RmsNormOnepassKernel<{args}>::run")],
    )


def _can_use_h200_norm_infer(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    is_rms_norm: bool,
    out: Optional[torch.Tensor],
) -> bool:
    return (
        not is_rms_norm
        and out is None
        and x.is_cuda
        and _is_h200(x)
        and x.dtype == torch.float32
        and x.shape[-1] == _H200_LN_N
        and x.is_contiguous()
        and _same_device_dtype_1d(weight, x, _H200_LN_N)
        and _same_device_dtype_1d(bias, x, _H200_LN_N)
    )


def _can_use_b200_norm_infer(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    is_rms_norm: bool,
    out: Optional[torch.Tensor],
) -> bool:
    if not (
        not is_rms_norm
        and out is None
        and x.is_cuda
        and _is_blackwell(x)
        and x.dtype == torch.float32
        and x.dim() == 2
        and x.is_contiguous()
    ):
        return False
    m, n = int(x.shape[0]), int(x.shape[1])
    return (
        (m, n) in _B200_SUPPORTED_LN
        and _same_device_dtype_1d(weight, x, n)
        and _same_device_dtype_1d(bias, x, n)
        and _aligned(x, _B200_LN_ALIGN)
        and _aligned(weight, _B200_LN_ALIGN)
        and _aligned(bias, _B200_LN_ALIGN)
    )


def can_use_native_norm_infer(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    is_rms_norm: bool,
    out: Optional[torch.Tensor],
) -> bool:
    return _can_use_h200_norm_infer(
        x, weight, bias, is_rms_norm, out
    ) or _can_use_b200_norm_infer(x, weight, bias, is_rms_norm, out)


def can_use_native_one_pass_rms_norm(x: torch.Tensor, w: torch.Tensor) -> bool:
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and w.device == x.device
        and w.dtype == x.dtype
        and w.dim() == 1
        and w.numel() == _H200_RMS_DIM
        and w.is_contiguous()
    ):
        return False
    if _is_h200(x):
        return x.shape[-1] == _H200_RMS_DIM and x.is_contiguous()
    if _is_blackwell(x):
        return (
            x.dim() == 2
            and x.is_contiguous()
            and tuple(int(v) for v in x.shape) in _B200_SUPPORTED_RMS
            and _aligned(x, _B200_RMS_ALIGN)
            and _aligned(w, _B200_RMS_ALIGN)
        )
    return False


@register_custom_op(op_name="diffusion_native_norm_infer_cuda", out_shape="x")
def _native_norm_infer_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = torch.empty_like(x)
    if _is_h200(x):
        _h200_ln_module(x.dtype, True).layer_norm(
            x.reshape(-1, _H200_LN_N),
            weight,
            bias,
            y.reshape(-1, _H200_LN_N),
            float(eps),
        )
        return y
    _b200_ln_module(x.dtype).norm_infer_ln(x, weight, bias, y, float(eps))
    return y


@register_custom_op(op_name="diffusion_native_one_pass_rms_norm_cuda", out_shape="x")
def _native_one_pass_rms_norm_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    shape = x.shape
    y = torch.empty_like(x)
    x2d = x.reshape(-1, shape[-1])
    y2d = y.reshape(-1, shape[-1])
    if _is_h200(x):
        _h200_rms_module(x.dtype).rms_norm(x2d, w, y2d, float(eps))
    else:
        unroll = _B200_RMS_LARGE_UNROLL if x2d.shape[0] >= _B200_RMS_LARGE_S else 1
        _b200_rms_module(shape[-1], unroll, x.dtype).rms_onepass(
            x2d, w, y2d, float(eps)
        )
    return y


def try_native_norm_infer(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    is_rms_norm: bool = False,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if can_use_native_norm_infer(x, weight, bias, is_rms_norm, out):
        assert weight is not None and bias is not None
        return _native_norm_infer_cuda(x, weight, bias, eps)
    return None


def try_native_one_pass_rms_norm(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> Optional[torch.Tensor]:
    if can_use_native_one_pass_rms_norm(x, w):
        return _native_one_pass_rms_norm_cuda(x, w, eps)
    return None


__all__ = [
    "can_use_native_norm_infer",
    "can_use_native_one_pass_rms_norm",
    "try_native_norm_infer",
    "try_native_one_pass_rms_norm",
]
