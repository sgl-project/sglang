from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

_SUPPORTED_X_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_SUPPORTED_SCALE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _is_blackwell(t: torch.Tensor) -> bool:
    return torch.cuda.get_device_capability(t.device)[0] >= 10


def _same_cuda_device(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return False
    dev = tensors[0].device
    return all(t.is_cuda and t.device == dev for t in tensors)


def _torch_extension_flags() -> tuple[list[str], list[str]]:
    from torch.utils import cpp_extension as tce

    try:
        include_paths = list(tce.include_paths(device_type="cuda"))
        library_paths = list(tce.library_paths(device_type="cuda"))
    except TypeError:
        include_paths = list(tce.include_paths())
        library_paths = list(tce.library_paths())
    ldflags = [f"-L{p}" for p in library_paths]
    ldflags += [f"-Wl,-rpath,{p}" for p in library_paths]
    ldflags += ["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda"]
    return include_paths, ldflags


@cache_once
def _b200_module():
    include_paths, ldflags = _torch_extension_flags()
    return load_jit(
        "diffusion_native_scale_shift_b200",
        "v1",
        cuda_files=["diffusion/scale_shift_b200.cu"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        header_only=False,
    )


def _can_use_native_scale_shift(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(scale, torch.Tensor)
        and isinstance(shift, torch.Tensor)
        and x.is_cuda
        and _is_blackwell(x)
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dtype in _SUPPORTED_X_DTYPES
        and scale.dtype in _SUPPORTED_SCALE_DTYPES
        and shift.dtype == scale.dtype
        and _same_cuda_device(x, scale, shift)
        and x.dim() == 3
        and x.is_contiguous()
    ):
        return False
    if x.dtype == torch.float32 and scale.dtype != torch.float32:
        return False
    if x.dtype == torch.bfloat16 and scale.dtype == torch.float16:
        return False
    if x.dtype == torch.float16 and scale.dtype == torch.bfloat16:
        return False
    return True


def _can_use_native_select01(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
) -> bool:
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and _is_blackwell(x)
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dtype in _SUPPORTED_X_DTYPES
        and x.dim() == 3
        and x.is_contiguous()
        and all(
            isinstance(t, torch.Tensor)
            and t.dtype == x.dtype
            and t.dim() == 2
            and t.shape == (x.shape[0], x.shape[2])
            and t.is_cuda
            and t.device == x.device
            for t in (scale0, shift0, gate0, scale1, shift1, gate1)
        )
        and isinstance(index, torch.Tensor)
        and index.is_cuda
        and index.device == x.device
        and index.dim() == 2
        and index.shape == (x.shape[0], x.shape[1])
        and index.dtype in (torch.int32, torch.int64)
    ):
        return False
    if weight is not None and not (
        isinstance(weight, torch.Tensor)
        and weight.dtype == x.dtype
        and weight.dim() == 1
        and weight.shape == (x.shape[2],)
        and weight.is_cuda
        and weight.device == x.device
    ):
        return False
    if bias is not None and not (
        isinstance(bias, torch.Tensor)
        and bias.dtype == x.dtype
        and bias.dim() == 1
        and bias.shape == (x.shape[2],)
        and bias.is_cuda
        and bias.device == x.device
    ):
        return False
    return True


def _fake_pair(
    x,
    weight,
    bias,
    scale0,
    shift0,
    gate0,
    scale1,
    shift1,
    gate1,
    index,
    eps,
):
    return x.new_empty(x.shape), x.new_empty(x.shape)


def _fake_triple(
    x,
    residual,
    residual_gate,
    weight,
    bias,
    scale0,
    shift0,
    gate0,
    scale1,
    shift1,
    gate1,
    index,
    eps,
):
    return x.new_empty(x.shape), x.new_empty(x.shape), x.new_empty(x.shape)


@register_custom_op(op_name="diffusion_native_fuse_scale_shift_cuda", out_shape="x")
def _native_fuse_scale_shift_cuda(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _b200_module().fuse_scale_shift(x, scale, shift, float(scale_constant), out)
    return out


@register_custom_op(
    op_name="diffusion_native_layernorm_scale_shift_gate_select01_cuda",
    fake_impl=_fake_pair,
)
def _native_layernorm_scale_shift_gate_select01_cuda(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    gate_out = torch.empty_like(x)
    _b200_module().fuse_layernorm_scale_shift_gate_select01(
        x,
        weight,
        bias,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        float(eps),
        out,
        gate_out,
    )
    return out, gate_out


@register_custom_op(
    op_name="diffusion_native_residual_layernorm_scale_shift_gate_select01_cuda",
    fake_impl=_fake_triple,
)
def _native_residual_layernorm_scale_shift_gate_select01_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    gate_out = torch.empty_like(x)
    _b200_module().fuse_residual_layernorm_scale_shift_gate_select01(
        x,
        residual,
        residual_gate,
        weight,
        bias,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        float(eps),
        out,
        residual_out,
        gate_out,
    )
    return out, residual_out, gate_out


def try_native_fuse_scale_shift_kernel(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float,
) -> Optional[torch.Tensor]:
    if _can_use_native_scale_shift(x, scale, shift):
        try:
            return _native_fuse_scale_shift_cuda(x, scale, shift, float(scale_constant))
        except Exception:
            return None
    return None


def try_native_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if _can_use_native_select01(
        x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index
    ):
        try:
            return _native_layernorm_scale_shift_gate_select01_cuda(
                x,
                weight,
                bias,
                scale0,
                shift0,
                gate0,
                scale1,
                shift1,
                gate1,
                index,
                float(eps),
            )
        except Exception:
            return None
    return None


def try_native_residual_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not (
        isinstance(residual, torch.Tensor)
        and isinstance(residual_gate, torch.Tensor)
        and residual.shape == x.shape
        and residual_gate.shape == x.shape
        and residual.dtype == x.dtype
        and residual_gate.dtype == x.dtype
        and residual.is_cuda
        and residual_gate.is_cuda
        and residual.device == x.device
        and residual_gate.device == x.device
        and residual.is_contiguous()
        and residual_gate.is_contiguous()
    ):
        return None
    if _can_use_native_select01(
        x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index
    ):
        try:
            return _native_residual_layernorm_scale_shift_gate_select01_cuda(
                x,
                residual,
                residual_gate,
                weight,
                bias,
                scale0,
                shift0,
                gate0,
                scale1,
                shift1,
                gate1,
                index,
                float(eps),
            )
        except Exception:
            return None
    return None


__all__ = [
    "try_native_fuse_scale_shift_kernel",
    "try_native_layernorm_scale_shift_gate_select01",
    "try_native_residual_layernorm_scale_shift_gate_select01",
]
