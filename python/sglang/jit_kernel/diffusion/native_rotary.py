from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

_USE_PDL = False
_STD_TOKENS = 27030
_STD_HEADS = 24
_STD_HEAD_DIM = 128
_LTX2_HEADS = 32
_B200_LTX2_SIGS = frozenset(
    {
        (1, 1536, 4096),
        (1, 126, 2048),
        (1, 1536, 2048),
        (1, 6144, 4096),
        (1, 6144, 2048),
        (2, 6144, 4096),
        (2, 126, 2048),
        (2, 6144, 2048),
        (1, 24576, 4096),
        (1, 24576, 2048),
    }
)
_H200_LTX2_SIGS = frozenset(
    {
        (1, 126, 2048),
        (1, 1536, 2048),
        (1, 1536, 4096),
        (1, 6144, 2048),
        (1, 6144, 4096),
    }
)


def _capability(t: torch.Tensor) -> tuple[int, int]:
    return torch.cuda.get_device_capability(t.device)


def _is_h200(t: torch.Tensor) -> bool:
    return _capability(t) == (9, 0)


def _is_blackwell(t: torch.Tensor) -> bool:
    return _capability(t)[0] >= 10


def _same_cuda_device(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return False
    device = tensors[0].device
    return all(t.is_cuda and t.device == device for t in tensors)


def _aligned16(t: torch.Tensor) -> bool:
    return t.storage_offset() == 0


@cache_once
def _b200_standard_module(head_dim: int, dtype: torch.dtype, use_pdl: bool):
    args = make_cpp_args(head_dim, use_pdl, dtype)
    return load_jit(
        "diffusion_native_rotary_b200_standard",
        *args,
        cuda_files=["diffusion/rotary_embedding_b200.cuh"],
        cuda_wrappers=[
            (
                "apply_rotary",
                f"kda_diffusion_rotary::StandardRotaryKernel<{args}>::run",
            )
        ],
    )


@cache_once
def _b200_ltx2_module(half_dim: int, dtype: torch.dtype, use_pdl: bool):
    args = make_cpp_args(half_dim, use_pdl, dtype)
    return load_jit(
        "diffusion_native_rotary_b200_ltx2",
        *args,
        cuda_files=["diffusion/rotary_embedding_b200.cuh"],
        cuda_wrappers=[
            (
                "apply_ltx2",
                f"kda_diffusion_rotary::Ltx2SplitRotaryKernel<{args}>::run",
            )
        ],
    )


@cache_once
def _h200_module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_native_rotary_h200",
        *args,
        cuda_files=["diffusion/rotary_embedding_h200.cuh"],
        cuda_wrappers=[
            ("standard_rope", f"StandardRopeKernel<{args}>::run"),
            ("ltx2_split_rope", f"Ltx2SplitRopeKernel<{args}>::run"),
        ],
    )


def _standard_common_dtype_device(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
) -> bool:
    return (
        not interleaved
        and _same_cuda_device(x, cos, sin)
        and x.dtype == torch.bfloat16
        and cos.dtype == torch.float32
        and sin.dtype == torch.float32
    )


def _can_use_b200_standard(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
) -> bool:
    if not _standard_common_dtype_device(x, cos, sin, interleaved):
        return False
    if x.dim() == 4:
        if tuple(x.shape) != (1, _STD_TOKENS, _STD_HEADS, _STD_HEAD_DIM):
            return False
    elif x.dim() == 3:
        if tuple(x.shape) != (_STD_TOKENS, _STD_HEADS, _STD_HEAD_DIM):
            return False
    else:
        return False
    if (
        x.stride(-1) != 1
        or x.stride(-2) != _STD_HEAD_DIM
        or x.stride(-3) != _STD_HEADS * _STD_HEAD_DIM
    ):
        return False
    half = _STD_HEAD_DIM // 2
    return (
        _aligned16(x)
        and _aligned16(cos)
        and _aligned16(sin)
        and tuple(cos.shape) == (_STD_TOKENS, half)
        and tuple(sin.shape) == (_STD_TOKENS, half)
        and tuple(cos.stride()) == (half, 1)
        and tuple(sin.stride()) == (half, 1)
    )


def _can_use_h200_standard(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
) -> bool:
    half = _STD_HEAD_DIM // 2
    return (
        _standard_common_dtype_device(x, cos, sin, interleaved)
        and tuple(x.shape) == (1, _STD_TOKENS, _STD_HEADS, _STD_HEAD_DIM)
        and x.is_contiguous()
        and _aligned16(x)
        and _aligned16(cos)
        and _aligned16(sin)
        and tuple(cos.shape) == (_STD_TOKENS, half)
        and tuple(sin.shape) == (_STD_TOKENS, half)
        and cos.is_contiguous()
        and sin.is_contiguous()
    )


def can_use_native_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> bool:
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    if _is_blackwell(x):
        return _can_use_b200_standard(x, cos, sin, interleaved)
    if _is_h200(x):
        return _can_use_h200_standard(x, cos, sin, interleaved)
    return False


def _ltx2_common_dtype_device(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> bool:
    return (
        _same_cuda_device(x, cos, sin)
        and x.dtype == torch.bfloat16
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and x.dim() == 3
        and cos.dim() == 4
        and sin.dim() == 4
    )


def _can_use_ltx2_with_sigs(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sigs: frozenset[tuple[int, int, int]],
) -> bool:
    if not _ltx2_common_dtype_device(x, cos, sin):
        return False
    b, seq, inner = (int(v) for v in x.shape)
    if (b, seq, inner) not in sigs:
        return False
    half = inner // (2 * _LTX2_HEADS)
    if half not in (32, 64) or 2 * _LTX2_HEADS * half != inner:
        return False
    if tuple(x.stride()) != (seq * inner, inner, 1):
        return False
    want_shape = (b, _LTX2_HEADS, seq, half)
    want_stride = (seq * _LTX2_HEADS * half, half, _LTX2_HEADS * half, 1)
    return (
        _aligned16(x)
        and _aligned16(cos)
        and _aligned16(sin)
        and tuple(cos.shape) == want_shape
        and tuple(sin.shape) == want_shape
        and tuple(cos.stride()) == want_stride
        and tuple(sin.stride()) == want_stride
    )


def can_use_native_ltx2_split_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> bool:
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    if _is_blackwell(x):
        return _can_use_ltx2_with_sigs(x, cos, sin, _B200_LTX2_SIGS)
    if _is_h200(x):
        return _can_use_ltx2_with_sigs(x, cos, sin, _H200_LTX2_SIGS)
    return False


@register_custom_op(op_name="diffusion_native_rotary_embedding_cuda", out_shape="x")
def _native_rotary_embedding_cuda(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    del interleaved
    out = torch.empty_like(x)
    heads, head_dim = int(x.shape[-2]), int(x.shape[-1])
    x3 = x.reshape(-1, heads, head_dim)
    out3 = out.reshape(-1, heads, head_dim)
    if _is_h200(x):
        _h200_module(x.dtype).standard_rope(out3, x3, cos, sin)
    else:
        _b200_standard_module(head_dim, x.dtype, _USE_PDL).apply_rotary(
            out3, x3, cos, sin
        )
    return out


@register_custom_op(op_name="diffusion_native_ltx2_split_rotary_cuda", out_shape="x")
def _native_ltx2_split_rotary_cuda(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    if _is_h200(x):
        _h200_module(x.dtype).ltx2_split_rope(out, x, cos, sin)
    else:
        _b200_ltx2_module(cos.shape[-1], x.dtype, _USE_PDL).apply_ltx2(
            out, x, cos, sin
        )
    return out


def try_native_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> Optional[torch.Tensor]:
    if can_use_native_rotary_embedding(x, cos, sin, interleaved):
        return _native_rotary_embedding_cuda(x, cos, sin, interleaved)
    return None


def try_native_ltx2_split_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Optional[torch.Tensor]:
    if can_use_native_ltx2_split_rotary_emb(x, cos, sin):
        return _native_ltx2_split_rotary_cuda(x, cos, sin)
    return None


__all__ = [
    "can_use_native_rotary_embedding",
    "can_use_native_ltx2_split_rotary_emb",
    "try_native_rotary_embedding",
    "try_native_ltx2_split_rotary_emb",
]
