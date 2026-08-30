from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_HEAD_DIM = 128
_ALIGN = 32


@cache_once
def flux2_qkv_epilogue_module() -> Module:
    return load_jit(
        "flux2_qkv_epilogue_bf16",
        cuda_files=["diffusion/flux2_qkv_epilogue.cuh"],
        cuda_wrappers=[
            (
                "flux2_qkv_epilogue",
                "flux2_qkv_epilogue::Flux2QKVEpilogueKernel::run",
            )
        ],
    )


def _qkv_tensor(tensor: torch.Tensor, like: torch.Tensor | None = None) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.dtype == torch.bfloat16
        and tensor.ndim == 4
        and tensor.shape[0] == 1
        and tensor.shape[-1] == _HEAD_DIM
        and tensor.numel() > 0
        and tensor.stride(-1) == 1
        and tensor.stride(-2) == _HEAD_DIM
        and tensor.data_ptr() % _ALIGN == 0
        and (
            like is None
            or (
                tensor.device == like.device
                and tensor.shape == like.shape
                and tensor.stride(1) == like.stride(1)
            )
        )
    )


def try_fused_flux2_qkv_epilogue(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    img_eps: float,
    txt_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fuse FLUX.2 Q/K norm, RoPE, QKV packing, and text/image concat."""
    if torch.compiler.is_compiling():
        return None
    if not (
        _qkv_tensor(img_q)
        and _qkv_tensor(img_k, img_q)
        and _qkv_tensor(img_v, img_q)
        and _qkv_tensor(txt_q)
        and _qkv_tensor(txt_k, txt_q)
        and _qkv_tensor(txt_v, txt_q)
        and img_q.shape[2] == txt_q.shape[2]
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(img_q.device)[0] >= 10
    ):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None

    weights = []
    for tensor in (img_q_weight, img_k_weight, txt_q_weight, txt_k_weight):
        if not (
            isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and tensor.device == img_q.device
            and tensor.dtype == torch.bfloat16
            and tensor.shape == (_HEAD_DIM,)
            and tensor.is_contiguous()
            and tensor.data_ptr() % _ALIGN == 0
        ):
            return None
        weights.append(tensor)

    total_tokens = txt_q.shape[1] + img_q.shape[1]
    if not (
        isinstance(cos_sin_cache, torch.Tensor)
        and cos_sin_cache.is_cuda
        and cos_sin_cache.device == img_q.device
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[0] >= total_tokens
        and cos_sin_cache.shape[1] == _HEAD_DIM
        and cos_sin_cache.is_contiguous()
    ):
        return None

    heads = img_q.shape[2]
    joint_shape = (1, total_tokens, heads, _HEAD_DIM)
    joint_q = torch.empty(joint_shape, dtype=img_q.dtype, device=img_q.device)
    joint_k = torch.empty_like(joint_q)
    joint_v = torch.empty_like(joint_q)
    flux2_qkv_epilogue_module().flux2_qkv_epilogue(
        joint_q.view(-1, heads, _HEAD_DIM),
        joint_k.view(-1, heads, _HEAD_DIM),
        joint_v.view(-1, heads, _HEAD_DIM),
        img_q.view(-1, heads, _HEAD_DIM),
        img_k.view(-1, heads, _HEAD_DIM),
        img_v.view(-1, heads, _HEAD_DIM),
        txt_q.view(-1, heads, _HEAD_DIM),
        txt_k.view(-1, heads, _HEAD_DIM),
        txt_v.view(-1, heads, _HEAD_DIM),
        *weights,
        cos_sin_cache,
        float(img_eps),
        float(txt_eps),
    )
    return joint_q, joint_k, joint_v
