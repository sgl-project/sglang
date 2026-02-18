from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rotary_embedding_cos_sin_module(rot: int) -> Module:
    return load_jit(
        "rotary_embedding_cos_sin",
        str(rot),
        cuda_files=["rotary_embedding_cos_sin.cuh"],
        cuda_wrappers=[
            (
                "rotary_embedding_cos_sin_q",
                f"RotaryEmbeddingCosSinKernel<{rot}>::run_q",
            ),
            (
                "rotary_embedding_cos_sin_qk",
                f"RotaryEmbeddingCosSinKernel<{rot}>::run_qk",
            ),
        ],
    )


@cache_once
def can_use_rotary_embedding_cos_sin(rot: int) -> bool:
    if rot <= 0:
        logging.getLogger(__name__).warning(f"Invalid rot={rot}")
        return False
    try:
        _jit_rotary_embedding_cos_sin_module(rot)
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(f"JIT compile failed (rot={rot}): {e}")
        return False


def rotary_embedding_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
    positions: Optional[torch.Tensor] = None,
    *,
    is_neox: Optional[bool] = None,
) -> None:
    if is_neox is not None:
        if interleaved is not None and interleaved != is_neox:
            raise ValueError(f"Mismatch: is_neox={is_neox}, interleaved={interleaved}")
        interleaved = is_neox
    interleaved = interleaved if interleaved is not None else True

    if head_size == 0:
        if query.dim() < 3:
            raise ValueError("head_size must be provided when query is 2D")
        head_size = query.shape[-1]

    def _prepare(t: torch.Tensor, name: str) -> torch.Tensor:
        if t.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.dtype != query.dtype:
            raise ValueError(f"{name} dtype mismatch")

        # Reshape to [tokens, heads, head_size]
        if t.dim() == 2:
            return t.view(t.shape[0], -1, head_size)
        if t.dim() == 3:
            if t.shape[-1] != head_size:
                raise ValueError(f"{name} head_size mismatch")
            return t
        if t.dim() == 4:
            if t.shape[-1] != head_size:
                raise ValueError(f"{name} head_size mismatch")
            return t.flatten(0, 1)
        raise ValueError(f"{name} must be 2D, 3D or 4D")

    q_3d = _prepare(query, "query")
    k_3d = _prepare(key, "key") if key is not None else None

    if cos.device != query.device or sin.device != query.device:
        raise ValueError("cos/sin device mismatch")
    if cos.dtype != query.dtype or sin.dtype != query.dtype:
        raise ValueError("cos/sin dtype mismatch")
    if positions is None and cos.shape[0] != q_3d.shape[0]:
        raise ValueError(f"cos/sin shape {cos.shape} mismatches tokens {q_3d.shape[0]}")

    if interleaved and cos.shape[1] == head_size:
        if head_size % 2 != 0:
            raise ValueError("interleaved layout requires even head_size")
        half = head_size // 2
        cos = cos.view(cos.shape[0], half, 2).select(2, 0).contiguous()
        sin = sin.view(sin.shape[0], half, 2).select(2, 1).contiguous()
    else:
        cos, sin = cos.contiguous(), sin.contiguous()

    rot_dim = cos.shape[1]
    if rot_dim <= 0:
        raise ValueError("rot_dim must be > 0")

    if interleaved:
        if 2 * rot_dim > head_size:
            raise ValueError(f"rot_dim {rot_dim} too large for interleaved")
        embed_dim = rot_dim
    else:
        if rot_dim % 2 != 0:
            raise ValueError("non-interleaved requires even rot_dim")
        if rot_dim > head_size:
            raise ValueError(f"rot_dim {rot_dim} too large")
        embed_dim = rot_dim // 2

    module = _jit_rotary_embedding_cos_sin_module(embed_dim)
    if k_3d is not None:
        module.rotary_embedding_cos_sin_qk(
            cos, sin, q_3d, k_3d, positions, head_size, interleaved
        )
    else:
        module.rotary_embedding_cos_sin_q(
            cos, sin, q_3d, positions, head_size, interleaved
        )
