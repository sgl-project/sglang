from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rotary_embedding_cos_sin_module(rot: int) -> Module:
    # Compile one specialized template per rotary embed dim (ROT).
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
    logger = logging.getLogger(__name__)
    if rot <= 0:
        logger.warning(f"Invalid rot={rot} for rotary_embedding_cos_sin")
        return False
    try:
        _jit_rotary_embedding_cos_sin_module(rot)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT rotary_embedding_cos_sin (rot={rot}): {e}")
        return False


def rotary_embedding_cos_sin_q(
    *,
    rot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    head_size: int,
    interleaved: bool,
    positions: Optional[torch.Tensor] = None,
) -> None:
    module = _jit_rotary_embedding_cos_sin_module(rot)
    module.rotary_embedding_cos_sin_q(cos, sin, query, positions, head_size, interleaved)


def rotary_embedding_cos_sin_qk(
    *,
    rot: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    interleaved: bool,
    positions: Optional[torch.Tensor] = None,
) -> None:
    module = _jit_rotary_embedding_cos_sin_module(rot)
    module.rotary_embedding_cos_sin_qk(
        cos, sin, query, key, positions, head_size, interleaved
    )


def _resolve_interleaved(interleaved: Optional[bool], is_neox: Optional[bool]) -> bool:
    if interleaved is not None and is_neox is not None and interleaved != is_neox:
        raise ValueError(
            f"is_neox({is_neox}) and interleaved({interleaved}) mismatch; keep only one or make them equal."
        )
    if is_neox is not None:
        return is_neox
    if interleaved is not None:
        return interleaved
    return True


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
    effective_interleaved = _resolve_interleaved(interleaved, is_neox)

    if query.device.type != "cuda":
        raise ValueError("query must be a CUDA tensor")
    if key is not None and key.device.type != "cuda":
        raise ValueError("key must be a CUDA tensor")
    if cos.device != query.device or sin.device != query.device:
        raise ValueError("cos/sin must be on the same device as query")

    if query.dtype != cos.dtype or query.dtype != sin.dtype:
        raise ValueError("cos/sin dtype must match query dtype")
    if key is not None and key.dtype != query.dtype:
        raise ValueError("key dtype must match query dtype")

    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError("cos/sin must be 2D tensors: [num_tokens, rot_dim]")
    if cos.shape != sin.shape:
        raise ValueError("cos/sin shape mismatch")
    if query.dim() == 4:
        expected_tokens = int(query.shape[0] * query.shape[1])
    elif query.dim() in (2, 3):
        expected_tokens = int(query.shape[0])
    else:
        raise ValueError("query must be a 2D, 3D or 4D tensor")
    if positions is None and cos.shape[0] != expected_tokens:
        raise ValueError(
            f"cos/sin num_tokens mismatch with query: cos.shape[0]={cos.shape[0]} vs expected_tokens={expected_tokens}"
        )

    if head_size == 0:
        if query.dim() in (3, 4):
            head_size = int(query.shape[-1])
        else:
            raise ValueError("head_size must be provided when query is 2D")

    if not query.is_contiguous():
        raise ValueError("query must be contiguous (in-place kernel)")
    if key is not None and (not key.is_contiguous()):
        raise ValueError("key must be contiguous (in-place kernel)")

    # Optional downsample for interleaved caches that are stored as full-dim (repeat format).
    # Keep this consistent with AOT kernel behavior.
    if effective_interleaved and cos.shape[1] == head_size:
        if head_size % 2 != 0:
            raise ValueError("interleaved layout requires even head_size")
        half = head_size // 2
        cos_to_use = cos.view(cos.shape[0], half, 2).select(2, 0).contiguous()
        sin_to_use = sin.view(sin.shape[0], half, 2).select(2, 1).contiguous()
    else:
        cos_to_use = cos.contiguous()
        sin_to_use = sin.contiguous()

    rot_dim = int(cos_to_use.shape[1])
    if rot_dim <= 0:
        raise ValueError("cos/sin rot_dim must be > 0")
    if effective_interleaved:
        if 2 * rot_dim > head_size:
            raise ValueError(
                f"rotate dim exceeds head_size for interleaved=True: 2*rot_dim={2*rot_dim} > head_size={head_size}"
            )
    else:
        if rot_dim % 2 != 0:
            raise ValueError(
                f"non-interleaved requires even rot_dim (cos.shape[1]), got rot_dim={rot_dim}"
            )
        if rot_dim > head_size:
            raise ValueError(
                f"rotate dim exceeds head_size for interleaved=False: rot_dim={rot_dim} > head_size={head_size}"
            )

    embed_dim_for_rotation = rot_dim if effective_interleaved else (rot_dim // 2)

    def _as_3d(x: torch.Tensor, h: int) -> torch.Tensor:
        if x.dim() == 4:
            # [bsz, seqlen, num_heads, head_dim] -> [tokens, num_heads, head_dim]
            if x.shape[-1] != head_size:
                raise ValueError("head_size mismatch with query/key last dim")
            return x.flatten(0, 1)
        if x.dim() == 3:
            if x.shape[-1] != head_size:
                raise ValueError("head_size mismatch with query/key last dim")
            return x
        if x.dim() != 2:
            raise ValueError("query/key must be 2D, 3D or 4D tensors")
        if x.shape[1] % head_size != 0:
            raise ValueError("hidden_size is not divisible by head_size")
        return x.view(x.shape[0], h, head_size)

    if query.dim() == 4:
        num_heads = int(query.shape[2])
    elif query.dim() == 3:
        num_heads = int(query.shape[1])
    else:
        num_heads = int(query.shape[1] // head_size)
    q3 = _as_3d(query, num_heads)
    if not q3.is_contiguous():
        raise ValueError("query must be contiguous (view must stay contiguous)")

    k3 = None
    if key is not None:
        if key.dim() == 4:
            num_kv_heads = int(key.shape[2])
        elif key.dim() == 3:
            num_kv_heads = int(key.shape[1])
        else:
            num_kv_heads = int(key.shape[1] // head_size)
        k3 = _as_3d(key, num_kv_heads)
        if not k3.is_contiguous():
            raise ValueError("key must be contiguous (view must stay contiguous)")

    if k3 is None:
        rotary_embedding_cos_sin_q(
            rot=embed_dim_for_rotation,
            cos=cos_to_use,
            sin=sin_to_use,
            query=q3,
            head_size=head_size,
            interleaved=effective_interleaved,
            positions=positions,
        )
    else:
        rotary_embedding_cos_sin_qk(
            rot=embed_dim_for_rotation,
            cos=cos_to_use,
            sin=sin_to_use,
            query=q3,
            key=k3,
            head_size=head_size,
            interleaved=effective_interleaved,
            positions=positions,
        )
