from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit


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


@cache_once
def _jit_rotary_embedding_cos_sin_module():
    return load_jit(
        "rotary_embedding_cos_sin",
        cuda_files=["rotary_embedding_cos_sin.cuh"],
        cuda_wrappers=[
            ("rotary_embedding_cos_sin_q", "RotaryEmbeddingCosSinKernel::run_q"),
            ("rotary_embedding_cos_sin_qk", "RotaryEmbeddingCosSinKernel::run_qk"),
        ],
    )


def rotary_embedding_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
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
    if cos.shape[0] != query.shape[0]:
        raise ValueError(
            f"cos/sin num_tokens mismatch with query: cos.shape[0]={cos.shape[0]} vs query.shape[0]={query.shape[0]}"
        )

    if head_size == 0:
        if query.dim() == 3:
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
        # Pairwise layout: rotates 2*rot_dim elements.
        if 2 * rot_dim > head_size:
            raise ValueError(
                f"rotate dim exceeds head_size for interleaved=True: 2*rot_dim={2*rot_dim} > head_size={head_size}"
            )
    else:
        # Split-halves layout: kernel interprets cos/sin as [rot_dim] where embed_dim=rot_dim/2.
        if rot_dim % 2 != 0:
            raise ValueError(
                f"non-interleaved requires even rot_dim (cos.shape[1]), got rot_dim={rot_dim}"
            )
        if rot_dim > head_size:
            raise ValueError(
                f"rotate dim exceeds head_size for interleaved=False: rot_dim={rot_dim} > head_size={head_size}"
            )

    def _as_3d(x: torch.Tensor, h: int) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[-1] != head_size:
                raise ValueError("head_size mismatch with query/key last dim")
            return x
        if x.dim() != 2:
            raise ValueError("query/key must be 2D or 3D tensors")
        if x.shape[1] % head_size != 0:
            raise ValueError("hidden_size is not divisible by head_size")
        return x.view(x.shape[0], h, head_size)

    if query.dim() == 3:
        num_heads = int(query.shape[1])
    else:
        num_heads = int(query.shape[1] // head_size)
    q3 = _as_3d(query, num_heads)
    if not q3.is_contiguous():
        # The CUDA kernel assumes contiguous [T, H, D] layout for fixed-stride addressing.
        raise ValueError("query must be contiguous (view must stay contiguous)")

    k3 = None
    if key is not None:
        if key.dim() == 3:
            num_kv_heads = int(key.shape[1])
        else:
            num_kv_heads = int(key.shape[1] // head_size)
        k3 = _as_3d(key, num_kv_heads)
        if not k3.is_contiguous():
            raise ValueError("key must be contiguous (view must stay contiguous)")

    module = _jit_rotary_embedding_cos_sin_module()
    if k3 is None:
        module.rotary_embedding_cos_sin_q(
            cos_to_use, sin_to_use, q3, head_size, effective_interleaved
        )
    else:
        module.rotary_embedding_cos_sin_qk(
            cos_to_use, sin_to_use, q3, k3, head_size, effective_interleaved
        )
