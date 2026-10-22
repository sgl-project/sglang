# SPDX-License-Identifier: Apache-2.0
# ViT runtime helpers for the MiniMax H3 visual VAE.
from collections.abc import Sequence
from typing import Tuple

import torch


def create_token_ids(
    patch_dims, device, dtype, id_type="length_normalized", flatten=True
):
    coords_list = []

    if isinstance(id_type, str):
        id_type_list = [id_type] * len(patch_dims)
    elif isinstance(id_type, list):
        id_type_list = id_type
        if len(id_type_list) != len(patch_dims):
            raise ValueError("id_type list must match patch_dims")
    else:
        raise ValueError("id_type must be a string or a list")

    if "area_normalized" in id_type_list or id_type == "area_normalized":
        raise NotImplementedError(
            "area_normalized id_type is not supported in this inference-only bundle"
        )

    for _dim_size, _id_type in zip(patch_dims, id_type_list):
        if isinstance(_dim_size, torch.Tensor):
            coords_list.append(_dim_size.to(device=device, dtype=dtype))
            continue

        if _id_type == "length_normalized":
            coords = torch.arange(0.5, _dim_size, dtype=dtype, device=device)
            coords = coords / _dim_size
            coords = 2.0 * coords - 1.0
        else:
            coords = torch.arange(_dim_size, dtype=dtype, device=device)

        coords_list.append(coords)

    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    if flatten:
        coords = coords.flatten(0, len(patch_dims) - 1)

    return coords.unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_impl(
    t: torch.Tensor, rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = rotary_pos_emb[:2]

    if cos.dim() != 4:
        raise ValueError(f"cos must be [B, N, 1, D], got {cos.shape}")

    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    rot_dim = cos.shape[-1]
    t_dim = t.shape[-1]

    if rot_dim < t_dim:
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        scaled = t_rot * cos
        scaled.add_(_rotate_half(t_rot) * sin)
        t_rot = scaled
        t = torch.cat((t_rot, t_pass), dim=-1)
    else:
        scaled = t * cos
        scaled.add_(_rotate_half(t) * sin)
        t = scaled

    return t


def prepare_rotary_pos_emb(
    rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    *,
    dtype: torch.dtype,
    batch: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Prebuild the native Q/K rotary cache once per ViT decoder forward; its
    positions cover ``batch`` samples sharing one token grid."""
    cos, sin = rotary_pos_emb
    if (
        not cos.is_cuda
        or dtype not in (torch.float16, torch.bfloat16)
        or cos.shape != sin.shape
        or cos.dim() != 4
        or cos.shape[0] != 1
        or cos.shape[2] != 1
        or cos.shape[-1] % 2
    ):
        return cos, sin

    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)
    half = cos.shape[-1] // 2
    # RotaryEmbeddingND repeats each half. The native kernel consumes the
    # compact NeoX cache [cos_half | sin_half].
    cache = torch.cat(
        (cos[0, :, 0, :half], sin[0, :, 0, :half]),
        dim=-1,
    ).contiguous()
    positions = torch.arange(
        cos.shape[1],
        dtype=torch.long,
        device=cos.device,
    ).repeat(batch)
    return cos, sin, cache, positions


def apply_rotary_pos_emb(
    t: torch.Tensor, rotary_pos_emb: Sequence[torch.Tensor]
) -> torch.Tensor:
    return _apply_rotary_pos_emb_impl(t, rotary_pos_emb)


def native_rope_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_pos_emb: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(cache, positions)`` from ``prepare_rotary_pos_emb`` when the native
    kernels can consume it for this Q/K, else None. ``positions`` covers the
    batch-major rows of ``query.flatten(0, 1)``."""
    if not (
        len(rotary_pos_emb) == 4
        and query.is_cuda
        and query.shape == key.shape
        and query.dtype == key.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dim() == 4
        and not torch.compiler.is_compiling()
    ):
        return None
    _, _, cache, positions = rotary_pos_emb
    if not (
        cache.is_cuda
        and cache.dtype == query.dtype
        and cache.dim() == 2
        and cache.shape[0] == query.shape[1]
        and cache.shape[1] <= query.shape[-1]
        and positions.is_cuda
        and positions.shape in ((query.shape[1],), (query.shape[0] * query.shape[1],))
    ):
        return None
    # Accept both main's shared sequence positions and the pre-expanded
    # batch-major positions used by the decoder cache and fused fast path.
    if positions.shape == (query.shape[1],) and query.shape[0] > 1:
        positions = positions.repeat(query.shape[0])
    return cache, positions


def apply_rotary_pos_emb_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_pos_emb: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact native NeoX rotary kernel to Q/K together when possible."""
    native = native_rope_cache(query, key, rotary_pos_emb)
    if native is not None:
        from sgl_kernel import rotary_embedding

        cache, positions = native
        query = query.contiguous()
        key = key.contiguous()
        rotary_embedding(
            positions,
            query.view(positions.shape[0], -1),
            key.view(positions.shape[0], -1),
            query.shape[-1],
            cache,
            True,
        )
        return query, key

    return (
        apply_rotary_pos_emb(query, rotary_pos_emb),
        apply_rotary_pos_emb(key, rotary_pos_emb),
    )
