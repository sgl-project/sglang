# SPDX-License-Identifier: Apache-2.0
# ViT runtime helpers for the MiniMax H3 visual VAE.
import os
from collections.abc import Sequence
from typing import Tuple

import torch
from diffusers.utils import logging


def _env_flag(name, default="0"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_optional_bool(name, default=""):
    value = str(os.environ.get(name, default)).strip().lower()
    if value in ("", "default", "auto", "none", "unset"):
        return None
    return value not in ("0", "false", "no", "off", "disabled")


def _vit_torch_compile_kwargs(prefix):
    kwargs = {}
    backend = os.environ.get(f"{prefix}_BACKEND", "inductor").strip()
    mode = os.environ.get(f"{prefix}_MODE", "reduce-overhead").strip()
    if backend and backend.lower() not in ("default", "none"):
        kwargs["backend"] = backend
    if mode and mode.lower() not in ("default", "none"):
        kwargs["mode"] = mode
    kwargs["fullgraph"] = _env_flag(f"{prefix}_FULLGRAPH", "0")
    dynamic = _env_optional_bool(f"{prefix}_DYNAMIC")
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    return kwargs


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
) -> tuple[torch.Tensor, ...]:
    """Prebuild the native Q/K rotary cache once per ViT decoder forward."""
    cos, sin = rotary_pos_emb
    if (
        not cos.is_cuda
        or dtype not in (torch.float16, torch.bfloat16)
        or cos.shape != sin.shape
        or cos.dim() != 4
        or cos.shape[0] != 1
        or cos.shape[2] != 1
        or cos.shape[-1] % 2
        or _env_flag("MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE", "0")
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
    )
    return cos, sin, cache, positions


_COMPILED_APPLY_ROTARY_POS_EMB = None
_APPLY_ROTARY_POS_EMB_COMPILE_DISABLED = False


def _get_apply_rotary_pos_emb_impl():
    global _COMPILED_APPLY_ROTARY_POS_EMB, _APPLY_ROTARY_POS_EMB_COMPILE_DISABLED
    if _APPLY_ROTARY_POS_EMB_COMPILE_DISABLED or not _env_flag(
        "MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE", "0"
    ):
        return _apply_rotary_pos_emb_impl
    if _COMPILED_APPLY_ROTARY_POS_EMB is not None:
        return _COMPILED_APPLY_ROTARY_POS_EMB
    if not hasattr(torch, "compile"):
        message = (
            "torch.compile is unavailable; falling back to eager ViT rotary embedding"
        )
        if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE_FATAL", "0"):
            raise RuntimeError(message)
        logger.warning(f"[ViTRope] {message}")
        _APPLY_ROTARY_POS_EMB_COMPILE_DISABLED = True
        return _apply_rotary_pos_emb_impl

    kwargs = _vit_torch_compile_kwargs("MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE")
    try:
        _COMPILED_APPLY_ROTARY_POS_EMB = torch.compile(
            _apply_rotary_pos_emb_impl, **kwargs
        )
        logger.info(f"[ViTRope] torch.compile enabled kwargs={kwargs}")
    except Exception as exc:
        if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE_FATAL", "0"):
            raise
        logger.warning(
            f"[ViTRope] torch.compile setup failed: {type(exc).__name__}: {exc}; "
            "falling back to eager"
        )
        _APPLY_ROTARY_POS_EMB_COMPILE_DISABLED = True
        _COMPILED_APPLY_ROTARY_POS_EMB = None
        return _apply_rotary_pos_emb_impl
    return _COMPILED_APPLY_ROTARY_POS_EMB


def apply_rotary_pos_emb(
    t: torch.Tensor, rotary_pos_emb: Sequence[torch.Tensor]
) -> torch.Tensor:
    global _COMPILED_APPLY_ROTARY_POS_EMB, _APPLY_ROTARY_POS_EMB_COMPILE_DISABLED
    fn = _get_apply_rotary_pos_emb_impl()
    try:
        return fn(t, rotary_pos_emb)
    except Exception as exc:
        if fn is _COMPILED_APPLY_ROTARY_POS_EMB and not _env_flag(
            "MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE_FATAL", "0"
        ):
            logger.warning(
                f"[ViTRope] compiled call failed: {type(exc).__name__}: {exc}; "
                "disabling compile and retrying eager"
            )
            _APPLY_ROTARY_POS_EMB_COMPILE_DISABLED = True
            _COMPILED_APPLY_ROTARY_POS_EMB = None
            return _apply_rotary_pos_emb_impl(t, rotary_pos_emb)
        raise


def apply_rotary_pos_emb_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_pos_emb: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact native NeoX rotary kernel to Q/K together when possible."""
    if (
        len(rotary_pos_emb) == 4
        and query.is_cuda
        and query.shape == key.shape
        and query.dtype == key.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dim() == 4
        and query.shape[0] == 1
        and not torch.compiler.is_compiling()
    ):
        _, _, cache, positions = rotary_pos_emb
        if (
            cache.is_cuda
            and cache.dtype == query.dtype
            and cache.dim() == 2
            and cache.shape[0] == query.shape[1]
            and cache.shape[1] <= query.shape[-1]
            and positions.is_cuda
            and positions.shape == (query.shape[1],)
        ):
            from sgl_kernel import rotary_embedding

            query = query.contiguous()
            key = key.contiguous()
            rotary_embedding(
                positions,
                query.view(query.shape[1], -1),
                key.view(key.shape[1], -1),
                query.shape[-1],
                cache,
                True,
            )
            return query, key

    return (
        apply_rotary_pos_emb(query, rotary_pos_emb),
        apply_rotary_pos_emb(key, rotary_pos_emb),
    )
