# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""Z-Image breakable CUDA graph (BCG) prompt padding."""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)


def is_zimage_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return (
        bcg_utils.transformer_class_name_matches(current_model, "zimage")
        and "encoder_hidden_states" in call_kwargs
        and "freqs_cis" in call_kwargs
    )


def _first_caption_tensor(encoder_hidden_states: Any) -> torch.Tensor | None:
    tensor = bcg_utils.first_tensor(encoder_hidden_states)
    if not torch.is_tensor(tensor):
        return None
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() == 3:
        return tensor[0]
    return None


def _caption_seq_len(tensor: torch.Tensor) -> int:
    if tensor.dim() == 2:
        return int(tensor.shape[0])
    if tensor.dim() == 3:
        return int(tensor.shape[1])
    raise ValueError("Z-Image caption tensor must have rank 2 or 3")


def _pad_caption(obj: Any, *, target: int) -> Any:
    if torch.is_tensor(obj):
        if obj.dim() == 2:
            return bcg_utils.pad_tensor_dim(obj, 0, target)
        if obj.dim() == 3:
            return bcg_utils.pad_tensor_dim(obj, 1, target)
        return obj
    if isinstance(obj, list):
        return [_pad_caption(item, target=target) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_pad_caption(item, target=target) for item in obj)
    return obj


def _unwrap_model(current_model: Any) -> Any:
    for attr in ("module", "_orig_mod"):
        wrapped = getattr(current_model, attr, None)
        if wrapped is not None:
            current_model = wrapped
    return current_model


def _build_caption_freqs(current_model: Any, *, target: int, device: torch.device):
    rotary_emb = getattr(_unwrap_model(current_model), "rotary_emb", None)
    if rotary_emb is None:
        return None

    axes = [
        torch.arange(1, target + 1, dtype=torch.int32, device=device),
        torch.zeros(target, dtype=torch.int32, device=device),
        torch.zeros(target, dtype=torch.int32, device=device),
    ]
    cap_pos_ids = torch.stack(axes, dim=-1)
    return rotary_emb(cap_pos_ids)


def _pad_caption_freqs(freqs_cis: Any, current_model: Any, *, target: int) -> Any:
    if not isinstance(freqs_cis, (tuple, list)) or len(freqs_cis) != 2:
        return freqs_cis

    cap_cache, image_cache = freqs_cis
    cap_tensor = bcg_utils.first_tensor(cap_cache)
    if torch.is_tensor(cap_tensor) and cap_tensor.dim() >= 1:
        cap_freqs = _build_caption_freqs(
            current_model, target=target, device=cap_tensor.device
        )
        if cap_freqs is not None:
            cap_cache = cap_freqs

    if isinstance(freqs_cis, tuple):
        return (cap_cache, image_cache)
    return [cap_cache, image_cache]


def _caption_mask(
    call_kwargs: dict, *, caption: torch.Tensor, seq: int, bucket: int
) -> torch.Tensor:
    mask = bcg_utils.first_tensor(call_kwargs.get("encoder_hidden_states_mask"))
    if not torch.is_tensor(mask):
        mask = bcg_utils.first_tensor(call_kwargs.get("encoder_attention_mask"))
    if torch.is_tensor(mask):
        if mask.dim() == 1:
            mask = mask[:seq].unsqueeze(0)
        elif mask.dim() >= 2:
            mask = mask[:, :seq]
        mask = mask.to(device=caption.device, dtype=torch.bool)
    else:
        batch = int(caption.shape[0]) if caption.dim() == 3 else 1
        mask = torch.ones((batch, seq), device=caption.device, dtype=torch.bool)
    return bcg_utils.pad_tensor_dim(mask, 1, bucket)


def pad_zimage_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    caption = _first_caption_tensor(call_kwargs.get("encoder_hidden_states"))
    if caption is None:
        return call_kwargs

    seq = _caption_seq_len(caption)
    cap_freq = None
    freqs_cis = call_kwargs.get("freqs_cis")
    if isinstance(freqs_cis, (tuple, list)) and len(freqs_cis) == 2:
        cap_freq = bcg_utils.first_tensor(freqs_cis[0])
    cap_freq_len = int(cap_freq.shape[0]) if torch.is_tensor(cap_freq) else seq

    bucket = bcg_utils.select_text_bucket(max(seq, cap_freq_len), buckets)
    if bucket is None:
        return call_kwargs

    out = {
        key: value
        for key, value in call_kwargs.items()
        if key
        in {
            "hidden_states",
            "timestep",
            "guidance",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "encoder_hidden_states_mask",
            "freqs_cis",
            "image_seq_len_target",
            "patch_size",
            "f_patch_size",
        }
    }

    if seq < bucket:
        out["encoder_hidden_states"] = _pad_caption(
            out["encoder_hidden_states"], target=bucket
        )

    caption_mask = _caption_mask(call_kwargs, caption=caption, seq=seq, bucket=bucket)
    out["encoder_hidden_states_mask"] = caption_mask
    out["caption_valid_lens"] = caption_mask.sum(dim=1).to(dtype=torch.long)
    out["_use_caption_valid_mask"] = True
    if out.get("encoder_attention_mask") is not None:
        out["encoder_attention_mask"] = out["encoder_hidden_states_mask"]
    out["freqs_cis"] = _pad_caption_freqs(
        out.get("freqs_cis"), current_model, target=bucket
    )
    return out


bcg_utils.register_prompt_padder(is_zimage_transformer, pad_zimage_prompt_kwargs)
