# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model-agnostic helpers for breakable CUDA graph (BCG) prompt padding.

These are the shared, model-independent primitives used to pad prompt
conditioning up to a sequence-length bucket so that prompts of different
lengths reuse one captured graph. Model-specific padders (Qwen, Z-Image, ...)
live next to their model in ``model_specific_stages`` and register themselves
through :func:`register_prompt_padder`; the generic masked padder here is the
default fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

# Prompt-conditioning kwarg keys, grouped by which dim carries the text length.
PROMPT_MASK_KEYS = (
    "encoder_attention_mask",
    "encoder_hidden_states_mask",
    "attention_mask",
    "text_mask",
    "prompt_attention_mask",
    "negative_attention_mask",
    "prompt_embeds_mask",
    "negative_prompt_embeds_mask",
)
TEXT_DIM1_KEYS = (
    "encoder_hidden_states",
    "encoder_hidden_states_2",
    "encoder_attention_mask",
    "encoder_hidden_states_mask",
    "attention_mask",
    "text_mask",
    "text_ids",
    "text_pos_ids",
    "txt_ids",
    "prompt_embeds",
    "negative_prompt_embeds",
    "prompt_attention_mask",
    "negative_attention_mask",
    "prompt_embeds_mask",
    "negative_prompt_embeds_mask",
    "audio_encoder_hidden_states",
    "audio_encoder_attention_mask",
)
TEXT_DIM0_KEYS = (
    "txt_freqs_cis",
    "text_freqs_cis",
)
TEXT_SEQ_LEN_KEYS = (
    "txt_seq_lens",
    "text_seq_lens",
)


def first_tensor(obj: Any) -> torch.Tensor | None:
    """First tensor leaf found by depth-first traversal (dicts in sorted-key
    order), or ``None``."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(obj, dict):
        for key in sorted(obj):
            tensor = first_tensor(obj[key])
            if tensor is not None:
                return tensor
    return None


def select_text_bucket(seq: int, buckets: tuple[int, ...]) -> int | None:
    """Smallest bucket that fits ``seq``; ``None`` (and a warning) when ``seq``
    exceeds the largest bucket so the caller runs that length eagerly."""
    for bucket in buckets:
        if seq <= bucket:
            return bucket
    logger.warning(
        "[Diffusion BCG] text length %d exceeds max bucket %d; not padding "
        "(this length captures its own graph). Raise --bcg-text-buckets.",
        seq,
        buckets[-1],
    )
    return None


def pad_tensor_dim(
    tensor: Any, dim: int, target: int, value: float = 0
) -> Any:
    if not torch.is_tensor(tensor) or tensor.dim() <= dim:
        return tensor
    seq = tensor.shape[dim]
    if seq >= target:
        return tensor
    pad = [0, 0] * tensor.dim()
    pad_index = 2 * (tensor.dim() - dim - 1) + 1
    pad[pad_index] = target - seq
    return torch.nn.functional.pad(tensor, tuple(pad), value=value)


def pad_nested_dim(
    obj: Any,
    *,
    dim: int,
    source: int,
    target: int,
    value: float = 0,
) -> Any:
    if torch.is_tensor(obj):
        if obj.dim() > dim and obj.shape[dim] == source:
            return pad_tensor_dim(obj, dim, target, value)
        return obj
    if isinstance(obj, list):
        return [
            pad_nested_dim(item, dim=dim, source=source, target=target, value=value)
            for item in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            pad_nested_dim(item, dim=dim, source=source, target=target, value=value)
            for item in obj
        )
    return obj


def bucket_txt_seq_lens(txt_seq_lens: Any, bucket: int) -> Any:
    if txt_seq_lens is None:
        return txt_seq_lens
    if torch.is_tensor(txt_seq_lens):
        return torch.full_like(txt_seq_lens, bucket)
    if isinstance(txt_seq_lens, list):
        return [bucket_txt_seq_lens(seq_len, bucket) for seq_len in txt_seq_lens]
    if isinstance(txt_seq_lens, tuple):
        return tuple(bucket_txt_seq_lens(seq_len, bucket) for seq_len in txt_seq_lens)
    if isinstance(txt_seq_lens, int):
        return bucket
    return txt_seq_lens


def prompt_seq_and_dim(call_kwargs: dict) -> tuple[int, int] | None:
    """Return ``(text_seq_len, seq_dim)`` inferred from the prompt embeddings or
    a prompt mask, or ``None`` when no text conditioning is present."""
    ehs_tensor = first_tensor(call_kwargs.get("encoder_hidden_states"))
    if torch.is_tensor(ehs_tensor) and ehs_tensor.dim() >= 2:
        if ehs_tensor.dim() == 2:
            return int(ehs_tensor.shape[0]), 0
        return int(ehs_tensor.shape[1]), 1

    for key in PROMPT_MASK_KEYS:
        tensor = first_tensor(call_kwargs.get(key))
        if torch.is_tensor(tensor) and tensor.dim() >= 2:
            if tensor.shape[0] == 1:
                return int(tensor.shape[1]), 1
            return int(tensor.shape[0]), 0
    return None


def pad_nested_text_dim(
    obj: Any,
    *,
    source: int,
    target: int,
    preferred_dim: int,
) -> Any:
    if torch.is_tensor(obj):
        if obj.dim() > preferred_dim and obj.shape[preferred_dim] == source:
            return pad_tensor_dim(obj, preferred_dim, target)
        for dim in (1, 0):
            if dim != preferred_dim and obj.dim() > dim and obj.shape[dim] == source:
                return pad_tensor_dim(obj, dim, target)
        return obj
    if isinstance(obj, list):
        return [
            pad_nested_text_dim(
                item, source=source, target=target, preferred_dim=preferred_dim
            )
            for item in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            pad_nested_text_dim(
                item, source=source, target=target, preferred_dim=preferred_dim
            )
            for item in obj
        )
    if isinstance(obj, dict):
        return {
            key: pad_nested_text_dim(
                value, source=source, target=target, preferred_dim=preferred_dim
            )
            for key, value in obj.items()
        }
    return obj


def bucket_text_seq_lens(obj: Any, *, target: int) -> Any:
    if isinstance(obj, int) and not isinstance(obj, bool):
        return target
    if isinstance(obj, list):
        return [bucket_text_seq_lens(item, target=target) for item in obj]
    if isinstance(obj, tuple):
        return tuple(bucket_text_seq_lens(item, target=target) for item in obj)
    return obj


def pad_masked_prompt_kwargs(call_kwargs: dict, buckets: tuple[int, ...]) -> dict:
    """Generic, model-agnostic prompt padding for models that pass a prompt
    attention mask alongside their text embeddings."""
    seq_and_dim = prompt_seq_and_dim(call_kwargs)
    if seq_and_dim is None:
        return call_kwargs
    seq, seq_dim = seq_and_dim
    has_mask = any(
        first_tensor(call_kwargs.get(key)) is not None for key in PROMPT_MASK_KEYS
    )
    if not has_mask:
        return call_kwargs
    bucket = select_text_bucket(seq, buckets)
    if bucket is None or seq == bucket:
        return call_kwargs

    out = dict(call_kwargs)
    for key in TEXT_DIM1_KEYS:
        if key in out and out[key] is not None:
            out[key] = pad_nested_text_dim(
                out[key], source=seq, target=bucket, preferred_dim=seq_dim
            )
    for key in TEXT_DIM0_KEYS:
        if key in out and out[key] is not None:
            out[key] = pad_nested_dim(out[key], dim=0, source=seq, target=bucket)
    for key in TEXT_SEQ_LEN_KEYS:
        if key in out and out[key] is not None:
            out[key] = bucket_text_seq_lens(out[key], target=bucket)
    return out


def transformer_class_name_matches(current_model: Any, needle: str) -> bool:
    """True when ``current_model`` (or its ``module`` / ``_orig_mod`` wrapper)
    is a transformer whose qualified class name contains ``needle``."""
    candidates = [current_model]
    for attr in ("module", "_orig_mod"):
        wrapped = getattr(current_model, attr, None)
        if wrapped is not None:
            candidates.append(wrapped)
    for candidate in candidates:
        cls = type(candidate)
        name = f"{cls.__module__}.{cls.__qualname__}".lower()
        if needle in name:
            return True
    return False


# --- Model-specific prompt-padder registry ------------------------------- #
# Each model that needs custom prompt padding registers a (predicate, padder)
# pair from its own module in ``model_specific_stages`` so the base denoising
# stage stays model-agnostic. ``padder(call_kwargs, current_model, buckets)``
# returns the padded kwargs.
PromptPadder = Callable[[dict, Any, tuple], dict]
_PROMPT_PADDERS: list[tuple[Callable[[Any, dict], bool], PromptPadder]] = []


def register_prompt_padder(
    predicate: Callable[[Any, dict], bool], padder: PromptPadder
) -> None:
    _PROMPT_PADDERS.append((predicate, padder))


def select_prompt_padder(current_model: Any, call_kwargs: dict) -> PromptPadder | None:
    """Return the registered model-specific padder for ``current_model``, or
    ``None`` to fall back to :func:`pad_masked_prompt_kwargs`."""
    _ensure_model_padders_registered()
    for predicate, padder in _PROMPT_PADDERS:
        if predicate(current_model, call_kwargs):
            return padder
    return None


_model_padders_registered = False


def _ensure_model_padders_registered() -> None:
    """Import the model-specific padder modules once so they register."""
    global _model_padders_registered
    if _model_padders_registered:
        return
    _model_padders_registered = True
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (  # noqa: F401
        qwen_image_bcg,
        zimage_bcg,
    )
