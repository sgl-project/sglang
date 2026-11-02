# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL text encoder configuration for BRIA Fibo 2.0.

Fibo 2.0 conditions its transformer on six "caption bundles", each made of three
Qwen3-VL hidden states concatenated along the channel axis after a per-token
float32 RMS normalization. Bundle 0 feeds the Perceiver resampler, bundles 1..5
feed the injection blocks in order.

Two things here are easy to get silently wrong, so both are asserted:

* **Layer index semantics.** The reference indexes the ``output_hidden_states``
  tuple the way ``transformers`` builds it: index 0 is the embedding output and
  index ``i`` is the input of decoder layer ``i``, for ``n_layers + 1`` entries.
  The in-tree Qwen3-VL text tower instead appends *after* each layer and never
  appends the final norm, so it yields ``n_layers`` entries where entry ``i`` is
  the output of layer ``i``. Hence ``sglang[i] == hf[i + 1]`` and the reference
  indices must be shifted down by one. ``SGLANG_HIDDEN_STATE_OFFSET`` records
  that, and the length assertion turns any upstream change into an error instead
  of wrong weights.
* **Architecture defaults.** The inherited defaults describe Qwen3-VL-8B, not the
  4B variant Fibo ships. ``post_diffusers_config_update`` flattens the real
  per-component values out of ``text_config`` / ``vision_config`` rather than
  relying on any default.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLArchConfig
from sglang.multimodal_gen.configs.models.fsdp import (
    is_block,
    is_embed_tokens,
    is_final_norm,
    is_layer,
)

# Qwen3-VL layers behind the transformer's six text inputs, in `transformers`
# index space: the Perceiver's bundle first, then one bundle per injection block
# (injection_layer_ids = (6, 14, 22, 30, 38)).
TEXT_ENCODER_OUT_LAYERS: tuple[tuple[int, ...], ...] = (
    (9, 20, 31),
    (5, 16, 27),
    (7, 18, 29),
    (9, 20, 31),
    (11, 22, 33),
    (13, 24, 35),
)

# The in-tree text tower drains decoder-layer outputs only, so its tuple is one
# entry shorter and every reference index maps one lower. Never drop this shift.
SGLANG_HIDDEN_STATE_OFFSET = 1

PROMPT_TEMPLATE_ENCODE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

CAP_FEAT_DIM = 7680


def unique_hf_layers(
    out_layers: Sequence[Sequence[int]] = TEXT_ENCODER_OUT_LAYERS,
) -> list[int]:
    """The distinct reference layer indices, sorted."""
    return sorted({layer for bundle in out_layers for layer in bundle})


def sglang_layer_indices(
    out_layers: Sequence[Sequence[int]] = TEXT_ENCODER_OUT_LAYERS,
) -> tuple[tuple[int, ...], ...]:
    """Reference layer indices translated into the in-tree tower's tuple."""
    return tuple(
        tuple(layer - SGLANG_HIDDEN_STATE_OFFSET for layer in bundle)
        for bundle in out_layers
    )


def build_caption_bundles(
    hidden_states: Sequence[torch.Tensor],
    num_hidden_layers: int,
    dtype: torch.dtype | None = None,
    out_layers: Sequence[Sequence[int]] = TEXT_ENCODER_OUT_LAYERS,
) -> torch.Tensor:
    """Stack the six caption bundles from the text tower's hidden states.

    Args:
        hidden_states: the text tower's ``output_hidden_states`` tuple, in the
            in-tree convention (outputs of every decoder layer, no embedding or
            final-norm entry), length ``num_hidden_layers``.
        num_hidden_layers: layer count the tower was built with, used to check
            the tuple length so an upstream collection change fails loudly.
        dtype: dtype the normalized states are cast back to; defaults to the
            dtype of the input states.

    Returns:
        ``(batch, 6, text_len, 3 * hidden_size)``, i.e. ``(B, 6, T, 7680)`` for
        the released checkpoints.
    """
    if len(hidden_states) != num_hidden_layers:
        raise ValueError(
            "BriaFibo2 expects the text tower to return one hidden state per "
            f"decoder layer ({num_hidden_layers}), got {len(hidden_states)}. "
            "Changing how Qwen3-VL collects `all_hidden_states` changes the "
            f"layer mapping (currently offset {SGLANG_HIDDEN_STATE_OFFSET})."
        )

    target_dtype = dtype if dtype is not None else hidden_states[0].dtype
    normalized: dict[int, torch.Tensor] = {}
    for hf_layer in unique_hf_layers(out_layers):
        index = hf_layer - SGLANG_HIDDEN_STATE_OFFSET
        states = hidden_states[index].float()
        # unit RMS per token, matching the reference
        normalized[hf_layer] = (
            states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ).to(target_dtype)

    bundles = [
        torch.cat([normalized[layer] for layer in bundle], dim=-1)
        for bundle in out_layers
    ]
    return torch.stack(bundles, dim=1)


def _read(container: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a config object or a plain dict."""
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


@dataclass
class BriaFibo2Qwen3VLArchConfig(Qwen3VLArchConfig):
    """Qwen3-VL-4B as Fibo 2.0 uses it.

    The inherited values describe the 8B encoder used by JoyImage; everything
    that matters is overwritten from the checkpoint config in
    ``post_diffusers_config_update``, so these defaults only have to be harmless.
    """

    # Fibo reads three hidden states per bundle, six bundles in total
    out_layers: tuple[tuple[int, ...], ...] = TEXT_ENCODER_OUT_LAYERS
    cap_feat_dim: int = CAP_FEAT_DIM

    # Reference prompt handling
    prompt_template_encode: str = PROMPT_TEMPLATE_ENCODE
    max_sequence_length: int = 4096
    max_num_images: int = 5
    text_encoder_image_size: int = 768
    text_len: int = 4096

    # The Fibo encoder has no pooled output and needs every layer
    output_hidden_states: bool = True

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_layer, is_block, is_embed_tokens, is_final_norm]
    )

    def __post_init__(self) -> None:
        # Deliberately not calling super().__post_init__(): `Qwen3VLArchConfig`
        # installs JoyImage's tokenizer kwargs there, whose `max_length` is
        # `text_len + prompt_template_encode_start_idx["image"]` (34 extra
        # tokens). Fibo's template is a plain chat turn with no such offset, so
        # the inherited value would silently mis-size every prompt.
        self.tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_sequence_length,
            "return_tensors": "pt",
        }


@dataclass
class BriaFibo2Qwen3VLConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(
        default_factory=BriaFibo2Qwen3VLArchConfig
    )

    def post_diffusers_config_update(self) -> None:
        """Flatten the real encoder shape out of the checkpoint config.

        `update_model_arch` copies only top-level keys, so the nested
        `text_config` / `vision_config` never reach the flat fields and the 8B
        defaults would survive. Reading them here keeps the 4B variant correct
        without hardcoding a layer count.
        """
        arch = self.arch_config

        text_config = _read(arch, "text_config")
        vision_config = _read(arch, "vision_config")

        arch.hidden_size = int(_read(text_config, "hidden_size", arch.hidden_size))
        arch.intermediate_size = int(
            _read(text_config, "intermediate_size", arch.intermediate_size)
        )
        arch.num_hidden_layers = int(
            _read(text_config, "num_hidden_layers", arch.num_hidden_layers)
        )
        arch.num_attention_heads = int(
            _read(text_config, "num_attention_heads", arch.num_attention_heads)
        )
        arch.num_key_value_heads = int(
            _read(text_config, "num_key_value_heads", arch.num_key_value_heads or 0)
            or arch.num_key_value_heads
        )
        arch.head_dim = int(_read(text_config, "head_dim", arch.head_dim or 0) or 0)
        arch.vocab_size = int(_read(text_config, "vocab_size", arch.vocab_size))
        arch.rms_norm_eps = float(_read(text_config, "rms_norm_eps", arch.rms_norm_eps))
        arch.tie_word_embeddings = bool(
            _read(text_config, "tie_word_embeddings", arch.tie_word_embeddings)
        )
        rope_theta = _read(text_config, "rope_theta")
        if rope_theta is None:
            rope_theta = _read(_read(text_config, "rope_parameters"), "rope_theta")
        if rope_theta is not None:
            arch.rope_theta = float(rope_theta)

        if vision_config is not None:
            arch.vision_config = vision_config

        if arch.head_dim in (None, 0):
            arch.head_dim = arch.hidden_size // arch.num_attention_heads

        # A three-layer bundle times the encoder width must match cap_feat_dim,
        # which is also the cross-attention key/value width of the transformer
        expected = 3 * arch.hidden_size
        if expected != arch.cap_feat_dim:
            raise ValueError(
                "cap_feat_dim must be 3 text-encoder hidden sizes: "
                f"3 * {arch.hidden_size} != {arch.cap_feat_dim}"
            )


__all__ = [
    "BriaFibo2Qwen3VLArchConfig",
    "BriaFibo2Qwen3VLConfig",
    "PROMPT_TEMPLATE_ENCODE",
    "SGLANG_HIDDEN_STATE_OFFSET",
    "TEXT_ENCODER_OUT_LAYERS",
    "VISION_MARKER",
    "build_caption_bundles",
    "sglang_layer_indices",
    "unique_hf_layers",
]
