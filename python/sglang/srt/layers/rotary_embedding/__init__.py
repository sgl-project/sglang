# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py
"""Rotary Positional Embeddings - public API (drop-in replacement for rotary_embedding.py)."""

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.factory import get_rope, get_rope_wrapper
from sglang.srt.layers.rotary_embedding.mrope import (
    Ernie4_5_VLRotaryEmbedding,
    MRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb
from sglang.srt.layers.rotary_embedding.yarn import (
    yarn_find_correction_range,
    yarn_get_mscale_simple,
    yarn_linear_ramp_mask,
)

_yarn_find_correction_range = yarn_find_correction_range
_yarn_get_mscale = yarn_get_mscale_simple
_yarn_linear_ramp_mask = yarn_linear_ramp_mask

__all__ = [
    "RotaryEmbedding",
    "get_rope",
    "get_rope_wrapper",
    "MRotaryEmbedding",
    "Ernie4_5_VLRotaryEmbedding",
    "apply_rotary_pos_emb",
    "_yarn_find_correction_range",
    "_yarn_get_mscale",
    "_yarn_linear_ramp_mask",
]
