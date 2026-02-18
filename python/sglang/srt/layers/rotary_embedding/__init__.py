# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py
"""Rotary Positional Embeddings - unified public API."""

from sglang.srt.layers.rotary_embedding._utils import (
    _rotate_neox,
    _rotate_gptj,
    _apply_rotary_emb,
    apply_rotary_pos_emb_native,
    apply_rotary_pos_emb_npu,
    apply_rotary_pos_emb,
)
from sglang.srt.layers.rotary_embedding._base import (
    RotaryEmbedding,
    LinearScalingRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    apply_interleaved_rope,
)
from sglang.srt.layers.rotary_embedding._yarn import (
    YaRNScalingRotaryEmbedding,
    _yarn_find_correction_dim,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    _yarn_get_mscale,
)
from sglang.srt.layers.rotary_embedding._mrope import (
    MRotaryEmbedding,
    YaRNScalingMRotaryEmbedding,
    Ernie4_5_VLRotaryEmbedding,
    triton_mrope_fused,
    triton_ernie45_rope_fused_inplace,
)
from sglang.srt.layers.rotary_embedding._special import (
    yarn_get_mscale,
    Phi3LongRoPEScaledRotaryEmbedding,
    FourierRotaryEmbedding,
    DeepseekScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    Llama4VisionRotaryEmbedding,
    DualChunkRotaryEmbedding,
    rotate_half,
)
from sglang.srt.layers.rotary_embedding._factory import (
    get_rope,
    get_rope_cpu,
    get_rope_wrapper,
    _ROPE_DICT,
)

__all__ = [
    # utils
    "_rotate_neox",
    "_rotate_gptj",
    "_apply_rotary_emb",
    "apply_rotary_pos_emb_native",
    "apply_rotary_pos_emb_npu",
    "apply_rotary_pos_emb",
    "rotate_half",
    # base
    "RotaryEmbedding",
    "LinearScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
    "DynamicNTKAlphaRotaryEmbedding",
    "apply_interleaved_rope",
    # yarn
    "YaRNScalingRotaryEmbedding",
    "_yarn_find_correction_dim",
    "_yarn_find_correction_range",
    "_yarn_linear_ramp_mask",
    "_yarn_get_mscale",
    # mrope
    "MRotaryEmbedding",
    "YaRNScalingMRotaryEmbedding",
    "Ernie4_5_VLRotaryEmbedding",
    "triton_mrope_fused",
    "triton_ernie45_rope_fused_inplace",
    # special
    "yarn_get_mscale",
    "Phi3LongRoPEScaledRotaryEmbedding",
    "FourierRotaryEmbedding",
    "DeepseekScalingRotaryEmbedding",
    "Llama3RotaryEmbedding",
    "Llama4VisionRotaryEmbedding",
    "DualChunkRotaryEmbedding",
    # factory
    "get_rope",
    "get_rope_cpu",
    "get_rope_wrapper",
    "_ROPE_DICT",
]
