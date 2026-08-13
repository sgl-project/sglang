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

import math
from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig

from sglang.srt.configs.muse_glimmer_processing import MuseGlimmerProcessor
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)

_ARCH = "muse-glimmer"


class MuseGlimmerAssistantConfig(PretrainedConfig):

    model_type = "muse_glimmer_assistant"
    is_causal = False
    # The DFlash draft has no head; draft_worker_common borrows the target's.
    vocab_size = None


class MuseGlimmerVisionConfig(PretrainedConfig):

    model_type = "muse_glimmer_vision"

    def __init__(
        self,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 16,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        attention_types: Optional[List[str]] = None,
        max_position_embeddings: int = 1024,
        merge_size: int = 2,
        patch_size: int = 14,
        patch_temporal: int = 2,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_types = attention_types
        self.max_position_embeddings = max_position_embeddings
        self.merge_size = merge_size
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.rope_theta = rope_theta
        super().__init__(**kwargs)


@register_customized_processor(MuseGlimmerProcessor)
class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"
    sub_configs = {"vision_config": MuseGlimmerVisionConfig}

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 6656,
        intermediate_size: int = 19968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 16384,
        rms_norm_eps: float = 1e-5,
        post_norm_eps: float = 1e-8,
        rope_theta: float = 500000.0,
        sliding_window: int = 2048,
        layer_types: Optional[List[str]] = None,
        no_rope_layers: Optional[List[int]] = None,
        use_qk_norm: bool = True,
        use_attn_output_gate: bool = True,
        qk_scale_factor: float = 43.7840518911,
        rope_is_neox_style: bool = False,
        normalize_tok_embeddings: bool = True,
        output_multiplier: float = 0.19611613513818404,
        output_soft_cap_temp: Optional[float] = 20.0,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 200000,
        eos_token_id: int = 200001,
        vision_config: Optional[Dict[str, Any]] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        out_hidden_size: int = 6144,
        projector_hidden_act: str = "gelu",
        projector_hidden_size: int = 4096,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.no_rope_layers = no_rope_layers
        self.use_qk_norm = use_qk_norm
        self.use_attn_output_gate = use_attn_output_gate
        self.qk_scale_factor = qk_scale_factor
        self.rope_is_neox_style = rope_is_neox_style
        self.normalize_tok_embeddings = normalize_tok_embeddings
        self.output_multiplier = output_multiplier
        self.output_soft_cap_temp = output_soft_cap_temp
        if isinstance(vision_config, dict):
            vision_config = MuseGlimmerVisionConfig(**vision_config)
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.out_hidden_size = out_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "MuseGlimmerConfig":
        return cls(**muse_glimmer_config_kwargs_from_gguf(gguf_path))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        return super().from_dict(
            muse_glimmer_config_kwargs_from_hf(config_dict), **kwargs
        )


_HF_TEXT_KEYS_TRANSLATED = frozenset(
    {
        "final_logit_softcapping",
        "hidden_activation",
        "layer_rope_theta",
        "model_type",
        "qk_scale_factor",
        "rope_parameters",
    }
)

_HF_NESTED_KEYS = ("text_config", "vision_config")

_HF_VISION_KEYS_TRANSLATED = frozenset({"layer_types", "model_type", "rope_parameters"})


def muse_glimmer_config_kwargs_from_hf(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    if "text_config" not in config_dict:
        return config_dict

    text = config_dict["text_config"]
    kwargs = {k: v for k, v in config_dict.items() if k not in _HF_NESTED_KEYS}
    kwargs.update({k: v for k, v in text.items() if k not in _HF_TEXT_KEYS_TRANSLATED})
    kwargs.update(
        hidden_act=text["hidden_activation"],
        rope_theta=text["rope_parameters"]["rope_theta"],
        no_rope_layers=[1 if theta else 0 for theta in text["layer_rope_theta"]],
        output_soft_cap_temp=text["final_logit_softcapping"],
        qk_scale_factor=text["qk_scale_factor"] * math.sqrt(text["head_dim"]),
        rope_is_neox_style=True,
    )
    if "vision_config" in config_dict:
        kwargs["vision_config"] = muse_glimmer_vision_config_kwargs_from_hf(
            config_dict["vision_config"]
        )
    return kwargs


def muse_glimmer_vision_config_kwargs_from_hf(
    vision_config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate the vendor's ``vision_config`` into ``MuseGlimmerVisionConfig`` kwargs."""
    kwargs = {
        k: v
        for k, v in vision_config_dict.items()
        if k not in _HF_VISION_KEYS_TRANSLATED
    }
    kwargs["rope_theta"] = vision_config_dict["rope_parameters"]["rope_theta"]
    kwargs["attention_types"] = vision_config_dict["layer_types"]
    return kwargs


def _f(v):
    return None if v is None else float(v)


def _i(v):
    return None if v is None else int(v)


def _mul_sqrt(v, head_dim):
    return None if v is None else float(v) * math.sqrt(head_dim)


def muse_glimmer_config_kwargs_from_gguf(gguf_path: str) -> Dict[str, Any]:
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}
    shapes = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}
    tensor_names = set(shapes)

    def get(suffix):
        return meta[f"{_ARCH}.{suffix}"]

    def opt(suffix):
        """None when this converter generation did not emit the key."""
        return meta.get(f"{_ARCH}.{suffix}")

    head_dim = int(get("attention.key_length"))
    swa_pattern = [bool(x) for x in get("attention.sliding_window_pattern")]

    return dict(
        # token_embd is stored [n_embd, n_vocab] in ggml's reversed order.
        vocab_size=shapes["token_embd.weight"][1],
        hidden_size=int(get("embedding_length")),
        intermediate_size=int(get("feed_forward_length")),
        num_hidden_layers=int(get("block_count")),
        num_attention_heads=int(get("attention.head_count")),
        num_key_value_heads=int(get("attention.head_count_kv")),
        head_dim=head_dim,
        max_position_embeddings=int(get("context_length")),
        rms_norm_eps=float(get("attention.layer_norm_rms_epsilon")),
        rope_theta=float(get("rope.freq_base")),
        sliding_window=int(get("attention.sliding_window")),
        layer_types=[
            "sliding_attention" if s else "full_attention" for s in swa_pattern
        ],
        no_rope_layers=[1 if s else 0 for s in swa_pattern],
        use_qk_norm=any(n.endswith("attn_q_norm.weight") for n in tensor_names),
        use_attn_output_gate=any(n.endswith("attn_gate.weight") for n in tensor_names),
        tie_word_embeddings="output.weight" not in tensor_names,
        architectures=["MuseGlimmerForCausalLM"],
        dtype="bfloat16",
        # Converter generations differ in which of these they emit, and every one
        # is an architecture constant that MuseGlimmerConfig already defaults to,
        # so an absent key falls back rather than raising. attention.scale is
        # stored pre-divided by sqrt(head_dim); the class stores it before that.
        **{
            k: v
            for k, v in (
                ("post_norm_eps", _f(opt("attention.post_norm_rms_epsilon"))),
                ("qk_scale_factor", _mul_sqrt(opt("attention.scale"), head_dim)),
                ("output_multiplier", _f(opt("logit_scale"))),
                ("output_soft_cap_temp", _f(opt("final_logit_softcapping"))),
                ("bos_token_id", _i(meta.get("tokenizer.ggml.bos_token_id"))),
                ("eos_token_id", _i(meta.get("tokenizer.ggml.eos_token_id"))),
            )
            if v is not None
        },
    )
