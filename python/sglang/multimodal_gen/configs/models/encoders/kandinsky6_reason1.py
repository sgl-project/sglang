# SPDX-License-Identifier: Apache-2.0
"""Reason1 text encoder, reusing the Qwen2.5-VL runtime without its vision tower.

Defaults describe the 3584-wide language trunk; checkpoint config overrides them."""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import (
    is_embed_tokens,
    is_final_norm,
    is_layer,
)


@dataclass
class Reason1ArchConfig(TextEncoderArchConfig):
    """Architecture metadata (defaults match Qwen2.5-VL-7B-Instruct)."""

    architectures: list[str] = field(
        default_factory=lambda: ["Qwen2_5_VLForConditionalGeneration"]
    )

    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = False
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0

    rope_theta: float = 1_000_000.0
    # 3D mRoPE (temporal/height/width) section split; Qwen2.5-VL-specific.
    rope_scaling: dict[str, Any] | None = field(
        default_factory=lambda: {"type": "mrope", "mrope_section": [16, 24, 24]}
    )

    # Kandinsky6's text-conditioning stack always consumes the last hidden
    # state (see kandinsky6_qwen_postprocess_text in
    # configs/pipeline_configs/kandinsky6.py), so hidden states must be kept.
    output_hidden_states: bool = True
    hidden_state_skip_layer: int = 0

    # text_len caps user tokens; tokenizer length also includes the 129-token template
    text_len: int = 512

    bos_token_id: int = 151643
    pad_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656

    # Same fused-projection naming as the sglang qwen2_5vl/qwen_image runtime
    # encoders (MergedColumnParallelLinear "gate_up_proj", fused "qkv_proj").
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),  # type: ignore
            (".gate_up_proj", ".up_proj", 1),  # type: ignore
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_layer, is_embed_tokens, is_final_norm]
    )


@dataclass
class Reason1Config(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=Reason1ArchConfig)
    prefix: str = "reason1"


__all__ = ["Reason1ArchConfig", "Reason1Config"]
