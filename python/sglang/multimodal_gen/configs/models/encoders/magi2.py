# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


@dataclass
class Magi2TextEncoderArchConfig(TextEncoderArchConfig):
    """KNOWN DEVIATION: loads through transformers, not srt/models/qwen3_5.py, which needs a KV cache and ForwardBatch."""

    hidden_size: int = 5120
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 17408
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attn_output_gate: bool = True
    full_attention_interval: int = 4
    eos_token_id: int = 248044

    text_len: int = 7000

    # Conditioning is hidden_states[-(skip_layer + 1)]; the last layer shifts guidance.
    skip_layer: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs = {
            "padding": "longest",
            "truncation": True,
            "max_length": self.text_len,
            "return_tensors": "pt",
        }


@dataclass
class Magi2TextEncoderConfig(TextEncoderConfig):
    arch_config: Magi2TextEncoderArchConfig = field(
        default_factory=Magi2TextEncoderArchConfig
    )
