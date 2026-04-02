import copy
from dataclasses import dataclass

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


@dataclass
class MiMoAudioConfig(Qwen2Config):
    def __init__(
        self,
        *,
        speech_vocab_size: str | int = "1025-1025-129-129-129-129-129-129",
        speech_zeroemb_idx: str | int = "1024-1024-128-128-128-128-128-128",
        delay_pattern: str = "0-1-2-3-4-5-6-7",
        head_dim: int = 128,
        group_size: int = 4,
        audio_channels: int = 8,
        local_dim: int = 1024,
        local_layers: int = 16,
        local_attn_heads: int = 64,
        local_ffn_dim: int = 4096,
        local_attn_dropout: float = 0.1,
        input_local_layers: int = 6,
        input_local_dim: int | None = None,
        input_full_attention: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.speech_vocab_size = speech_vocab_size
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.delay_pattern = delay_pattern

        self.head_dim = head_dim

        self.group_size = group_size
        self.audio_channels = audio_channels

        self.local_dim = local_dim
        self.local_layers = local_layers
        self.local_attn_heads = local_attn_heads
        self.local_ffn_dim = local_ffn_dim
        self.local_attn_dropout = local_attn_dropout

        self.input_local_layers = input_local_layers
        self.input_local_dim = input_local_dim or local_dim

        self.input_full_attention = input_full_attention

    def _parse_maybe_list(self, value: str | int, length: int) -> list[int]:
        if isinstance(value, str) and "-" in value:
            return [int(s) for s in value.split("-")]
        return [int(value)] * length

    def parsed_speech_empty_ids(self):
        return self._parse_maybe_list(self.speech_zeroemb_idx, self.audio_channels)

    def parsed_speech_vocab_sizes(self):
        return self._parse_maybe_list(self.speech_vocab_size, self.audio_channels)

    def parsed_delay_pattern(self):
        return self._parse_maybe_list(self.delay_pattern, self.audio_channels)

    def local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.local_dim
        config.num_hidden_layers = self.local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = self.local_ffn_dim
        config.attention_dropout = self.local_attn_dropout

        return config

    def input_local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.input_local_dim
        config.num_hidden_layers = self.input_local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = config.hidden_size * 4
        config.attention_dropout = self.local_attn_dropout

        return config
