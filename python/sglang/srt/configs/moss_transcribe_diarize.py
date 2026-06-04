from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.whisper.configuration_whisper import WhisperConfig


class MossTranscribeDiarizeConfig(PretrainedConfig):
    model_type = "moss_transcribe_diarize"
    sub_configs = {"text_config": Qwen3Config, "audio_config": WhisperConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_token_id: int = 151671,
        audio_merge_size: int = 4,
        adaptor_input_dim: int | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Qwen3Config(
                vocab_size=151936,
                hidden_size=1024,
                intermediate_size=3072,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=40960,
                tie_word_embeddings=tie_word_embeddings,
                rope_theta=1_000_000.0,
                layer_types=["full_attention"] * 28,
            )
        elif isinstance(text_config, dict):
            text_config = self.sub_configs["text_config"](**text_config)

        if audio_config is None:
            audio_config = WhisperConfig(
                num_mel_bins=80,
                d_model=1024,
                encoder_layers=24,
                encoder_attention_heads=16,
                encoder_ffn_dim=4096,
                max_source_positions=1500,
                dropout=0.0,
                attention_dropout=0.0,
                activation_dropout=0.0,
                activation_function="gelu",
                encoder_layerdrop=0.0,
                scale_embedding=False,
            )
        elif isinstance(audio_config, dict):
            audio_config = self.sub_configs["audio_config"](**audio_config)

        text_config.tie_word_embeddings = tie_word_embeddings
        if not getattr(text_config, "layer_types", None):
            text_config.layer_types = ["full_attention"] * text_config.num_hidden_layers

        self.text_config = text_config
        self.audio_config = audio_config
        self.audio_token_id = audio_token_id
        self.audio_merge_size = audio_merge_size
        self.adaptor_input_dim = (
            adaptor_input_dim or audio_config.d_model * audio_merge_size
        )
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.text_config


AutoConfig.register("moss_transcribe_diarize", MossTranscribeDiarizeConfig)
