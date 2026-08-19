from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.adapter.base import (
    AdapterArchConfig,
    AdapterConfig,
)

# Diffusers names the per-modality projections `video_text_proj_in` /
# `audio_text_proj_in`; SGLang follows ltx-core (`*_aggregate_embed`). Every
# other connector weight already matches.
LTX2_CONNECTOR_PARAM_NAMES_MAPPING: dict[str, str] = {
    r"^video_text_proj_in\.(.*)$": r"video_aggregate_embed.\1",
    r"^audio_text_proj_in\.(.*)$": r"audio_aggregate_embed.\1",
}


@dataclass
class LTX2ConnectorArchConfig(AdapterArchConfig):
    param_names_mapping: dict = field(
        default_factory=lambda: dict(LTX2_CONNECTOR_PARAM_NAMES_MAPPING)
    )

    audio_connector_attention_head_dim: int = 128
    audio_connector_num_attention_heads: int = 30
    audio_connector_num_layers: int = 2
    audio_connector_num_learnable_registers: int = 128
    audio_feature_extractor_out_features: int = 0
    caption_channels: int = 3840
    causal_temporal_positioning: bool = False
    connector_rope_base_seq_len: int = 4096
    connector_apply_gated_attention: bool = False
    feature_extractor_in_features: int = 0
    rope_double_precision: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "split"
    text_proj_in_factor: int = 49
    video_feature_extractor_out_features: int = 0
    video_connector_attention_head_dim: int = 128
    video_connector_num_attention_heads: int = 30
    video_connector_num_layers: int = 2
    video_connector_num_learnable_registers: int = 128

    # `update_model_arch` copies `connectors/config.json` verbatim onto this
    # object, so declare its names here and derive the SGLang-side fields in
    # `__post_init__`. LTX-2.0 leaves `per_modality_projections` false and keeps
    # one shared `text_proj_in`; LTX-2.3 / 2.5 set it.
    per_modality_projections: bool = False
    video_hidden_dim: int = 4096
    audio_hidden_dim: int = 2048
    video_gated_attn: bool = False
    audio_gated_attn: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.per_modality_projections:
            self.feature_extractor_in_features = (
                self.caption_channels * self.text_proj_in_factor
            )
            self.video_feature_extractor_out_features = self.video_hidden_dim
            self.audio_feature_extractor_out_features = self.audio_hidden_dim

        # Upstream gates these separately; released checkpoints always pair them.
        if self.video_gated_attn or self.audio_gated_attn:
            self.connector_apply_gated_attention = True


@dataclass
class LTX2ConnectorConfig(AdapterConfig):

    arch_config: AdapterArchConfig = field(default_factory=LTX2ConnectorArchConfig)

    prefix: str = "LTX2"
