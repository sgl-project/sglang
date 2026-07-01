# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/NVlabs/LongLive

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)


@dataclass
class LongLive2ArchConfig(WanVideoArchConfig):
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^model\.patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^model\.text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^model\.text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^model\.time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^model\.time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^model\.time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^model\.blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
            r"^model\.blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^model\.blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^model\.blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^model\.blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
            r"^model\.blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^model\.blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^model\.blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
            r"^model\.blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^model\.blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^model\.blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^model\.blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^model\.blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
            r"^model\.blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
            r"^model\.blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^model\.blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^model\.head\.modulation$": r"scale_shift_table",
            r"^model\.head\.head\.(.*)$": r"proj_out.\1",
        }
    )
    reverse_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.proj\.(.*)$": r"model.patch_embedding.\1",
            r"^condition_embedder\.text_embedder\.fc_in\.(.*)$": r"model.text_embedding.0.\1",
            r"^condition_embedder\.text_embedder\.fc_out\.(.*)$": r"model.text_embedding.2.\1",
            r"^condition_embedder\.time_embedder\.mlp\.fc_in\.(.*)$": r"model.time_embedding.0.\1",
            r"^condition_embedder\.time_embedder\.mlp\.fc_out\.(.*)$": r"model.time_embedding.2.\1",
            r"^condition_embedder\.time_modulation\.linear\.(.*)$": r"model.time_projection.1.\1",
            r"^blocks\.(\d+)\.scale_shift_table$": r"model.blocks.\1.modulation",
            r"^blocks\.(\d+)\.to_q\.(.*)$": r"model.blocks.\1.self_attn.q.\2",
            r"^blocks\.(\d+)\.to_k\.(.*)$": r"model.blocks.\1.self_attn.k.\2",
            r"^blocks\.(\d+)\.to_v\.(.*)$": r"model.blocks.\1.self_attn.v.\2",
            r"^blocks\.(\d+)\.to_out\.(.*)$": r"model.blocks.\1.self_attn.o.\2",
            r"^blocks\.(\d+)\.norm_q\.(.*)$": r"model.blocks.\1.self_attn.norm_q.\2",
            r"^blocks\.(\d+)\.norm_k\.(.*)$": r"model.blocks.\1.self_attn.norm_k.\2",
            r"^blocks\.(\d+)\.self_attn_residual_norm\.norm\.(.*)$": r"model.blocks.\1.norm3.\2",
            r"^blocks\.(\d+)\.attn2\.to_q\.(.*)$": r"model.blocks.\1.cross_attn.q.\2",
            r"^blocks\.(\d+)\.attn2\.to_k\.(.*)$": r"model.blocks.\1.cross_attn.k.\2",
            r"^blocks\.(\d+)\.attn2\.to_v\.(.*)$": r"model.blocks.\1.cross_attn.v.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.(.*)$": r"model.blocks.\1.cross_attn.o.\2",
            r"^blocks\.(\d+)\.attn2\.norm_q\.(.*)$": r"model.blocks.\1.cross_attn.norm_q.\2",
            r"^blocks\.(\d+)\.attn2\.norm_k\.(.*)$": r"model.blocks.\1.cross_attn.norm_k.\2",
            r"^blocks\.(\d+)\.ffn\.fc_in\.(.*)$": r"model.blocks.\1.ffn.0.\2",
            r"^blocks\.(\d+)\.ffn\.fc_out\.(.*)$": r"model.blocks.\1.ffn.2.\2",
            r"^scale_shift_table$": r"model.head.modulation",
            r"^proj_out\.(.*)$": r"model.head.head.\1",
        }
    )
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    ffn_dim: int = 14336
    num_layers: int = 30
    local_attn_size: int = 32
    sink_size: int = 8
    num_frames_per_block: int = 8
    sliding_window_num_frames: int = 32

@dataclass
class LongLive2VideoConfig(WanVideoConfig):
    arch_config: DiTArchConfig = field(default_factory=LongLive2ArchConfig)
