# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from: https://github.com/Robbyant/lingbot-world

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class LingBotWorldArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^patch_embedding_wancamctrl\.(.*)$": r"patch_embedding_wancamctrl.proj.\1",
            r"^c2ws_hidden_states_layer1\.(.*)$": r"c2ws_mlp.fc_in.\1",
            r"^c2ws_hidden_states_layer2\.(.*)$": r"c2ws_mlp.fc_out.\1",
            r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.cam_injector_layer1\.(.*)$": r"blocks.\1.cam_conditioner.cam_injector.fc_in.\2",
            r"^blocks\.(\d+)\.cam_injector_layer2\.(.*)$": r"blocks.\1.cam_conditioner.cam_injector.fc_out.\2",
            r"^blocks\.(\d+)\.cam_scale_layer\.(.*)$": r"blocks.\1.cam_conditioner.cam_scale_layer.\2",
            r"^blocks\.(\d+)\.cam_shift_layer\.(.*)$": r"blocks.\1.cam_conditioner.cam_shift_layer.\2",
            r"^head\.modulation$": r"scale_shift_table",
            r"^head\.head\.(.*)$": r"proj_out.\1",
        }
    )
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})
    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 36
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])
    boundary_ratio: float | None = None
    local_attn_size: int = -1
    sink_size: int = 3
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 45

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class LingBotWorldVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LingBotWorldArchConfig)

    prefix: str = "Wan"
