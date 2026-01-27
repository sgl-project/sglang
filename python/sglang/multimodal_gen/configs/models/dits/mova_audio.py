# Copied and adapted from: mossVG/mova/diffusion/models/wan_audio_dit.py
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class MOVAAudioArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^text_embedding\.0\.(.*)$": r"text_embedding.fc_in.\1",
            r"^text_embedding\.2\.(.*)$": r"text_embedding.fc_out.\1",
            r"^time_embedding\.0\.(.*)$": r"time_embedding.fc_in.\1",
            r"^time_embedding\.2\.(.*)$": r"time_embedding.fc_out.\1",
            r"^img_emb\.proj\.1\.(.*)$": r"img_emb.fc_in.\1",
            r"^img_emb\.proj\.3\.(.*)$": r"img_emb.fc_out.\1",
        }
    )
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    dim: int = 1536
    in_dim: int = 128
    ffn_dim: int = 6144
    out_dim: int = 128
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_heads: int = 12
    num_layers: int = 30
    has_image_input: bool = False
    has_image_pos_emb: bool = False
    has_ref_conv: bool = False
    add_control_adapter: bool = False
    in_dim_control_adapter: int = 24
    seperated_timestep: bool = False
    require_vae_embedding: bool = False
    require_clip_embedding: bool = False
    fuse_vae_embedding_in_latents: bool = False
    vae_type: str = "dac"

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.dim
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.out_dim
        assert (
            not self.has_image_input
        ), "has_image_input must be False; it's a config from Diffsynth Studio, which means the model uses CLIP for image encoding (we don't)."


@dataclass
class MOVAAudioConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MOVAAudioArchConfig)
    prefix: str = "mova_audio"
