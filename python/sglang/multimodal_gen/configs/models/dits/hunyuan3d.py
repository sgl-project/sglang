# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Hunyuan3DDiTArchConfig(DiTArchConfig):
    """Architecture config for Hunyuan3D DiT (Flux-style for Hunyuan3D-2.0)."""

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Strip leading "model." prefix used by some exports
            r"^model\.(.*)$": r"\1",
            # MLP linear renames (double-stream blocks)
            r"^(double_blocks\.\d+\.img_mlp)\.0\.(.*)$": r"\1.fc_in.\2",
            r"^(double_blocks\.\d+\.img_mlp)\.2\.(.*)$": r"\1.fc_out.\2",
            r"^(double_blocks\.\d+\.txt_mlp)\.0\.(.*)$": r"\1.fc_in.\2",
            r"^(double_blocks\.\d+\.txt_mlp)\.2\.(.*)$": r"\1.fc_out.\2",
            # Double-stream attention: fuse split Q/K/V into fused qkv for both txt_attn and img_attn
            r"^(double_blocks\.\d+\.(?:txt_attn|img_attn))\.(?:to_q|q_proj|query)\.(.*)$": (
                r"\1.qkv.\2",
                0,
                3,
            ),
            r"^(double_blocks\.\d+\.(?:txt_attn|img_attn))\.(?:to_k|k_proj|key)\.(.*)$": (
                r"\1.qkv.\2",
                1,
                3,
            ),
            r"^(double_blocks\.\d+\.(?:txt_attn|img_attn))\.(?:to_v|v_proj|value)\.(.*)$": (
                r"\1.qkv.\2",
                2,
                3,
            ),
            # Double-stream out projection (image/text): to_out[.0].{weight,bias} and
            # txt_attn.to_add_out[.0].{weight,bias} -> proj.{weight,bias}
            r"^(double_blocks\.\d+\.(?:txt_attn|img_attn))\.to_out(?:\.0)?\.(weight|bias)$": r"\1.proj.\2",
            r"^(double_blocks\.\d+\.txt_attn)\.to_add_out(?:\.0)?\.(weight|bias)$": r"\1.proj.\2",
            # Double-stream Q/K norm aliases and convert HF 'weight' to internal 'scale'
            r"^(double_blocks\.\d+\.(?:txt_attn|img_attn))\.norm_q\.(.*)$": r"\1.norm.query_norm.\2",
            r"^(double_blocks\.\d+\.(?:txt_attn|img_attn))\.norm_k\.(.*)$": r"\1.norm.key_norm.\2",
            r"^(.*norm\.query_norm)\.weight$": r"\1.scale",
            r"^(.*norm\.key_norm)\.weight$": r"\1.scale",
            # Single-stream blocks: pack Q/K/V and MLP into linear1 ([Q, K, V, MLP]) and map out-proj to linear2
            # Apply to both single_blocks.* and single_transformer_blocks.* exports
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.attn\.(?:to_q|q_proj|query)\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                0,
                4,
            ),
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.attn\.(?:to_k|k_proj|key)\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                1,
                4,
            ),
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.attn\.(?:to_v|v_proj|value)\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                2,
                4,
            ),
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.(?:proj_mlp|mlp_fc1)\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                3,
                4,
            ),
            # Single-stream out projection variants -> linear2 (only weight/bias)
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.(?:proj_out|out_proj)(?:\.0)?\.(weight|bias)$": r"single_blocks.\1.linear2.\2",
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.attn\.to_out(?:\.0)?\.(weight|bias)$": r"single_blocks.\1.linear2.\2",
            # Single-stream Q/K norm aliases
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.attn\.norm_q\.(.*)$": r"single_blocks.\1.norm.query_norm.\2",
            r"^(?:single_blocks|single_transformer_blocks)\.(\d+)\.attn\.norm_k\.(.*)$": r"single_blocks.\1.norm.key_norm.\2",
        }
    )

    in_channels: int = 64
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_layers: int = 16
    num_single_layers: int = 32
    mlp_ratio: float = 4.0
    context_in_dim: int = 1536
    axes_dim: tuple[int, ...] = (64,)
    theta: int = 10000
    qkv_bias: bool = True
    guidance_embed: bool = False
    time_factor: float = 1000.0

    def __post_init__(self) -> None:
        if self.num_channels_latents == 0:
            self.num_channels_latents = self.in_channels
        super().__post_init__()


@dataclass
class Hunyuan3DDiTConfig(DiTConfig):
    """DiT configuration for Hunyuan3D shape generation (Flux-style)."""

    arch_config: Hunyuan3DDiTArchConfig = field(default_factory=Hunyuan3DDiTArchConfig)
    subfolder: str = "hunyuan3d-dit-v2-0"
