# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class FluxArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    # nunchaku checkpoint uses different weight names; map to sglang flux layout
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # HF diffusers format
            r"^transformer\.(\w*)\.(.*)$": r"\1.\2",
            # transformer_blocks nunchaku format (raw export - before internal conversion)
            r"^transformer_blocks\.(\d+)\.mlp_fc1\.(.*)$": r"transformer_blocks.\1.ff.net.0.proj.\2",
            r"^transformer_blocks\.(\d+)\.mlp_fc2\.(.*)$": r"transformer_blocks.\1.ff.net.2.\2",
            r"^transformer_blocks\.(\d+)\.mlp_context_fc1\.(.*)$": r"transformer_blocks.\1.ff_context.net.0.proj.\2",
            r"^transformer_blocks\.(\d+)\.mlp_context_fc2\.(.*)$": r"transformer_blocks.\1.ff_context.net.2.\2",
            r"^transformer_blocks\.(\d+)\.qkv_proj\.(.*)$": r"transformer_blocks.\1.attn.to_qkv.\2",
            r"^transformer_blocks\.(\d+)\.qkv_proj_context\.(.*)$": r"transformer_blocks.\1.attn.to_added_qkv.\2",
            r"^transformer_blocks\.(\d+)\.out_proj\.(.*)$": r"transformer_blocks.\1.attn.to_out.0.\2",
            r"^transformer_blocks\.(\d+)\.out_proj_context\.(.*)$": r"transformer_blocks.\1.attn.to_add_out.\2",
            r"^transformer_blocks\.(\d+)\.norm_q\.(.*)$": r"transformer_blocks.\1.attn.norm_q.\2",
            r"^transformer_blocks\.(\d+)\.norm_k\.(.*)$": r"transformer_blocks.\1.attn.norm_k.\2",
            r"^transformer_blocks\.(\d+)\.norm_added_q\.(.*)$": r"transformer_blocks.\1.attn.norm_added_q.\2",
            r"^transformer_blocks\.(\d+)\.norm_added_k\.(.*)$": r"transformer_blocks.\1.attn.norm_added_k.\2",
            # transformer_blocks nunchaku format (already converted with convert_flux_state_dict)
            r"^transformer_blocks\.(\d+)\.attn\.add_qkv_proj\.(.*)$": r"transformer_blocks.\1.attn.to_added_qkv.\2",
            # single_transformer_blocks nunchaku format (raw export - before internal conversion)
            r"^single_transformer_blocks\.(\d+)\.qkv_proj\.(.*)$": r"single_transformer_blocks.\1.attn.to_qkv.\2",
            r"^single_transformer_blocks\.(\d+)\.out_proj\.(.*)$": r"single_transformer_blocks.\1.attn.to_out.0.\2",
            r"^single_transformer_blocks\.(\d+)\.norm_q\.(.*)$": r"single_transformer_blocks.\1.attn.norm_q.\2",
            r"^single_transformer_blocks\.(\d+)\.norm_k\.(.*)$": r"single_transformer_blocks.\1.attn.norm_k.\2",
            # nunchaku quantization parameter name conversions (apply to all blocks)
            r"^(.*)\.smooth_orig$": r"\1.smooth_factor_orig",
            r"^(.*)\.smooth$": r"\1.smooth_factor",
            r"^(.*)\.lora_down$": r"\1.proj_down",
            r"^(.*)\.lora_up$": r"\1.proj_up",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class FluxConfig(DiTConfig):

    arch_config: DiTArchConfig = field(default_factory=FluxArchConfig)

    prefix: str = "Flux"
