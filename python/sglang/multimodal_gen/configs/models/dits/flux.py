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
    guidance_embeds: bool = True
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    exclude_lora_layers: list[str] = field(
        default_factory=lambda: [
            "time_guidance_embed.timestep_embedder.linear_1",
            "time_guidance_embed.timestep_embedder.linear_2",
            "time_guidance_embed.guidance_embedder.linear_1",
            "time_guidance_embed.guidance_embedder.linear_2",
        ]
    )

    # nunchaku checkpoint uses different weight names; map to sglang flux layout
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # HF diffusers format: strip leading "transformer." prefix
            r"^transformer\.(\w*)\.(.*)$": r"\1.\2",
            # FLUX2-nvfp4 format: double blocks - image attention QKV (packed, fused)
            r"^double_blocks\.(\d+)\.img_attn\.qkv\.(.*)$": r"transformer_blocks.\1.attn.to_qkv.\2",
            r"^double_blocks\.(\d+)\.img_attn\.proj\.(.*)$": r"transformer_blocks.\1.attn.to_out.0.\2",
            r"^double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.(.*)$": r"transformer_blocks.\1.attn.norm_q.\2",
            r"^double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.(.*)$": r"transformer_blocks.\1.attn.norm_k.\2",
            # FLUX2-nvfp4 format: double blocks - text/context attention QKV (packed, fused)
            r"^double_blocks\.(\d+)\.txt_attn\.qkv\.(.*)$": r"transformer_blocks.\1.attn.to_added_qkv.\2",
            r"^double_blocks\.(\d+)\.txt_attn\.proj\.(.*)$": r"transformer_blocks.\1.attn.to_add_out.\2",
            r"^double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.(.*)$": r"transformer_blocks.\1.attn.norm_added_q.\2",
            r"^double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.(.*)$": r"transformer_blocks.\1.attn.norm_added_k.\2",
            # FLUX2-nvfp4  format: double blocks - image MLP
            r"^double_blocks\.(\d+)\.img_mlp\.0\.(.*)$": r"transformer_blocks.\1.ff.linear_in.\2",
            r"^double_blocks\.(\d+)\.img_mlp\.2\.(.*)$": r"transformer_blocks.\1.ff.linear_out.\2",
            # FLUX2-nvfp4  format: double blocks - text/context MLP
            r"^double_blocks\.(\d+)\.txt_mlp\.0\.(.*)$": r"transformer_blocks.\1.ff_context.linear_in.\2",
            r"^double_blocks\.(\d+)\.txt_mlp\.2\.(.*)$": r"transformer_blocks.\1.ff_context.linear_out.\2",
            # FLUX2-nvfp4  format: single blocks
            r"^single_blocks\.(\d+)\.linear1\.(.*)$": r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.\2",
            r"^single_blocks\.(\d+)\.linear2\.(.*)$": r"single_transformer_blocks.\1.attn.to_out.\2",
            r"^single_blocks\.(\d+)\.norm\.query_norm\.(.*)$": r"single_transformer_blocks.\1.attn.norm_q.\2",
            r"^single_blocks\.(\d+)\.norm\.key_norm\.(.*)$": r"single_transformer_blocks.\1.attn.norm_k.\2",
            # FLUX2-nvfp4  format: non-block input/output projections
            r"^img_in\.(.*)$": r"x_embedder.\1",
            r"^txt_in\.(.*)$": r"context_embedder.\1",
            r"^time_in\.in_layer\.(.*)$": r"time_guidance_embed.timestep_embedder.linear_1.\1",
            r"^time_in\.out_layer\.(.*)$": r"time_guidance_embed.timestep_embedder.linear_2.\1",
            r"^guidance_in\.in_layer\.(.*)$": r"time_guidance_embed.guidance_embedder.linear_1.\1",
            r"^guidance_in\.out_layer\.(.*)$": r"time_guidance_embed.guidance_embedder.linear_2.\1",
            r"^double_stream_modulation_img\.lin\.(.*)$": r"double_stream_modulation_img.linear.\1",
            r"^double_stream_modulation_txt\.lin\.(.*)$": r"double_stream_modulation_txt.linear.\1",
            r"^single_stream_modulation\.lin\.(.*)$": r"single_stream_modulation.linear.\1",
            r"^final_layer\.adaLN_modulation\.1\.(.*)$": r"norm_out.linear.\1",
            r"^final_layer\.linear\.(.*)$": r"proj_out.\1",
            # FLUX2-nvfp4 format: RMSNorm uses "scale" parameter; rename to "weight" (model uses .weight)
            r"^(.*)\.scale$": r"\1.weight",
            # transformer_blocks nunchaku format (raw export - before internal conversion)
            r"^transformer_blocks\.(\d+)\.mlp_fc1\.(.*)$": r"transformer_blocks.\1.ff.net.0.proj.\2",
            r"^transformer_blocks\.(\d+)\.mlp_fc2\.(.*)$": r"transformer_blocks.\1.ff.net.2.\2",
            r"^transformer_blocks\.(\d+)\.mlp_context_fc1\.(.*)$": r"transformer_blocks.\1.ff_context.net.0.proj.\2",
            r"^transformer_blocks\.(\d+)\.mlp_context_fc2\.(.*)$": r"transformer_blocks.\1.ff_context.net.2.\2",
            # nunchaku packed QKV → fused to_qkv / to_added_qkv (matches use_fused_qkv in model)
            r"^transformer_blocks\.(\d+)\.qkv_proj\.(.*)$": r"transformer_blocks.\1.attn.to_qkv.\2",
            r"^transformer_blocks\.(\d+)\.qkv_proj_context\.(.*)$": r"transformer_blocks.\1.attn.to_added_qkv.\2",
            r"^transformer_blocks\.(\d+)\.out_proj\.(.*)$": r"transformer_blocks.\1.attn.to_out.0.\2",
            r"^transformer_blocks\.(\d+)\.out_proj_context\.(.*)$": r"transformer_blocks.\1.attn.to_add_out.\2",
            r"^transformer_blocks\.(\d+)\.norm_q\.(.*)$": r"transformer_blocks.\1.attn.norm_q.\2",
            r"^transformer_blocks\.(\d+)\.norm_k\.(.*)$": r"transformer_blocks.\1.attn.norm_k.\2",
            r"^transformer_blocks\.(\d+)\.norm_added_q\.(.*)$": r"transformer_blocks.\1.attn.norm_added_q.\2",
            r"^transformer_blocks\.(\d+)\.norm_added_k\.(.*)$": r"transformer_blocks.\1.attn.norm_added_k.\2",
            # nunchaku format (already converted): add_qkv_proj → fused to_added_qkv
            r"^transformer_blocks\.(\d+)\.attn\.add_qkv_proj\.(.*)$": r"transformer_blocks.\1.attn.to_added_qkv.\2",
            # single_transformer_blocks nunchaku format (raw export - before internal conversion)
            r"^single_transformer_blocks\.(\d+)\.qkv_proj\.(.*)$": r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.\2",
            r"^single_transformer_blocks\.(\d+)\.out_proj\.(.*)$": r"single_transformer_blocks.\1.attn.to_out.\2",
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
