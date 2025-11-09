# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_double_block(n: str, m) -> bool:
    return "double" in n and str.isdigit(n.split(".")[-1])


def is_single_block(n: str, m) -> bool:
    return "single" in n and str.isdigit(n.split(".")[-1])


def is_refiner_block(n: str, m) -> bool:
    return "refiner" in n and str.isdigit(n.split(".")[-1])


def is_txt_in(n: str, m) -> bool:
    return n.split(".")[-1] == "txt_in"


@dataclass
class HunyuanVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_double_block, is_single_block, is_refiner_block]
    )

    _compile_conditions: list = field(
        default_factory=lambda: [is_double_block, is_single_block, is_txt_in]
    )

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # 1. context_embedder.time_text_embed submodules (specific rules, applied first):
            r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_1\.(.*)$": r"txt_in.t_embedder.mlp.fc_in.\1",
            r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_2\.(.*)$": r"txt_in.t_embedder.mlp.fc_out.\1",
            r"^context_embedder\.proj_in\.(.*)$": r"txt_in.input_embedder.\1",
            r"^context_embedder\.time_text_embed\.text_embedder\.linear_1\.(.*)$": r"txt_in.c_embedder.fc_in.\1",
            r"^context_embedder\.time_text_embed\.text_embedder\.linear_2\.(.*)$": r"txt_in.c_embedder.fc_out.\1",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm1\.(.*)$": r"txt_in.refiner_blocks.\1.norm1.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm2\.(.*)$": r"txt_in.refiner_blocks.\1.norm2.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_q\.(.*)$": (
                r"txt_in.refiner_blocks.\1.self_attn_qkv.\2",
                0,
                3,
            ),
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_k\.(.*)$": (
                r"txt_in.refiner_blocks.\1.self_attn_qkv.\2",
                1,
                3,
            ),
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_v\.(.*)$": (
                r"txt_in.refiner_blocks.\1.self_attn_qkv.\2",
                2,
                3,
            ),
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$": r"txt_in.refiner_blocks.\1.self_attn_proj.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0(?:\.proj)?\.(.*)$": r"txt_in.refiner_blocks.\1.mlp.fc_in.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2(?:\.proj)?\.(.*)$": r"txt_in.refiner_blocks.\1.mlp.fc_out.\2",
            r"^context_embedder\.token_refiner\.refiner_blocks\.(\d+)\.norm_out\.linear\.(.*)$": r"txt_in.refiner_blocks.\1.adaLN_modulation.linear.\2",
            # 3. x_embedder mapping:
            r"^x_embedder\.proj\.(.*)$": r"img_in.proj.\1",
            # 4. Top-level time_text_embed mappings:
            r"^time_text_embed\.timestep_embedder\.linear_1\.(.*)$": r"time_in.mlp.fc_in.\1",
            r"^time_text_embed\.timestep_embedder\.linear_2\.(.*)$": r"time_in.mlp.fc_out.\1",
            r"^time_text_embed\.guidance_embedder\.linear_1\.(.*)$": r"guidance_in.mlp.fc_in.\1",
            r"^time_text_embed\.guidance_embedder\.linear_2\.(.*)$": r"guidance_in.mlp.fc_out.\1",
            r"^time_text_embed\.text_embedder\.linear_1\.(.*)$": r"vector_in.fc_in.\1",
            r"^time_text_embed\.text_embedder\.linear_2\.(.*)$": r"vector_in.fc_out.\1",
            # 5. transformer_blocks mapping:
            r"^transformer_blocks\.(\d+)\.norm1\.linear\.(.*)$": r"double_blocks.\1.img_mod.linear.\2",
            r"^transformer_blocks\.(\d+)\.norm1_context\.linear\.(.*)$": r"double_blocks.\1.txt_mod.linear.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$": r"double_blocks.\1.img_attn_q_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$": r"double_blocks.\1.img_attn_k_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$": (
                r"double_blocks.\1.img_attn_qkv.\2",
                0,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$": (
                r"double_blocks.\1.img_attn_qkv.\2",
                1,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$": (
                r"double_blocks.\1.img_attn_qkv.\2",
                2,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.add_q_proj\.(.*)$": (
                r"double_blocks.\1.txt_attn_qkv.\2",
                0,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.add_k_proj\.(.*)$": (
                r"double_blocks.\1.txt_attn_qkv.\2",
                1,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.add_v_proj\.(.*)$": (
                r"double_blocks.\1.txt_attn_qkv.\2",
                2,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$": r"double_blocks.\1.img_attn_proj.\2",
            # Corrected: merge attn.to_add_out into the main projection.
            r"^transformer_blocks\.(\d+)\.attn\.to_add_out\.(.*)$": r"double_blocks.\1.txt_attn_proj.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_q\.(.*)$": r"double_blocks.\1.txt_attn_q_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_added_k\.(.*)$": r"double_blocks.\1.txt_attn_k_norm.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0(?:\.proj)?\.(.*)$": r"double_blocks.\1.img_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2(?:\.proj)?\.(.*)$": r"double_blocks.\1.img_mlp.fc_out.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.0(?:\.proj)?\.(.*)$": r"double_blocks.\1.txt_mlp.fc_in.\2",
            r"^transformer_blocks\.(\d+)\.ff_context\.net\.2(?:\.proj)?\.(.*)$": r"double_blocks.\1.txt_mlp.fc_out.\2",
            # 6. single_transformer_blocks mapping:
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$": r"single_blocks.\1.q_norm.\2",
            r"^single_transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$": r"single_blocks.\1.k_norm.\2",
            r"^single_transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                0,
                4,
            ),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                1,
                4,
            ),
            r"^single_transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                2,
                4,
            ),
            r"^single_transformer_blocks\.(\d+)\.proj_mlp\.(.*)$": (
                r"single_blocks.\1.linear1.\2",
                3,
                4,
            ),
            # Corrected: map proj_out to modulation.linear rather than a separate proj_out branch.
            r"^single_transformer_blocks\.(\d+)\.proj_out\.(.*)$": r"single_blocks.\1.linear2.\2",
            r"^single_transformer_blocks\.(\d+)\.norm\.linear\.(.*)$": r"single_blocks.\1.modulation.linear.\2",
            # 7. Final layers mapping:
            r"^norm_out\.linear\.(.*)$": r"final_layer.adaLN_modulation.linear.\1",
            r"^proj_out\.(.*)$": r"final_layer.linear.\1",
        }
    )

    # Reverse mapping for saving checkpoints: custom -> hf
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    patch_size: int = 2
    patch_size_t: int = 1
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    mlp_ratio: float = 4.0
    num_layers: int = 20
    num_single_layers: int = 40
    num_refiner_layers: int = 2
    rope_axes_dim: tuple[int, int, int] = (16, 56, 56)
    guidance_embeds: bool = False
    dtype: torch.dtype | None = None
    text_embed_dim: int = 4096
    pooled_projection_dim: int = 768
    rope_theta: int = 256
    qk_norm: str = "rms_norm"
    exclude_lora_layers: list[str] = field(
        default_factory=lambda: ["img_in", "txt_in", "time_in", "vector_in"]
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size: int = self.attention_head_dim * self.num_attention_heads
        self.num_channels_latents: int = self.in_channels


@dataclass
class HunyuanVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HunyuanVideoArchConfig)

    prefix: str = "Hunyuan"
