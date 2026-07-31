# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for BAGEL generation and multimodal understanding.

The architecture values follow the official BAGEL implementation and the
``ByteDance-Seed/BAGEL-7B-MoT`` checkpoint. Image-producing pipelines load
both mixture-of-transformer experts, while Understanding can omit the
generation-only expert and latent projection modules.

Source: https://github.com/ByteDance-Seed/Bagel/tree/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class BagelDiTArchConfig(DiTArchConfig):
    """Architecture configuration for BAGEL's Qwen2 mixture-of-transformers denoiser."""

    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    attention_head_dim: int = 128
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    latent_patch_size: int = 2
    max_latent_size: int = 64
    latent_channel: int = 16
    latent_downsample: int = 16
    timestep_frequency_embedding_size: int = 256
    latent_position_embedding_rows: int = 4096

    # These IDs are stable in the official tokenizer. The pipeline validates
    # them and passes them through the request-local BagelContext.
    start_of_image_token_id: int = 151652
    end_of_image_token_id: int = 151653

    param_names_mapping: dict[str, str] = field(
        default_factory=lambda: {
            r"^language_model\.model\.embed_tokens\.(.*)$": r"embed_tokens.\1",
            r"^language_model\.model\.layers\.(\d+)\.input_layernorm_moe_gen\.(.*)$": r"layers.\1.gen_in_norm.\2",
            r"^language_model\.model\.layers\.(\d+)\.input_layernorm\.(.*)$": r"layers.\1.und_in_norm.\2",
            r"^language_model\.model\.layers\.(\d+)\.post_attention_layernorm_moe_gen\.(.*)$": r"layers.\1.gen_post_norm.\2",
            r"^language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.(.*)$": r"layers.\1.und_post_norm.\2",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj_moe_gen\.(.*)$": r"layers.\1.attn.gen_\2_proj.\3",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.o_proj_moe_gen\.(.*)$": r"layers.\1.attn.gen_o_proj.\2",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.(q|k)_norm_moe_gen\.(.*)$": r"layers.\1.attn.gen_\2_norm.\3",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(.*)$": r"layers.\1.attn.und_\2_proj.\3",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.(.*)$": r"layers.\1.attn.und_o_proj.\2",
            r"^language_model\.model\.layers\.(\d+)\.self_attn\.(q|k)_norm\.(.*)$": r"layers.\1.attn.und_\2_norm.\3",
            r"^language_model\.model\.layers\.(\d+)\.mlp_moe_gen\.(gate|up|down)_proj\.(.*)$": r"layers.\1.mlp.gen_\2.\3",
            r"^language_model\.model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.(.*)$": r"layers.\1.mlp.und_\2.\3",
            r"^language_model\.model\.norm_moe_gen\.(.*)$": r"gen_final_norm.\1",
            r"^language_model\.model\.norm\.(.*)$": r"und_final_norm.\1",
            r"^language_model\.lm_head\.(.*)$": r"lm_head.\1",
            # Reuse SGLang's Apache-2.0 TimestepEmbedder.  Its MLP names are
            # fc_in/fc_out rather than the checkpoint's Sequential indices.
            r"^time_embedder\.mlp\.0\.(.*)$": r"time_embedder.mlp.fc_in.\1",
            r"^time_embedder\.mlp\.2\.(.*)$": r"time_embedder.mlp.fc_out.\1",
        }
    )

    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )

    def __post_init__(self) -> None:
        """Validate dimensions and populate base DiT fields."""
        super().__post_init__()
        if self.num_attention_heads * self.attention_head_dim != self.hidden_size:
            raise ValueError(
                "num_attention_heads * attention_head_dim must equal hidden_size"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        expected_rows = self.max_latent_size**2
        if self.latent_position_embedding_rows != expected_rows:
            raise ValueError(
                "latent_position_embedding_rows must equal max_latent_size squared; "
                f"got {self.latent_position_embedding_rows} and {self.max_latent_size}"
            )
        self.num_channels_latents = self.latent_patch_size**2 * self.latent_channel


@dataclass
class BagelDiTConfig(DiTConfig):
    """Runtime configuration for the BAGEL denoiser.

    ``load_lm_head`` is opt-in so the baseline T2I and Editing pipelines do
    not keep the checkpoint's roughly 1 GiB language head resident.
    ``load_generation_expert`` lets the text-only Understanding pipeline avoid
    roughly 12 GiB of generation-expert weights that it never executes.
    """

    arch_config: BagelDiTArchConfig = field(default_factory=BagelDiTArchConfig)
    prefix: str = "bagel"
    load_lm_head: bool = False
    load_generation_expert: bool = True
