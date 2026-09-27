# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class QwenImageArchConfig(DiTArchConfig):
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
    zero_cond_t: bool = False

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Merge the short text-stream projections into one tensor-parallel
            # GEMM. The loader only applies these rules when the fused target
            # exists, so quantization backends that keep the original modules
            # continue to load their unfused parameters.
            r"^(.*\.attn)\.add_q_proj\.(.+)$": (
                r"\1.to_added_qkv.\2",
                0,
                3,
            ),
            r"^(.*\.attn)\.add_k_proj\.(.+)$": (
                r"\1.to_added_qkv.\2",
                1,
                3,
            ),
            r"^(.*\.attn)\.add_v_proj\.(.+)$": (
                r"\1.to_added_qkv.\2",
                2,
                3,
            ),
            # LoRA mappings
            r"^(transformer_blocks\.\d+\.attn\..*\.lora_[AB])\.default$": r"\1",
            # SVDquant mappings
            r"(.*)\.add_qkv_proj\.(.+)$": r"\1.to_added_qkv.\2",
            r"(transformer_blocks\.\d+\.(img_mlp|txt_mlp)\..*\.(smooth_factor_orig|wcscales))$": r"\1",
            r".*\.wtscale$": r"",
        }
    )

    # Serialized ModelOpt checkpoints keep the added Q/K/V projections as
    # separate modules, including their BF16 fallback layers. Do not apply the
    # runtime-only fused mapping while inferring their quantized tensor layout.
    quant_param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class QwenImageEditPlus_2511_ArchConfig(QwenImageArchConfig):
    zero_cond_t: bool = True


@dataclass
class QwenImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=QwenImageArchConfig)

    prefix: str = "qwenimage"


@dataclass
class QwenImageEditPlus_2511_DitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(
        default_factory=QwenImageEditPlus_2511_ArchConfig
    )

    prefix: str = "qwenimageedit"
