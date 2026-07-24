# SPDX-License-Identifier: Apache-2.0
"""Architecture and runtime configuration for SANA-Video."""

from dataclasses import dataclass, field
from typing import Literal

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class SanaVideoArchConfig(DiTArchConfig):
    """Diffusers-compatible SANA-Video transformer configuration."""

    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 20
    attention_head_dim: int = 112
    num_attention_heads: int = 20
    num_cross_attention_heads: int = 20
    cross_attention_head_dim: int = 112
    cross_attention_dim: int = 2240
    caption_channels: int = 2304
    mlp_ratio: float = 3.0
    dropout: float = 0.0
    attention_bias: bool = False
    sample_size: int = 30
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    guidance_embeds: bool = False
    guidance_embeds_scale: float = 0.1
    qk_norm: str | None = "rms_norm_across_heads"
    rope_max_seq_len: int = 1024

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Fuse self-attention Q/K/V into one projection.
            r"^(transformer_blocks\.\d+\.attn1)\.to_q\.(.*)$": (
                r"\1.to_qkv.\2",
                0,
                3,
            ),
            r"^(transformer_blocks\.\d+\.attn1)\.to_k\.(.*)$": (
                r"\1.to_qkv.\2",
                1,
                3,
            ),
            r"^(transformer_blocks\.\d+\.attn1)\.to_v\.(.*)$": (
                r"\1.to_qkv.\2",
                2,
                3,
            ),
            # Cross-attention K/V share the step-invariant text input.
            r"^(transformer_blocks\.\d+\.attn2)\.to_k\.(.*)$": (
                r"\1.to_kv.\2",
                0,
                2,
            ),
            r"^(transformer_blocks\.\d+\.attn2)\.to_v\.(.*)$": (
                r"\1.to_kv.\2",
                1,
                2,
            ),
            r"^transformer\.(.*)$": r"\1",
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.patch_size = tuple(self.patch_size)
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class SanaVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaVideoArchConfig)
    prefix: str = "SanaVideo"
    # Keep the dense torch.compile graph free of cache-control graph breaks
    # unless the deployment explicitly opts into EasyCache.
    enable_easycache: bool = False
    # The reference implementation promotes the two linear-attention
    # aggregation matmuls to fp32. ``bf16`` keeps them on tensor cores and is
    # the optional Sol-Engine fast path.
    linear_attention_aggregation_precision: Literal["fp32", "bf16"] = "fp32"
    # Sol-Engine found the generic max-autotune mode can stall on the very
    # large grouped convolution used by GLUMBTempConv. Keep the portable mode
    # as the model default; users can still override it explicitly.
    torch_compile_mode: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.enable_easycache, bool):
            raise TypeError(
                f"enable_easycache must be bool, got {self.enable_easycache!r}"
            )
        if self.linear_attention_aggregation_precision not in {"fp32", "bf16"}:
            raise ValueError(
                "linear_attention_aggregation_precision must be 'fp32' or 'bf16', "
                f"got {self.linear_attention_aggregation_precision!r}"
            )
