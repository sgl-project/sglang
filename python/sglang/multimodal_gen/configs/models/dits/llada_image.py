# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class LLaDAImageArchConfig(DiTArchConfig):
    all_patch_size: tuple[int, ...] = (1,)
    all_f_patch_size: tuple[int, ...] = (1,)
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (32768, 1024, 1024)
    cap_feat_dim: int = 2560
    dim: int = 3840
    in_channels: int = 128
    n_heads: int = 30
    n_layers: int = 30
    n_refiner_layers: int = 2
    norm_eps: float = 1e-5
    qk_norm: bool = True
    rope_theta: float = 256.0
    semantic_feat_dim: int = 4096
    t_scale: float = 1000.0
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"(.*)\.attention\.to_q\.weight$": (r"\1.attention.to_qkv.weight", 0, 3),
            r"(.*)\.attention\.to_k\.weight$": (r"\1.attention.to_qkv.weight", 1, 3),
            r"(.*)\.attention\.to_v\.weight$": (r"\1.attention.to_qkv.weight", 2, 3),
            r"(.*)\.feed_forward\.w1\.weight$": (r"\1.feed_forward.w13.weight", 0, 2),
            r"(.*)\.feed_forward\.w3\.weight$": (r"\1.feed_forward.w13.weight", 1, 2),
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.dim
        self.num_attention_heads = self.n_heads
        self.num_channels_latents = self.in_channels


@dataclass
class LLaDAImageDitConfig(DiTConfig):
    arch_config: LLaDAImageArchConfig = field(default_factory=LLaDAImageArchConfig)
    prefix: str = "llada_image"
