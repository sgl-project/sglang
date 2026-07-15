# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_layer
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class Ideogram4DiTArchConfig(DiTArchConfig):
    adaln_dim: int = 512
    attention_head_dim: int = 256
    in_channels: int = 128
    intermediate_size: int = 12288
    llm_features_dim: int = 53248
    mrope_section: tuple[int, int, int] | list[int] = (24, 20, 20)
    norm_eps: float = 1e-5
    num_attention_heads: int = 18
    num_layers: int = 34
    rope_theta: int = 5_000_000
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(layers\.\d+\.attention)\.to_q\.(.*)$": (
                r"\1.qkv.\2",
                0,
                3,
            ),
            r"^(layers\.\d+\.attention)\.to_k\.(.*)$": (
                r"\1.qkv.\2",
                1,
                3,
            ),
            r"^(layers\.\d+\.attention)\.to_v\.(.*)$": (
                r"\1.qkv.\2",
                2,
                3,
            ),
            r"^(layers\.\d+\.attention)\.to_out\.0\.(.*)$": r"\1.o.\2",
        }
    )
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_layer])
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels


@dataclass
class Ideogram4DiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Ideogram4DiTArchConfig)
    prefix: str = "ideogram4"
    # The official FP8 checkpoint stores row-wise FP8 weights without a
    # quantization_config, so its native loader intentionally defaults to the
    # dedicated weight-only FP8 linears. Distilled fal checkpoints instead
    # store floating-point weights and must use ordinary TP-aware linears.
    use_weight_only_fp8_linears: bool = True


@dataclass
class Ideogram4DistilledDiTConfig(Ideogram4DiTConfig):
    use_weight_only_fp8_linears: bool = False
