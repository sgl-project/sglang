# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_boogu_image_layer


@dataclass
class BooguImageArchConfig(DiTArchConfig):
    all_patch_size: Tuple[int, ...] = (2,)
    all_f_patch_size: Tuple[int, ...] = (1,)
    in_channels: int = 16
    out_channels: int | None = None
    dim: int = 3360
    num_layers: int = 40
    num_double_stream_layers: int = 8
    n_refiner_layers: int = 2
    num_attention_heads: int = 28
    n_kv_heads: int = 7
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 4096
    num_instruction_feature_layers: int = 1
    instruction_reduce_type: str = "mean"
    rope_theta: float = 10000.0
    t_scale: float = 1000.0
    axes_dims: Tuple[int, int, int] = (40, 40, 40)
    axes_lens: Tuple[int, int, int] = (2048, 1664, 1664)
    max_ref_images: int = 5

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_boogu_image_layer])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"(.*_feed_forward|.*\.feed_forward)\.linear_1\.weight$": (
                r"\1.w13.weight",
                0,
                2,
            ),
            r"(.*_feed_forward|.*\.feed_forward)\.linear_3\.weight$": (
                r"\1.w13.weight",
                1,
                2,
            ),
            r"(.*_feed_forward|.*\.feed_forward)\.linear_2\.weight$": r"\1.w2.weight",
        }
    )

    @property
    def num_single_stream_layers(self) -> int:
        return self.num_layers - self.num_double_stream_layers

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.num_channels_latents = self.in_channels
        self.hidden_size = self.dim
        head_dim = self.dim // self.num_attention_heads
        if head_dim != sum(self.axes_dims):
            raise ValueError(
                f"head_dim ({head_dim}) must equal sum(axes_dims) ({sum(self.axes_dims)})"
            )
        if self.instruction_reduce_type not in ("mean", "cat"):
            raise ValueError(
                f"unsupported instruction_reduce_type: {self.instruction_reduce_type}"
            )
        self.preprocessed_cap_feat_dim = (
            self.cap_feat_dim * self.num_instruction_feature_layers
            if self.instruction_reduce_type == "cat"
            else self.cap_feat_dim
        )


@dataclass
class BooguImageDitConfig(DiTConfig):
    arch_config: BooguImageArchConfig = field(default_factory=BooguImageArchConfig)

    prefix: str = "boogu_image"
