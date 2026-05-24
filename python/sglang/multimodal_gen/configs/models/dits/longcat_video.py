# Copied and adapted from: ../LongCat-Video/longcat_video/modules/longcat_video_dit.py

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class LongCatVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    # LongCat-Video only implements SDPA and FlashAttn2/3 attention paths.
    # SAGE_ATTN, AITER, VIDEO_SPARSE_ATTN, etc. are not implemented in
    # LongCatSingleStreamBlock and would silently fall back to SDPA or crash.
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FA,
        }
    )

    # Keep official LongCat checkpoint keys unchanged.
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    in_channels: int = 16
    out_channels: int = 16
    hidden_size: int = 4096
    num_layers: int = 48
    num_attention_heads: int = 32
    caption_channels: int = 4096
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_tokens_zero_pad: bool = True
    enable_flashattn2: bool = False
    enable_bsa: bool = False
    cp_split_hw: list[int] | None = None

    mlp_ratio: int = 4
    adaln_tembed_dim: int = 512
    frequency_embedding_size: int = 256
    enable_flashattn3: bool = False
    enable_xformers: bool = False
    bsa_params: dict | None = field(
        default_factory=lambda: {
            "sparsity": 0.9375,
            "chunk_3d_shape_q": [4, 4, 4],
            "chunk_3d_shape_k": [4, 4, 4],
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.patch_size = tuple(self.patch_size)
        self.num_channels_latents = self.out_channels


@dataclass
class LongCatVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LongCatVideoArchConfig)

    prefix: str = "LongCatVideo"
