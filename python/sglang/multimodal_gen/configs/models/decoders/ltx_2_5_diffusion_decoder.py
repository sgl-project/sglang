# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig


@dataclass
class LTX25DiffusionDecoderArchConfig(ArchConfig):
    """LTX-2.5 diffusion video decoder.

    Field names match `diffusion_decoder/config.json` verbatim so
    `update_model_arch` populates this straight from the checkpoint.
    """

    latent_channels: int = 128
    out_channels: int = 3
    patch_size: int = 4
    spatial_compression_ratio: int = 32
    temporal_compression_ratio: int = 8
    scaling_factor: float = 1.0

    decoder_head_dim: int = 64
    decoder_t_emb_dim: int = 384
    decoder_model_output_type: str = "x0"
    decoder_num_inference_steps: int = 1
    decoder_timestep_scale_multiplier: float = 1000.0

    decoder_stage_channels: list[int] = field(
        default_factory=lambda: [2048, 1024, 512, 512, 256]
    )
    decoder_stage_depths: list[int] = field(default_factory=lambda: [4, 6, 4, 2, 8])
    decoder_stage_kernels: list[list[int]] = field(
        default_factory=lambda: [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5]]
    )
    decoder_stage5_kernel: list[int] = field(default_factory=lambda: [11, 11, 11])
    decoder_upsample_strides: list[list[int]] = field(
        default_factory=lambda: [[1, 2, 2], [2, 1, 1], [2, 2, 2], [2, 2, 2]]
    )
    decoder_upsample_channel_reductions: list[int] = field(
        default_factory=lambda: [2, 2, 1, 2]
    )


@dataclass
class LTX25DiffusionDecoderConfig(ModelConfig):
    arch_config: LTX25DiffusionDecoderArchConfig = field(
        default_factory=LTX25DiffusionDecoderArchConfig
    )
