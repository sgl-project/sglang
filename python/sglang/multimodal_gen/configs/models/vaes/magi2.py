# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig
from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)


@dataclass
class Magi2VideoVAEArchConfig(WanVAEArchConfig):
    """Wan VAE arch verbatim; only the 48-channel latent and its normalization statistics differ from the Wan2.1 default."""

    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    is_residual: bool = True
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 4
    patch_size: int | None = 2

    # Post-patchify: 2x2 blocks fold into channels, so convs see 3 * 2**2 = 12.
    in_channels: int = 12
    out_channels: int = 12

    latents_mean: tuple[float, ...] = (
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.1570,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.1230,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.0520,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    )
    latents_std: tuple[float, ...] = (
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    )


@dataclass
class Magi2VideoVAEConfig(WanVAEConfig):
    arch_config: Magi2VideoVAEArchConfig = field(
        default_factory=Magi2VideoVAEArchConfig
    )


@dataclass
class Magi2TurboVAEArchConfig(VAEArchConfig):
    """Values mirror TurboV3-Wan22-TinyShallow_7_7.json; same 48-channel latent, 16x spatial decode, decode half only."""

    latent_channels: int = 48
    in_channels: int = 3
    out_channels: int = 3
    scaling_factor: float = 1.0
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 4

    decoder_block_out_channels: tuple[int, ...] = (64, 128, 256, 512)
    decoder_layers_per_block: tuple[int, ...] = (2, 2, 2, 3, 3)
    decoder_spatio_temporal_scaling: tuple[bool, ...] = (False, True, True, True)
    decoder_spatio_only: tuple[bool, ...] = (False, True, False, False)
    decoder_is_dw_conv: tuple[bool, ...] = (False, False, False, False, False)
    decoder_dw_kernel_size: int = 5
    decoder_causal: bool = False
    resnet_norm_eps: float = 1e-6

    aligned_feature_projection_mode: str = "conv-2layer"
    aligned_feature_projection_dim: tuple[tuple[int, int], ...] = (
        (512, 1024),
        (512, 1024),
    )
    aligned_blks_indices: tuple[int, ...] = (0, 1)

    patch_size: int = 2
    patch_size_t: int = 1
    use_unpatchify: bool = True

    # Sliding causal decode window; keeps a 10s 1080p clip in memory.
    first_chunk_size: int = 7
    step_size: int = 7


@dataclass
class Magi2TurboVAEConfig(VAEConfig):
    arch_config: Magi2TurboVAEArchConfig = field(
        default_factory=Magi2TurboVAEArchConfig
    )
    load_encoder: bool = False
    # The sliding window already bounds decode memory; tiling would blend twice.
    use_tiling: bool = False
    use_temporal_tiling: bool = False


@dataclass
class Magi2AudioVAEArchConfig(VAEArchConfig):
    """stable-audio-open-1.0 oobleck decoder (model_config.json)."""

    latent_dim: int = 64
    io_channels: int = 2
    channels: int = 128
    c_mults: tuple[int, ...] = (1, 2, 4, 8, 16)
    strides: tuple[int, ...] = (2, 4, 4, 8, 8)
    use_snake: bool = True
    final_tanh: bool = False
    scaling_factor: float = 1.0

    # prod(strides); the audio VAE is 1-D so the video ratios do not apply.
    downsampling_ratio: int = 2048
    spatial_compression_ratio: int = 1
    temporal_compression_ratio: int = 1

    # Latents run at the video latent rate: decode emits 25 * 2048 = 51200 Hz,
    # resampled to 44100 for the container.
    latent_fps: int = 25
    native_sample_rate: int = 51200
    output_sample_rate: int = 44100


@dataclass
class Magi2AudioVAEConfig(VAEConfig):
    arch_config: Magi2AudioVAEArchConfig = field(
        default_factory=Magi2AudioVAEArchConfig
    )
    load_encoder: bool = False
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_decode: bool = False
