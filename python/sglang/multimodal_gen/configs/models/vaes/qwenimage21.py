# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class QwenImage21VAEArchConfig(VAEArchConfig):
    base_dim: int = 96
    decoder_base_dim: int = 144
    z_dim: int = 64
    dim_mult: tuple = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    attn_scales: tuple = ()
    temperal_downsample: tuple = (False, True, True, True)
    dropout: float = 0.0
    latents_mean: tuple = (
        0.5126,
        0.7721,
        -0.0631,
        1.3506,
        -0.7855,
        -2.1025,
        -0.3458,
        1.3722,
        1.8873,
        -1.7177,
        -0.651,
        0.2732,
        0.7562,
        -0.6163,
        -1.0277,
        3.8363,
        2.021,
        0.0472,
        0.932,
        2.0087,
        2.4954,
        -0.1391,
        -1.4249,
        1.8464,
        -0.5236,
        1.2826,
        3.7046,
        -1.3035,
        2.7286,
        -1.4518,
        -1.9036,
        -1.9955,
        -0.0342,
        -1.0265,
        -0.7636,
        3.0555,
        0.0746,
        -3.0751,
        -0.1076,
        1.7376,
        -1.0914,
        -1.9435,
        -0.2784,
        -1.368,
        0.4809,
        -0.4433,
        0.3764,
        0.5729,
        -2.0595,
        1.096,
        -1.326,
        -2.0211,
        -5.0179,
        0.5275,
        4.0162,
        1.8505,
        0.3026,
        1.9373,
        1.4937,
        0.2632,
        0.5547,
        -1.7121,
        -0.1562,
        0.0304,
    )
    latents_std: tuple = (
        3.2001,
        3.2936,
        3.4321,
        3.0091,
        3.1061,
        4.0379,
        4.0705,
        3.791,
        3.0785,
        3.65,
        3.9308,
        3.0904,
        2.8778,
        3.7675,
        3.732,
        5.0756,
        3.2864,
        4.0397,
        3.1317,
        4.0443,
        2.9249,
        3.9454,
        3.0988,
        4.2489,
        3.4896,
        3.8513,
        3.9323,
        3.4719,
        3.7498,
        4.283,
        3.5694,
        4.2467,
        3.9037,
        3.2947,
        5.077,
        3.5075,
        3.27,
        3.4767,
        2.8063,
        5.1125,
        3.5327,
        4.7833,
        3.1286,
        4.1819,
        3.8527,
        3.8312,
        3.5605,
        4.3875,
        3.9624,
        4.0168,
        3.5643,
        4.055,
        5.5614,
        4.2963,
        4.408,
        3.4959,
        3.8747,
        3.7608,
        3.5735,
        3.149,
        3.7662,
        3.6746,
        3.4563,
        3.8161,
    )
    is_residual: bool = True
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int | None = None
    scale_factor_temporal: int = 8
    scale_factor_spatial: int = 8
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 1
    vae_scale_factor: int = 16


@dataclass
class QwenImage21VAEConfig(VAEConfig):
    arch_config: QwenImage21VAEArchConfig = field(
        default_factory=QwenImage21VAEArchConfig
    )
    parallel_decode_mode: str = "tiled"
    use_temporal_tiling: bool = False
