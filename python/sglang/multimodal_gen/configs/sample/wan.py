# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import WanTeaCacheParams

# Magnitude ratio arrays from the reference implementation:
# https://github.com/Zehong-Ma/MagCache/blob/df81cb181776c2c61477c08e1d21f87fda1cd938/MagCache4Wan2.1/magcache_generate.py
T2V_13B_MAG_RATIOS = [
    1.0,
    1.0,
    1.0124,
    1.02213,
    1.00166,
    1.0041,
    0.99791,
    1.00061,
    0.99682,
    0.99762,
    0.99634,
    0.99685,
    0.99567,
    0.99586,
    0.99416,
    0.99422,
    0.99578,
    0.99575,
    0.9957,
    0.99563,
    0.99511,
    0.99506,
    0.99535,
    0.99531,
    0.99552,
    0.99549,
    0.99541,
    0.99539,
    0.9954,
    0.99536,
    0.99489,
    0.99485,
    0.99518,
    0.99514,
    0.99484,
    0.99478,
    0.99481,
    0.99479,
    0.99415,
    0.99413,
    0.99419,
    0.99416,
    0.99396,
    0.99393,
    0.99388,
    0.99386,
    0.99349,
    0.99349,
    0.99309,
    0.99304,
    0.9927,
    0.9927,
    0.99228,
    0.99226,
    0.99171,
    0.9917,
    0.99137,
    0.99135,
    0.99068,
    0.99063,
    0.99005,
    0.99003,
    0.98944,
    0.98942,
    0.98849,
    0.98849,
    0.98758,
    0.98757,
    0.98644,
    0.98643,
    0.98504,
    0.98503,
    0.9836,
    0.98359,
    0.98202,
    0.98201,
    0.97977,
    0.97978,
    0.97717,
    0.97718,
    0.9741,
    0.97411,
    0.97003,
    0.97002,
    0.96538,
    0.96541,
    0.9593,
    0.95933,
    0.95086,
    0.95089,
    0.94013,
    0.94019,
    0.92402,
    0.92414,
    0.90241,
    0.9026,
    0.86821,
    0.86868,
    0.81838,
    0.81939,
]


@dataclass
class WanT2V_1_3B_SamplingParams(SamplingParams):
    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 3.0
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )
    num_inference_steps: int = 50

    # Wan T2V 1.3B supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (832, 480),  # 16:9
            (480, 832),  # 9:16
        ]
    )

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.08,
            ret_steps_coeffs=[
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            non_ret_steps_coeffs=[
                2.39676752e03,
                -1.31110545e03,
                2.01331979e02,
                -8.29855975e00,
                1.37887774e-01,
            ],
        )
    )

    magcache_params: MagCacheParams = field(
        default_factory=lambda: MagCacheParams(
            threshold=0.12,
            max_skip_steps=4,
            skip_start_step=10,
            skip_end_step=0,
            mag_ratios=T2V_13B_MAG_RATIOS,
        )
    )


@dataclass
class WanT2V_14B_SamplingParams(SamplingParams):
    # Video parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 5.0
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )
    num_inference_steps: int = 50

    # Wan T2V 14B supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),  # 16:9
            (720, 1280),  # 9:16
            (832, 480),  # 16:9
            (480, 832),  # 9:16
        ]
    )

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.20,
            use_ret_steps=False,
            ret_steps_coeffs=[
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            non_ret_steps_coeffs=[
                -5784.54975374,
                5449.50911966,
                -1811.16591783,
                256.27178429,
                -13.02252404,
            ],
        )
    )


@dataclass
class WanI2V_14B_480P_SamplingParam(WanT2V_1_3B_SamplingParams):
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    # num_inference_steps: int = 40

    # Wan I2V 480P supported resolutions (override parent)
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (832, 480),  # 16:9
            (480, 832),  # 9:16
        ]
    )

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.26,
            ret_steps_coeffs=[
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            non_ret_steps_coeffs=[
                -5784.54975374,
                5449.50911966,
                -1811.16591783,
                256.27178429,
                -13.02252404,
            ],
        )
    )


@dataclass
class WanI2V_14B_720P_SamplingParam(WanT2V_14B_SamplingParams):
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    # num_inference_steps: int = 40

    # Wan I2V 720P supported resolutions (override parent)
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),  # 16:9
            (720, 1280),  # 9:16
            (832, 480),  # 16:9
            (480, 832),  # 9:16
        ]
    )

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.3,
            ret_steps_coeffs=[
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            non_ret_steps_coeffs=[
                -5784.54975374,
                5449.50911966,
                -1811.16591783,
                256.27178429,
                -13.02252404,
            ],
        )
    )


@dataclass
class FastWanT2V480PConfig(WanT2V_1_3B_SamplingParams):
    # DMD parameters
    # dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 757, 522])
    num_inference_steps: int = 3
    num_frames: int = 61
    height: int = 448
    width: int = 832
    fps: int = 16


# =============================================
# ============= Wan2.1 Fun Models =============
# =============================================
@dataclass
class Wan2_1_Fun_1_3B_InP_SamplingParams(SamplingParams):
    """Sampling parameters for Wan2.1 Fun 1.3B InP model."""

    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    negative_prompt: str | None = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )
    guidance_scale: float = 6.0
    num_inference_steps: int = 50


# =============================================
# ============= Wan2.2 TI2V Models =============
# =============================================
@dataclass
class Wan2_2_Base_SamplingParams(SamplingParams):
    """Sampling parameters for Wan2.2 TI2V 5B model."""

    negative_prompt: str | None = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )


@dataclass
class Wan2_2_TI2V_5B_SamplingParam(Wan2_2_Base_SamplingParams):
    """Sampling parameters for Wan2.2 TI2V 5B model."""

    height: int = 704
    width: int = 1280
    num_frames: int = 121
    fps: int = 24
    guidance_scale: float = 5.0
    num_inference_steps: int = 50

    # Wan2.2 TI2V 5B supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 704),  # 16:9-ish
            (704, 1280),  # 9:16-ish
        ]
    )


@dataclass
class Wan2_2_T2V_A14B_SamplingParam(Wan2_2_Base_SamplingParams):
    guidance_scale: float = 4.0  # high_noise
    guidance_scale_2: float = 3.0  # low_noise
    num_inference_steps: int = 40
    fps: int = 16

    num_frames: int = 81

    # Wan2.2 T2V A14B supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),  # 16:9
            (720, 1280),  # 9:16
            (832, 480),  # 16:9
            (480, 832),  # 9:16
        ]
    )


@dataclass
class Wan2_2_I2V_A14B_SamplingParam(Wan2_2_Base_SamplingParams):
    guidance_scale: float = 3.5  # high_noise
    guidance_scale_2: float = 3.5  # low_noise
    num_inference_steps: int = 40
    fps: int = 16

    num_frames: int = 81

    # Wan2.2 I2V A14B supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),  # 16:9
            (720, 1280),  # 9:16
            (832, 480),  # 16:9
            (480, 832),  # 9:16
        ]
    )


@dataclass
class Turbo_Wan2_2_I2V_A14B_SamplingParam(Wan2_2_Base_SamplingParams):
    guidance_scale: float = 3.5  # high_noise
    guidance_scale_2: float = 3.5  # low_noise
    num_inference_steps: int = 4
    fps: int = 16


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWanT2V480PConfig(WanT2V_1_3B_SamplingParams):
    pass
