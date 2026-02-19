# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import WanTeaCacheParams
from sglang.multimodal_gen.configs.sample.magcache import WanMagCacheParams


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

    magcache_params: WanMagCacheParams = field(
        default_factory=lambda: WanMagCacheParams(
            threshold=0.12,
            max_skip_steps=4,
            retention_ratio=0.2,
            use_ret_steps=True,
            mag_ratios=torch.tensor([1.0000, 1.0000, 1.0211, 1.0229, 0.9903, 0.9907, 0.9877, 0.9883, 0.9903,
                      0.9900, 0.9878, 0.9879, 0.9911, 0.9909, 0.9937, 0.9935, 0.9892, 0.9890,
                      0.9921, 0.9920, 0.9885, 0.9885, 0.9917, 0.9916, 0.9910, 0.9910, 0.9892,
                      0.9893, 0.9912, 0.9912, 0.9877, 0.9877, 0.9913, 0.9913, 0.9885, 0.9886,
                      0.9888, 0.9888, 0.9889, 0.9889, 0.9885, 0.9884, 0.9870, 0.9870, 0.9878,
                      0.9878, 0.9870, 0.9869, 0.9880, 0.9880, 0.9857, 0.9857, 0.9852, 0.9852,
                      0.9846, 0.9846, 0.9838, 0.9838, 0.9825, 0.9825, 0.9836, 0.9835, 0.9794,
                      0.9794, 0.9795, 0.9795, 0.9783, 0.9783, 0.9767, 0.9766, 0.9763, 0.9762,
                      0.9729, 0.9728, 0.9744, 0.9743, 0.9676, 0.9675, 0.9671, 0.9670, 0.9642,
                      0.9640, 0.9599, 0.9598, 0.9570, 0.9568, 0.9535, 0.9533, 0.9462, 0.9460,
                      0.9405, 0.9403, 0.9356, 0.9355, 0.9272, 0.9269, 0.9342, 0.9337, 0.9733,
                      0.9719]),
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

    magcache_params: WanMagCacheParams = field(
        default_factory=lambda: WanMagCacheParams(
            threshold=0.06,
            max_skip_steps=3,
            use_ret_steps=True,
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

    magcache_params: WanMagCacheParams = field(
        default_factory=lambda: WanMagCacheParams(
            threshold=0.06,
            max_skip_steps=3,
            use_ret_steps=True,
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
    # NOTE(will): default boundary timestep is tracked by PipelineConfig, but
    # can be overridden during sampling

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
    # NOTE(will): default boundary timestep is tracked by PipelineConfig, but
    # can be overridden during sampling

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
