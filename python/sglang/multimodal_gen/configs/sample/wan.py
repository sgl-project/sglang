# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _wan_1_3b_coefficients(p: TeaCacheParams) -> list[float]:
    if p.use_ret_steps:
        # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L883
        return [
            -5.21862437e04,
            9.23041404e03,
            -5.28275948e02,
            1.36987616e01,
            -4.99875664e-02,
        ]
    # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L890
    return [
        2.39676752e03,
        -1.31110545e03,
        2.01331979e02,
        -8.29855975e00,
        1.37887774e-01,
    ]


def _wan_14b_coefficients(p: TeaCacheParams) -> list[float]:
    if p.use_ret_steps:
        # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L885
        return [
            -3.03318725e05,
            4.90537029e04,
            -2.65530556e03,
            5.87365115e01,
            -3.15583525e-01,
        ]
    # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L892
    return [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]


@dataclass
class WanT2V_1_3B_SamplingParams(SamplingParams):
    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 3.0
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_inference_steps: int = 50

    # Wan T2V 1.3B supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (832, 480),  # 16:9
            (480, 832),  # 9:16
        ]
    )

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.08,
            use_ret_steps=True,
            coefficients_callback=_wan_1_3b_coefficients,
            start_skipping=5,
            end_skipping=1.0,
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
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
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

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.20,
            use_ret_steps=False,
            coefficients_callback=_wan_14b_coefficients,
            start_skipping=1,
            end_skipping=-1,
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

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.26,
            use_ret_steps=True,
            coefficients_callback=_wan_14b_coefficients,
            start_skipping=5,
            end_skipping=1.0,
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

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.3,
            use_ret_steps=True,
            coefficients_callback=_wan_14b_coefficients,
            start_skipping=5,
            end_skipping=1.0,
        )
    )


@dataclass
class FastWanT2V480PConfig(WanT2V_1_3B_SamplingParams):
    # DMD parameters
    # dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 757, 522])
    num_inference_steps: int = 3
    num_frames: int = 61
    height: int = 480
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

    # TODO(Wan2.2): TeaCache coefficients need to be calibrated for Wan2.2 by
    # profiling L1 distances across timesteps. Until then, teacache_params is None
    # and enable_teacache will be accepted but silently no-op.
    # Consider using Cache-DiT (SGLANG_CACHE_DIT_ENABLED=1) as an alternative.


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
class Wan_Animate_2_14B_SamplingParam(Wan2_2_Base_SamplingParams):
    """Sampling defaults for Wan-Animate-2 14B."""

    # Single expert: no guidance_scale_2 / boundary switching.
    guidance_scale: float = 3.0
    num_inference_steps: int = 40  # official README demo default

    # The reference video is resampled to this rate and the output MP4 is written at it.
    fps: int = 16
    # Carry the reference video's audio track on the output, as the official pipeline does:
    # the track is decoded in-process and the shared MP4 writer encodes it. False gives a
    # silent video. A track that cannot be decoded is logged and the output stays silent.
    enable_audio: bool = True

    # Per-clip denoising length, 4k+1 (VAE stride 4); larger = better temporal coherence
    # but more VRAM. Output length follows the reference video, not this.
    clip_len: int = 37
    # Text for the reference-video branch (prompt_ref_embeds). The official demo conditions it on a
    # fixed description of the reference video, not on the character caption.
    prompt_ref: str = "人物动作的参考视频"

    # The reference image comes via the base image_path and the reference video via the
    # base video_path; neither needs a model-specific field.

    # (width, height) pairs; upstream Wan-Animate default is 640x800.
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (640, 800),  # upstream Wan-Animate default
            (720, 1280),  # 9:16
        ]
    )

    def __post_init__(self) -> None:
        # The in-context denoising loop runs every block at every step; the shared DiT's
        # cache heuristics are never consulted.
        if self.enable_teacache or self.enable_spectrum:
            raise ValueError(
                "Wan-Animate-2 does not support enable_teacache or enable_spectrum."
            )
        requested_clip_len = self.clip_len
        self.clip_len = _round_clip_len(self.clip_len)
        if self.clip_len < MIN_CLIP_LEN:
            raise ValueError(
                f"clip_len={requested_clip_len!r} is not supported: a clip is 4k+1 frames "
                f"with k >= 1, so clip_len must be an int >= {MIN_CLIP_LEN}"
            )
        if self.clip_len != requested_clip_len:
            logger.info(
                "clip_len=%s is not 4k+1; using clip_len=%s",
                requested_clip_len,
                self.clip_len,
            )
        # The output length follows the reference video and the stages read clip_len, so
        # `num_frames` is not a length control here; the multi-GPU frame alignment must
        # leave it alone, and it must stay writable so the warmup probe can shrink.
        self.adjust_frames = False
        super().__post_init__()

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        # Settable over the /v1/videos API: the per-clip length, the reference-branch
        # text (the official gradio exposes prompt_ref as well) and the audio track. The
        # reference video is the base video_path field, so it needs no entry here.
        return frozenset({"clip_len", "prompt_ref", "enable_audio"})

    def prepare_synthetic_warmup_request_for_queue(
        self, req: Any, server_args: Any
    ) -> None:
        """Give the synthetic warmup request the driving video the generic builder does not
        supply, sized so the request denoises exactly one clip."""
        del server_args
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.preprocess import (
            single_clip_reference_video_len,
            write_reference_video,
        )

        image_path = req.image_path
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else None
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(
                "Wan-Animate-2 synthetic warmup requires the warmup reference image"
            )

        # The builder sizes warmup by num_frames (the class default or its bounded cap), but
        # the stage reads clip_len and the compiled attention is specialized per clip
        # geometry, so warm up at the production clip length unless more frames were asked.
        clip_len = _round_clip_len(max(int(req.num_frames), self.clip_len))
        num_video_frames = single_clip_reference_video_len(clip_len)
        fps = int(req.fps)

        # Same directory and lifetime as the warmup image; the name carries the frame count
        # because every warmup request is built before the first one runs.
        video_path = os.path.join(
            os.path.dirname(image_path),
            f"warmup_driving_video_{num_video_frames}f_{fps}fps.mp4",
        )
        write_reference_video(
            video_path,
            _synthetic_driving_video_frames(num_video_frames),
            fps=fps,
        )

        req.video_path = video_path
        req.clip_len = clip_len
        req.num_frames = clip_len
        # The synthetic driving video has no audio track.
        req.enable_audio = False

    def _adjust_visual_fields(self, server_args: Any, pipeline_config: Any) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        if self.num_frames != type(self).num_frames:
            logger.info(
                "Wan-Animate-2 ignores num_frames=%s (the /v1/videos `seconds`): the output "
                "length follows the reference video; clip_len=%s is the per-clip chunk",
                self.num_frames,
                self.clip_len,
            )


# Smallest clip: one conditioning frame plus one VAE temporal stride (4k+1, k >= 1).
MIN_CLIP_LEN = 5


def _round_clip_len(clip_len: int) -> int:
    """Clips are 4k+1 pixel frames (Wan VAE temporal stride 4). Other values map to the
    4k+1 with the same k: 4k rounds up by one frame, 4k+2 and 4k+3 round down."""
    if clip_len % 4 != 1:
        clip_len = clip_len // 4 * 4 + 1
    return clip_len


# Pixel size of the synthetic warmup driving video; the stage rescales every frame to the
# request area anyway, so this only has to be a size the shared video writer takes as is
# and cheap to encode.
_WARMUP_DRIVING_VIDEO_SIZE = 64


def _synthetic_driving_video_frames(num_frames: int) -> np.ndarray:
    """``[num_frames, 64, 64, 3]`` uint8 frames: a gradient that drifts one row per frame,
    so consecutive frames differ like real motion."""
    size = _WARMUP_DRIVING_VIDEO_SIZE
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    rows = np.broadcast_to(ramp[:, None], (size, size))
    cols = np.broadcast_to(ramp[None, :], (size, size))
    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    for index in range(num_frames):
        frames[index, ..., 0] = np.roll(rows, index, axis=0)
        frames[index, ..., 1] = cols
        frames[index, ..., 2] = 127
    return frames


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
