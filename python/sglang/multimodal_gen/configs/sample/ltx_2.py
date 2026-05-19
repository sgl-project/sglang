import dataclasses
from dataclasses import field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclasses.dataclass
class LTX2SamplingParams(SamplingParams):
    """Sampling parameters for LTX-2."""

    # Match the reference defaults used by ltx-pipelines (one-stage).
    # See: LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
    seed: int = 10
    generator_device: str = "cpu"

    # Video parameters
    height: int = 512
    width: int = 768
    num_frames: int = 121
    fps: int = 24

    # Audio specific
    generate_audio: bool = True

    # Denoising parameters
    guidance_scale: float = 4.0
    num_inference_steps: int = 40

    # Match ltx-pipelines default negative prompt (covers video + audio artifacts).
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )


@dataclasses.dataclass
class LTX23SamplingParams(LTX2SamplingParams):
    """Sampling parameters matching official LTX-2.3 one-stage defaults."""

    seed: int = 42
    generator_device: str = "cuda"
    guidance_scale: float = 3.0
    num_inference_steps: int = 30

    video_cfg_scale: float = 3.0
    video_stg_scale: float = 1.0
    video_rescale_scale: float = 0.7
    video_modality_scale: float = 3.0
    video_skip_step: int = 0
    video_stg_blocks: list[int] = field(default_factory=lambda: [28])

    audio_cfg_scale: float = 7.0
    audio_stg_scale: float = 1.0
    audio_rescale_scale: float = 0.7
    audio_modality_scale: float = 3.0
    audio_skip_step: int = 0
    audio_stg_blocks: list[int] = field(default_factory=lambda: [28])
    skip_v2a_cross_attn_for_video_gt: bool = False

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        extra["ltx2_stage1_guider_params"] = {
            "video_cfg_scale": self.video_cfg_scale,
            "video_stg_scale": self.video_stg_scale,
            "video_rescale_scale": self.video_rescale_scale,
            "video_modality_scale": self.video_modality_scale,
            "video_skip_step": self.video_skip_step,
            "video_stg_blocks": self.video_stg_blocks,
            "audio_cfg_scale": self.audio_cfg_scale,
            "audio_stg_scale": self.audio_stg_scale,
            "audio_rescale_scale": self.audio_rescale_scale,
            "audio_modality_scale": self.audio_modality_scale,
            "audio_skip_step": self.audio_skip_step,
            "audio_stg_blocks": self.audio_stg_blocks,
        }
        if self.skip_v2a_cross_attn_for_video_gt:
            extra["ltx2_skip_v2a_cross_attn_for_video_gt"] = True
        return extra


@dataclasses.dataclass
class LTX23HQSamplingParams(LTX23SamplingParams):
    """Sampling parameters matching official LTX-2.3 HQ two-stage defaults."""

    height: int = 1088
    width: int = 1920
    num_inference_steps: int = 15
    distilled_lora_strength_stage_1: float = 0.25
    distilled_lora_strength_stage_2: float = 0.5

    video_cfg_scale: float = 3.0
    video_stg_scale: float = 0.0
    video_rescale_scale: float = 0.45
    video_modality_scale: float = 3.0
    video_skip_step: int = 0
    video_stg_blocks: list[int] = field(default_factory=list)

    audio_cfg_scale: float = 7.0
    audio_stg_scale: float = 0.0
    audio_rescale_scale: float = 1.0
    audio_modality_scale: float = 3.0
    audio_skip_step: int = 0
    audio_stg_blocks: list[int] = field(default_factory=list)

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        extra["ltx2_distilled_lora_strength_stage_1"] = float(
            self.distilled_lora_strength_stage_1
        )
        extra["ltx2_distilled_lora_strength_stage_2"] = float(
            self.distilled_lora_strength_stage_2
        )
        return extra
