# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

DEFAULT_NEGATIVE_PROMPT = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'


@dataclass
class LingBotVideoMoESamplingParams(SamplingParams):
    # prompt must be a structured-JSON caption; raw free-text is out-of-distribution.
    num_frames: int = 81
    height: int = 480
    width: int = 480
    fps: int = 16
    num_inference_steps: int = 40
    guidance_scale: float = 6.0
    flow_shift: float = 3.0
    negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT
    seed: int = 0
