# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

REFINER_PIPELINE_NAME = "LingBotVideoRefinerPipeline"

DEFAULT_NEGATIVE_PROMPT = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'

DEFAULT_NEGATIVE_PROMPT_IMAGE = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "pillarboxed", "side bars", "portrait image in landscape frame"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"]}}'


@dataclass
class LingBotVideoMoESamplingParams(SamplingParams):
    """T2V defaults; also serves I2V when ``image_path`` is set and T2I when ``num_frames`` is 1.

    ``prompt`` must be a structured-JSON caption; raw free text is out of distribution.
    """

    num_frames: int = 81
    height: int = 480
    width: int = 480
    fps: int = 16
    num_inference_steps: int = 40
    guidance_scale: float = 6.0
    flow_shift: float = 3.0
    negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT
    seed: int = 0

    def _explicitly_set(self, field: str) -> bool:
        explicit_fields = getattr(self, "_explicit_fields", None)
        return explicit_fields is None or field in explicit_fields

    def _set_output_file_name(self) -> None:
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()

    def _adjust(self, server_args) -> None:
        if self.num_frames == 1:
            if server_args.pipeline_class_name == REFINER_PIPELINE_NAME:
                raise ValueError(
                    "The refiner is trained on video and does not support "
                    "single-frame requests. Serve the base pipeline for text to image."
                )
            # Aligning latent frames to the GPU count would grow a still into a clip.
            self.adjust_frames = False
            if not self._explicitly_set("negative_prompt"):
                self.negative_prompt = DEFAULT_NEGATIVE_PROMPT_IMAGE
        super()._adjust(server_args)
