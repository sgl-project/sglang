# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

REFINER_PIPELINE_NAME = "LingBotVideoRefinerPipeline"
# The reference base pass for the refiner.
REFINER_BASE_HEIGHT, REFINER_BASE_WIDTH = 480, 832
# Arbitrary; admits 480x832, which is 1.8% off the 1088x1920 refiner aspect.
REFINER_ASPECT_TOLERANCE = 0.05

DEFAULT_NEGATIVE_PROMPT = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'

DEFAULT_NEGATIVE_PROMPT_IMAGE = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "pillarboxed", "side bars", "portrait image in landscape frame"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"]}}'


@dataclass
class LingBotVideoMoESamplingParams(SamplingParams):
    """T2V defaults; also serves I2V when ``image_path`` is set and T2I when ``num_frames`` is 1.

    ``prompt`` must be a structured-JSON caption; raw free text is out of distribution.
    """

    _explicit_fields: ClassVar[set[str] | None] = None

    num_frames: int = 81
    height: int = 480
    width: int = 480
    fps: int = 16
    num_inference_steps: int = 40
    guidance_scale: float = 6.0
    flow_shift: float = 3.0
    negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT
    seed: int = 0

    def expand_video_request_outputs_for_queue(self, req: Any) -> list[Any]:
        # The image and text conditioning stages build one sample,
        # so each output becomes its own request, as the offline path does.
        from sglang.multimodal_gen.runtime.entrypoints.utils import (
            expand_request_outputs,
        )

        return expand_request_outputs(req)

    def _explicitly_set(self, field: str) -> bool:
        return self._explicit_fields is None or field in self._explicit_fields

    def _set_output_file_name(self) -> None:
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()

    def prepare_synthetic_warmup_request_for_queue(
        self, req: Any, server_args: Any
    ) -> None:
        # Warmup skips _adjust, so it would otherwise run the square default.
        if (
            server_args.pipeline_class_name == REFINER_PIPELINE_NAME
            and server_args.warmup_resolutions is None
            and req.num_frames != 1
        ):
            req.height, req.width = REFINER_BASE_HEIGHT, REFINER_BASE_WIDTH

    def _adjust_refiner_geometry(self, pipeline_config) -> None:
        if not (self._explicitly_set("height") or self._explicitly_set("width")):
            self.height, self.width = REFINER_BASE_HEIGHT, REFINER_BASE_WIDTH
        base_aspect = self.width / self.height
        refiner_aspect = pipeline_config.refiner_width / pipeline_config.refiner_height
        # The refiner resizes the base clip to its own geometry without cropping.
        if abs(base_aspect / refiner_aspect - 1.0) > REFINER_ASPECT_TOLERANCE:
            raise ValueError(
                f"The refiner resizes the {self.height}x{self.width} base clip to "
                f"{pipeline_config.refiner_height}x{pipeline_config.refiner_width} "
                "without cropping; request a base geometry of the same aspect ratio, "
                f"such as {REFINER_BASE_HEIGHT}x{REFINER_BASE_WIDTH}."
            )

    def _adjust(self, server_args) -> None:
        if (
            server_args.pipeline_class_name == REFINER_PIPELINE_NAME
            and self.num_frames != 1
        ):
            self._adjust_refiner_geometry(server_args.pipeline_config)
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
        # The DiT shards the joint sequence itself, so keep the flag on by default.
        enable_sequence_shard = self.enable_sequence_shard
        if enable_sequence_shard is None or enable_sequence_shard:
            self.adjust_frames = False
        super()._adjust(server_args)
        if enable_sequence_shard is None or enable_sequence_shard:
            self.enable_sequence_shard = True
            self.adjust_frames = False
