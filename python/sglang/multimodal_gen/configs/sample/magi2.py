# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
)

# 1088 rather than 1080: the VAE stride forces multiples of 16.
MAGI2_RESOLUTIONS = [(1920, 1088), (896, 512)]

MAGI2_CLIP_SECONDS = 10.0

# (63 - 1) * 4 + 1, from a causal decoder over 63 refiner latent frames.
MAGI2_NUM_FRAMES = 249

MAGI2_PREVIEW_FPS = 12.5
MAGI2_REFINER_FPS = 25

MAGI2_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards"
    ", low quality, worst quality, poor quality, noise, background noise, "
    "hiss, hum, buzz, crackle, static, compression artifacts, MP3 artifacts, "
    "digital clipping, distortion, muffled, muddy, unclear, echo, reverb, "
    "room echo, over-reverberated, hollow sound, distant, washed out, harsh, "
    "shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, "
    "over-compressed, abrupt cut, jarring transition, sudden silence, looping "
    "artifact, music, instrumental, sirens, alarms, crowd noise, unrelated "
    "sound effects, chaotic, disorganized, messy, cheap sound"
    ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, "
    "mechanical, monotone, flat intonation, undynamic, boring, reading from a "
    "script, AI voice, synthetic, text-to-speech, TTS, insincere, fake "
    "emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, "
    "hesitant, unconfident, tired, weak voice, stuttering, stammering, "
    "mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal "
    "fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy "
    "breathing, audible inhales, plosives, p-pops, coughing, clearing throat, "
    "sneezing, speaking too fast, rushed, speaking too slow, dragged out, "
    "unnatural pauses, awkward silence, choppy, disconnected, multiple "
    "speakers, two voices, background talking, out of tune, off-key, autotune "
    "artifacts"
)


@dataclass
class Magi2SamplingParams(SamplingParams):
    """Request parameters for MAGI-2-preview; defaults are the shipped two-stage
    1080p tier (896x512 preview, 1920x1088 refiner)."""

    _default_width: ClassVar[int | None] = 1920
    _default_height: ClassVar[int | None] = 1088
    # Fixed-length clips at two fixed sizes: a halved probe fails _validate.
    supports_warmup_lightening: ClassVar[bool] = False

    negative_prompt: str = MAGI2_NEGATIVE_PROMPT

    num_inference_steps: int = 100
    refiner_num_inference_steps: int = 5

    preview_width: int = 896
    preview_height: int = 512

    num_frames: int = MAGI2_NUM_FRAMES
    fps: int = MAGI2_REFINER_FPS

    guidance_scale: float = 5.0
    audio_guidance_scale: float = 7.0
    refiner_guidance_scale: float = 2.0
    refiner_audio_guidance_scale: float = 5.0

    use_skimmed_guidance: bool = False
    skimmed_guidance_scale: float = 3.0

    generate_audio: bool = True

    def __post_init__(self) -> None:
        self.adjust_frames = False  # clip length is fixed by the model
        if self.supported_resolutions is None:
            self.supported_resolutions = list(MAGI2_RESOLUTIONS)
        super().__post_init__()

    def _adjust(self, server_args: Any) -> None:
        explicit = getattr(self, "_explicit_fields", None) or set()
        if not getattr(
            server_args.pipeline_config, "enable_refiner", True
        ) and not explicit & {"width", "height"}:
            self.width = self.preview_width
            self.height = self.preview_height
        super()._adjust(server_args)

    def _validate(self) -> None:
        super()._validate()

        if (self.width, self.height) not in MAGI2_RESOLUTIONS:
            raise ValueError(
                f"MAGI-2 supports {MAGI2_RESOLUTIONS}, got {self.width}x{self.height}"
            )

        if (self.preview_width, self.preview_height) not in MAGI2_RESOLUTIONS:
            raise ValueError(
                f"preview_width/preview_height must be one of {MAGI2_RESOLUTIONS}; "
                f"got {self.preview_width}x{self.preview_height}"
            )

        if (self.num_frames, self.fps) != (MAGI2_NUM_FRAMES, MAGI2_REFINER_FPS):
            raise ValueError(
                f"MAGI-2 only supports {MAGI2_NUM_FRAMES} frames at {MAGI2_REFINER_FPS} fps; "
                f"got num_frames={self.num_frames}, fps={self.fps}"
            )

        if self.num_inference_steps is not None and self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if self.refiner_num_inference_steps < 1:
            raise ValueError("refiner_num_inference_steps must be >= 1")

        for name in ("attention_backend_override", "skip_softmax_params"):
            if getattr(self, name, None) is not None:
                raise ValueError(
                    f"MAGI-2 does not support per-request {name}; set it to None"
                )

    def _validate_with_pipeline_config(self, pipeline_config) -> None:
        super()._validate_with_pipeline_config(pipeline_config)
        if not getattr(pipeline_config, "enable_refiner", True) and (
            self.width,
            self.height,
        ) != (self.preview_width, self.preview_height):
            raise ValueError(
                f"This server runs MAGI-2 in preview-only mode and can only deliver "
                f"{self.preview_width}x{self.preview_height}; "
                f"got {self.width}x{self.height}"
            )

    def prepare_synthetic_warmup_request_for_queue(
        self, req: Any, server_args: Any
    ) -> None:
        if server_args.warmup_resolutions is not None:
            return
        req.width, req.height = self._default_width, self._default_height
        if server_args.warmup_num_frames is not None:
            return
        req.num_frames = MAGI2_NUM_FRAMES

    @classmethod
    def lower_video_request_kwargs(
        cls,
        request: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # The video API synthesizes num_frames as fps * seconds from its own 24fps
        # 4s defaults, which a fixed-length 10s model rejects. Drop both unless the
        # caller set them, so this type's 249 @ 25 applies.
        kwargs = dict(kwargs)
        if request.fps is None:
            kwargs.pop("fps", None)
        if request.num_frames is None:
            kwargs.pop("num_frames", None)
        return kwargs
