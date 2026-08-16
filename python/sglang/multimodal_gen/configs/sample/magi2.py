# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

# 1088 rather than 1080: the VAE stride forces multiples of 16.
MAGI2_RESOLUTIONS = [(1920, 1088), (896, 512)]

MAGI2_CLIP_SECONDS = 10.0

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

    negative_prompt: str = MAGI2_NEGATIVE_PROMPT

    num_inference_steps: int = 100
    refiner_num_inference_steps: int = 5

    preview_width: int = 896
    preview_height: int = 512

    # (63 - 1) * 4 + 1, from a causal decoder over 63 refiner latent frames.
    num_frames: int = 249
    fps: int = MAGI2_REFINER_FPS

    guidance_scale: float = 5.0
    audio_guidance_scale: float = 7.0
    refiner_guidance_scale: float = 2.0
    refiner_audio_guidance_scale: float = 5.0

    use_skimmed_guidance: bool = False
    skimmed_guidance_scale: float = 3.0

    generate_audio: bool = True

    def __post_init__(self) -> None:
        if self.supported_resolutions is None:
            self.supported_resolutions = list(MAGI2_RESOLUTIONS)
        super().__post_init__()

    def _validate(self) -> None:
        super()._validate()

        if (self.width, self.height) not in MAGI2_RESOLUTIONS:
            raise ValueError(
                f"MAGI-2 supports {MAGI2_RESOLUTIONS}, got "
                f"{self.width}x{self.height}"
            )

        # Tolerance: causal decode yields 4k+1 frames, and num_frames is rounded up
        # so the latent count divides the sequence-parallel degree.
        nominal_frames = MAGI2_CLIP_SECONDS * self.fps
        if abs(self.num_frames - nominal_frames) > 0.05 * nominal_frames:
            raise ValueError(
                f"MAGI-2 only generates {MAGI2_CLIP_SECONDS:g}s clips, i.e. "
                f"about {round(nominal_frames)} frames at fps={self.fps}; got "
                f"{self.num_frames}"
            )

        if self.num_inference_steps is not None and self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if self.refiner_num_inference_steps < 1:
            raise ValueError("refiner_num_inference_steps must be >= 1")
