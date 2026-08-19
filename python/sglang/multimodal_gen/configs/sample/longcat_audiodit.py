from dataclasses import dataclass
from typing import ClassVar, Optional

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class LongCatAudioDiTSamplingParams(SamplingParams):
    """Sampling parameters for LongCat-AudioDiT TTS / voice-cloning inference.

    The model generates audio via Conditional Flow Matching (ODE) — there is no
    traditional image-space height/width.  Audio-specific fields are added here.
    """

    # Mark output as audio so the framework uses the correct extension and
    # post-processing path.
    data_type: DataType = DataType.AUDIO

    # ── audio-specific ────────────────────────────────────────────────────────
    # prompt_audio_path, prompt_text and guidance_method are inherited from SamplingParams
    duration_seconds: Optional[float] = None

    # ── denoising ─────────────────────────────────────────────────────────────
    num_inference_steps: int = 16
    guidance_scale: float = 4.0

    _default_height: ClassVar[Optional[int]] = None
    _default_width: ClassVar[Optional[int]] = None
