# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


def _require_positive_duration_seconds(duration_seconds: Any) -> None:
    if duration_seconds is None:
        return
    if isinstance(duration_seconds, bool) or not isinstance(
        duration_seconds, (int, float)
    ):
        raise ValueError(f"duration_seconds must be a number, got {duration_seconds!r}")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(
            "duration_seconds must be a positive finite number, "
            f"got {duration_seconds!r}"
        )


@dataclass
class LongCatAudioDiTSamplingParams(SamplingParams):
    """Sampling parameters for LongCat-AudioDiT TTS / voice-cloning inference.

    The model generates audio via Conditional Flow Matching (ODE) — there is no
    traditional image-space height/width.
    """

    data_type: DataType = DataType.AUDIO

    num_inference_steps: int = 16
    guidance_scale: float = 4.0

    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    guidance_method: str = "cfg"
    duration_seconds: float | None = None

    _default_height: ClassVar[Optional[int]] = None
    _default_width: ClassVar[Optional[int]] = None

    def _validate(self):
        super()._validate()
        if self.guidance_method not in ("cfg", "apg"):
            raise ValueError(
                f"Unknown guidance_method '{self.guidance_method}', "
                "must be 'cfg' or 'apg'"
            )
        _require_positive_duration_seconds(self.duration_seconds)
