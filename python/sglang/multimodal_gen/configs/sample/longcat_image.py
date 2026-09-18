from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

# Calibrated via tools/teacache_calibrate.py; shared by LongCat-Image T2I and
# Edit (same DiT).
_LONGCAT_IMAGE_COEFFICIENTS = [
    -48.33117963214599,
    65.13385568407884,
    -27.579648889361575,
    5.5492815393017,
    -0.041339016370571656,
]
_LONGCAT_IMAGE_TEACACHE_THRESH = 0.13


def _longcat_teacache_params() -> TeaCacheParams:
    return TeaCacheParams(
        teacache_thresh=_LONGCAT_IMAGE_TEACACHE_THRESH,
        coefficients=list(_LONGCAT_IMAGE_COEFFICIENTS),
    )


@dataclass
class LongCatImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    height: int = 1024
    width: int = 1024
    enable_cfg_renorm: bool = True
    cfg_renorm_min: float = 0.0
    enable_prompt_rewrite: bool = True
    teacache_params: Any = field(default_factory=_longcat_teacache_params)

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "cfg_renorm_min",
                "enable_cfg_renorm",
                "enable_prompt_rewrite",
            }
        )


@dataclass
class LongCatImageEditSamplingParams(SamplingParams):
    """Defaults for LongCat-Image-Edit (mirrors diffusers LongCatImageEditPipeline).

    Output height/width are derived from the condition image (~1MP), so no
    defaults are set here. The reference uses an empty negative prompt and
    plain CFG (no renorm, no prompt rewrite).
    """

    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    negative_prompt: str = ""
    enable_cfg_renorm: bool = False
    enable_prompt_rewrite: bool = False
    teacache_params: Any = field(default_factory=_longcat_teacache_params)


@dataclass
class LongCatImageEditTurboSamplingParams(LongCatImageEditSamplingParams):
    """LongCat-Image-Edit-Turbo: distilled, 8 steps, CFG disabled."""

    num_inference_steps: int = 8
    guidance_scale: float = 1.0
