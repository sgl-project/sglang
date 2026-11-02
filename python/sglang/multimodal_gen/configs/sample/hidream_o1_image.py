# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for the HiDream-O1-Image family.

One pipeline serves all three checkpoints; the recipe is resolved per served
model path in _adjust. The Dev checkpoints are guidance-distilled and run the
official fixed 28-step flash schedule with per-variant noise scales; the full
checkpoint runs the reference 50-step CFG 5.0 UniPC recipe. Noise scales for
the Dev variants are a deliberate deployment deviation from the official
README checkpoint table.
"""

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _resolve_variant(model_path: str | None) -> str:
    """Map a served model path to "dev-2604", "dev", or "full"."""
    lowered = (model_path or "").lower()
    if "dev-2604" in lowered:
        return "dev-2604"
    if "dev" in lowered:
        return "dev"
    return "full"


_VARIANT_DEFAULTS = {
    "dev-2604": {
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "noise_scale_start": 8.0,
        "noise_scale_end": 8.0,
        "noise_clip_std": 2.5,
        "num_inference_steps": 28,
    },
    "dev": {
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "noise_scale_start": 8.0,
        "noise_scale_end": 8.0,
        "noise_clip_std": 2.5,
        "num_inference_steps": 28,
    },
    "full": {
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler_name": "default",
        "noise_scale_start": 8.0,
        "noise_scale_end": 8.0,
        "noise_clip_std": 2.5,
        "num_inference_steps": 50,
    },
}


# (width, height) pairs the model was trained on. The pixel patches carry no
# scale invariance, so anything else degrades quality even when it is a legal
# multiple of the 32-pixel patch size.
HIDREAM_O1_SUPPORTED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]


@dataclass
class HiDreamO1ImageSamplingParams(SamplingParams):
    # The reference implementation conditions the unconditional branch on a
    # single space rather than the empty string.
    negative_prompt: str = " "

    num_frames: int = 1
    height: int = 2048
    width: int = 2048
    seed: int = 42

    # Sentinel resolved in _adjust from the variant table (50 FULL / fixed
    # 28 DEV); an explicit request value always wins.
    num_inference_steps: Optional[int] = None
    guidance_scale: float = None
    shift: Optional[float] = None
    scheduler_name: Optional[str] = None

    # linearly interpolated across steps (flash scheduler only)
    noise_scale_start: Optional[float] = None
    noise_scale_end: Optional[float] = None
    noise_clip_std: Optional[float] = None

    # Reference images for TI2I editing (Dev checkpoints)
    ref_images: Optional[List[str]] = None

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: list(HIDREAM_O1_SUPPORTED_RESOLUTIONS)
    )

    _default_height: ClassVar[int] = 2048
    _default_width: ClassVar[int] = 2048

    _RECIPE_FIELDS = (
        "num_inference_steps",
        "guidance_scale",
        "shift",
        "scheduler_name",
        "noise_scale_start",
        "noise_scale_end",
        "noise_clip_std",
    )

    def _adjust(self, server_args):
        explicit = getattr(self, "_explicit_fields", None)
        if explicit is None:
            unset = {name: getattr(self, name) is None for name in self._RECIPE_FIELDS}
        else:
            unset = {name: name not in explicit for name in self._RECIPE_FIELDS}
        super()._adjust(server_args)
        variant = _resolve_variant(server_args.model_path)
        defaults = _VARIANT_DEFAULTS[variant]
        for name, was_unset in unset.items():
            if was_unset:
                setattr(self, name, defaults[name])
        if (
            variant != "full"
            and self.num_inference_steps != defaults["num_inference_steps"]
        ):
            logger.warning(
                "HiDream DEV generation always uses num_inference_steps=%d "
                "(the official fixed DEFAULT_TIMESTEPS schedule); ignoring "
                "the requested %d.",
                defaults["num_inference_steps"],
                self.num_inference_steps,
            )

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Synthetic warmup starts from raw dataclass defaults rather than the
        # adjusted request parameters, so resolve the variant recipe here too.
        self._adjust(server_args)
