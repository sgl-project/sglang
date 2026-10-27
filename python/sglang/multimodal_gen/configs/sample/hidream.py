"""Sampling parameters for the HiDream-O1-Image model family.

One pipeline serves three checkpoints; the recipe is resolved per served
model path in _adjust. Noise scales are a deliberate deployment deviation
from the official README checkpoint table.
"""

from dataclasses import dataclass
from typing import List, Optional

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


def _resolve_variant(model_path: str | None) -> str:
    """Map a served model path to "dev-2604", "dev", or "full"."""
    lowered = (model_path or "").lower()
    if "dev-2604" in lowered:
        return "dev-2604"
    if "dev" in lowered:
        return "dev"
    return "full"


def _is_dev_2604(model_path: str | None) -> bool:
    return _resolve_variant(model_path) == "dev-2604"


def _is_full_model(model_path: str | None) -> bool:
    return _resolve_variant(model_path) == "full"


_VARIANT_DEFAULTS = {
    "dev-2604": {
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "model_type": "dev",
        "torch_dtype": "bfloat16",
        "noise_scale_start": 7.5,
        "noise_scale_end": 7.5,
        "noise_clip_std": 2.5,
    },
    "dev": {
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "model_type": "dev",
        "torch_dtype": "float32",
        "noise_scale_start": 8.0,
        "noise_scale_end": 8.0,
        "noise_clip_std": 8.0,
    },
    "full": {
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler_name": "default",
        "model_type": "full",
        "torch_dtype": "bfloat16",
        "noise_scale_start": 7.5,
        "noise_scale_end": 7.5,
        "noise_clip_std": 2.5,
    },
}

@dataclass
class HiDreamSamplingParams(SamplingParams):
    """Sampling parameters for the HiDream-O1-Image family.

    Recipe fields are None sentinels resolved in _adjust from the served
    model path; an explicit per-request value (non-None) keeps precedence.
    """

    num_inference_steps: Optional[int] = None
    num_frames: int = 1
    height: int = 2048
    width: int = 2048
    guidance_scale: Optional[float] = None
    shift: Optional[float] = None
    seed: int = 42
    negative_prompt: str = " "

    # "flash" (DEV), "flow_match" (DEV editing), "default" (UniPC/FULL)
    scheduler_name: Optional[str] = None

    # linearly interpolated across steps
    noise_scale_start: Optional[float] = None
    noise_scale_end: Optional[float] = None
    noise_clip_std: Optional[float] = None

    # Reference images for I2I/TI2I
    ref_images: Optional[List[str]] = None
    layout_bboxes: Optional[str] = None

    model_type: Optional[str] = None

    # captured before super()._adjust() (base fills guidance_scale=1.0)
    _RECIPE_FIELDS = (
        "guidance_scale",
        "shift",
        "scheduler_name",
        "model_type",
        "noise_scale_start",
        "noise_scale_end",
        "noise_clip_std",
    )

    def _adjust(self, server_args):
        unset = {
            name: getattr(self, name) is None for name in self._RECIPE_FIELDS
        }
        super()._adjust(server_args)
        # serve --num-inference-steps default; explicit request wins.
        if self.num_inference_steps is None:
            self.num_inference_steps = int(
                server_args.pipeline_config.num_inference_steps
            )

        defaults = _VARIANT_DEFAULTS[_resolve_variant(server_args.model_path)]
        for name, was_unset in unset.items():
            if was_unset:
                setattr(self, name, defaults[name])
