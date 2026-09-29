# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


@dataclass
class Flux3ActionSamplingParams(ActionSamplingParams):
    """Sampling parameters for FLUX 3 Action policies.

    ``num_inference_steps``, ``guidance_scale``, ``guidance_scale_action`` and
    ``seed`` default to the recipe of the served policy package (its
    ``inference_seed`` for the noise) when left unset.
    """

    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    guidance_scale_action: float | None = None
    seed: int | list[int] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    action_horizon: int | None = None
    action_dim: int | None = None
    output_format: str = "list"
    return_timing: bool = True
    # False bypasses the per-caption text context cache for this request.
    enable_prefix_cache: bool = True
    # Camera frames keyed by camera name (``wrist``, ``left``, ...), or a
    # prebuilt ``composite``; HWC uint8 arrays, PIL images or tensors.
    images: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    state: Any = field(default=None, metadata={"batch_sig_exclude": True})
    observation: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        observation = dict(self.observation or {})
        if self.images is not None:
            observation["images"] = self.images
        if self.state is not None:
            observation["state"] = self.state
        if self.prompt is not None:
            observation["prompt"] = self.prompt
        extra["vla"] = {
            "observation": observation,
            "options": {
                "output_format": self.output_format,
                "return_timing": self.return_timing,
                "guidance_scale": self.guidance_scale,
                "guidance_scale_action": self.guidance_scale_action,
                "enable_prefix_cache": self.enable_prefix_cache,
            },
        }
        return extra

    def _adjust(self, server_args):
        super()._adjust(server_args)
        config = server_args.pipeline_config
        if self.num_inference_steps is None:
            self.num_inference_steps = config.default_num_inference_steps
        if self.seed is None:
            self.seed = config.inference_seed

    def _validate(self):
        steps, seed = self.num_inference_steps, self.seed
        # None selects the package defaults; the base checks need ints.
        self.num_inference_steps = 1 if steps is None else steps
        self.seed = 0 if seed is None else seed
        try:
            super()._validate()
        finally:
            self.num_inference_steps, self.seed = steps, seed
        if isinstance(seed, list) and len(seed) != 1:
            raise ValueError("FLUX 3 Action takes one seed per request")
        if self.num_outputs_per_prompt != 1:
            raise ValueError("FLUX 3 Action returns one action chunk per request")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.output_format not in ("list", "numpy"):
            raise ValueError("output_format must be 'list' or 'numpy'")
        for name, value in (
            ("guidance_scale", self.guidance_scale),
            ("guidance_scale_action", self.guidance_scale_action),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def _set_output_file_name(self):
        if self.output_file_name is None:
            self.output_file_name = "flux3_action"
        super()._set_output_file_name()
