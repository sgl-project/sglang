# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.praxis import RIGS, PraxisRig
from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


@dataclass
class PraxisSamplingParams(ActionSamplingParams):
    """Sampling parameters for Praxis-1 policies.

    ``num_inference_steps``, ``guidance_scale`` (text guidance on the video),
    ``guidance_scale_action`` and ``seed`` default to the recipe of the served
    checkpoint when left unset.
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
    images: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    state: Any = field(default=None, metadata={"batch_sig_exclude": True})
    observation: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )

    _rig: PraxisRig = field(default=RIGS["single_arm"], init=False, repr=False)

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
            },
        }
        return extra

    def _adjust(self, server_args):
        super()._adjust(server_args)
        config = server_args.pipeline_config
        # The required views and state depend on the checkpoint, which only the server args name.
        self._rig = RIGS[config.embodiment]
        if self.observation is not None:
            self._validate_observation()
        if self.num_inference_steps is None:
            self.num_inference_steps = config.default_num_inference_steps
        if self.seed is None:
            self.seed = config.inference_seed

    def _validate(self):
        steps, seed = self.num_inference_steps, self.seed
        # None selects the checkpoint defaults; the base checks need ints.
        self.num_inference_steps = 1 if steps is None else steps
        self.seed = 0 if seed is None else seed
        try:
            super()._validate()
        finally:
            self.num_inference_steps, self.seed = steps, seed
        if isinstance(seed, list) and len(seed) != 1:
            raise ValueError("Praxis takes one seed per request")
        if self.num_outputs_per_prompt != 1:
            raise ValueError("Praxis returns one action chunk per request")
        if (self.observation or {}).get(
            "return_video"
        ) and self.output_format != "numpy":
            raise ValueError("return_video needs output_format 'numpy' (msgpack)")
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

    def _validate_observation(self):
        images = self.images or {}
        missing = [
            f"{view} image"
            for view in self._rig.image_keys
            if images.get(view) is None
            and self.observation.get(f"image.{view}") is None
        ]
        if self.state is None and self._rig.state_keys[0] not in self.observation:
            missing.append(f"state ({self._rig.state_name})")
        if missing:
            raise ValueError(f"Praxis request is missing: {', '.join(missing)}")

    def _set_output_file_name(self):
        if self.output_file_name is None:
            self.output_file_name = "praxis"
        super()._set_output_file_name()
