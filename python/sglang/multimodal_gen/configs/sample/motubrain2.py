# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for the Motubrain2 action policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


@dataclass
class Motubrain2Observation:
    inputs: dict[str, Any]
    session_id: str | None = None
    seed: int | None = None
    reset: bool = False
    rtc: Any = None
    return_native_outputs: bool = False


@dataclass(frozen=True)
class _ConnectionState:
    """Frontend-only marker that decoded client messages cannot construct."""

    values: dict


def validate_observation_options(observation: dict[str, Any], config: Any):
    if (
        observation.get("noise") is not None
        or observation.get("observation.noise") is not None
    ):
        raise ValueError(
            "Motubrain2 accepts a seed to generate joint video and action noise"
        )
    aliases = config.camera_aliases
    cameras = tuple(config.image_keys)
    order = observation.get("camera_order")
    if order is not None:
        if not isinstance(order, (list, tuple)) or not all(
            isinstance(name, str) for name in order
        ):
            raise ValueError("camera_order must be a list of camera names")
        if tuple(aliases.get(name, name) for name in order) != cameras:
            raise ValueError(f"camera_order must match the served profile: {cameras}")

    masks = observation.get("image_masks")
    if masks is not None:
        if not isinstance(masks, dict):
            raise ValueError("image_masks must map camera names to booleans")
        for name, present in masks.items():
            if aliases.get(name, name) not in cameras:
                raise ValueError(f"image_masks contains unknown camera {name!r}")
            if not isinstance(present, (bool, np.bool_)) or not present:
                raise ValueError(f"Motubrain2 requires an unmasked image for {name!r}")

    session_id = observation.get("session_id")
    if config.rtc_enabled or session_id is not None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("observation.session_id must be a nonempty string for RTC")
    if not isinstance(observation.get("reset", False), (bool, np.bool_)):
        raise ValueError("observation.reset must be a boolean")
    spec = observation.get("rtc")
    anchored = isinstance(spec, dict) and spec.get("anchor") is not None
    if (
        config.require_state
        and not anchored
        and all(
            observation.get(name) is None for name in ("state", "observation.state")
        )
    ):
        raise ValueError("this robot profile requires observation.state")
    if config.require_camera_extrinsic and observation.get("camera_extrinsic") is None:
        raise ValueError("this robot profile requires observation.camera_extrinsic")
    if spec is None:
        return None
    if (
        not isinstance(spec, dict)
        or not {"future", "delay"}.issubset(spec)
        or spec.keys() - {"anchor", "future", "delay"}
    ):
        raise ValueError(
            "observation.rtc requires future and delay, with an optional anchor"
        )
    from motubrain2.inference.rtc import TrainRTC

    future = spec["future"]
    if isinstance(future, list) and not future:
        # JSON cannot preserve the wire width of a zero-row array.
        future = np.empty((0, config.output_action_dim), dtype=np.float64)
    rtc = TrainRTC(anchor=spec.get("anchor"), future=future, delay=spec["delay"])
    try:
        rtc.prefix(config.output_action_dim, config.action_horizon)
    except (TypeError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"invalid observation.rtc: {exc}") from exc
    return rtc


@dataclass
class Motubrain2SamplingParams(ActionSamplingParams):
    """Per-request parameters for Motubrain2 policy inference."""

    # None selects the served checkpoint's recipe in ``_adjust``.
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    guidance_scale_action: float | None = None
    seed: int | list[int] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    _implicit_seed = False
    action_horizon: int | None = None
    action_dim: int | None = None

    output_format: str = "list"
    return_timing: bool = True
    return_native_outputs: bool | None = None
    enable_prefix_cache: bool = True
    enable_cuda_graph: bool = False

    state: Any = field(default=None, metadata={"batch_sig_exclude": True})
    images: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    image_masks: dict[str, bool] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    camera_order: list[str] | tuple[str, ...] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    noise: Any = field(default=None, metadata={"batch_sig_exclude": True})
    observation: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )

    def __post_init__(self) -> None:
        if self.return_native_outputs is None:
            self.return_native_outputs = (self.observation or {}).get(
                "return_native_outputs", False
            )
        super().__post_init__()

    def _request_observation(self) -> dict[str, Any]:
        observation = dict(self.observation or {})
        observation.pop("_motubrain2_connection_state", None)
        observation["return_native_outputs"] = self.return_native_outputs
        if self.images is not None:
            observation["images"] = self.images
        if self.image_masks is not None:
            observation["image_masks"] = self.image_masks
        if self.state is not None:
            observation["state"] = self.state
        if self.camera_order is not None:
            observation["camera_order"] = self.camera_order
        if self.prompt is not None:
            observation["prompt"] = self.prompt
        if self.noise is not None:
            observation["noise"] = self.noise
        seed = self.seed
        if isinstance(seed, list):
            seed = seed[0] if seed else None
        observation["_motubrain2_seed"] = (
            None if self._implicit_seed or seed is None else int(seed)
        )
        return observation

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        extra["motubrain2"] = {
            "observation": self._request_observation(),
            "options": {
                "output_format": self.output_format,
                "return_timing": self.return_timing,
                "enable_prefix_cache": self.enable_prefix_cache,
            },
        }
        return extra

    def apply_request_extra(self, req: Any) -> None:
        # Move the trusted state into the worker envelope before serializing the request.
        state = (self.observation or {}).pop("_motubrain2_connection_state", None)
        super().apply_request_extra(req)
        if isinstance(state, _ConnectionState):
            req.extra["motubrain2"]["connection_state"] = state.values

    def _adjust(self, server_args) -> None:
        super()._adjust(server_args)
        config = server_args.pipeline_config
        if self.num_inference_steps is None:
            self.num_inference_steps = config.default_num_inference_steps
        if self.action_horizon is None:
            self.action_horizon = config.action_horizon
        elif self.action_horizon != config.action_horizon:
            raise ValueError(
                f"this policy requires action_horizon={config.action_horizon}"
            )
        if self.action_dim is None:
            self.action_dim = config.action_dim
        elif self.action_dim not in (config.action_dim, config.output_action_dim):
            raise ValueError(
                f"action_dim must match the profile's wire width "
                f"({config.output_action_dim}) or padded width ({config.action_dim})"
            )
        if self.seed is None:
            # The generic scheduler needs an integer. The policy must still
            # distinguish an omitted seed from an explicit seed of zero.
            self._implicit_seed = True
            self.seed = config.inference_seed
        validate_observation_options(self._request_observation(), config)

    def _validate(self) -> None:
        steps, seed = self.num_inference_steps, self.seed
        horizon, dim = self.action_horizon, self.action_dim
        # None selects the checkpoint defaults; the base checks need ints.
        self.num_inference_steps = 1 if steps is None else steps
        self.seed = 0 if seed is None else seed
        try:
            super()._validate()
        finally:
            self.num_inference_steps = steps
            self.seed = seed

        if horizon is not None and horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if dim is not None and dim <= 0:
            raise ValueError("action_dim must be positive")
        if isinstance(seed, list) and len(seed) != 1:
            raise ValueError("Motubrain2 takes one seed per request")
        if type(self.return_native_outputs) is not bool:
            raise ValueError("return_native_outputs must be a boolean")
        if self.num_outputs_per_prompt != 1:
            raise ValueError("Motubrain2 returns one action chunk per request")
        if self.output_format not in ("list", "numpy"):
            raise ValueError("output_format must be 'list' or 'numpy'")
        if self.enable_cuda_graph:
            raise ValueError("Motubrain2 does not implement CUDA graphs")
        for name, value in (
            ("guidance_scale", self.guidance_scale),
            ("guidance_scale_action", self.guidance_scale_action),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if value != 1.0:
                raise ValueError(f"Motubrain2 does not implement CFG; {name} must be 1")

    def _set_output_file_name(self) -> None:
        if self.output_file_name is None:
            self.output_file_name = "motubrain2_action"
        super()._set_output_file_name()
