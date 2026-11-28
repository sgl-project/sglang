# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


def _validate_tensor(value: Any, shape: tuple[int, ...], name: str) -> None:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"LingBot {name} must be a numeric tensor") from exc
    if tuple(tensor.shape) != shape:
        raise ValueError(f"LingBot {name} must have shape {shape}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"LingBot {name} must be finite")


@dataclass
class LingBotVLAV2SamplingParams(ActionSamplingParams):
    action_horizon: int | None = None
    action_dim: int | None = None
    output_format: str = "list"
    return_timing: bool = True
    enable_cuda_graph: bool = True
    images: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    state: Any = field(default=None, metadata={"batch_sig_exclude": True})
    noise: Any = field(default=None, metadata={"batch_sig_exclude": True})

    def _adjust(self, server_args):
        from sglang.multimodal_gen.runtime.models.vlas.lingbot_vla_v2.processor_lingbot_vla_v2 import (
            validate_camera_image,
        )

        super()._adjust(server_args)
        config = server_args.pipeline_config
        if self.action_horizon not in (None, config.action_horizon):
            raise ValueError(f"LingBot action_horizon must be {config.action_horizon}")
        if self.action_dim not in (None, config.action_dim):
            raise ValueError(f"LingBot noise action_dim must be {config.action_dim}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("LingBot requires a nonempty prompt or task string")
        if self.state is None:
            raise ValueError("LingBot observation requires state")
        _validate_tensor(self.state, (config.policy.state_dim,), "state")
        if self.noise is not None:
            _validate_tensor(
                self.noise, (config.action_horizon, config.action_dim), "noise"
            )
        if self.images is not None and not isinstance(self.images, dict):
            raise ValueError("LingBot images must map camera names to RGB frames")
        images = {}
        for name, image in (self.images or {}).items():
            if isinstance(name, str):
                key = (
                    name
                    if name.startswith("observation.images.")
                    else f"observation.images.{name}"
                )
                images[config.policy.image_key_map.get(key, key)] = image
        cameras = config.policy.camera_keys
        present = [key for key in cameras if key in images]
        if not present:
            raise ValueError(
                f"Observation requires at least one configured camera: {cameras}"
            )
        for key in present:
            validate_camera_image(images[key])

    def _validate(self):
        super()._validate()
        if self.num_outputs_per_prompt != 1 or isinstance(self.seed, list):
            raise ValueError("LingBot accepts one seed and returns one action chunk")
        if self.output_format not in ("list", "numpy"):
            raise ValueError("output_format must be 'list' or 'numpy'")

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        extra["vla"] = {
            "observation": {
                "images": self.images,
                "state": self.state,
                "noise": self.noise,
                "prompt": self.prompt,
            },
            "options": {
                "output_format": self.output_format,
                "return_timing": self.return_timing,
                "enable_cuda_graph": self.enable_cuda_graph,
            },
        }
        return extra
