# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


@dataclass
class DM05SamplingParams(ActionSamplingParams):
    """Continuous DM05 action sampling; deployment geometry comes from the recipe."""

    action_horizon: int | None = None
    action_dim: int | None = None
    output_format: str = "list"
    return_timing: bool = True
    enable_prefix_cache: bool = True
    enable_cuda_graph: bool = True
    images: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    state: Any = field(default=None, metadata={"batch_sig_exclude": True})
    noise: Any = field(default=None, metadata={"batch_sig_exclude": True})
    image_masks: dict[str, bool] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    camera_order: list[str] | tuple[str, ...] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    history_images: list[Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    observation: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )

    def _validate(self):
        super()._validate()
        if self.num_outputs_per_prompt != 1:
            raise ValueError("DM05 supports one action candidate per observation")
        if type(self.num_inference_steps) is not int:
            raise ValueError("num_inference_steps must be a positive integer")
        for name in ("action_horizon", "action_dim"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.output_format not in ("list", "numpy"):
            raise ValueError("output_format must be list or numpy")

    def _adjust(self, server_args):
        config = server_args.pipeline_config
        for name in ("action_horizon", "action_dim"):
            configured = getattr(config, name)
            if getattr(self, name) is None:
                setattr(self, name, configured)
            elif getattr(self, name) != configured:
                raise ValueError(
                    f"{name} must match the loaded DM05 deployment ({configured})"
                )
        super()._adjust(server_args)

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        observation = dict(self.observation or {})
        for name in (
            "images",
            "state",
            "noise",
            "image_masks",
            "camera_order",
            "history_images",
        ):
            value = getattr(self, name)
            if value is not None:
                observation[name] = value
        if self.prompt is not None:
            observation.setdefault("prompt", self.prompt)
        extra["vla"] = {
            "observation": observation,
            "options": {
                "output_format": self.output_format,
                "return_timing": self.return_timing,
                "enable_prefix_cache": self.enable_prefix_cache,
                "enable_cuda_graph": self.enable_cuda_graph,
            },
        }
        return extra
