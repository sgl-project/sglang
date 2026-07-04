# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
    _sanitize_filename,
)


@dataclass
class Pi05SamplingParams(SamplingParams):
    """Sampling parameters for Pi0.5 flow-matching action inference."""

    data_type: DataType = DataType.ACTION
    prompt: str | list[str] | None = field(
        default="", metadata={"batch_sig_exclude": True}
    )
    negative_prompt: str | None = None
    num_inference_steps: int = 10
    guidance_scale: float = 1.0
    num_frames: int = 1
    fps: int = 1
    save_output: bool = False
    return_file_paths_only: bool = False

    action_horizon: int = 50
    action_dim: int = 32
    return_timing: bool = True
    enable_prefix_cache: bool = True
    enable_cuda_graph: bool = True

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
    normalization_config: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    discretization_config: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    adapter: str | None = field(default=None, metadata={"batch_sig_exclude": True})

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        observation = dict(self.observation or {})
        if self.images is not None:
            observation["images"] = self.images
        if self.image_masks is not None:
            observation["image_masks"] = self.image_masks
        if self.state is not None:
            observation["state"] = self.state
        if self.camera_order is not None:
            observation["camera_order"] = tuple(self.camera_order)
        if self.prompt is not None:
            observation["prompt"] = self.prompt
        if self.noise is not None:
            observation["noise"] = self.noise

        extra["pi05_observation"] = observation
        extra["pi05_options"] = {
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "num_inference_steps": self.num_inference_steps,
            "return_timing": self.return_timing,
            "enable_prefix_cache": self.enable_prefix_cache,
            "enable_cuda_graph": self.enable_cuda_graph,
            "normalization_config": self.normalization_config,
            "discretization_config": self.discretization_config,
            "adapter": self.adapter,
        }
        return extra

    def _validate(self):
        super()._validate()
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")

    def _set_output_file_name(self):
        if self.output_file_name is None:
            self.output_file_name = "pi05_action"
        self.output_file_name = _sanitize_filename(self.output_file_name)
        self._set_output_file_ext()
