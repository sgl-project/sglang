# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class DreamZeroSamplingParams(SamplingParams):
    """Request parameters for DreamZero observation-to-action inference."""

    data_type: DataType = DataType.ACTION
    prompt: str | list[str] | None = None
    negative_prompt: str = ""
    image_path: str | list[str] | None = None

    num_frames: int = 1
    fps: int = 1
    num_inference_steps: int = 4
    guidance_scale: float = 5.0
    save_output: bool = False
    return_file_paths_only: bool = False
    adjust_frames: bool = False

    dreamzero_obs: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    session_id: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    reset_session: bool = field(default=False, metadata={"batch_sig_exclude": True})
    embodiment_tag: str | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    language: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    action_horizon: int = 24
    relative_action_per_horizon: bool = False

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        if self.dreamzero_obs is not None:
            extra["dreamzero_obs"] = self.dreamzero_obs
        if self.session_id is not None:
            extra["dreamzero_session_id"] = self.session_id
        if self.reset_session:
            extra["dreamzero_reset_session"] = True
        if self.embodiment_tag is not None:
            extra["dreamzero_embodiment_tag"] = self.embodiment_tag
        if self.language is not None:
            extra["dreamzero_language"] = self.language
        extra["dreamzero_action_horizon"] = self.action_horizon
        extra["dreamzero_relative_action_per_horizon"] = (
            self.relative_action_per_horizon
        )
        return extra

    def _set_output_file_name(self):
        if self.output_file_name is None:
            self.output_file_name = "dreamzero_action"
        self._set_output_file_ext()
