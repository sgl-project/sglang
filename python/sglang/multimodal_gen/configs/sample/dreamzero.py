# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.vla import VLASamplingParams


@dataclass
class DreamZeroSamplingParams(VLASamplingParams):
    """Request parameters for DreamZero observation-to-action inference."""

    prompt: str | list[str] | None = None
    negative_prompt: str = ""

    num_inference_steps: int = 4
    guidance_scale: float = 5.0

    observation: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    session_ids: list[str] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    reset_mask: list[bool] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    negative_prompts: list[str] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    dreamzero_prompts: str | list[str] | None = field(
        default=None, init=False, metadata={"batch_sig_exclude": True}
    )

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        if self.observation is not None:
            extra["dreamzero_normalized_input"] = self.observation
        if self.session_ids is not None:
            extra["dreamzero_session_ids"] = self.session_ids
        if self.reset_mask is not None:
            extra["dreamzero_reset_mask"] = self.reset_mask
        if self.dreamzero_prompts is not None:
            extra["dreamzero_prompts"] = self.dreamzero_prompts
        if self.negative_prompts is not None:
            extra["dreamzero_negative_prompts"] = self.negative_prompts
        return extra

    def _adjust(self, server_args):
        self.dreamzero_prompts = self.prompt
        if isinstance(self.prompt, list):
            self.prompt = self.prompt[0] if self.prompt else ""
        elif self.prompt is None:
            self.prompt = ""
        super()._adjust(server_args)

    def _set_output_file_name(self):
        if self.output_file_name is None:
            self.output_file_name = "dreamzero_action"
        super()._set_output_file_name()
