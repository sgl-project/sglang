# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


@dataclass
class LingBotVASamplingParams(ActionSamplingParams):
    num_inference_steps: int = 50
    images: dict | None = field(default=None, metadata={"batch_sig_exclude": True})
    observation: dict | None = field(default=None, metadata={"batch_sig_exclude": True})
    output_format: str = "list"

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        obs = dict(self.observation or {})
        if self.images:
            obs["images"] = self.images
        extra["vla"] = {
            "observation": obs,
            "options": {"output_format": self.output_format},
        }
        return extra
