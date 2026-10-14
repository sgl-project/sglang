# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class MOSSImageSamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    seed: int | list[int] = field(default=0, metadata={"batch_sig_exclude": True})
    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: str = ""
    # Formal checkpoints skip this joint DiT layer in the unconditional branch.
    # An empty list restores ordinary two-branch CFG.
    uncond_skip_layers: list[int] = field(default_factory=lambda: [22])

    def should_use_classifier_free_guidance(self) -> bool:
        # The reference evaluates both branches even for guidance below one.
        if self.negative_prompt is None:
            self.negative_prompt = ""
        return self.guidance_scale != 1.0

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset({"uncond_skip_layers"})
