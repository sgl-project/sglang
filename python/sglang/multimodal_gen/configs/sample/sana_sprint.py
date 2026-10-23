# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-Sprint few-step image generation."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sana import SanaSamplingParams


@dataclass
class SanaSprintSamplingParams(SanaSamplingParams):
    """Defaults follow Diffusers' ``SanaSprintPipeline``.

    SANA-Sprint is guidance-distilled: ``guidance_scale`` feeds the DiT's
    guidance embedding and no unconditional pass runs, so negative prompts
    are not used.
    """

    num_inference_steps: int = 2
    guidance_scale: float = 4.5
    negative_prompt: str | None = None

    def should_use_classifier_free_guidance(self) -> bool:
        return False
