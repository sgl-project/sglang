# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for SANA-Sprint (few-step, guidance-distilled SANA)."""

from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sana import SanaPipelineConfig


@dataclass
class SanaSprintPipelineConfig(SanaPipelineConfig):
    # Prompts are padded to the instruction-prefixed length, then windowed.
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            {
                "padding": "max_length",
                "return_attention_mask": True,
                "add_special_tokens": True,
            }
        ]
    )

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        # The SCM update runs on fp32 latents, as in Diffusers.
        return torch.float32


def register():
    from sglang.multimodal_gen.configs.sample.sana_sprint import (
        SanaSprintSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=SanaSprintSamplingParams,
        pipeline_config_cls=SanaSprintPipelineConfig,
        hf_model_paths=[
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        ],
        model_detectors=[
            lambda hf_id: (
                "sana_sprint" in hf_id.lower() or "sana-sprint" in hf_id.lower()
            )
        ],
    )
