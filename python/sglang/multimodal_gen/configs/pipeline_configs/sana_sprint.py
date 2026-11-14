# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for SANA-Sprint (few-step, guidance-distilled SANA)."""

from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sana import SanaPipelineConfig

# Diffusers' ASPECT_RATIO_1024_BIN (pipelines/pixart_alpha), which SanaSprintPipeline
# bins to; copied because importing it pulls in the whole PixArt pipeline.
ASPECT_RATIO_1024_BIN: dict[str, tuple[int, int]] = {
    "0.25": (512, 2048),
    "0.28": (512, 1856),
    "0.32": (576, 1792),
    "0.33": (576, 1728),
    "0.35": (576, 1664),
    "0.4": (640, 1600),
    "0.42": (640, 1536),
    "0.48": (704, 1472),
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "3.0": (1728, 576),
    "4.0": (2048, 512),
}


def classify_height_width_bin(height: int, width: int) -> tuple[int, int]:
    """The (height, width) bin whose aspect ratio is closest, as Diffusers picks it."""
    aspect_ratio = height / width
    closest = min(ASPECT_RATIO_1024_BIN, key=lambda r: abs(float(r) - aspect_ratio))
    return ASPECT_RATIO_1024_BIN[closest]


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

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height, width = batch.height, batch.width
        if batch.use_resolution_binning:
            height, width = classify_height_width_bin(height, width)
        compression = self.vae_config.arch_config.spatial_compression_ratio
        num_channels = self.dit_config.arch_config.num_channels_latents
        return (batch_size, num_channels, height // compression, width // compression)

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
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        ],
        model_detectors=[
            lambda hf_id: (
                "sana_sprint" in hf_id.lower() or "sana-sprint" in hf_id.lower()
            )
        ],
    )
