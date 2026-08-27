# SPDX-License-Identifier: Apache-2.0
"""
Generic pipeline configuration for diffusers backend.

This module provides a minimal pipeline configuration that works with the diffusers backend.
Since diffusers handles its own model loading and configuration, this config is intentionally minimal.
"""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class DiffusersGenericPipelineConfig(PipelineConfig):
    """
    Generic pipeline configuration for diffusers backend.

    This is a minimal configuration since the diffusers backend handles most
    configuration internally. It provides sensible defaults for the required fields.
    """

    # default to T2I since it's the most common
    task_type: ModelTaskType = ModelTaskType.T2I

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"

    should_use_guidance: bool = True
    embedded_cfg_scale: float = 1.0
    flow_shift: float | None = None
    disable_autocast: bool = True  # let diffusers handle dtype

    # diffusers handles its own loading
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp16",))

    # VAE settings
    vae_tiling: bool = False  # diffusers handles this
    vae_slicing: bool = False  # slice VAE decode for lower memory usage
    vae_sp: bool = False

    # Attention backend for diffusers models (e.g., "flash", "_flash_3_hub", "sage", "xformers")
    # See: https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
    diffusers_attention_backend: str | None = None

    # Quantization config for pipeline-level quantization
    # See: https://huggingface.co/docs/diffusers/main/en/quantization/overview
    # Use PipelineQuantizationConfig for component-level control:
    #   from diffusers.quantizers import PipelineQuantizationConfig
    #   quantization_config = PipelineQuantizationConfig(
    #       quant_backend="bitsandbytes_4bit",
    #       quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16},
    #       components_to_quantize=["transformer", "text_encoder_2"],
    #   )
    quantization_config: Any = None

    def check_pipeline_config(self) -> None:
        """
        Override to skip most validation since diffusers handles its own config.
        """
        pass

    def adjust_size(self, width, height, image):
        """
        Pass through - diffusers handles size adjustments.
        """
        return width, height

    def adjust_num_frames(self, num_frames):
        """
        Pass through - diffusers handles frame count.
        """
        return num_frames
