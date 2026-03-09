# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.helios import HeliosConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, T5Config
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Helios UMT5 max sequence length (used for both tokenizer and post-processing padding)
HELIOS_MAX_SEQUENCE_LENGTH = 226


def umt5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Post-process UMT5 text encoder outputs, padding to HELIOS_MAX_SEQUENCE_LENGTH tokens."""
    max_seq_len = HELIOS_MAX_SEQUENCE_LENGTH
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


@dataclass
class HeliosT2VConfig(PipelineConfig):
    """Configuration for the Helios T2V pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2V

    # DiT
    dit_config: DiTConfig = field(default_factory=HeliosConfig)

    # VAE (same as Wan)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 1.0

    # Text encoding stage (UMT5 is T5-compatible)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (
            T5Config(arch_config=T5ArchConfig(text_len=HELIOS_MAX_SEQUENCE_LENGTH)),
        )
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (umt5_postprocess_text,))
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    # Helios-specific chunked denoising params
    num_latent_frames_per_chunk: int = 9
    history_sizes: list[int] = field(default_factory=lambda: [16, 2, 1])
    is_cfg_zero_star: bool = False
    zero_steps: int = 1
    keep_first_frame: bool = True

    # Stage 2 (Pyramid SR) & Stage 3 (DMD) params
    is_enable_stage2: bool = False
    pyramid_num_stages: int = 3
    pyramid_num_inference_steps_list: list[int] = field(
        default_factory=lambda: [10, 10, 10]
    )
    is_distilled: bool = False
    is_amplify_first_chunk: bool = False
    scheduler_type: str = "unipc"
    gamma: float = 1 / 3

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class HeliosMidConfig(HeliosT2VConfig):
    """Configuration for Helios-Mid (Stage 1 + Stage 2 pyramid SR)."""

    is_enable_stage2: bool = True
    is_cfg_zero_star: bool = True
    pyramid_num_inference_steps_list: list[int] = field(
        default_factory=lambda: [20, 20, 20]
    )


@dataclass
class HeliosDistilledConfig(HeliosT2VConfig):
    """Configuration for Helios-Distilled (Stage 1 + Stage 2 + Stage 3 DMD)."""

    is_enable_stage2: bool = True
    is_distilled: bool = True
    is_amplify_first_chunk: bool = True
    scheduler_type: str = "dmd"
    pyramid_num_inference_steps_list: list[int] = field(
        default_factory=lambda: [10, 10, 10]
    )
