from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import LongCatVideoConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, T5Config
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def longcat_text_postprocess(
    outputs: BaseEncoderOutput, _text_inputs
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_state = outputs.last_hidden_state
    attention_mask = getattr(outputs, "attention_mask", None)
    if attention_mask is None and _text_inputs is not None:
        attention_mask = _text_inputs.get("attention_mask")
    if hidden_state.ndim != 3:
        raise ValueError(
            "LongCat text encoder output must be [batch, seq, hidden], got "
            f"{tuple(hidden_state.shape)}"
        )
    if attention_mask is None:
        attention_mask = torch.ones(
            hidden_state.shape[:2], device=hidden_state.device, dtype=torch.long
        )
    return hidden_state.unsqueeze(1), attention_mask


@dataclass
class LongCatVideoPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2V

    dit_config: DiTConfig = field(default_factory=LongCatVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput, Any], tuple[torch.Tensor, torch.Tensor]], ...
    ] = field(default_factory=lambda: (longcat_text_postprocess,))

    precision: str = "bf16"
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    def __post_init__(self) -> None:
        # Guard with hasattr in case vae_config is replaced with a VAE type
        # that does not support selective encoder/decoder loading.
        if hasattr(self.vae_config, "load_encoder"):
            self.vae_config.load_encoder = False
        if hasattr(self.vae_config, "load_decoder"):
            self.vae_config.load_decoder = True

    def adjust_num_frames(self, num_frames: int) -> int:
        if (num_frames - 1) % 4 == 0:
            return int(num_frames)
        adjusted = num_frames // 4 * 4 + 1
        logger.warning(
            "LongCat requires (num_frames - 1) divisible by 4; adjusting %s to %s.",
            num_frames,
            adjusted,
        )
        return int(adjusted)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        # num_frames here is already the latent frame count (after temporal compression
        # by LatentPreparationStage.adjust_video_length). Do not compress again.
        return [
            batch_size,
            16,
            int(num_frames),
            int(batch.height) // 8,
            int(batch.width) // 8,
        ]

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        return torch.bfloat16

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is not None:
            return sigmas
        return torch.linspace(1.0, 0.001, int(num_inference_steps)).tolist()

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {"encoder_attention_mask": batch.prompt_attention_mask[0]}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {"encoder_attention_mask": batch.negative_attention_mask[0]}

