# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import LingBotVideoMoEConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    Qwen3VLConfig,
)
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


def _qwen3vl_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> list[torch.Tensor]:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    return [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]


@dataclass
class LingBotVideoMoEPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2V
    dit_config: DiTConfig = field(default_factory=LingBotVideoMoEConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False
    flow_shift: float | None = 3.0
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3VLConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (None,)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (_qwen3vl_postprocess_text,))
    )
    precision: str = "bf16"
    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    embedded_cfg_scale: float = 6.0

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(dit_layerwise_offload_modes=("memory",))

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        return torch.float32

    def get_decode_scale_and_shift(self, device, dtype, vae):
        arch = self.vae_config.arch_config
        mean = torch.tensor(arch.latents_mean, device=device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(arch.latents_std, device=device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        return 1.0 / std, mean
