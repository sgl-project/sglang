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
    task_type: ModelTaskType = ModelTaskType.TI2V
    # Qwen3-VL needs the condition frame as a PIL image, not the generic geometry rewrite.
    skip_input_image_preprocess: bool = True
    dit_config: DiTConfig = field(default_factory=LingBotVideoMoEConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False
    flow_shift: float | None = 3.0
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3VLConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (_qwen3vl_postprocess_text,))
    )
    precision: str = "bf16"
    vae_precision: str = "bf16"
    # Second-pass settings, used only by LingBotVideoRefinerPipeline.
    refiner_height: int = 1088
    refiner_width: int = 1920
    refiner_num_inference_steps: int = 8
    refiner_guidance_scale: float = 3.0
    refiner_flow_shift: float = 3.0
    refiner_t_thresh: float = 0.85
    refiner_sigma_tail_steps: int = 2
    # Prompt rewriter, off unless a URL or a local base model is configured.
    rewriter_url: str | None = None
    # Falls back to rewriter_url when the mapping turn shares the endpoint.
    rewriter_map_url: str | None = None
    rewriter_expand_model: str = "lingbot-rewriter-base"
    rewriter_map_model: str = "lingbot-rewriter-lora"
    rewriter_timeout: float = 300.0
    rewriter_model_path: str | None = None
    rewriter_adapter_path: str | None = None
    rewriter_device_map: str = "auto"
    rewriter_max_new_tokens: int = 6144
    rewriter_auto_negative: bool = False

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.rewriter_auto_negative and not self.has_rewriter:
            raise ValueError(
                "rewriter_auto_negative needs a rewriter backend: set rewriter_url "
                "to serve one, or rewriter_model_path and rewriter_adapter_path to "
                "load it in this process."
            )

    @property
    def has_rewriter(self) -> bool:
        return self.rewriter_url is not None or self.rewriter_model_path is not None

    def supports_dynamic_batching(self) -> bool:
        # A merged request carries one negative prompt for several captions.
        return not self.rewriter_auto_negative

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(dit_layerwise_offload_modes=("memory",))

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

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
