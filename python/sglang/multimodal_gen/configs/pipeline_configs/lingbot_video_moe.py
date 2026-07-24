# Adapted from: https://github.com/Robbyant/lingbot-video
# Reference (upstream): /vllm-workspace/lingbot-video/lingbot_video

# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for the LingBot-Video MoE 30B text-to-video model.

MVP scope: single-GPU, T2V base-only, batch size 1, structured-JSON captions.

IMPORTANT (parity / OOD contract): the DiT was trained on *structured JSON*
captions produced by a separate prompt-rewriter. The caption string rides in
the standard ``prompt`` field of :class:`LingBotVideoMoESamplingParams`. Raw
natural-language prompts are out-of-distribution and produce garbage output;
the prompt-rewriter is a follow-up phase.
"""

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
    """Trim Qwen3-VL hidden states to the valid (attention) token range.

    The DiT consumes a *list* of per-prompt embeddings (``batch.prompt_embeds[0]``
    via :meth:`LingBotVideoMoEPipelineConfig.get_pos_prompt_embeds`). The native
    ``LingBotVideoBeforeDenoisingStage`` populates ``batch.prompt_embeds``
    directly (it runs its own ``_compute_crop_start`` trimming), so this function
    only matters if the standard text-encoding stage is used as a fallback.
    """
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    return [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]


@dataclass
class LingBotVideoMoEPipelineConfig(PipelineConfig):
    """Configuration for the LingBot-Video MoE 30B T2V pipeline.

    Mirrors :class:`WanT2V480PConfig` (Wan T2V 1.3B) in structure, but uses the
    LingBot-Video MoE DiT config, Wan VAE (same 16-channel ``AutoencoderKLWan``
    with the 16-element ``latents_mean``/``latents_std``), and Qwen3-VL text
    encoder.
    """

    task_type: ModelTaskType = ModelTaskType.T2V

    # --- DiT ---
    dit_config: DiTConfig = field(default_factory=LingBotVideoMoEConfig)

    # --- VAE (AutoencoderKLWan; z_dim=16, 4x temporal / 8x spatial) ---
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # --- Denoising stage ---
    flow_shift: float | None = 3.0

    # --- Text encoding stage ---
    # Qwen3-VL is the text encoder. The native BeforeDenoisingStage runs
    # ``Qwen3VLForConditionalGeneration`` directly (with its own prompt
    # template / ``_compute_crop_start``), so the SGLang encoder config here is
    # metadata for component loading; the real arch is read from the HF config.
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

    # --- Precision ---
    precision: str = "bf16"
    vae_precision: str = "bf16"

    # --- Guidance ---
    should_use_guidance: bool = True
    embedded_cfg_scale: float = 6.0

    def __post_init__(self):
        # T2V only needs the VAE decoder.
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # 30B MoE: enable layerwise DiT offload so the model fits on a single
        # 32GB GPU (and stays resident on 80GB).
        return ModelDeploymentConfig(auto_dit_layerwise_offload=True)

    # ---- conditioning ----
    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # The DiT derives its complex64 RoPE internally from
        # ``encoder_attention_mask`` (joint 3D position ids), so there is no
        # pipeline-side rotary_emb to forward. The DenoisingStage injects
        # ``encoder_hidden_states`` + ``encoder_attention_mask`` directly.
        return {}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    # ---- decoding (latents -> pixel) ----
    def get_decode_scale_and_shift(self, device, dtype, vae):
        """De-normalize latents before VAE decode, matching upstream
        ``_dit_latent_to_vae``: ``latents = latents * std + mean``.

        The shared ``DecodingStage`` applies ``latents / scaling + shift``, so
        we return ``scaling = 1/std`` and ``shift = mean`` broadcast to
        ``[1, C, 1, 1, 1]``. The 16-element ``latents_mean``/``latents_std``
        live on the Wan VAE arch config (the same vectors the upstream diffusers
        ``AutoencoderKLWan`` carries on ``vae.config``), guaranteeing parity.
        """
        arch = self.vae_config.arch_config
        latents_mean = arch.latents_mean
        latents_std = arch.latents_std
        mean = torch.tensor(latents_mean, device=device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        return 1.0 / std, mean
