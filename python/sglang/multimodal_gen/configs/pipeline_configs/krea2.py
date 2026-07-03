# Krea-2 (K2) pipeline config.
from dataclasses import dataclass, field

import torch
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.krea2 import Krea2DitConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)

# Resolution-interpolation endpoints for the time-shift `mu` (reference sampler):
# mu is linear in image-token count between (min_res, 0.5) and (max_res, 1.15).
_MU_MIN_RES = 256
_MU_MAX_RES = 1280
_MU_Y1 = 0.5
_MU_Y2 = 1.15


@dataclass
class Krea2PipelineConfig(ImagePipelineConfig):
    """Krea-2 single-stream MMDiT, text-to-image.

    Reuses the Qwen-Image VAE (same checkpoint) and its latent pack/unpack +
    decode de-normalization. K2-specific pieces are the 3-axis joint-stream RoPE
    positions and key-padding mask (``prepare_pos_cond_kwargs``), the 4-D
    layer-stacked text conditioning (``get_pos_prompt_embeds``), and the flow
    time-shift ``mu``.
    """

    task_type: ModelTaskType = ModelTaskType.T2I
    should_use_guidance: bool = False
    enable_autocast: bool = False
    vae_tiling: bool = False
    vae_sp: bool = False
    vae_precision: str = "bf16"
    # The released Qwen3-VL text encoder is bf16 (text_encoder/config.json dtype);
    # the base loader defaults to fp32, which perturbs the conditioning embeddings.
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    dit_config: DiTConfig = field(default_factory=Krea2DitConfig)
    vae_config: VAEConfig = field(default_factory=QwenImageVAEConfig)

    # Pinned time-shift mu (distilled `oss_turbo`); set None to derive from resolution.
    pinned_mu: float | None = 1.15

    def __post_init__(self):
        self.vae_scale_factor = self.vae_config.get_vae_scale_factor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def get_vae_scale_factor(self):
        return self.vae_config.get_vae_scale_factor()

    # --- text conditioning: K2 feeds the raw layer-stacked encoder states ---
    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    # --- joint-stream RoPE positions + key-padding mask ---
    def _build_pos_and_mask(self, context, text_mask, batch, device):
        b, txt_len = context.shape[0], context.shape[1]
        patch = self.dit_config.arch_config.patch
        vsf = self.get_vae_scale_factor()
        h_tok = int(batch.height) // vsf // patch
        w_tok = int(batch.width) // vsf // patch

        img_ids = torch.zeros(h_tok, w_tok, 3, device=device)
        img_ids[..., 1] = torch.arange(h_tok, device=device)[:, None]
        img_ids[..., 2] = torch.arange(w_tok, device=device)[None, :]
        img_pos = img_ids.reshape(h_tok * w_tok, 3).unsqueeze(0).expand(b, -1, -1)
        txt_pos = torch.zeros(b, txt_len, 3, device=device)
        pos = torch.cat([txt_pos, img_pos], dim=1)

        img_mask = torch.ones(b, h_tok * w_tok, dtype=torch.bool, device=device)
        if text_mask is None:
            txt_mask = torch.ones(b, txt_len, dtype=torch.bool, device=device)
        else:
            txt_mask = text_mask.to(device=device).bool()
        mask = torch.cat([txt_mask, img_mask], dim=1)
        return {"pos": pos, "mask": mask}

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        text_mask = batch.prompt_embeds_mask[0] if batch.prompt_embeds_mask else None
        return self._build_pos_and_mask(
            batch.prompt_embeds[0], text_mask, batch, device
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        text_mask = (
            batch.negative_prompt_embeds_mask[0]
            if batch.negative_prompt_embeds_mask
            else None
        )
        return self._build_pos_and_mask(
            batch.negative_prompt_embeds[0], text_mask, batch, device
        )

    # --- timestep shift ---
    def compute_mu(self, image_seq_len: int) -> float:
        if self.pinned_mu is not None:
            return self.pinned_mu
        patch = self.dit_config.arch_config.patch
        vsf = self.get_vae_scale_factor()
        x1 = (_MU_MIN_RES // (vsf * patch)) ** 2
        x2 = (_MU_MAX_RES // (vsf * patch)) ** 2
        slope = (_MU_Y2 - _MU_Y1) / (x2 - x1)
        return slope * image_seq_len + (_MU_Y1 - slope * x1)

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    # --- VAE decode (same as Qwen-Image: latents * std + mean) ---
    def get_decode_scale_and_shift(self, device, dtype, vae):
        vae_arch = self.vae_config.arch_config
        scaling_factor = 1.0 / torch.tensor(vae_arch.latents_std, device=device).view(
            1, vae_arch.z_dim, 1, 1, 1
        ).to(device, dtype)
        shift_factor = (
            torch.tensor(vae_arch.latents_mean)
            .view(1, vae_arch.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        return scaling_factor, shift_factor

    def post_denoising_loop(self, latents, batch):
        latents, batch_size, channels, height, width = self._unpad_and_unpack_latents(
            latents, batch
        )
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents
