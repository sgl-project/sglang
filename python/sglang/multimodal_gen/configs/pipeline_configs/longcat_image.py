from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.longcat_image import (
    LongCatImageDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.longcat_image import (
    LongCatImageVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression, plus 2x packing factor
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), h, w)
    return latents


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )
    return latents


def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@dataclass
class LongCatImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the LongCat-Image T2I pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2I

    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    vae_tiling: bool = False
    vae_sp: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=LongCatImageDitConfig)
    vae_config: VAEConfig = field(default_factory=LongCatImageVAEConfig)

    # --- LatentPreparationStage hooks ---

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.vae_config.get_vae_scale_factor()
        # LongCat packs 2x2 patches: effective spatial resolution after packing
        h = 2 * (int(batch.height) // (vae_scale_factor * 2))
        w = 2 * (int(batch.width) // (vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.num_channels_latents
        # Unpacked shape — maybe_pack_latents will fold into tokens
        return (batch_size, num_channels_latents, h, w)

    def maybe_pack_latents(self, latents, batch_size, batch):
        num_channels_latents = self.dit_config.arch_config.num_channels_latents
        _, _, h, w = latents.shape
        return _pack_latents(latents, batch_size, num_channels_latents, h, w)

    def maybe_prepare_latent_ids(self, latents):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_image import (
            TOKENIZER_MAX_LENGTH,
            _prepare_pos_ids,
        )

        # latents shape after packing: [B, (h//2)*(w//2), C*4]
        # We need h//2 and w//2 — derive from the unpacked shape stored on the config.
        # latents is still unpacked here (called before maybe_pack_latents in LatentPreparationStage)
        _, _, h, w = latents.shape
        return _prepare_pos_ids(
            modality_id=1,
            token_type="image",
            start=(TOKENIZER_MAX_LENGTH, TOKENIZER_MAX_LENGTH),
            height=h // 2,
            width=w // 2,
        )

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        # Generate in float32 then cast to bfloat16, matching diffusers behavior.
        return torch.float32

    # --- TimestepPreparationStage hook ---

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        return sigmas

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "txt_ids": batch.txt_ids,
            "img_ids": batch.img_ids,
            "image_rotary_emb": batch.image_rotary_emb,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "txt_ids": batch.negative_txt_ids,
            "img_ids": batch.img_ids,
            "image_rotary_emb": batch.image_rotary_emb,
        }

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # LongCat uses standard AutoencoderKL: latents = (latents / scaling_factor) + shift_factor
        # DecodingStage does: latents = latents / scaling_factor + shift_factor
        # So return (scaling_factor, shift_factor) directly.
        vae_config = getattr(vae, "config", None)
        if vae_config is not None:
            sf = getattr(vae_config, "scaling_factor", 1.0)
            shift = getattr(vae_config, "shift_factor", 0.0)
        else:
            sf = 1.0
            shift = 0.0
        return sf, shift

    def post_denoising_loop(self, latents, batch):
        vae_scale_factor = self.vae_config.get_vae_scale_factor()
        latents = _unpack_latents(latents, batch.height, batch.width, vae_scale_factor)
        # Add frames dimension for DecodingStage compatibility: [B, C, H, W] -> [B, C, 1, H, W]
        latents = latents.unsqueeze(2)
        return latents

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Remove frames dimension before VAE decode: [B, C, 1, H, W] -> [B, C, H, W]."""
        if latents.dim() == 5 and latents.shape[2] == 1:
            latents = latents.squeeze(2)
        return latents

    def postprocess_cfg_noise(
        self,
        batch,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor,
    ) -> torch.Tensor:
        enable_cfg_renorm = getattr(batch, "enable_cfg_renorm", True)
        cfg_renorm_min = getattr(batch, "cfg_renorm_min", 0.0)
        if not enable_cfg_renorm:
            return noise_pred
        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return noise_pred * scale
