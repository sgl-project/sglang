# SPDX-License-Identifier: Apache-2.0
"""
Wan video progressive-resolution denoising stage.

Extends ProgressiveDenoisingStage for the Wan T2V video model:
  - Latent format: [B, C, T, H, W] (already spatial — no pack/unpack required)
  - Upsample: spatial H×W dims only; T (temporal frames) is fixed across all stages
  - No RoPE / freqs_cis update needed (Wan T2V uses no spatial positional embeddings
    that depend on H or W in the context)

Power-law spectrum constants fitted on VChitect dataset (9050 videos), spatial
spectrum P(ω) = A * |ω|^(-β):
  A    = 219.484718
  β    = 2.422687
"""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
    is_progressive_resolution_mode,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Power-law spectrum constants for WAN 2.1 VAE
# Fitted on VChitect (9050 videos): P(ω) = A * |ω|^(-β)
WAN_SPECTRUM_A: float = 219.484718
WAN_SPECTRUM_BETA: float = 2.422687


class WanProgressiveDenoisingStage(ProgressiveDenoisingStage):
    """Wan T2V–specific progressive denoising stage.

    Differences from the FLUX progressive stage:
    - Wan latent is [B, C, T, H, W] — no patchify pack/unpack needed.
    - Progressive upsample grows only the spatial H×W plane; T stays fixed.
    - Wan T2V has no spatial RoPE freqs_cis that depends on H/W, so
      _on_resolution_change is a no-op.
    - Initial noise must carry the temporal dimension T_lat.
    """

    def __init__(self, transformer, scheduler, pipeline=None, vae=None) -> None:
        super().__init__(
            transformer,
            scheduler,
            pipeline=pipeline,
            vae=vae,
            spectrum_A=WAN_SPECTRUM_A,
            spectrum_beta=WAN_SPECTRUM_BETA,
        )

    # ------------------------------------------------------------------
    # Latent scale factor  (WanVAEArchConfig uses spatial_compression_ratio)
    # ------------------------------------------------------------------

    def _latent_scale_factor(self, server_args: ServerArgs) -> int:
        arch = server_args.pipeline_config.vae_config.arch_config
        return getattr(arch, "vae_scale_factor", None) or getattr(
            arch, "spatial_compression_ratio", 8
        )

    # ------------------------------------------------------------------
    # Pack / Unpack overrides  (Wan latent is already [B, C, T, H, W])
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return latent

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return x_spatial

    # ------------------------------------------------------------------
    # Resolution-change hook  (no-op for Wan T2V)
    # ------------------------------------------------------------------

    def _on_resolution_change(
        self,
        ctx,
        batch: Req,
        server_args: ServerArgs,
        new_h_pixel: int,
        new_w_pixel: int,
    ) -> None:
        """Wan T2V has no spatial positional embeddings that require updating."""
        pass

    # ------------------------------------------------------------------
    # Resolution alignment  (Wan patch embedding requires even spatial dims)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Snap batch.height / batch.width to even multiples before progressive loop.

        Wan's patch embedding is Conv3d(stride=(1,2,2)), so each progressive
        stage latent must have even H and W.  At L levels of downsampling the
        initial latent is H_lat//(2^L) × W_lat//(2^L); if either is odd the
        patchification fails with a tensor size mismatch.

        Fix: align batch.height/width down to the nearest multiple of
        vae_scale_factor * 2^L * 2  (= vae_scale * align_unit) so that every
        stage latent is guaranteed even.  For 480p L=1 this is a no-op (60 is
        already divisible by 4).  For 720p L=1: 90→88 latent rows (704 px).
        """
        mode = getattr(batch, "progressive_mode", "fullres") or "fullres"
        if not is_progressive_resolution_mode(mode):
            return super().forward(batch, server_args)

        levels = int(getattr(batch, "progressive_levels", 1))
        arch = server_args.pipeline_config.vae_config.arch_config
        vae_scale = getattr(arch, "vae_scale_factor", None) or getattr(
            arch, "spatial_compression_ratio", 8
        )
        # Each stage halves the spatial dims; Wan needs even dims at every stage.
        # Required: H_lat divisible by 2^L * patch_spatial (= 2^L * 2).
        align_pixels = vae_scale * (2**levels) * 2
        h_aligned = max((batch.height // align_pixels) * align_pixels, align_pixels)
        w_aligned = max((batch.width // align_pixels) * align_pixels, align_pixels)

        if h_aligned != batch.height or w_aligned != batch.width:
            logger.info(
                "WanProgressiveDenoisingStage: aligning resolution %dx%d → %dx%d "
                "so all progressive stage latents have even spatial dims (patch=2, L=%d)",
                batch.height,
                batch.width,
                h_aligned,
                w_aligned,
                levels,
            )
            batch.height = h_aligned
            batch.width = w_aligned

        return super().forward(batch, server_args)

    # ------------------------------------------------------------------
    # Initial noise  (must include the temporal dim T_lat)
    # ------------------------------------------------------------------

    def _generate_initial_noise(
        self,
        batch: Req,
        server_args: ServerArgs,
        h_lat: int,
        w_lat: int,
        seed,
    ) -> torch.Tensor:
        """Generate low-res initial noise [1, C, T_lat, h_lat, w_lat].

        The base-class version generates 4-D [1, C, H, W] noise and uses
        in_channels // 4 for the channel count (FLUX patchify convention).
        Wan operates directly on the 5-D latent, so we override to:
          - Use z_dim (= 16) as the correct latent channel count.
          - Preserve T_lat from the original full-res latent in batch.latents,
            since progressive upsample only grows spatial H×W.
        """
        device = get_local_torch_device()
        C = server_args.pipeline_config.vae_config.arch_config.z_dim
        # batch.latents still holds the full-res latent from LatentPreparationStage
        # at this call site, so shape[2] gives the fixed T_lat.
        T_lat = batch.latents.shape[2]
        dtype = server_args.pipeline_config.get_latent_dtype(
            batch.prompt_embeds[0].dtype if batch.prompt_embeds else torch.bfloat16
        )
        noise = randn_tensor(
            (self._initial_noise_batch_size(batch), C, T_lat, h_lat, w_lat),
            generator=self._get_initial_noise_generator(batch, seed, device),
            device=device,
            dtype=dtype,
        )
        return noise
