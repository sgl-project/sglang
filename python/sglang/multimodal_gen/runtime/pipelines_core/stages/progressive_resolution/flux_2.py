# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-specific progressive-resolution denoising stage.

Provides pack/unpack for FLUX.2's simple row-major token format and updates
both batch.latent_ids and freqs_cis when the latent resolution changes
between progressive stages.

FLUX.2 latent layout (before packing):
  spatial: [B, C, H_lat, W_lat]  where H_lat = H_pixel // (vae_scale_factor * 2)
  packed:  [B, H_lat * W_lat, C] (row-major reshape)

This differs from FLUX.1 which uses a 2×2 patchification to interleave spatial
blocks into packed tokens.
"""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.flux import _prepare_latent_ids
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Power-law spectrum constants — using FLUX.1-dev VAE values as a placeholder
# until FLUX.2-specific coefficients are fitted.
# Fitted on Aesthetics-Train-V2 (105k images) for the FLUX.1-dev VAE.
FLUX_SPECTRUM_A: float = 203.615097
FLUX_SPECTRUM_BETA: float = 1.915461


def _flux2_unpack(latent: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Packed [B, H_lat*W_lat, C] → spatial [B, C, H_lat, W_lat] (row-major)."""
    B, _S, C = latent.shape
    return latent.permute(0, 2, 1).reshape(B, C, h_lat, w_lat)


def _flux2_pack(x: torch.Tensor) -> torch.Tensor:
    """Spatial [B, C, H_lat, W_lat] → packed [B, H_lat*W_lat, C] (row-major)."""
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).permute(0, 2, 1)


class Flux2ProgressiveDenoisingStage(ProgressiveDenoisingStage):
    """FLUX.2-specific progressive denoising stage.

    Handles:
    - FLUX.2 row-major pack/unpack
    - latent_ids update on resolution change (needed for 4-D RoPE in FLUX.2)
    - freqs_cis cache and branch update on resolution change
    """

    def __init__(self, transformer, scheduler, pipeline=None, vae=None) -> None:
        super().__init__(
            transformer,
            scheduler,
            pipeline=pipeline,
            vae=vae,
            spectrum_A=FLUX_SPECTRUM_A,
            spectrum_beta=FLUX_SPECTRUM_BETA,
        )
        self._freqs_cis_cache: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    # ------------------------------------------------------------------
    # Scale factor override
    # ------------------------------------------------------------------

    def _latent_scale_factor(self, server_args: ServerArgs) -> int:
        # FLUX.2 latent spatial dimensions are at 1/(vae_scale_factor * 2) of
        # pixel resolution due to the extra patchification step.
        return server_args.pipeline_config.vae_config.arch_config.vae_scale_factor * 2

    # ------------------------------------------------------------------
    # Pack / Unpack overrides
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return _flux2_unpack(latent, h_lat, w_lat)

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return _flux2_pack(x_spatial)

    # ------------------------------------------------------------------
    # Initial noise generation
    # ------------------------------------------------------------------

    def _generate_initial_noise(
        self,
        batch: Req,
        server_args: ServerArgs,
        h_lat: int,
        w_lat: int,
        seed,
    ) -> torch.Tensor:
        """Generate low-res noise, set batch.latent_ids, and return packed latent.

        FLUX.2 uses in_channels directly (no //4) because the spatial latent
        already incorporates the patchification channel expansion.
        """
        device = get_local_torch_device()
        C = server_args.pipeline_config.dit_config.arch_config.in_channels
        dtype = server_args.pipeline_config.get_latent_dtype(
            batch.prompt_embeds[0].dtype if batch.prompt_embeds else torch.bfloat16
        )
        noise_spatial = randn_tensor(
            (self._initial_noise_batch_size(batch), C, h_lat, w_lat),
            generator=self._get_initial_noise_generator(batch, seed, device),
            device=device,
            dtype=dtype,
        )

        # latent_ids are derived from the spatial shape; _prepare_denoising_loop
        # will read batch.latent_ids when building freqs_cis.
        latent_ids = _prepare_latent_ids(noise_spatial)
        batch.latent_ids = latent_ids.to(device)

        return _flux2_pack(noise_spatial)

    # ------------------------------------------------------------------
    # Resolution-change hook
    # ------------------------------------------------------------------

    def _on_resolution_change(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        new_h_pixel: int,
        new_w_pixel: int,
    ) -> None:
        """Update batch.latent_ids and freqs_cis for the new latent resolution.

        Called after the upsampled latent is stored in ctx.latents and
        batch.height/width are updated to new_h_pixel/new_w_pixel.
        """
        if ctx.cfg_policy is None:
            return

        latent_scale = self._latent_scale_factor(server_args)
        new_h_lat = new_h_pixel // latent_scale
        new_w_lat = new_w_pixel // latent_scale
        key = (new_h_lat, new_w_lat)

        # Update batch.latent_ids so that prepare_pos_cond_kwargs sees the
        # correct grid coordinates for the upsampled resolution.
        C = server_args.pipeline_config.dit_config.arch_config.in_channels
        dummy = ctx.latents.new_zeros(1, C, new_h_lat, new_w_lat)
        latent_ids = _prepare_latent_ids(dummy)
        batch.latent_ids = latent_ids.to(ctx.latents.device)

        if key not in self._freqs_cis_cache:
            new_pos_kwargs = self._prepare_resolution_pos_cond_kwargs(
                ctx, batch, server_args
            )
            freqs_cis = new_pos_kwargs.get("freqs_cis")
            if freqs_cis is not None:
                self._freqs_cis_cache[key] = freqs_cis

        cached = self._freqs_cis_cache.get(key)
        if cached is None:
            logger.warning(
                "freqs_cis not available for %dx%d latent; skipping update",
                new_h_lat,
                new_w_lat,
            )
            return

        self._update_cfg_branch_kwargs(ctx, {"freqs_cis": cached})

        logger.info(
            "Updated latent_ids and freqs_cis for %dx%d latent (pixel %dx%d) "
            "across %d branch(es)",
            new_h_lat,
            new_w_lat,
            new_h_pixel,
            new_w_pixel,
            len(ctx.cfg_policy.branches),
        )
