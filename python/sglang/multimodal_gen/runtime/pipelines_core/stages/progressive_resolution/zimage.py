# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-specific progressive-resolution denoising stage.

Provides pack/unpack for Z-Image's 5-D latent format [B, C, F, H, W] and updates
the RoPE positional embeddings (freqs_cis) when the latent resolution changes
between progressive stages.
"""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor

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

# Power-law spectrum constants for Z-Image.
# Z-Image uses the same VAE as FLUX.1-dev (FluxVAEConfig), so the spectrum
# constants fitted on Aesthetics-Train-V2 (105k images) apply directly.
ZIMAGE_SPECTRUM_A: float = 203.615097
ZIMAGE_SPECTRUM_BETA: float = 1.915461


def _zimage_unpack(latent: torch.Tensor) -> torch.Tensor:
    """5-D latent [B, C, 1, H_lat, W_lat] → spatial [B, C, H_lat, W_lat]."""
    return latent.squeeze(2)


def _zimage_repack(x: torch.Tensor) -> torch.Tensor:
    """Spatial [B, C, H_lat, W_lat] → 5-D latent [B, C, 1, H_lat, W_lat]."""
    return x.unsqueeze(2)


class ZImageProgressiveDenoisingStage(ProgressiveDenoisingStage):
    """Z-Image-specific progressive denoising stage.

    Handles:
    - Z-Image 5-D latent pack/unpack [B, C, 1, H, W] ↔ [B, C, H, W]
    - freqs_cis (RoPE caption + image position embeddings) update on resolution change
    """

    def __init__(self, transformer, scheduler, pipeline=None, vae=None) -> None:
        super().__init__(
            transformer,
            scheduler,
            pipeline=pipeline,
            vae=vae,
            spectrum_A=ZIMAGE_SPECTRUM_A,
            spectrum_beta=ZIMAGE_SPECTRUM_BETA,
        )

    # ------------------------------------------------------------------
    # Initial noise
    # ------------------------------------------------------------------

    def _generate_initial_noise(
        self,
        batch: Req,
        server_args: ServerArgs,
        h_lat: int,
        w_lat: int,
        seed,
    ) -> torch.Tensor:
        """Generate low-res initial noise in Z-Image's native 5-D format [B, C, 1, H, W].

        The base class uses in_channels // 4 which is correct for FLUX (64 // 4 = 16),
        but Z-Image's in_channels = 16 already refers to the spatial channel count.
        We use it directly and return 5-D via _repack_latent.
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
        return self._repack_latent(noise_spatial, h_lat, w_lat, batch, server_args)

    # ------------------------------------------------------------------
    # Pack / Unpack overrides
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return _zimage_unpack(latent)

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return _zimage_repack(x_spatial)

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
        """Recompute freqs_cis for the new resolution and update all CFG branches.

        Z-Image freqs_cis is a (cap_freqs_cis, x_freqs_cis) tuple.  The image
        position offsets depend on caption length, so the full tuple must be
        recomputed rather than only the image portion.

        batch.height / batch.width are already updated to new_h_pixel / new_w_pixel
        by the time this hook is called, so prepare_pos_cond_kwargs uses the
        correct new resolution automatically.

        CFGBranch.kwargs is a shallow copy made at build() time; updating
        ctx.pos_cond_kwargs alone does NOT reach the transformer.  We must
        update branch.kwargs["freqs_cis"] directly in every branch.
        """
        if ctx.cfg_policy is None:
            return

        new_pos_kwargs = self._prepare_resolution_pos_cond_kwargs(
            ctx, batch, server_args
        )
        freqs_cis = new_pos_kwargs.get("freqs_cis")
        if freqs_cis is None:
            logger.warning(
                "freqs_cis not available for pixel %dx%d; skipping update",
                new_h_pixel,
                new_w_pixel,
            )
            return

        vae_scale_factor = (
            server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
        )
        new_h_lat = new_h_pixel // vae_scale_factor
        new_w_lat = new_w_pixel // vae_scale_factor

        self._update_cfg_branch_kwargs(ctx, {"freqs_cis": freqs_cis})

        logger.info(
            "Updated freqs_cis for %dx%d latent (pixel %dx%d) across %d branch(es)",
            new_h_lat,
            new_w_lat,
            new_h_pixel,
            new_w_pixel,
            len(ctx.cfg_policy.branches),
        )
