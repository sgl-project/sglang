# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-specific progressive-resolution denoising stage.

Provides pack/unpack for FLUX's patchify format and updates the RoPE
positional embeddings (freqs_cis) when the latent resolution changes
between progressive stages.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
    pack_2x2_latent,
    unpack_2x2_latent,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Power-law spectrum constants for FLUX.1-dev VAE
# Fitted on Aesthetics-Train-V2 (105k images)
FLUX_SPECTRUM_A: float = 203.615097
FLUX_SPECTRUM_BETA: float = 1.915461


class FluxProgressiveDenoisingStage(ProgressiveDenoisingStage):
    """FLUX-specific progressive denoising stage.

    Handles:
    - FLUX patchify pack/unpack
    - freqs_cis (RoPE image position embeddings) update on resolution change
    - img_ids cache keyed on (h_lat, w_lat) to avoid redundant computation
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
        # Cache freqs_cis per latent resolution (h_lat, w_lat) to avoid
        # redundant rotary embedding recomputation between requests.
        self._freqs_cis_cache: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    # ------------------------------------------------------------------
    # Pack / Unpack overrides
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return unpack_2x2_latent(latent, h_lat, w_lat)

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return pack_2x2_latent(x_spatial, h_lat, w_lat)

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

        CFGBranch.kwargs is a shallow copy made at build() time; updating
        ctx.pos_cond_kwargs alone does NOT reach the transformer.  We must
        update branch.kwargs["freqs_cis"] directly in every branch.
        """
        if ctx.cfg_policy is None:
            return

        vae_scale_factor = (
            server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
        )
        new_h_lat = new_h_pixel // vae_scale_factor
        new_w_lat = new_w_pixel // vae_scale_factor
        key = (new_h_lat, new_w_lat)

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
            "Updated freqs_cis for %dx%d latent (pixel %dx%d) across %d branch(es)",
            new_h_lat,
            new_w_lat,
            new_h_pixel,
            new_w_pixel,
            len(ctx.cfg_policy.branches),
        )
