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
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Power-law spectrum constants for FLUX.1-dev VAE
# Fitted on Aesthetics-Train-V2 (105k images)
FLUX_SPECTRUM_A: float = 203.615097
FLUX_SPECTRUM_BETA: float = 1.915461


def _flux_unpack(latent: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Packed [B, S, 64] → spatial [B, 16, H_lat, W_lat]."""
    B = latent.shape[0]
    x = latent.view(B, h_lat // 2, w_lat // 2, 16, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 16, h_lat, w_lat)


def _flux_pack(x: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Spatial [B, 16, H_lat, W_lat] → packed [B, S, 64]."""
    B = x.shape[0]
    x = x.view(B, 16, h_lat // 2, 2, w_lat // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (h_lat // 2) * (w_lat // 2), 64)


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
        return _flux_unpack(latent, h_lat, w_lat)

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return _flux_pack(x_spatial, h_lat, w_lat)

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
            rotary_emb = self._get_transformer_attr("rotary_emb")
            new_pos_kwargs = server_args.pipeline_config.prepare_pos_cond_kwargs(
                batch,
                self.device,
                rotary_emb,
                dtype=ctx.target_dtype,
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

        # Update every CFG branch — this is what _predict_noise_with_cfg reads.
        for branch in ctx.cfg_policy.branches:
            if "freqs_cis" in branch.kwargs:
                branch.kwargs["freqs_cis"] = cached

        # Keep ctx.pos_cond_kwargs in sync (used for neg branch rebuild if needed).
        if "freqs_cis" in ctx.pos_cond_kwargs:
            ctx.pos_cond_kwargs["freqs_cis"] = cached

        logger.info(
            "Updated freqs_cis for %dx%d latent (pixel %dx%d) across %d branch(es)",
            new_h_lat,
            new_w_lat,
            new_h_pixel,
            new_w_pixel,
            len(ctx.cfg_policy.branches),
        )
