# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-specific progressive-resolution denoising stage.

Provides pack/unpack for Qwen-Image's patchify format and updates the RoPE
positional embeddings (freqs_cis) and image shape metadata (img_shapes) when
the latent resolution changes between progressive stages.

Qwen-Image uses the same patchify convention as FLUX.1-dev:
  - in_channels = 64, spatial channels C = in_channels // 4 = 16
  - 2×2 patchification → packed [B, S, 64] where S = (H_lat/2) * (W_lat/2)

The Qwen DiT forward() uses both ``freqs_cis`` (RoPE) and ``img_shapes``
(for build_modulate_index), so _on_resolution_change updates both.

Extension points (from ProgressiveDenoisingStage base class):
  _unpack_latent   : [B, S, 64] → [B, 16, H_lat, W_lat]
  _repack_latent   : [B, 16, H_lat, W_lat] → [B, S, 64]
  _on_resolution_change : update freqs_cis + img_shapes in every CFG branch
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

# Power-law spectrum constants P(ω) = A·|ω|^{-β} for Qwen-Image VAE latents.
# TODO: fit these from Qwen-Image VAE latent statistics on a representative
#       dataset (e.g. Aesthetics-Train-V2). Using FLUX.1-dev fitted values as
#       a reasonable starting point — both VAEs compress 2-D images into a
#       16-channel spatial latent with similar frequency roll-off.
QWEN_IMAGE_SPECTRUM_A: float = 203.615097
QWEN_IMAGE_SPECTRUM_BETA: float = 1.915461


def _qwen_image_unpack(latent: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Packed [B, S, 64] → spatial [B, 16, H_lat, W_lat].

    Inverse of _pack_latents() in QwenImagePipelineConfig.
    Identical to FLUX unpack: in_channels=64, C=16, 2×2 patches.
    """
    B = latent.shape[0]
    x = latent.view(B, h_lat // 2, w_lat // 2, 16, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 16, h_lat, w_lat)


def _qwen_image_pack(x: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Spatial [B, 16, H_lat, W_lat] → packed [B, S, 64].

    Matches _pack_latents() in QwenImagePipelineConfig with
    num_channels_latents=16 and 2×2 patchification.
    """
    B = x.shape[0]
    x = x.view(B, 16, h_lat // 2, 2, w_lat // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (h_lat // 2) * (w_lat // 2), 64)


class QwenImageProgressiveDenoisingStage(ProgressiveDenoisingStage):
    """Qwen-Image progressive denoising stage.

    Inherits the full coarse-to-fine denoising loop from
    ProgressiveDenoisingStage and overrides three model-specific hooks:

    * _unpack_latent / _repack_latent — Qwen's 2×2 patchify format
    * _on_resolution_change           — update freqs_cis AND img_shapes in
                                        every CFG branch so the Qwen DiT's
                                        build_modulate_index sees the right
                                        spatial dimensions at each stage

    When progressive_mode == "fullres" (the default) the stage delegates
    entirely to DenoisingStage.forward(), so existing non-progressive
    requests are completely unaffected.
    """

    def __init__(
        self,
        transformer,
        scheduler,
        pipeline=None,
        vae=None,
    ) -> None:
        super().__init__(
            transformer,
            scheduler,
            pipeline=pipeline,
            vae=vae,
            spectrum_A=QWEN_IMAGE_SPECTRUM_A,
            spectrum_beta=QWEN_IMAGE_SPECTRUM_BETA,
        )

    # ------------------------------------------------------------------
    # Pack / Unpack overrides
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return _qwen_image_unpack(latent, h_lat, w_lat)

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return _qwen_image_pack(x_spatial, h_lat, w_lat)

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
        """Update freqs_cis and img_shapes for the new latent resolution.

        batch.height / batch.width are already set to new_h_pixel / new_w_pixel
        by the base class before this hook fires, so prepare_pos_cond_kwargs
        computes the correct RoPE cache and img_shapes for the new resolution.

        Both freqs_cis (RoPE) and img_shapes (build_modulate_index) are updated
        in every CFG branch because CFGBranch.kwargs is a shallow copy made at
        build() time — updating ctx.pos_cond_kwargs alone does not reach the
        transformer.
        """
        if ctx.cfg_policy is None:
            return

        rotary_emb = self._get_transformer_attr("rotary_emb")
        new_pos_kwargs = server_args.pipeline_config.prepare_pos_cond_kwargs(
            batch,
            self.device,
            rotary_emb,
            dtype=ctx.target_dtype,
        )
        freqs_cis = new_pos_kwargs.get("freqs_cis")
        img_shapes = new_pos_kwargs.get("img_shapes")

        if freqs_cis is None:
            logger.warning(
                "freqs_cis not available for pixel %dx%d; skipping update",
                new_h_pixel,
                new_w_pixel,
            )
            return

        for branch in ctx.cfg_policy.branches:
            if "freqs_cis" in branch.kwargs:
                branch.kwargs["freqs_cis"] = freqs_cis
            if img_shapes is not None and "img_shapes" in branch.kwargs:
                branch.kwargs["img_shapes"] = img_shapes

        if "freqs_cis" in ctx.pos_cond_kwargs:
            ctx.pos_cond_kwargs["freqs_cis"] = freqs_cis
        if img_shapes is not None and "img_shapes" in ctx.pos_cond_kwargs:
            ctx.pos_cond_kwargs["img_shapes"] = img_shapes

        vae_scale_factor = (
            server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
        )
        new_h_lat = new_h_pixel // vae_scale_factor
        new_w_lat = new_w_pixel // vae_scale_factor

        logger.info(
            "Updated freqs_cis + img_shapes for %dx%d latent (pixel %dx%d)"
            " across %d branch(es)",
            new_h_lat,
            new_w_lat,
            new_h_pixel,
            new_w_pixel,
            len(ctx.cfg_policy.branches),
        )
