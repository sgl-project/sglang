# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image 2.1 progressive-resolution denoising stage.

Qwen-Image 2.1 shares the coarse-to-fine denoising loop with the other
progressive-resolution models, but its latent layout differs from v1:

  * VAE compresses 16× and the DiT consumes the latent directly
    (``in_channels == 64``), so packed latents are ``[B, H_lat*W_lat, 64]``
    with ``H_lat = H_pixel / 16``.  There is no 2×2 patch packing, which is
    why ``unpack_2x2_latent`` / ``pack_2x2_latent`` cannot be reused.
  * Positional state is carried by ``batch.extra["qwen21_positive"]`` (and the
    negative counterpart): ``layouts / condition_latents / prefix_caches``.
    Only ``layouts[i]["target_rope"]`` depends on the target resolution — the
    prefix grid (text + reference-image tokens) and its K/V caches do not.

The class is a multiple-inheritance composition:

    QwenImage21ProgressiveDenoisingStage
      ├─ QwenImage21DenoisingStage   → _predict_noise (BCG prefill handling)
      └─ ProgressiveDenoisingStage   → forward + resolution hooks

MRO: QwenImage21DenoisingStage → ProgressiveDenoisingStage → DenoisingStage,
so ``_predict_noise`` comes from 2.1 while ``forward`` and the hooks come from
the progressive base.  With ``progressive_mode == "fullres"`` the router runs
the standard ``QwenImage21DenoisingStage`` and this class is never used.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
    build_target_rope,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.qwen_image import (
    QWEN_IMAGE_SPECTRUM_A,
    QWEN_IMAGE_SPECTRUM_BETA,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class QwenImage21ProgressiveDenoisingStage(
    QwenImage21DenoisingStage, ProgressiveDenoisingStage
):
    """Qwen-Image 2.1 coarse-to-fine denoising with resolution rewinding."""

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
    # Latent layout (no patch packing)
    # ------------------------------------------------------------------

    def _latent_channels(self, server_args: ServerArgs) -> int:
        # 2.1 feeds the full 64-channel latent to the DiT (no 2×2 patchify).
        return server_args.pipeline_config.dit_config.arch_config.in_channels

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return latent.transpose(1, 2).reshape(latent.shape[0], -1, h_lat, w_lat)

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return x_spatial.flatten(2).transpose(1, 2)

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
        """Rebuild the target-grid RoPE for the new latent resolution.

        ``batch.extra["qwen21_positive"]`` (and ``qwen21_negative``) hold the
        precomputed layouts.  Everything except each layout's ``target_rope``
        is resolution independent: the prefix grid positions, the condition
        latents, and the prefix K/V caches stay valid.  ``CFGPolicy.build``
        shallow-copies the cond kwargs into ``branch.kwargs``, so the layout
        dicts are shared and updating them in place reaches the transformer.

        Both cond-kwarg dicts are scanned instead of gating on
        ``ctx.cfg_policy`` so that CFG-disabled runs still update the target
        grid.
        """
        latent_scale = self._latent_scale_factor(server_args)
        new_h_lat = new_h_pixel // latent_scale
        new_w_lat = new_w_pixel // latent_scale
        axes_dims = server_args.pipeline_config.dit_config.arch_config.axes_dims_rope

        updated = 0
        for cond_kwargs in (ctx.pos_cond_kwargs, ctx.neg_cond_kwargs):
            if not cond_kwargs:
                continue
            for layout in cond_kwargs.get("layouts") or ():
                layout["target_rope"] = build_target_rope(
                    position=layout["target_position"],
                    height=new_h_lat,
                    width=new_w_lat,
                    axes_dims=axes_dims,
                    device=layout["target_rope"].device,
                )
                updated += 1

        if updated == 0:
            logger.warning(
                "No qwen21 layouts found to update for pixel %dx%d; "
                "progressive stages may use stale target RoPE",
                new_h_pixel,
                new_w_pixel,
            )
            return

        logger.info(
            "Updated target_rope for %dx%d latent (pixel %dx%d) across %d layout(s)",
            new_h_lat,
            new_w_lat,
            new_h_pixel,
            new_w_pixel,
            updated,
        )
