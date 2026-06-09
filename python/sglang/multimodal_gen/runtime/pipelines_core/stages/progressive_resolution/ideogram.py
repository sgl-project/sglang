# SPDX-License-Identifier: Apache-2.0
"""
Ideogram 4 progressive-resolution denoising stage.

Ideogram 4 latent layout:
  packed:  [B, grid_h * grid_w, in_channels]  (row-major, same as FLUX.2)
  spatial: [B, in_channels, grid_h, grid_w]

where grid_h = height // (patch_size * ae_scale_factor)  =  height // 16
      grid_w = width  // (patch_size * ae_scale_factor)  =  width  // 16

On each stage transition _on_resolution_change rebuilds the position_ids,
segment_ids, indicator, attention masks, and the zero neg_llm_features tensor
that Ideogram4DenoisingStage reads from batch.extra["ideogram4"] and ctx.extra.
"""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    refresh_context_on_transformer,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention import build_varlen_mask_meta
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ideogram import (
    IMAGE_POSITION_OFFSET,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4DenoisingStage,
    Ideogram4Scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Power-law spectrum constants.
# Using FLUX.1-dev VAE values as a placeholder until Ideogram-specific
# coefficients are fitted on a representative latent dataset.
IDEOGRAM_SPECTRUM_A: float = 203.615097
IDEOGRAM_SPECTRUM_BETA: float = 1.915461


def _ideogram4_unpack(latent: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Packed [B, grid_h*grid_w, C] → spatial [B, C, grid_h, grid_w] (row-major)."""
    B, _S, C = latent.shape
    return latent.permute(0, 2, 1).reshape(B, C, h_lat, w_lat)


def _ideogram4_pack(x: torch.Tensor) -> torch.Tensor:
    """Spatial [B, C, grid_h, grid_w] → packed [B, grid_h*grid_w, C] (row-major)."""
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).permute(0, 2, 1)


class Ideogram4ProgressiveDenoisingStage(
    ProgressiveDenoisingStage, Ideogram4DenoisingStage
):
    """Progressive-resolution denoising stage for Ideogram 4.

    Inherits the progressive loop from ProgressiveDenoisingStage and the
    Ideogram-specific dual-transformer forward pass from Ideogram4DenoisingStage
    via MRO.  __init__ calls DenoisingStage directly to avoid cooperative-init
    incompatibility between the two parent signatures.

    MRO for method resolution:
      Ideogram4ProgressiveDenoisingStage
        → ProgressiveDenoisingStage   (forward / _run_stage_steps / latent hooks)
        → Ideogram4DenoisingStage     (_prepare_denoising_loop / _run_denoising_step)
        → DenoisingStage              (shared infrastructure)
        → PipelineStage
    """

    def __init__(
        self,
        transformer,
        unconditional_transformer,
        pipeline=None,
    ) -> None:
        # Bypass cooperative __init__: the two parents have incompatible
        # signatures (ProgressiveDenoisingStage takes scheduler/spectrum args;
        # Ideogram4DenoisingStage takes unconditional_transformer).
        # Initialise DenoisingStage — the common ancestor — directly, then
        # set the attributes each parent __init__ would have added.
        DenoisingStage.__init__(
            self,
            transformer=transformer,
            scheduler=Ideogram4Scheduler(),
            pipeline=pipeline,
        )
        # ProgressiveDenoisingStage spectrum constants
        self._spectrum_A = IDEOGRAM_SPECTRUM_A
        self._spectrum_beta = IDEOGRAM_SPECTRUM_BETA
        # Ideogram4DenoisingStage extra transformer
        self.unconditional_transformer = unconditional_transformer
        self._maybe_enable_torch_compile(self.unconditional_transformer)

    # ------------------------------------------------------------------
    # Latent scale factor
    # ------------------------------------------------------------------

    def _latent_scale_factor(self, server_args: ServerArgs) -> int:
        # pixel → latent-grid: divide by patch_size (2) × ae_scale_factor (8) = 16
        cfg = server_args.pipeline_config
        return cfg.patch_size * cfg.ae_scale_factor

    # ------------------------------------------------------------------
    # Pack / Unpack
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return _ideogram4_unpack(latent, h_lat, w_lat)

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return _ideogram4_pack(x_spatial)

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
        """Generate low-res packed noise for the initial progressive stage.

        Uses in_channels directly (no //4) because the spatial latent already
        incorporates the patchification channel expansion (same as FLUX.2).
        Ideogram denoising steps cast latents to fp32 internally, so we
        generate fp32 noise to match.
        """
        device = get_local_torch_device()
        C = server_args.pipeline_config.dit_config.arch_config.in_channels
        noise_spatial = randn_tensor(
            (self._initial_noise_batch_size(batch), C, h_lat, w_lat),
            generator=self._get_initial_noise_generator(batch, seed, device),
            device=device,
            dtype=torch.float32,
        )
        return _ideogram4_pack(noise_spatial)

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
        """Rebuild Ideogram position IDs and attention masks for the new grid.

        Called after ctx.latents and batch.height/width are already updated to
        the upsampled resolution.  Patches batch.extra["ideogram4"] and ctx.extra
        in-place so that _run_denoising_step sees correctly-sized tensors.
        """
        cfg = server_args.pipeline_config
        patch = cfg.patch_size * cfg.ae_scale_factor  # 16
        grid_h = new_h_pixel // patch
        grid_w = new_w_pixel // patch
        num_image_tokens = grid_h * grid_w

        data = batch.extra["ideogram4"]
        max_text_tokens = data["max_text_tokens"]
        batch_size = ctx.latents.shape[0]
        device = ctx.latents.device

        # Build new image position IDs for the upsampled grid.
        h_idx = (
            torch.arange(grid_h, device=device)
            .view(-1, 1)
            .expand(grid_h, grid_w)
            .reshape(-1)
        )
        w_idx = (
            torch.arange(grid_w, device=device)
            .view(1, -1)
            .expand(grid_h, grid_w)
            .reshape(-1)
        )
        image_pos = (
            torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1)
            + IMAGE_POSITION_OFFSET
        )  # [S_new, 3]

        # Grow full-sequence tensors: the text portion (first max_text_tokens
        # positions) is invariant; only the image tail changes size.
        new_position_ids = torch.cat(
            [
                data["position_ids"][:, :max_text_tokens],  # [B, T, 3]
                image_pos.unsqueeze(0).expand(batch_size, -1, -1),  # [B, S_new, 3]
            ],
            dim=1,
        )
        new_segment_ids = torch.cat(
            [
                data["segment_ids"][:, :max_text_tokens],  # [B, T]
                torch.ones(
                    batch_size, num_image_tokens, dtype=torch.long, device=device
                ),
            ],
            dim=1,
        )
        new_indicator = torch.cat(
            [
                data["indicator"][:, :max_text_tokens],  # [B, T]
                torch.full(
                    (batch_size, num_image_tokens),
                    OUTPUT_IMAGE_INDICATOR,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )
        new_attn_mask = new_segment_ids > 0

        # Negative (unconditional) tensors span the image tokens only.
        neg_position_ids = new_position_ids[:, max_text_tokens:]
        neg_segment_ids = new_segment_ids[:, max_text_tokens:]
        neg_indicator = new_indicator[:, max_text_tokens:]
        neg_attn_mask = neg_segment_ids > 0

        llm_dim = ctx.extra["ideogram4_neg_llm_features"].shape[-1]
        neg_llm_features = ctx.extra["ideogram4_neg_llm_features"].new_zeros(
            batch_size, num_image_tokens, llm_dim
        )

        # Update batch.extra["ideogram4"] in-place.
        data["position_ids"] = new_position_ids
        data["segment_ids"] = new_segment_ids
        data["indicator"] = new_indicator
        data["num_image_tokens"] = num_image_tokens
        data["grid_h"] = grid_h
        data["grid_w"] = grid_w

        # Update ctx.extra in-place.
        ctx.extra.update(
            {
                "ideogram4_attn_mask": new_attn_mask,
                "ideogram4_attn_mask_meta": build_varlen_mask_meta(new_attn_mask),
                "ideogram4_neg_position_ids": neg_position_ids,
                "ideogram4_neg_segment_ids": neg_segment_ids,
                "ideogram4_neg_indicator": neg_indicator,
                "ideogram4_neg_attn_mask": neg_attn_mask,
                "ideogram4_neg_attn_mask_meta": build_varlen_mask_meta(neg_attn_mask),
                "ideogram4_neg_llm_features": neg_llm_features,
            }
        )

        logger.info(
            "Updated position_ids / attn_masks for %dx%d latent grid "
            "(%d image tokens) across %d batch item(s)",
            grid_h,
            grid_w,
            num_image_tokens,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Cache-DiT refresh override
    # ------------------------------------------------------------------

    def _refresh_cache_dit_context(
        self, n_remaining: int, scm_preset: str | None
    ) -> None:
        """Refresh both conditional and unconditional transformers."""
        refresh_context_on_transformer(
            self.transformer, n_remaining, scm_preset=scm_preset
        )
        refresh_context_on_transformer(
            self.unconditional_transformer, n_remaining, scm_preset=scm_preset
        )
