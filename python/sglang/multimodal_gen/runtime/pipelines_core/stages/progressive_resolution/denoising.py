# SPDX-License-Identifier: Apache-2.0
"""
Progressive-resolution denoising stage.

Extends DenoisingStage with a multi-stage coarse-to-fine denoising loop:
  Stage 1 runs at 1/(2^levels) of the full latent resolution.
  Between stages, the latent is upsampled via the spectral method selected by
  progressive_mode (default: "dct_rewind").
  Stage N runs at full resolution.

When progressive_mode == "fullres" (default), the stage delegates entirely to
DenoisingStage.forward() — existing behaviour is preserved unchanged.

Supported progressive_mode values
  "dct"         : DCT-II embed, IDCT upsample, no scheduler rewind
  "dct_rewind"  : DCT upsample + gamma scaling + scheduler sigma rewind (paper §3)

Extension hooks for model-specific subclasses
  _unpack_latent(latent, h_lat, w_lat)               → spatial [B, C, H, W]
  _repack_latent(x_spatial, h_lat, w_lat, batch)     → model-native latent
  _on_resolution_change(ctx, batch, srv, h_px, w_px) → update resolution-dep. state
"""

from __future__ import annotations

import time
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.scheduler_utils import (
    compute_stage_transitions,
    find_transition_steps,
    reset_scheduler_at_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.upsample import (
    apply_upsample,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_PROGRESSIVE_MODES = frozenset({"dct", "dct_rewind"})


class ProgressiveDenoisingStage(DenoisingStage):
    """DenoisingStage extended with progressive resolution growing.

    Subclass and override _unpack_latent / _repack_latent / _on_resolution_change
    for model-specific latent packing and positional-embedding updates.

    spectrum_A and spectrum_beta are the fitted power-law coefficients for
    P(ω) = A * |ω|^{-β} describing the latent frequency spectrum.
    """

    def __init__(
        self,
        transformer,
        scheduler,
        pipeline=None,
        transformer_2=None,
        vae=None,
        spectrum_A: float = 1.0,
        spectrum_beta: float = 2.0,
    ) -> None:
        super().__init__(transformer, scheduler, pipeline, transformer_2, vae)
        self._spectrum_A = spectrum_A
        self._spectrum_beta = spectrum_beta

    # ------------------------------------------------------------------
    # Extension hooks (override in model-specific subclasses)
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        """Convert model-native latent → spatial [B, C, H_lat, W_lat]."""
        return latent

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        """Convert spatial [B, C, H_lat, W_lat] → model-native latent."""
        return x_spatial

    def _on_resolution_change(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        new_h_pixel: int,
        new_w_pixel: int,
    ) -> None:
        """Called after each stage transition. Update resolution-dependent state."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_seed(batch: Req) -> int:
        seeds = getattr(batch, "seeds", None)
        if seeds:
            return int(seeds[0])
        sp = getattr(batch, "sampling_params", None)
        seed = getattr(sp, "seed", None) if sp is not None else None
        return int(seed) if seed is not None else 42

    def _generate_initial_noise(
        self,
        batch: Req,
        server_args: ServerArgs,
        h_lat: int,
        w_lat: int,
        seed: int,
    ) -> torch.Tensor:
        """Generate low-res initial noise and return in model-native format."""
        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device

        device = get_local_torch_device()
        C = server_args.pipeline_config.dit_config.arch_config.in_channels // 4
        dtype = server_args.pipeline_config.get_latent_dtype(
            batch.prompt_embeds[0].dtype if batch.prompt_embeds else torch.bfloat16
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        noise_spatial = torch.randn(
            1, C, h_lat, w_lat, generator=gen, dtype=dtype
        ).to(device)
        return self._repack_latent(noise_spatial, h_lat, w_lat, batch, server_args)

    def _run_stage_steps(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        timesteps_cpu: torch.Tensor,
        start_step: int,
        end_step: int,
    ) -> None:
        """Run denoising steps [start_step, end_step) using the parent infrastructure."""
        for step_index in range(start_step, end_step):
            t_host = timesteps_cpu[step_index]
            step = self._prepare_step_state(
                ctx, batch, server_args, step_index, t_host, timesteps_cpu
            )
            self._run_denoising_step(ctx, step, batch, server_args)

    # ------------------------------------------------------------------
    # Progressive forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        mode = getattr(batch, "progressive_mode", "fullres") or "fullres"

        # Existing path — no change in behavior
        if mode not in _PROGRESSIVE_MODES:
            return super().forward(batch, server_args)

        if get_sp_world_size() > 1:
            raise RuntimeError(
                "Progressive resolution growing is not compatible with sequence "
                "parallelism. Disable --ulysses-degree / --ring-degree or set "
                "progressive_mode='fullres'."
            )

        levels = int(getattr(batch, "progressive_levels", 1))
        delta = float(getattr(batch, "progressive_delta", 0.01))
        seed = self._get_seed(batch)

        vae_scale_factor = (
            server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
        )
        H_lat = batch.height // vae_scale_factor
        W_lat = batch.width // vae_scale_factor
        downsample = 2**levels
        init_h_lat = H_lat // downsample
        init_w_lat = W_lat // downsample

        # Compute stage transitions from the power-law spectrum
        stage_sigmas = compute_stage_transitions(
            delta, levels, self._spectrum_A, self._spectrum_beta, H_lat, W_lat
        )
        num_stages = len(stage_sigmas)

        logger.info(
            "Progressive denoising: mode=%s levels=%d delta=%.3f initial=%dx%d",
            mode, levels, delta, init_h_lat, init_w_lat,
        )

        # ── Prepare initial state ──────────────────────────────────────────────
        # Save the full-res dimensions that were set by LatentPreparationStage.
        orig_h, orig_w = batch.height, batch.width

        # Override batch with low-res initial noise; _prepare_denoising_loop
        # reads batch.latents and batch.height/width to build freqs_cis.
        batch.height = init_h_lat * vae_scale_factor
        batch.width = init_w_lat * vae_scale_factor
        batch.latents = self._generate_initial_noise(
            batch, server_args, init_h_lat, init_w_lat, seed
        )
        batch.raw_latent_shape = batch.latents.shape

        ctx = self._prepare_denoising_loop(batch, server_args)
        self._before_denoising_loop(ctx, batch, server_args)

        scheduler = ctx.scheduler
        n_steps = int(batch.num_inference_steps)
        timesteps_cpu = ctx.timesteps.cpu()

        transition_steps = find_transition_steps(scheduler.sigmas, stage_sigmas, n_steps)
        rewind = mode.endswith("_rewind")

        # For rewind mode we patch scheduler.sigmas/timesteps and ctx.timesteps
        # in-place at transition points.  The scheduler tensors may be inference
        # tensors (created inside torch.inference_mode), so clone them once now
        # to obtain normal mutable tensors.  timesteps_cpu is already a fresh
        # CPU tensor from .cpu(), so no clone is needed there.
        if rewind:
            scheduler.sigmas = scheduler.sigmas.clone()
            scheduler.timesteps = scheduler.timesteps.clone()
            ctx.timesteps = ctx.timesteps.clone()

        denoising_start = time.time()
        stage_start = 0
        cur_h_lat = init_h_lat
        cur_w_lat = init_w_lat

        # ── Stage loop ────────────────────────────────────────────────────────
        for stage in range(1, num_stages + 1):
            stage_end = transition_steps.get(stage + 1, n_steps)

            logger.info(
                "Stage %d/%d: %dx%d latent, steps [%d, %d)",
                stage, num_stages, cur_h_lat, cur_w_lat, stage_start, stage_end,
            )

            self._run_stage_steps(
                ctx, batch, server_args, timesteps_cpu, stage_start, stage_end
            )

            if stage == num_stages:
                break

            # ── Resolution transition ──────────────────────────────────────
            sigma_t = float(scheduler.sigmas[stage_end])
            upsample_seed = seed + stage * 10_000

            # Unpack → spatial, upsample, repack
            x_spatial = self._unpack_latent(ctx.latents, cur_h_lat, cur_w_lat)

            result = apply_upsample(x_spatial, sigma_t, upsample_seed, mode)

            if rewind:
                x_spatial_up, t_eff = result
                # Patch scheduler sigma/timestep at transition point for rewind
                scheduler.sigmas[stage_end] = t_eff
                scheduler.timesteps[stage_end] = t_eff * 1000
                ctx.timesteps[stage_end] = t_eff * 1000
                timesteps_cpu[stage_end] = t_eff * 1000
                logger.info(
                    "  rewind: sigma=%.4f → t_eff=%.4f at step %d",
                    sigma_t, t_eff, stage_end,
                )
            else:
                x_spatial_up = result

            new_h_lat = cur_h_lat * 2
            new_w_lat = cur_w_lat * 2
            ctx.latents = self._repack_latent(
                x_spatial_up, new_h_lat, new_w_lat, batch, server_args
            )

            # Update batch dimensions and model-specific state
            new_h_pixel = new_h_lat * vae_scale_factor
            new_w_pixel = new_w_lat * vae_scale_factor
            batch.height = new_h_pixel
            batch.width = new_w_pixel
            self._on_resolution_change(
                ctx, batch, server_args, new_h_pixel, new_w_pixel
            )

            reset_scheduler_at_step(scheduler, stage_end)
            cur_h_lat = new_h_lat
            cur_w_lat = new_w_lat
            stage_start = stage_end

        denoising_end = time.time()
        if not ctx.is_warmup:
            logger.info(
                "Progressive denoising done in %.2fs (avg %.4fs/step)",
                denoising_end - denoising_start,
                (denoising_end - denoising_start) / max(n_steps, 1),
            )

        # raw_latent_shape was set to the low-res initial noise shape when we
        # replaced batch.latents.  Update it to the final full-res latent so
        # maybe_unpad_latents in post_denoising_loop does not truncate tokens.
        batch.raw_latent_shape = ctx.latents.shape

        # Ensure batch resolution reflects the final full-res output
        batch.height = orig_h
        batch.width = orig_w

        self._finish_active_component_use()
        self._finalize_denoising_loop(ctx, batch, server_args)
        return batch
