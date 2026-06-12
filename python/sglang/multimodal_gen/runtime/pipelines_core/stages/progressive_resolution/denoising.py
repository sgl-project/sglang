# SPDX-License-Identifier: Apache-2.0
"""
Progressive-resolution denoising stage.

Extends DenoisingStage with a multi-stage coarse-to-fine denoising loop:
  Stage 1 runs at 1/(2^levels) of the full latent resolution.
  Between stages, the latent is upsampled via the spectral method selected by
  progressive_mode.
  Stage N runs at full resolution.

When progressive_mode == "fullres" (default), route the request to the standard
DenoisingStage instead of this stage.

Supported progressive_mode values
  "dct"         : DCT-II embed, IDCT upsample, no scheduler rewind
  "dct_rewind"  : DCT upsample + gamma scaling + scheduler sigma rewind (paper §3)

Extension hooks for model-specific subclasses
  _unpack_latent(latent, h_lat, w_lat)               → spatial [B, C, H, W]
  _repack_latent(x_spatial, h_lat, w_lat, batch)     → model-native latent
  _on_resolution_change(ctx, batch, srv, h_px, w_px) → update resolution-dep. state
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    refresh_context_on_dual_transformer,
    refresh_context_on_transformer,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.upsample import (
    apply_upsample,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

PROGRESSIVE_MODES = frozenset({"dct", "dct_rewind"})


def is_progressive_resolution_mode(mode: str | None) -> bool:
    return (mode or "fullres") in PROGRESSIVE_MODES


def unpack_2x2_latent(latent: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    batch_size, _seq_len, packed_channels = latent.shape
    spatial_channels = packed_channels // 4
    x = latent.view(batch_size, h_lat // 2, w_lat // 2, spatial_channels, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(batch_size, spatial_channels, h_lat, w_lat)


def pack_2x2_latent(x: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    batch_size, spatial_channels = x.shape[:2]
    x = x.view(batch_size, spatial_channels, h_lat // 2, 2, w_lat // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(batch_size, (h_lat // 2) * (w_lat // 2), spatial_channels * 4)


def _P_omega(w: float, A: float, beta: float) -> float:
    return A * abs(w) ** (-beta)


def _activation_time(P: float, delta: float) -> float:
    denom = P * (1.0 + P - delta)
    if denom <= 0 or delta >= 1.0 + P:
        raise ValueError(
            f"delta={delta} >= 1+P={1+P:.4f}; criterion trivially satisfied."
        )
    return 1.0 / (1.0 + math.sqrt(delta / denom))


def compute_stage_transitions(
    delta: float,
    n_levels: int,
    A: float,
    beta: float,
    H_lat: int,
    W_lat: int,
) -> dict[int, float]:
    stage_sigmas: dict[int, float] = {1: 1.0}
    num_stages = n_levels + 1
    for stage in range(2, num_stages + 1):
        H_prev = H_lat // (2 ** (num_stages - stage + 1))
        W_prev = W_lat // (2 ** (num_stages - stage + 1))
        w = min(H_prev, W_prev) // 2
        stage_sigmas[stage] = _activation_time(_P_omega(w, A, beta), delta)
    return stage_sigmas


def find_transition_steps(
    scheduler_sigmas: torch.Tensor,
    stage_sigmas: dict[int, float],
    n_steps: int,
) -> dict[int, int]:
    transition_steps: dict[int, int] = {}
    sigmas_list = scheduler_sigmas.cpu().tolist()
    for stage, threshold in stage_sigmas.items():
        if stage == 1:
            continue
        found = n_steps
        for step_index in range(n_steps):
            if sigmas_list[step_index] <= threshold:
                found = step_index
                break
        transition_steps[stage] = found
    return transition_steps


def reset_scheduler_at_step(scheduler: object, step_index: int) -> None:
    if hasattr(scheduler, "model_outputs"):
        solver_order = getattr(
            getattr(scheduler, "config", None),
            "solver_order",
            len(scheduler.model_outputs),
        )
        scheduler.model_outputs = [None] * solver_order
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "this_order"):
        scheduler.this_order = 0
    if hasattr(scheduler, "timestep_list"):
        solver_order = getattr(
            getattr(scheduler, "config", None),
            "solver_order",
            len(scheduler.timestep_list),
        )
        scheduler.timestep_list = [None] * solver_order
    scheduler._step_index = step_index


class ProgressiveDenoisingStageRouter(PipelineStage):
    def __init__(
        self,
        standard_stage: DenoisingStage,
        progressive_stage_factory: Callable[[], DenoisingStage],
    ) -> None:
        super().__init__()
        self.standard_stage = standard_stage
        self._progressive_stage_factory = progressive_stage_factory
        self._progressive_stage: DenoisingStage | None = None

    def _get_progressive_stage(self) -> DenoisingStage:
        if self._progressive_stage is None:
            stage = self._progressive_stage_factory()
            if self._component_residency_manager is not None:
                stage.set_component_residency_manager(self._component_residency_manager)
            if self._registered_stage_name is not None:
                stage.set_registered_stage_name(self._registered_stage_name)
            if self._profile_stage_name is not None:
                stage.set_profile_stage_name(self._profile_stage_name)
            self._progressive_stage = stage
        return self._progressive_stage

    @property
    def role_affinity(self):
        return RoleType.DENOISER

    @property
    def parallelism_type(self):
        return self.standard_stage.parallelism_type

    def set_component_residency_manager(self, manager) -> None:
        super().set_component_residency_manager(manager)
        self.standard_stage.set_component_residency_manager(manager)
        if self._progressive_stage is not None:
            self._progressive_stage.set_component_residency_manager(manager)

    def set_registered_stage_name(self, stage_name: str) -> None:
        super().set_registered_stage_name(stage_name)
        self.standard_stage.set_registered_stage_name(stage_name)
        if self._progressive_stage is not None:
            self._progressive_stage.set_registered_stage_name(stage_name)

    def set_profile_stage_name(self, stage_name: str) -> None:
        super().set_profile_stage_name(stage_name)
        self.standard_stage.set_profile_stage_name(stage_name)
        if self._progressive_stage is not None:
            self._progressive_stage.set_profile_stage_name(stage_name)

    def _active_profile_stage_name(self) -> str:
        # keep progressive requests under the canonical perf baseline stage name
        return "DenoisingStage"

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None):
        return self.standard_stage.component_uses(server_args, stage_name)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        mode = getattr(batch, "progressive_mode", "fullres") or "fullres"
        if is_progressive_resolution_mode(mode):
            return self._get_progressive_stage().forward(batch, server_args)
        if mode == "fullres":
            return self.standard_stage.forward(batch, server_args)
        raise ValueError(f"Unsupported progressive_mode: {mode!r}")


def _get_scm_preset() -> str | None:
    preset = envs.SGLANG_CACHE_DIT_SCM_PRESET
    return None if (preset is None or preset == "none") else preset


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

    def _latent_scale_factor(self, server_args: ServerArgs) -> int:
        """Pixel-to-latent scale factor used for spatial latent dimensions.

        Defaults to vae_scale_factor.  Models that apply an extra patchification
        step (e.g. FLUX.2 uses vae_scale_factor * 2) should override this.
        """
        return server_args.pipeline_config.vae_config.arch_config.vae_scale_factor

    def _spectrum_latent_dims(
        self, batch: Req, server_args: ServerArgs, H_lat: int, W_lat: int
    ) -> tuple[int, int]:
        """Physical spatial-latent dims for the Nyquist-frequency calculation.

        By default these equal the grid dims returned by _latent_scale_factor.
        Override for models (e.g. Ideogram 4) where patch packing causes the
        grid dimension to be smaller than the true spatial-latent dimension,
        so that the spectrum threshold is computed at the correct scale.
        """
        return H_lat, W_lat

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

    def _refresh_cache_dit_context(
        self, n_remaining: int, scm_preset: str | None
    ) -> None:
        """Refresh cache-dit activations and step counter at a stage transition.

        Override in model-specific subclasses that use more than one transformer
        (e.g. models with a separate unconditional branch).
        """
        if self.transformer_2 is not None:
            n_high = n_remaining // 2
            n_low = n_remaining - n_high
            refresh_context_on_dual_transformer(
                self.transformer,
                self.transformer_2,
                n_high,
                n_low,
                scm_preset=scm_preset,
            )
        else:
            refresh_context_on_transformer(
                self.transformer, n_remaining, scm_preset=scm_preset
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_resolution_pos_cond_kwargs(
        self,
        ctx: DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
    ) -> dict[str, Any]:
        rotary_emb = self._get_transformer_attr("rotary_emb")
        return server_args.pipeline_config.prepare_pos_cond_kwargs(
            batch,
            self.device,
            rotary_emb,
            dtype=ctx.target_dtype,
        )

    @staticmethod
    def _update_cfg_branch_kwargs(
        ctx: DenoisingContext,
        updates: dict[str, Any | None],
    ) -> None:
        assert ctx.cfg_policy is not None
        for branch in ctx.cfg_policy.branches:
            for name, value in updates.items():
                if value is not None and name in branch.kwargs:
                    branch.kwargs[name] = value

        for name, value in updates.items():
            if value is not None and name in ctx.pos_cond_kwargs:
                ctx.pos_cond_kwargs[name] = value

    @staticmethod
    def _get_seed(batch: Req) -> int:
        seeds = getattr(batch, "seeds", None)
        if seeds:
            return int(seeds[0])
        sp = getattr(batch, "sampling_params", None)
        seed = getattr(sp, "seed", None) if sp is not None else None
        return int(seed) if seed is not None else 42

    @staticmethod
    def _initial_noise_batch_size(batch: Req) -> int:
        try:
            return int(batch.batch_size)
        except AttributeError:
            prompt_embeds = getattr(batch, "prompt_embeds", None)
            if prompt_embeds:
                return int(prompt_embeds[0].shape[0])
            latents = getattr(batch, "latents", None)
            if latents is not None:
                return int(latents.shape[0])
            return 1

    def _get_seeds(self, batch: Req, seed: int | Sequence[int]) -> list[int]:
        batch_size = self._initial_noise_batch_size(batch)
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [int(item) for item in seed]
        else:
            batch_seeds = getattr(batch, "seeds", None)
            if batch_seeds:
                seeds = [int(item) for item in batch_seeds]
            else:
                seeds = [int(seed) + i for i in range(batch_size)]
        if len(seeds) != batch_size:
            raise ValueError(
                "progressive seeds length must match batch size: "
                f"{len(seeds)} vs {batch_size}"
            )
        return seeds

    def _get_initial_noise_generator(
        self, batch: Req, seed: int | Sequence[int], device: torch.device | str
    ):
        seeds = self._get_seeds(batch, seed)
        generators = [
            torch.Generator(device=device).manual_seed(seed) for seed in seeds
        ]
        if len(generators) == 1:
            return generators[0]
        return generators

    def _generate_initial_noise(
        self,
        batch: Req,
        server_args: ServerArgs,
        h_lat: int,
        w_lat: int,
        seed: int | Sequence[int],
    ) -> torch.Tensor:
        """Generate low-res initial noise and return in model-native format."""
        device = get_local_torch_device()
        C = server_args.pipeline_config.dit_config.arch_config.in_channels // 4
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

        if mode not in PROGRESSIVE_MODES:
            raise ValueError(
                "ProgressiveDenoisingStage requires progressive_mode to be "
                "'dct' or 'dct_rewind'. Route fullres requests to DenoisingStage."
            )

        if get_sp_world_size() > 1:
            raise RuntimeError(
                "Progressive resolution growing is not compatible with sequence "
                "parallelism. Disable --ulysses-degree / --ring-degree or set "
                "progressive_mode='fullres'."
            )

        levels = int(getattr(batch, "progressive_levels", 1))
        delta = float(getattr(batch, "progressive_delta", 0.01))
        seed = self._get_seed(batch)
        seeds = self._get_seeds(batch, seed)

        latent_scale = self._latent_scale_factor(server_args)
        H_lat = batch.height // latent_scale
        W_lat = batch.width // latent_scale
        downsample = 2**levels
        init_h_lat = H_lat // downsample
        init_w_lat = W_lat // downsample

        # Compute stage transitions from the power-law spectrum.
        # Use physical spatial-latent dims (may differ from grid dims for
        # patch-packed models like Ideogram 4).
        H_spec, W_spec = self._spectrum_latent_dims(batch, server_args, H_lat, W_lat)
        stage_sigmas = compute_stage_transitions(
            delta, levels, self._spectrum_A, self._spectrum_beta, H_spec, W_spec
        )
        num_stages = len(stage_sigmas)

        logger.info(
            "Progressive denoising: mode=%s levels=%d delta=%.3f initial=%dx%d",
            mode,
            levels,
            delta,
            init_h_lat,
            init_w_lat,
        )

        # ── Prepare initial state ──────────────────────────────────────────────
        # Save the full-res dimensions that were set by LatentPreparationStage.
        orig_h, orig_w = batch.height, batch.width

        # Override batch with low-res initial noise; _prepare_denoising_loop
        # reads batch.latents and batch.height/width to build freqs_cis.
        batch.height = init_h_lat * latent_scale
        batch.width = init_w_lat * latent_scale
        batch.latents = self._generate_initial_noise(
            batch, server_args, init_h_lat, init_w_lat, seed
        )
        batch.raw_latent_shape = batch.latents.shape

        ctx = self._prepare_denoising_loop(batch, server_args)
        self._before_denoising_loop(ctx, batch, server_args)

        scheduler = ctx.scheduler
        n_steps = int(batch.num_inference_steps)
        timesteps_cpu = ctx.timesteps.cpu()

        transition_steps = find_transition_steps(
            scheduler.sigmas, stage_sigmas, n_steps
        )
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
        # DenoisingStage.forward() wraps its denoising loop in torch.autocast;
        # we bypass that path, so we must apply the same context here.
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=ctx.target_dtype,
            enabled=ctx.autocast_enabled,
        ):
            for stage in range(1, num_stages + 1):
                stage_end = transition_steps.get(stage + 1, n_steps)

                logger.info(
                    "Stage %d/%d: %dx%d latent, steps [%d, %d)",
                    stage,
                    num_stages,
                    cur_h_lat,
                    cur_w_lat,
                    stage_start,
                    stage_end,
                )

                self._run_stage_steps(
                    ctx, batch, server_args, timesteps_cpu, stage_start, stage_end
                )

                if stage == num_stages:
                    break

                # ── Resolution transition ──────────────────────────────────────
                sigma_t = float(scheduler.sigmas[stage_end])
                upsample_seed = [item + stage * 10_000 for item in seeds]

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
                        sigma_t,
                        t_eff,
                        stage_end,
                    )
                else:
                    x_spatial_up = result

                new_h_lat = cur_h_lat * 2
                new_w_lat = cur_w_lat * 2
                ctx.latents = self._repack_latent(
                    x_spatial_up, new_h_lat, new_w_lat, batch, server_args
                )

                # Update batch dimensions and model-specific state
                new_h_pixel = new_h_lat * latent_scale
                new_w_pixel = new_w_lat * latent_scale
                batch.height = new_h_pixel
                batch.width = new_w_pixel
                self._on_resolution_change(
                    ctx, batch, server_args, new_h_pixel, new_w_pixel
                )

                reset_scheduler_at_step(scheduler, stage_end)

                # Refresh cache-dit context so its step counter and cached
                # activations start clean for the new resolution.  The coarse-
                # stage activations have the wrong shape and would corrupt the
                # residual-diff decision for the first full-res steps.
                if self._cache_dit_enabled:
                    n_remaining = n_steps - stage_end
                    self._refresh_cache_dit_context(n_remaining, _get_scm_preset())
                    logger.info(
                        "cache-dit context refreshed at stage transition "
                        "(step %d, %d steps remaining)",
                        stage_end,
                        n_remaining,
                    )

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
