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

from sglang.multimodal_gen.configs.sample.ideogram import IDEOGRAM4_PRESETS
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    refresh_context_on_dual_transformer,
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
    get_schedule_for_resolution,
    make_step_intervals,
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


def _adapt_llm_features(
    llm_features: torch.Tensor,
    max_text_tokens: int,
    new_num_image_tokens: int,
) -> torch.Tensor:
    """Adapt LLM features to a different image-token count.

    The text encoder produces [text_tokens | image_tokens] features at full-res.
    For progressive denoising the grid size changes between stages, so the
    image portion must be resized.  Following the reference Ideogram inference
    code (run_experiment.py), we keep the text portion intact and use zeros for
    image positions at the new resolution — the unconditional transformer always
    receives zero image features anyway, so the model is designed to handle this.
    """
    existing_image_tokens = llm_features.shape[1] - max_text_tokens
    if existing_image_tokens == new_num_image_tokens:
        return llm_features
    B, _T, D = llm_features.shape
    text_feat = llm_features[:, :max_text_tokens]
    image_feat_new = llm_features.new_zeros(B, new_num_image_tokens, D)
    return torch.cat([text_feat, image_feat_new], dim=1)


def _ideogram4_unpack(latent: torch.Tensor, h_lat: int, w_lat: int) -> torch.Tensor:
    """Packed [B, grid_h*grid_w, C] → spatial [B, C, grid_h, grid_w] (row-major)."""
    B, _S, C = latent.shape
    return latent.permute(0, 2, 1).reshape(B, C, h_lat, w_lat)


def _ideogram4_pack(x: torch.Tensor) -> torch.Tensor:
    """Spatial [B, C, grid_h, grid_w] → packed [B, grid_h*grid_w, C] (row-major)."""
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).permute(0, 2, 1)


def _build_ideogram4_seq_tensors(
    data: dict,
    grid_h: int,
    grid_w: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build full-sequence position_ids, segment_ids, indicator for a grid.

    Keeps the text prefix from data unchanged and appends new image-token rows.
    Returns (position_ids, segment_ids, indicator), each shaped [B, T+S, ...].
    """
    max_text_tokens = data["max_text_tokens"]
    num_image_tokens = grid_h * grid_w
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
    )
    position_ids = torch.cat(
        [
            data["position_ids"][:, :max_text_tokens],
            image_pos.unsqueeze(0).expand(batch_size, -1, -1),
        ],
        dim=1,
    )
    segment_ids = torch.cat(
        [
            data["segment_ids"][:, :max_text_tokens],
            torch.ones(batch_size, num_image_tokens, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    indicator = torch.cat(
        [
            data["indicator"][:, :max_text_tokens],
            torch.full(
                (batch_size, num_image_tokens),
                OUTPUT_IMAGE_INDICATOR,
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=1,
    )
    return position_ids, segment_ids, indicator


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
            transformer_2=unconditional_transformer,
            scheduler=Ideogram4Scheduler(),
            pipeline=pipeline,
        )
        # ProgressiveDenoisingStage spectrum constants
        self._spectrum_A = IDEOGRAM_SPECTRUM_A
        self._spectrum_beta = IDEOGRAM_SPECTRUM_BETA
        # Ideogram4DenoisingStage extra transformer
        self.unconditional_transformer = self.transformer_2

    # ------------------------------------------------------------------
    # Latent scale factor
    # ------------------------------------------------------------------

    def _latent_scale_factor(self, server_args: ServerArgs) -> int:
        # pixel → latent-grid: divide by patch_size (2) × ae_scale_factor (8) = 16
        cfg = server_args.pipeline_config
        return cfg.patch_size * cfg.ae_scale_factor

    def _spectrum_latent_dims(self, batch, server_args, H_lat: int, W_lat: int):
        # Ideogram 4 packs 2×2 latent patches per grid token.  H_lat here is
        # the grid dimension (= image_h // 16 = 64 for 1024-px input); the
        # physical spatial-latent dimension is grid × patch_size (= 128).
        # The Nyquist calculation must use physical dims to match the reference.
        patch = server_args.pipeline_config.patch_size  # 2
        return H_lat * patch, W_lat * patch

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
        if ctx.cfg_policy is None:
            return

        cfg = server_args.pipeline_config
        patch = cfg.patch_size * cfg.ae_scale_factor  # 16
        grid_h = new_h_pixel // patch
        grid_w = new_w_pixel // patch
        num_image_tokens = grid_h * grid_w

        data = batch.extra["ideogram4"]
        max_text_tokens = data["max_text_tokens"]
        batch_size = ctx.latents.shape[0]
        device = ctx.latents.device

        new_position_ids, new_segment_ids, new_indicator = _build_ideogram4_seq_tensors(
            data, grid_h, grid_w, batch_size, device
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

        # Adapt LLM features to the new grid: keep text portion, use zeros for
        # image positions at the new resolution (mirrors run_experiment.py).
        full_res_llm = ctx.extra.get("ideogram4_full_res_llm_features")
        if full_res_llm is not None:
            batch.prompt_embeds[0] = _adapt_llm_features(
                full_res_llm, max_text_tokens, num_image_tokens
            )

        logger.info(
            "Updated position_ids / attn_masks / llm_features for %dx%d latent grid "
            "(%d image tokens) across %d batch item(s)",
            grid_h,
            grid_w,
            num_image_tokens,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Denoising-loop preparation
    # ------------------------------------------------------------------

    def _prepare_denoising_loop(
        self, batch: Req, server_args: ServerArgs
    ) -> DenoisingContext:
        # ProgressiveDenoisingStage.forward() overrides batch.height/width to
        # the initial low-res pixel dimensions before calling this method.
        # batch.extra["ideogram4"] was built by the text-encoding stage at the
        # full-res grid size, so we must resize position_ids / segment_ids /
        # indicator / num_image_tokens to match the low-res grid BEFORE calling
        # Ideogram4DenoisingStage._prepare_denoising_loop, which reads them to
        # build the negative tensors, attn masks, and neg_llm_features.
        cfg = server_args.pipeline_config
        patch = cfg.patch_size * cfg.ae_scale_factor  # 16
        grid_h = batch.height // patch
        grid_w = batch.width // patch
        num_image_tokens = grid_h * grid_w

        data = batch.extra["ideogram4"]
        max_text_tokens = data["max_text_tokens"]
        device = data["position_ids"].device
        batch_size = data["position_ids"].shape[0]

        data["position_ids"], data["segment_ids"], data["indicator"] = (
            _build_ideogram4_seq_tensors(data, grid_h, grid_w, batch_size, device)
        )
        data["num_image_tokens"] = num_image_tokens
        data["grid_h"] = grid_h
        data["grid_w"] = grid_w

        # The text encoder ran at full-res, so batch.prompt_embeds[0] has shape
        # [B, max_text_tokens + full_res_image_tokens, d_llm].  The DiT forward
        # expects llm_features and x=pos_z to share the same sequence length, so
        # we bilinearly resize the image portion of llm_features to the low-res grid.
        # ProgressiveDenoisingStage.forward() sets batch.height = init_h_pixel =
        # (H_lat // downsample) * latent_scale before calling us, so:
        #   orig_grid_{h,w} = grid_{h,w} * downsample  (downsample = 2^levels)
        full_res_llm = batch.prompt_embeds[0]
        batch.prompt_embeds[0] = _adapt_llm_features(
            full_res_llm, max_text_tokens, num_image_tokens
        )

        ctx = Ideogram4DenoisingStage._prepare_denoising_loop(self, batch, server_args)
        # Persist original LLM features so _on_resolution_change can restore
        # them (with re-adapted image-token count) when the latent is upsampled.
        ctx.extra["ideogram4_full_res_llm_features"] = full_res_llm

        # _prepare_denoising_loop (parent) reads batch.height = init_h_pixel (low-res)
        # and computes the schedule at that resolution.  The reference always uses the
        # TARGET full-resolution schedule throughout denoising, even during coarse steps.
        # Recompute schedule_values and schedule_deltas at the original full-res.
        levels = int(getattr(batch, "progressive_levels", 1))
        orig_h = batch.height * (2**levels)
        orig_w = batch.width * (2**levels)
        preset = getattr(batch, "preset", "V4_DEFAULT_20")
        preset_cfg = IDEOGRAM4_PRESETS[preset]
        full_res_schedule = get_schedule_for_resolution(
            (orig_h, orig_w),
            known_mean=float(preset_cfg["mu"]),
            std=float(preset_cfg["std"]),
        )
        device = ctx.extra["ideogram4_schedule_values"].device
        step_intervals = make_step_intervals(int(preset_cfg["num_steps"])).to(device)
        full_res_schedule_values = full_res_schedule(step_intervals)
        ctx.extra["ideogram4_schedule_values"] = full_res_schedule_values
        ctx.extra["ideogram4_schedule_deltas"] = (
            full_res_schedule_values[:-1] - full_res_schedule_values[1:]
        )

        # Expose a sigma_NOISE tensor for stage-transition logic (find_transition_steps)
        # and DWT upsample (sigma_t = sigmas[stage_end]).
        #
        # schedule_values convention: sigma_clean, index 0 = clean end (≈1),
        # index N = noisy end (≈0). Step step_index k uses internal index i = N-1-k,
        # so sigma_NOISE at step k = 1 - schedule_values[N-k].
        # flip(1 - schedule_values) gives sigmas[k] = 1 - schedule_values[N-k],
        # which decreases from ≈1 (noisy, step 0) to ≈0 (clean, step N).
        ctx.scheduler.sigmas = torch.flip(1.0 - full_res_schedule_values, [0])
        return ctx

    # ------------------------------------------------------------------
    # Denoising step override
    # ------------------------------------------------------------------

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        # Ideogram4DenoisingStage uses step.t_int as a step index [0..N-1] into
        # schedule_values.  set_timesteps(N) produces timesteps = [N-1, N-2, ..., 0],
        # so the correct mapping is t_int = N-1-step_index.
        #
        # In dct_rewind mode the progressive base patches
        # ctx.timesteps[transition_step] = t_eff * 1000 (FLUX-convention noise level),
        # which corrupts int(timesteps[step_index]) to ~950 and causes an IndexError.
        # We bypass ctx.timesteps entirely and reconstruct from step_index.
        num_steps = len(ctx.timesteps)
        step.t_int = num_steps - 1 - step.step_index
        Ideogram4DenoisingStage._run_denoising_step(self, ctx, step, batch, server_args)

    # ------------------------------------------------------------------
    # Cache-DiT refresh override
    # ------------------------------------------------------------------

    def _refresh_cache_dit_context(
        self,
        n_remaining: int,
        scm_preset: str | None,
        ctx: DenoisingContext | None = None,
    ) -> None:
        """Refresh active Cache-DiT transformer contexts after a stage transition."""
        # Recompute the full SCM config here so custom compute/cache bins stay
        # active after a progressive-resolution stage transition.
        skip_unconditional = bool(
            ctx is not None and ctx.extra.get("ideogram4_skip_unconditional", False)
        )

        _, scm_policy, steps_computation_mask, steps_computation_mask_2 = (
            self._cache_dit_scm_masks(
                n_remaining, None if skip_unconditional else n_remaining
            )
        )
        if skip_unconditional:
            refresh_context_on_transformer(
                self.transformer,
                n_remaining,
                steps_computation_mask=steps_computation_mask,
                steps_computation_policy=scm_policy,
            )
            return

        refresh_context_on_dual_transformer(
            self.transformer,
            self.unconditional_transformer,
            n_remaining,
            n_remaining,
            steps_computation_mask=steps_computation_mask,
            steps_computation_mask_2=steps_computation_mask_2,
            steps_computation_policy=scm_policy,
        )
