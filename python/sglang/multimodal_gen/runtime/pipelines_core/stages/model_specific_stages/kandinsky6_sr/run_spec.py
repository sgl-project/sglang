# SPDX-License-Identifier: Apache-2.0
"""SR request state and sampler resolution.

batch.extra contains only CPU tensors and serializable data, never model objects."""

from typing import Any

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRArchConfig,
)
from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
    PiflowScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    DitSpec,
    SamplingSpec,
    module_dtype,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SR_VIDEO_KEY = (
    "kandinsky6_sr_video"  # [T, 3, H, W] uint8 clip (after pre-upscale + pad)
)
SR_TILING_SCALE_KEY = "kandinsky6_sr_tiling_scale"  # int tile scale (2 or 4)
SR_LR_LATENT_KEY = "kandinsky6_sr_lr_latent"  # [T', C, h, w] fp32 (LU path)
SR_PLAN_KEY = "kandinsky6_sr_tile_plan"  # tiled.TilePlan
SR_TILES_KEY = "kandinsky6_sr_tiles"  # list of uint8 [3, T, Hb, Wb]
# stages exchange immutable specs and CPU chunks, not resident model objects
SR_SAMPLING_SPEC_KEY = "kandinsky6_sr_sampling_spec"  # sampling.SamplingSpec
SR_DIT_SPEC_KEY = "kandinsky6_sr_dit_spec"  # sampling.DitSpec
SR_CHUNKS_KEY = (
    "kandinsky6_sr_chunks"  # list of fp32 [B, T', H', W', 2C+1] (pre-denoise)
)
SR_DENOISED_KEY = (
    "kandinsky6_sr_denoised"  # list of fp32 [B, T', H', W', C] (post-denoise)
)
# original (H, W), used to crop VAE alignment padding from the stitched output
SR_REQUESTED_HW_KEY = "kandinsky6_sr_requested_hw"


def uses_latent_path(latent_upscaler: Any, tiling_scale: int) -> bool:
    """LU path iff the bank has an entry for the tiling scale (else the pixel path)."""
    return (
        latent_upscaler is not None
        and latent_upscaler.for_scale(tiling_scale) is not None
    )


def build_dit_spec(dit: Any) -> DitSpec:
    """Resolved DiT input and conditioning layout."""
    return DitSpec(
        instruct_type=dit.instruct_type,
        visual_cond=bool(dit.visual_cond),
        in_visual_dim=dit.in_visual_dim,
        use_motion_score=bool(dit.use_motion_score),
        patch_size=tuple(dit.patch_size),
        dtype=module_dtype(dit),
    )


def build_sampling_spec(
    *,
    arch: Kandinsky6SRArchConfig,
    tiling_scale: int,
    seed: int,
    num_steps: int,
    tiles_batch_size: int,
    tile_min_overlap: float,
    scheduler: FlowMatchEulerDiscreteScheduler,
) -> SamplingSpec:
    """Validate the loaded scheduler and resolve DiT calls before encoding tiles."""
    if num_steps < 1:
        raise ValueError(
            f"Kandinsky6 SR: num_inference_steps must be >= 1, got {num_steps}"
        )
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError(
            "Kandinsky6 SR requires a FlowMatchEulerDiscreteScheduler or "
            "PiflowScheduler from the checkpoint"
        )
    is_piflow = isinstance(scheduler, PiflowScheduler)
    n_grid = scheduler.config.n_grid if is_piflow else 1
    expected_width = arch.in_visual_dim * n_grid
    if arch.out_visual_dim != expected_width:
        raise ValueError(
            f"Kandinsky6 SR: {type(scheduler).__name__} requires a {expected_width}-channel "
            f"head, but out_visual_dim={arch.out_visual_dim}; use the scheduler "
            "from the same checkpoint as the transformer"
        )
    if is_piflow:
        nfe = scheduler.config.nfe
        if nfe is None or nfe < 1:
            raise ValueError(
                f"Kandinsky6 SR: PiflowScheduler requires nfe >= 1, got {nfe}"
            )
        if arch.sr_cap_noise_timestep and arch.effective_override("instruct_type") in (
            "noise",
            "hybrid",
        ):
            raise NotImplementedError(
                "piflow sampler does not support cap_noise_timestep for noise/hybrid instruct"
            )
        if num_steps != nfe:
            logger.warning(
                "Kandinsky6 SR: ignoring num_inference_steps=%d; the pi-Flow checkpoint "
                "requires %d DiT calls per tile",
                num_steps,
                nfe,
            )
        num_steps = nfe
    table = arch.sr_scale_factor
    visual_size = arch.sr_visual_size[0]
    if table is None:
        raise ValueError(
            "transformer/config.json carries no sr_params (visual_size / scale_factor); "
            "an official Diffusers Kandinsky6SRPipeline repo is required"
        )
    values = table.get(str(visual_size), table.get(visual_size))
    if values is None:
        raise ValueError(
            f"sr_scale_factor has no entry for visual size {visual_size}: {table}"
        )
    return SamplingSpec(
        tiling_scale=tiling_scale,
        tiles_batch_size=tiles_batch_size,
        seed=seed,
        num_steps=num_steps,
        is_piflow=is_piflow,
        tile_min_overlap=tile_min_overlap,
        visual_size=visual_size,
        scale_factor=tuple(float(value) for value in values),
        lq_noise_scale=arch.sr_lq_noise_scale,
        lq_noise_type=arch.sr_lq_noise_type,
        lq_channel_noise_scale=arch.sr_lq_channel_noise_scale,
        cap_noise_timestep=arch.sr_cap_noise_timestep,
    )
