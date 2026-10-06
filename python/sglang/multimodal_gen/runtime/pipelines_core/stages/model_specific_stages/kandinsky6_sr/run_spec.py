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
    PiflowParams,
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
# Handoff between the three model-phase stages (latent prep -> denoise -> decode), split out
# of the single combined stage so the denoising stage can subclass the shared DenoisingStage
# (see denoising_stage.py): SamplingSpec / DitSpec are immutable msgspec.Struct values (plain
# data, like TilePlan above), and the chunk lists are CPU tensors throughout.
SR_SAMPLING_SPEC_KEY = "kandinsky6_sr_sampling_spec"  # sampling.SamplingSpec
SR_DIT_SPEC_KEY = "kandinsky6_sr_dit_spec"  # sampling.DitSpec
SR_CHUNKS_KEY = (
    "kandinsky6_sr_chunks"  # list of fp32 [B, T', H', W', 2C+1] (pre-denoise)
)
SR_DENOISED_KEY = (
    "kandinsky6_sr_denoised"  # list of fp32 [B, T', H', W', C] (post-denoise)
)
# (H, W) of the clip actually requested, before any VAE-spatial-factor alignment padding the
# input stage added so whole-video KVAE encoding survives every supported sr_resolution_scale
# (not only 2.25, whose pixel pre-upscale happens to already land on an aligned size). The
# output stage crops the stitched result from (padded_H * scale, padded_W * scale) down to
# (H * scale, W * scale) so the delivered video matches what was actually asked for.
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


def _rope_scale_factor(arch: Kandinsky6SRArchConfig) -> tuple[float, float, float]:
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
    return tuple(float(value) for value in values)


def piflow_params_from_scheduler(scheduler: Any) -> PiflowParams | None:
    """pi-Flow sampler values of a ``PiflowScheduler`` component (``None``: flow-Euler)."""
    if not isinstance(scheduler, PiflowScheduler):
        return None
    config = scheduler.config
    if config["nfe"] is None:
        return None
    return PiflowParams(
        nfe=int(config["nfe"]),
        num_policy_substeps=int(config["num_policy_substeps"]),
        final_step_size_scale=float(config["final_step_size_scale"]),
        shift=float(config["shift"]),
        n_grid=int(config["n_grid"]),
        eps=float(config["eps"]),
    )


def _describe_scheduler(scheduler: Any) -> str:
    if scheduler is None:
        return "missing"
    if isinstance(scheduler, PiflowScheduler):
        return "a PiflowScheduler without nfe"
    return f"a {type(scheduler).__name__}"


def _check_sampler_fits_head(
    *, arch: Kandinsky6SRArchConfig, piflow: PiflowParams | None, scheduler: Any
) -> None:
    """Reject incompatible scheduler and DiT head widths before encoding tiles."""
    head, grid = arch.head_width, arch.base_out_visual_dim
    if piflow is None:
        if head != grid:
            raise ValueError(
                f"Kandinsky6SR: the transformer head is {head} channels wide (a DX / "
                f"pi-Flow head: {grid} channels per latent), but the sampler resolved "
                "to flow-Euler: the scheduler component is "
                f"{_describe_scheduler(scheduler)}, and only a PiflowScheduler with "
                "nfe selects pi-Flow. Use the scheduler of the same repo as the "
                "transformer."
            )
        return
    if piflow.nfe < 1:
        raise ValueError(
            "Kandinsky6SR: the pi-Flow sampler needs nfe >= 1 model calls per tile, "
            f"got nfe={piflow.nfe}."
        )
    if head != piflow.n_grid * grid:
        raise ValueError(
            f"Kandinsky6SR: the PiflowScheduler (nfe {piflow.nfe}, n_grid "
            f"{piflow.n_grid}) needs a {piflow.n_grid} x {grid} = "
            f"{piflow.n_grid * grid} channel DX head, but the transformer head is "
            f"{head} channels wide (out_visual_dim {arch.out_visual_dim}). The "
            "scheduler and the transformer come from different repos."
        )


def build_sampling_spec(
    *,
    arch: Kandinsky6SRArchConfig,
    tiling_scale: int,
    seed: int,
    num_steps: int,
    tiles_batch_size: int,
    tile_min_overlap: float,
    scheduler: Any = None,
) -> SamplingSpec:
    """Resolve sampling from the bundle scheduler, falling back to legacy DiT fields.

    num_steps counts DiT calls, not Diffusers timestep grid points."""
    piflow = piflow_params_from_scheduler(scheduler)
    if piflow is None and arch.is_piflow:
        piflow = PiflowParams(
            nfe=arch.piflow_nfe,
            num_policy_substeps=arch.piflow_num_policy_substeps,
            final_step_size_scale=arch.piflow_final_step_size_scale,
            shift=arch.piflow_shift,
            n_grid=arch.n_grid,
            eps=arch.piflow_eps,
        )
    _check_sampler_fits_head(arch=arch, piflow=piflow, scheduler=scheduler)
    return SamplingSpec(
        tiling_scale=tiling_scale,
        tiles_batch_size=tiles_batch_size,
        seed=seed,
        num_steps=num_steps,
        tile_min_overlap=tile_min_overlap,
        visual_size=arch.sr_visual_size[0],
        scale_factor=_rope_scale_factor(arch),
        scheduler_scale=arch.sr_scheduler_scale,
        lq_noise_scale=arch.sr_lq_noise_scale,
        lq_noise_type=arch.sr_lq_noise_type,
        lq_channel_noise_scale=arch.sr_lq_channel_noise_scale,
        cap_noise_timestep=arch.sr_cap_noise_timestep,
        piflow=piflow,
    )


def effective_scheduler(spec: SamplingSpec, scheduler: Any) -> Any:
    """Use the loaded scheduler when compatible; synthesize one for legacy configs."""
    if spec.piflow is not None:
        if isinstance(scheduler, PiflowScheduler):
            return scheduler
        return PiflowScheduler(
            nfe=spec.piflow.nfe,
            n_grid=spec.piflow.n_grid,
            shift=spec.piflow.shift,
            eps=spec.piflow.eps,
            final_step_size_scale=spec.piflow.final_step_size_scale,
            num_policy_substeps=spec.piflow.num_policy_substeps,
        )
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return scheduler
    return FlowMatchEulerDiscreteScheduler(shift=spec.scheduler_scale)


def check_denoising_request(spec: SamplingSpec, num_inference_steps: int) -> None:
    """Validate the requested DiT-call count before denoising.

    Diffusers counts timestep grid points instead: its default 5 is 4 calls here."""
    if num_inference_steps < 1:
        raise ValueError(
            f"Kandinsky6 SR: num_inference_steps must be >= 1, got {num_inference_steps}"
        )
    if spec.piflow is not None and num_inference_steps != spec.piflow.nfe:
        logger.warning(
            "Kandinsky6 SR: num_inference_steps=%d was requested, but this is a pi-Flow "
            "bundle distilled for nfe=%d; num_inference_steps is ignored for pi-Flow "
            "bundles and every tile will run exactly nfe=%d DiT calls instead.",
            num_inference_steps,
            spec.piflow.nfe,
            spec.piflow.nfe,
        )
