# SPDX-License-Identifier: Apache-2.0
"""Scheduler-driven denoising for Kandinsky 6 SR tile chunks."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import msgspec
import torch

# (latent [B, T, H, W, C], model time [B] (already * 1000), step index) -> DiT output
DitFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
StepCallback = Callable[[], None]
StepContext = Callable[[int], AbstractContextManager]


class PiflowParams(msgspec.Struct, frozen=True, kw_only=True):
    """pi-Flow parameters from the scheduler or legacy flat DiT config."""

    nfe: int
    num_policy_substeps: int
    final_step_size_scale: float
    shift: float
    n_grid: int
    eps: float


class DitSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Post-override behaviour of the DiT that the sampling code needs."""

    instruct_type: str | None
    visual_cond: bool
    in_visual_dim: int
    use_motion_score: bool
    patch_size: tuple[int, int, int]
    # The DiT's loaded parameter dtype (``module_dtype(dit)``); the reference casts the
    # LQ latent to this dtype before building the initial latent, so a bf16-loaded
    # checkpoint's LQ-noise mix rounds in bf16, not fp32. ``None`` (parameter-less DiT,
    # e.g. a mocked one in a test) keeps the LQ latent's own dtype.
    dtype: torch.dtype | None = None


class SamplingSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Everything besides the components that defines one tiled SR run."""

    tiling_scale: int
    tiles_batch_size: int
    seed: int
    num_steps: int
    tile_min_overlap: float
    visual_size: int
    scale_factor: tuple[float, float, float]
    scheduler_scale: float
    lq_noise_scale: float
    lq_noise_type: str
    lq_channel_noise_scale: float
    cap_noise_timestep: bool
    piflow: PiflowParams | None

    @property
    def steps_per_chunk(self) -> int:
        """DiT calls per chunk of tiles: ``nfe`` (pi-Flow), else ``num_steps``."""
        return self.piflow.nfe if self.piflow is not None else self.num_steps


def bf16_autocast(device: torch.device | str) -> AbstractContextManager:
    """CUDA bf16 autocast (a no-op on other devices), as the reference wraps its models."""
    if torch.device(device).type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def module_dtype(module: torch.nn.Module) -> torch.dtype | None:
    """Dtype of the first parameter (``None`` for a parameter-less module)."""
    return next((param.dtype for param in module.parameters()), None)


def cast_to_module_dtype(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Cast a floating tensor to the module's parameter dtype (reference helper)."""
    dtype = module_dtype(module)
    if dtype is not None and value.is_floating_point() and value.dtype != dtype:
        return value.to(dtype=dtype)
    return value


def euler_start_timestep(
    *, cap_noise_timestep: bool, lq_noise_scale: float, instruct_type: str | None
) -> float:
    """Euler start time: the LQ noise level when capped for noise-type conditioning."""
    capped = cap_noise_timestep and instruct_type in (
        "noise",
        "hybrid",
        "hybrid_anchor",
    )
    return lq_noise_scale if capped else 1.0


def check_sampler_options(
    *, piflow: PiflowParams | None, cap_noise_timestep: bool, instruct_type: str | None
) -> None:
    """Reject option combinations the reference sampler rejects."""
    if (
        piflow is not None
        and cap_noise_timestep
        and instruct_type in ("noise", "hybrid")
    ):
        raise NotImplementedError(
            "piflow sampler does not support cap_noise_timestep for noise/hybrid "
            "instruct"
        )


def make_dit_fn(
    dit: Callable[..., torch.Tensor],
    *,
    latent_frames_hw: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    scale_factor: tuple[float, float, float],
    use_motion_score: bool,
    step_context: StepContext | None = None,
) -> DitFn:
    """Bind per-chunk RoPE/motion inputs and an optional per-step context."""
    frames, height, width = latent_frames_hw
    context = step_context or (lambda step: nullcontext())

    def call(x: torch.Tensor, model_time: torch.Tensor, step: int) -> torch.Tensor:
        rope_pos = [
            torch.arange(frames, device=x.device),
            torch.arange(height // patch_size[1], device=x.device),
            torch.arange(width // patch_size[2], device=x.device),
        ]
        motion = (
            torch.full((1,), 900.0, device=x.device, dtype=x.dtype)
            if use_motion_score
            else None
        )
        with context(step):
            return dit(
                x,
                model_time,
                rope_pos,
                scale_factor=scale_factor,
                motion_score=motion,
            )

    return call


@torch.no_grad()
def denoise_with_scheduler(
    x: torch.Tensor,
    dit_fn: DitFn,
    scheduler: Any,
    *,
    channels: int,
    is_piflow: bool,
    on_step: StepCallback | None = None,
) -> torch.Tensor:
    """Denoise one chunk after set_timesteps; keep conditioning channels fixed.

    Promote predictions to fp32 so flow-Euler does not downcast its running
    state to the bf16 prediction dtype. pi-Flow already preserves the state dtype."""
    state = x[..., :channels].float()
    cond = x[..., channels:]
    for step, t in enumerate(scheduler.timesteps):
        model_time = t.to(device=x.device, dtype=torch.float32).expand(state.shape[0])
        hidden_states = torch.cat([state, cond], dim=-1) if cond.shape[-1] else state
        prediction = dit_fn(hidden_states, model_time, step)
        if not is_piflow:
            prediction = prediction.float()
        state = scheduler.step(prediction, t, state, return_dict=False)[0]
        if on_step is not None:
            on_step()
    return state.float()


__all__ = [
    "DitFn",
    "DitSpec",
    "PiflowParams",
    "SamplingSpec",
    "StepCallback",
    "StepContext",
    "bf16_autocast",
    "cast_to_module_dtype",
    "check_sampler_options",
    "denoise_with_scheduler",
    "euler_start_timestep",
    "make_dit_fn",
    "module_dtype",
]
