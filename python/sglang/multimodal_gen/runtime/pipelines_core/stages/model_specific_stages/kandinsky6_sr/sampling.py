# SPDX-License-Identifier: Apache-2.0
"""Scheduler-driven denoising for Kandinsky 6 SR tile chunks."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import msgspec
import torch


class DitSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Post-override input layout and loaded parameter dtype."""

    instruct_type: str | None
    visual_cond: bool
    in_visual_dim: int
    use_motion_score: bool
    patch_size: tuple[int, int, int]
    dtype: torch.dtype | None = None


class SamplingSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Serializable settings shared by tile preparation and denoising."""

    tiling_scale: int
    tiles_batch_size: int
    seed: int
    num_steps: int
    is_piflow: bool
    tile_min_overlap: float
    visual_size: int
    scale_factor: tuple[float, float, float]
    lq_noise_scale: float
    lq_noise_type: str
    lq_channel_noise_scale: float
    cap_noise_timestep: bool


def bf16_autocast(device: torch.device | str) -> AbstractContextManager:
    if torch.device(device).type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def module_dtype(module: torch.nn.Module) -> torch.dtype | None:
    return next((param.dtype for param in module.parameters()), None)


def cast_to_module_dtype(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    dtype = module_dtype(module)
    return (
        value.to(dtype=dtype)
        if dtype is not None and value.is_floating_point()
        else value
    )


@torch.no_grad()
def _denoise_chunk(x, dit, scheduler, dit_spec, spec, step_context, on_step):
    # each chunk starts at step zero and releases its GPU temporaries before the next
    scheduler._step_index = None
    scheduler.set_begin_index(0)
    state = x[..., : dit_spec.in_visual_dim].float()
    cond = x[..., dit_spec.in_visual_dim :]
    frames, height, width = x.shape[1:4]
    rope_pos = [
        torch.arange(frames, device=x.device),
        torch.arange(height // dit_spec.patch_size[1], device=x.device),
        torch.arange(width // dit_spec.patch_size[2], device=x.device),
    ]
    motion = (
        torch.full((1,), 900.0, device=x.device, dtype=state.dtype)
        if dit_spec.use_motion_score and not spec.is_piflow
        else None
    )
    with bf16_autocast(x.device):
        for step, t in enumerate(scheduler.timesteps):
            model_time = t.to(device=x.device, dtype=torch.float32).expand(
                state.shape[0]
            )
            hidden_states = (
                torch.cat([state, cond], dim=-1) if cond.shape[-1] else state
            )
            with step_context(step) if step_context is not None else nullcontext():
                prediction = dit(
                    hidden_states,
                    model_time,
                    rope_pos,
                    scale_factor=spec.scale_factor,
                    motion_score=motion,
                )
            # Euler otherwise downcasts the running state to the prediction dtype
            if not spec.is_piflow:
                prediction = prediction.float()
            state = scheduler.step(prediction, t, state, return_dict=False)[0]
            if on_step is not None:
                on_step()
    return state.float()


def denoise_chunks(
    chunks: Sequence[torch.Tensor],
    dit: Callable[..., torch.Tensor],
    scheduler: Any,
    *,
    dit_spec: DitSpec,
    spec: SamplingSpec,
    device: torch.device,
    step_context: Callable[[int], AbstractContextManager] | None = None,
    on_step: Callable[[], None] | None = None,
) -> list[torch.Tensor]:
    """Denoise CPU chunks sequentially, preserving conditioning and fp32 state."""
    if spec.is_piflow:
        scheduler.set_timesteps(spec.num_steps, device=device)
    else:
        capped = spec.cap_noise_timestep and dit_spec.instruct_type in (
            "noise",
            "hybrid",
            "hybrid_anchor",
        )
        start = spec.lq_noise_scale if capped else 1.0
        sigmas = torch.linspace(start, 0.0, spec.num_steps + 1)[:-1].tolist()
    results = []
    for chunk in chunks:
        if not spec.is_piflow:
            scheduler.set_timesteps(sigmas=sigmas, device=device)
        results.append(
            _denoise_chunk(
                chunk.to(device), dit, scheduler, dit_spec, spec, step_context, on_step
            ).cpu()
        )
    return results
