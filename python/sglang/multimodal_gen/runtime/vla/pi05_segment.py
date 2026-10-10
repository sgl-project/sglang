# SPDX-License-Identifier: Apache-2.0
"""Request-local Euler state for opt-in pi0.5 completion-boundary execution.

This module has no scheduler policy: the caller chooses an admitted batch and
its execution horizon. Numerical state is committed only after a whole segment
succeeds, so partially executed work is never exposed as a completed action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)


@dataclass
class Pi05FlowState:
    x_t: torch.Tensor
    prefix: PrefixContext
    num_steps: int
    timesteps: torch.Tensor
    steps_done: int = 0

    @classmethod
    def create(cls, noise: torch.Tensor, prefix: PrefixContext, num_steps: int):
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("num_steps must be a positive integer")
        if noise.ndim != 3 or noise.shape[0] != 1:
            raise ValueError("Segmented pi0.5 requires one observation per request")
        return cls(
            x_t=noise.to(dtype=torch.float32).clone(),
            prefix=prefix,
            num_steps=num_steps,
            # Preserve native sample_actions' endpoints and dtype exactly.
            timesteps=torch.linspace(
                1.0,
                1.0 / num_steps,
                num_steps,
                dtype=torch.float32,
                device=noise.device,
            ),
        )

    @property
    def remaining(self) -> int:
        return self.num_steps - self.steps_done

    @property
    def dt(self) -> float:
        return -1.0 / self.num_steps


def prefix_compatibility_key(prefix: PrefixContext) -> tuple:
    """Ignore sequence length, but retain tensor and attention compatibility."""
    return (
        prefix.prefix_pad_masks.dtype,
        prefix.prefix_pad_masks.device,
        tuple(
            (
                k.shape[1:-2],
                k.shape[-1],
                k.dtype,
                k.device,
                v.shape[1:-2],
                v.shape[-1],
                v.dtype,
                v.device,
                window,
            )
            for k, v, window in prefix.past_key_values
        ),
    )


def collate_prefixes(prefixes: list[PrefixContext]) -> PrefixContext:
    if not prefixes:
        raise ValueError("Cannot collate an empty prefix batch")
    key = prefix_compatibility_key(prefixes[0])
    if any(prefix_compatibility_key(p) != key for p in prefixes):
        raise ValueError("Incompatible prefix tensor layouts")
    target = max(p.prefix_len for p in prefixes)
    for p in prefixes:
        if p.prefix_pad_masks.shape != (1, p.prefix_len):
            raise ValueError("Invalid request-local prefix mask")
        for k, v, _ in p.past_key_values:
            if (
                k.shape[0] != 1
                or v.shape[0] != 1
                or k.shape[-2] != p.prefix_len
                or v.shape[-2] != p.prefix_len
            ):
                raise ValueError("Invalid request-local prefix K/V shape")
    layers = []
    for i in range(len(prefixes[0].past_key_values)):
        keys = [
            F.pad(p.past_key_values[i][0], (0, 0, 0, target - p.prefix_len))
            for p in prefixes
        ]
        values = [
            F.pad(p.past_key_values[i][1], (0, 0, 0, target - p.prefix_len))
            for p in prefixes
        ]
        layers.append(
            (torch.cat(keys), torch.cat(values), prefixes[0].past_key_values[i][2])
        )
    masks = torch.cat(
        [
            F.pad(p.prefix_pad_masks, (0, target - p.prefix_len), value=False)
            for p in prefixes
        ]
    )
    return PrefixContext(
        VLADensePrefixCache(layers, read_only=True),
        masks,
        target,
        {
            "full_attention": all(
                p.prefix_len == target and p.layout.get("full_attention", False)
                for p in prefixes
            ),
            "cuda_graph_eligible": False,
        },
    )


@torch.inference_mode()
def advance_pi05_segment(model: Any, states: list[Pi05FlowState], length: int) -> None:
    """Advance compatible states without changing their original grids/budgets.

    The first implementation uses eager denoising. Graph replay and action
    sequence parallelism require separate validation before being enabled here.
    """
    if not states or len({id(s) for s in states}) != len(states):
        raise ValueError("Expected a nonempty batch of distinct flow states")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("Segment length must be a positive integer")
    for state in states:
        if not 0 <= state.steps_done < state.num_steps or length > state.remaining:
            raise ValueError("Segment exceeds the remaining request budget")
        if state.timesteps.shape != (state.num_steps,):
            raise ValueError("Original timestep grid was changed")
    shape = states[0].x_t.shape
    if any(
        s.x_t.shape != shape
        or s.x_t.device != states[0].x_t.device
        or s.x_t.dtype != states[0].x_t.dtype
        for s in states
    ):
        raise ValueError("Incompatible action tensors")
    prefix = collate_prefixes([s.prefix for s in states])
    x = torch.cat([s.x_t for s in states])
    layout = model.core_model.prepare_denoise_layout(
        prefix.prefix_pad_masks,
        x,
        bool(prefix.layout["full_attention"]),
        action_position_offset=0,
    )
    for offset in range(length):
        t = torch.stack([s.timesteps[s.steps_done + offset] for s in states])
        velocity = model.denoise_step(
            prefix,
            x,
            t,
            use_cuda_graph=False,
            action_position_offset=0,
            action_sp_enabled=False,
            denoise_layout=layout,
        )
        if all(s.num_steps == states[0].num_steps for s in states):
            x.add_(velocity, alpha=states[0].dt)
        else:
            for i, state in enumerate(states):
                x[i : i + 1].add_(velocity[i : i + 1], alpha=state.dt)
    for i, state in enumerate(states):
        state.x_t = x[i : i + 1]
        state.steps_done += length
