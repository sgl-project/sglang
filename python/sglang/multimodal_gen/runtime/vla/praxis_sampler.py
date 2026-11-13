# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 window-joint sampler: float64 time grid, seeded CPU noise, text guidance and the Euler loop.

Noising is ``x_t = (1 - t) * x0 + t * eps`` and the models predict the velocity ``v = eps - x0``. One Euler
step is ``x + dt * v`` with a negative Python-float ``dt``. The window holds the context chunks and the
chunk being served, all denoised together, video and actions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import msgspec
import torch

# The video noise and the action noise use separate generators.
VIDEO_SALT = 0
ACTION_SALT = 1

Velocity = Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]]
TraceFn = Callable[[str, torch.Tensor], None]


class SamplerSettings(msgspec.Struct, frozen=True):
    """``cfg_scale`` guides the actions and ``cfg_video_scale`` the video; 1.0 turns a guidance off."""

    num_steps: int = 2
    t_shift: float = 12.5
    cfg_scale: float = 1.0
    cfg_video_scale: float = 4.0

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        for name, value in (
            ("cfg_scale", self.cfg_scale),
            ("cfg_video_scale", self.cfg_video_scale),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite float >= 0, got {value!r}")

    @property
    def guided(self) -> bool:
        return not (self.cfg_scale == 1.0 and self.cfg_video_scale == 1.0)


def window_time_grid(*, num_steps: int, shift: float = 1.0) -> list[float]:
    """Descending float64 grid from 1 to 0: ``u = linspace(1, 0)`` warped by ``s * u / (1 + (s - 1) * u)``."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    ts = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    if shift != 1.0:
        ts = shift * ts / (1.0 + (shift - 1.0) * ts)
    return [float(t) for t in ts]


def seeded_normal(*, shape: Sequence[int], seed: int, salt: int) -> torch.Tensor:
    """Float32 standard normal noise drawn on the CPU from the generator seeded ``(seed * 1000003 + salt) & (2**63 - 1)``."""
    s = (int(seed) * 1000003 + int(salt)) & 0x7FFFFFFFFFFFFFFF
    generator = torch.Generator().manual_seed(s)
    return torch.randn(tuple(shape), generator=generator)


def guidance_combine(
    *, cond: torch.Tensor, uncond: torch.Tensor, scale: float
) -> torch.Tensor:
    """``v_u + scale * (v_c - v_u)`` in float32; scale 1.0 returns ``cond`` untouched to keep its bits."""
    if float(scale) == 1.0:
        return cond
    return uncond.float() + float(scale) * (cond.float() - uncond.float())


class CfgParallel(msgspec.Struct, frozen=True):
    """Splits the two passes of a guided step over two ranks.

    Rank 0 runs the guided pass and rank 1 the unguided one. ``share(tensor, src)`` broadcasts the tensor
    of rank ``src`` to the other rank and returns the received (or own) tensor.
    """

    rank: int
    share: Callable[[torch.Tensor, int], torch.Tensor]


Passes = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]


def _passes_in_sequence(
    *,
    velocity: Velocity,
    probe: Callable[[str, torch.Tensor], None],
    step: int,
    state: tuple[torch.Tensor, torch.Tensor, float],
    text: torch.Tensor,
    text_uncond: torch.Tensor | None,
    uncond_parts: tuple[bool, bool],
) -> Passes:
    """The guided pass, then the unguided one, on this rank."""
    xv, xa, t_cur = state
    probe("pass.marker", torch.tensor([step, 0]))
    v, a = velocity(text, xv, xa, t_cur, True)
    assert v is not None and a is not None
    if text_uncond is None:
        return v, a, None, None
    probe("pass.marker", torch.tensor([step, 1]))
    decode_uncond, act_uncond = uncond_parts
    v_u, a_u = velocity(text_uncond, xv, xa, t_cur, decode_uncond, act=act_uncond)
    return v, a, v_u, a_u


def _passes_split(
    *,
    velocity: Velocity,
    parallel: CfgParallel,
    state: tuple[torch.Tensor, torch.Tensor, float],
    text: torch.Tensor,
    text_uncond: torch.Tensor,
    uncond_parts: tuple[bool, bool],
) -> Passes:
    """Each rank runs one pass; both ranks end with the velocities of both."""
    xv, xa, t_cur = state
    decode_uncond, act_uncond = uncond_parts
    if parallel.rank == 0:
        v, a = velocity(text, xv, xa, t_cur, True)
        assert v is not None and a is not None
        v_u = a_u = None
    else:
        v = torch.empty_like(xv)
        a = torch.empty_like(xa)
        v_u, a_u = velocity(text_uncond, xv, xa, t_cur, decode_uncond, act=act_uncond)
    v, a = parallel.share(v, 0), parallel.share(a, 0)
    if decode_uncond:
        v_u = parallel.share(v_u if v_u is not None else torch.empty_like(xv), 1)
    if act_uncond:
        a_u = parallel.share(a_u if a_u is not None else torch.empty_like(xa), 1)
    return v, a, v_u, a_u


def sample_window_joint(
    *,
    velocity: Velocity,
    noise_video: torch.Tensor,
    noise_action: torch.Tensor,
    settings: SamplerSettings,
    text: torch.Tensor,
    text_uncond: torch.Tensor | None,
    trace: TraceFn | None = None,
    parallel: CfgParallel | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the video and the action noise of the whole window from t = 1 to t = 0.

    ``velocity(text, x_video, x_action, t_cur, decode, act)`` returns the video and the action velocity
    (None for a part that is not computed). A guided step makes a second, unguided pass with
    ``text_uncond`` that computes only what a guidance scale reads: its video with ``cfg_video_scale``
    not 1, its actions with ``cfg_scale`` not 1. A trace keeps both parts. With ``parallel`` the two
    passes of a guided step run on two ranks at once; it cannot be combined with a trace.
    """
    guided = settings.guided
    if guided and text_uncond is None:
        raise ValueError("text guidance needs the encoded blank caption")
    if parallel is not None and trace is not None:
        raise ValueError("a trace needs the passes in sequence on one rank")

    def probe(name: str, tensor: torch.Tensor) -> None:
        if trace is not None:
            trace(name, tensor)

    grid = window_time_grid(num_steps=settings.num_steps, shift=settings.t_shift)
    probe("sampler.time_grid", torch.tensor(grid, dtype=torch.float64))
    uncond_parts = (
        trace is not None or float(settings.cfg_video_scale) != 1.0,
        trace is not None or float(settings.cfg_scale) != 1.0,
    )
    uncond_text = text_uncond if guided else None
    xv, xa = noise_video, noise_action
    for step in range(settings.num_steps):
        t_cur, t_next = grid[step], grid[step + 1]
        state = (xv, xa, t_cur)
        if parallel is not None and guided:
            assert text_uncond is not None
            v, a, v_u, a_u = _passes_split(
                velocity=velocity,
                parallel=parallel,
                state=state,
                text=text,
                text_uncond=text_uncond,
                uncond_parts=uncond_parts,
            )
        else:
            v, a, v_u, a_u = _passes_in_sequence(
                velocity=velocity,
                probe=probe,
                step=step,
                state=state,
                text=text,
                text_uncond=uncond_text,
                uncond_parts=uncond_parts,
            )
        for kind, cond, uncond, scale in (
            ("video", v, v_u, settings.cfg_video_scale),
            ("action", a, a_u, settings.cfg_scale),
        ):
            if uncond is None:
                continue
            out = guidance_combine(cond=cond, uncond=uncond, scale=scale)
            probe(f"cfg_combine.{kind}.cond", cond)
            probe(f"cfg_combine.{kind}.uncond", uncond)
            probe(f"cfg_combine.{kind}.out", out)
            if kind == "video":
                v = out
            else:
                a = out
        dt = t_next - t_cur
        xv = xv + dt * v
        xa = xa + dt * a
    return xv, xa
