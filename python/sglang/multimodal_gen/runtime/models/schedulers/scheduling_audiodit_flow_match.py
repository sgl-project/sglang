# SPDX-License-Identifier: Apache-2.0
"""Simple Euler flow-matching scheduler for LongCat-AudioDiT.

LongCat-AudioDiT uses a t=0→1 convention (0=noise, 1=data) with a plain
Euler integrator: ``y = y + fn(t, y) * dt`` where ``dt > 0``.

The standard ``FlowMatchEulerDiscreteScheduler`` uses descending sigmas
(1→0) with ``dt < 0``, and the model-output sign convention differs
(``v_longcat = -v_diffusers``).  Rather than negate the prediction, this
scheduler mirrors the original ``odeint_euler`` exactly: ascending
timesteps, positive ``dt``, same sign.
"""

import torch

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler


class AudioDiTFlowMatchScheduler(BaseScheduler):
    """Euler flow-matching scheduler with ascending t (0→1)."""

    order = 1
    num_train_timesteps = 1

    def __init__(self, num_train_timesteps: int = 1, **kwargs) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None
        super().__init__()

    # -- BaseScheduler interface ------------------------------------------------

    def set_shift(self, shift: float) -> None:
        """No-op — audio flow matching does not use sigma shifting."""
        pass

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> None:
        """Produce *N* ascending timesteps ``linspace(0, 1, N+1)[:-1]``.

        The sigmas array has *N+1* entries (terminal = 1.0) so that
        ``step()`` can always read ``sigmas[step_index + 1]``.
        """
        grid = torch.linspace(0, 1, num_inference_steps + 1, device=device)
        self.timesteps = grid[:-1].clone()  # N points: 0, 1/N, ..., (N-1)/N
        self.sigmas = grid  # N+1 points: 0, 1/N, ..., 1.0
        self._step_index = None

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | torch.Tensor | None = None
    ) -> torch.Tensor:
        return sample

    # -- Step -------------------------------------------------------------------

    def _init_step_index(self, timestep: torch.Tensor) -> None:
        if self._begin_index is not None:
            self._step_index = self._begin_index
            return
        # Match by value (timesteps are unique and ascending).
        self._step_index = self.index_for_timestep(timestep)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        """One Euler step: ``prev = sample + dt * model_output`` (dt > 0)."""
        if self._step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        dt = sigma_next - sigma  # > 0 (ascending)

        prev_sample = sample.to(torch.float32) + dt * model_output.to(torch.float32)
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1
        return (prev_sample,)

    # -- Misc (required by denoising loop) --------------------------------------

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def index_for_timestep(
        self, timestep, schedule_timesteps: torch.Tensor | None = None
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        if len(indices) == 0:
            return int((schedule_timesteps - timestep).abs().argmin().item())
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()


EntryClass = AudioDiTFlowMatchScheduler
