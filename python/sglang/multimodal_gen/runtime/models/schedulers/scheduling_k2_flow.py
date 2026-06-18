"""Rectified-flow Euler scheduler for Krea-2 (K2).

The denoising timesteps are kept on the [0, 1] flow scale that the K2 DiT's
timestep embedding expects (it applies its own 1000x factor internally), unlike
the diffusers FlowMatch scheduler which reports timesteps on a 0..1000 scale.

The sampling grid is a uniform 1->0 schedule reshaped by an exponential
time-shift `mu`:  t' = exp(mu) / (exp(mu) + (1/t - 1) ** sigma).
A constant `mu` may be pinned (distilled checkpoint) or derived from the image
sequence length. Integration is Euler over the flow ODE.
"""

import math
from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class K2FlowSchedulerOutput:
    prev_sample: torch.Tensor


class K2FlowMatchScheduler:
    order = 1
    init_noise_sigma = 1.0

    def __init__(self, num_train_timesteps: int = 1000, sigma: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_exp = sigma
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def step_index(self):
        return self._step_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device=None,
        mu: float | None = None,
        sigmas=None,
        timesteps=None,
    ):
        if num_inference_steps is None:
            raise ValueError("K2FlowMatchScheduler requires num_inference_steps")
        ts = torch.linspace(1, 0, num_inference_steps + 1, dtype=torch.float64)
        if mu is not None:
            ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** self.sigma_exp)
        ts = ts.to(device=device, dtype=torch.float32)
        self.sigmas = ts
        self.timesteps = ts[:-1].clone()
        self._step_index = None
        self._begin_index = None

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample

    def _init_step_index(self):
        self._step_index = self._begin_index if self._begin_index is not None else 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        if self._step_index is None:
            self._init_step_index()
        i = self._step_index
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1
        if return_dict:
            return K2FlowSchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


EntryClass = K2FlowMatchScheduler
