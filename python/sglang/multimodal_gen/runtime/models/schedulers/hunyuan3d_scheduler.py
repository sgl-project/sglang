# Copied and adapted from: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class Hunyuan3DFlowMatchSchedulerOutput(BaseOutput):
    """Output class for the scheduler's step function."""

    prev_sample: torch.FloatTensor


class Hunyuan3DFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Euler discrete scheduler for flow matching."""

    # External module path aliases for compatibility with Hunyuan3D configs
    _aliases = [
        "hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler",
        "hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler",
    ]

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
    ):
        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        ).copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self) -> Optional[int]:
        """The index counter for current timestep."""
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        """The index for the first timestep."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """Set the begin index for the scheduler.

        Args:
            begin_index: The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_model_input(
        self,
        sample: torch.FloatTensor,
        timestep: Optional[Union[float, torch.FloatTensor]] = None,
    ) -> torch.FloatTensor:
        """Identity operation for flow matching (no input scaling needed)."""
        return sample

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Forward process in flow-matching (add noise to sample)."""
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timestep
            ]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample
        return sample

    def _sigma_to_t(self, sigma: float) -> float:
        """Convert sigma to timestep."""
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        """Apply time shift transformation."""
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """Set the discrete timesteps for the diffusion chain."""
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                "Must pass a value for `mu` when `use_dynamic_shifting` is True"
            )

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max),
                self._sigma_to_t(self.sigma_min),
                num_inference_steps,
            )
            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self, timestep: float, schedule_timesteps: Optional[torch.Tensor] = None
    ) -> int:
        """Find the index for a given timestep."""
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep: Union[float, torch.Tensor]):
        """Initialize step index from timestep."""
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[Hunyuan3DFlowMatchSchedulerOutput, Tuple]:
        """Predict the sample from the previous timestep."""
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Pass one of `scheduler.timesteps` as a timestep."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return Hunyuan3DFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return self.config.num_train_timesteps


@dataclass
class Hunyuan3DConsistencyFlowMatchSchedulerOutput(BaseOutput):
    """Output for consistency flow matching scheduler."""

    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor


class Hunyuan3DConsistencyFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Consistency Flow Matching Euler Discrete Scheduler."""

    # External module path aliases for compatibility with Hunyuan3D configs
    _aliases = [
        "hy3dshape.schedulers.Hunyuan3DConsistencyFlowMatchEulerDiscreteScheduler",
    ]

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        pcm_timesteps: int = 50,
    ):
        sigmas = np.linspace(0, 1, num_train_timesteps)
        step_ratio = num_train_timesteps // pcm_timesteps

        euler_timesteps = (np.arange(1, pcm_timesteps) * step_ratio).round().astype(
            np.int64
        ) - 1
        euler_timesteps = np.asarray([0] + euler_timesteps.tolist())

        self.euler_timesteps = euler_timesteps
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas = torch.from_numpy(self.sigmas.copy()).to(dtype=torch.float32)
        self.timesteps = self.sigmas * num_train_timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def scale_model_input(
        self,
        sample: torch.FloatTensor,
        timestep: Optional[Union[float, torch.FloatTensor]] = None,
    ) -> torch.FloatTensor:
        """Identity operation for flow matching (no input scaling needed)."""
        return sample

    def _sigma_to_t(self, sigma: float) -> float:
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
    ):
        """Set timesteps for inference."""
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else len(sigmas)
        )
        inference_indices = np.linspace(
            0, self.config.pcm_timesteps, num=self.num_inference_steps, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = torch.from_numpy(inference_indices).long()

        self.sigmas_ = self.sigmas[inference_indices]
        timesteps = self.sigmas_ * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas_ = torch.cat(
            [self.sigmas_, torch.ones(1, device=self.sigmas_.device)]
        )

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self, timestep: float, schedule_timesteps: Optional[torch.Tensor] = None
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep: Union[float, torch.Tensor]):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[Hunyuan3DConsistencyFlowMatchSchedulerOutput, Tuple]:
        """Perform one step of the consistency flow matching scheduler."""
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError("Passing integer indices as timesteps is not supported.")

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        sigma = self.sigmas_[self.step_index]
        sigma_next = self.sigmas_[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        pred_original_sample = sample + (1.0 - sigma) * model_output
        pred_original_sample = pred_original_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return Hunyuan3DConsistencyFlowMatchSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    def __len__(self) -> int:
        return self.config.num_train_timesteps


# Entry class for model registry
EntryClass = [
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DConsistencyFlowMatchEulerDiscreteScheduler,
]
