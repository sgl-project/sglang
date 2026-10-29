# SPDX-License-Identifier: Apache-2.0
#
# Euler discrete scheduler wrapper for Stable Diffusion XL.
#
# Delegates all numerical work to diffusers' implementation and only adapts
# the interface for sglang's denoising stage.

import torch
from diffusers import EulerDiscreteScheduler as DiffusersEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler


class EulerDiscreteScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """Euler discrete scheduler wrapper for sglang's BaseScheduler interface."""

    order = 1
    num_train_timesteps = 1000

    # **kwargs absorbs legacy keys (clip_sample, skip_prk_steps, ...) that SDXL
    # scheduler configs carry but diffusers' EulerDiscreteScheduler does not take.
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas=None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        final_sigmas_type: str = "zero",
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        self._inner = DiffusersEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            prediction_type=prediction_type,
            interpolation_type=interpolation_type,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            timestep_spacing=timestep_spacing,
            timestep_type=timestep_type,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            final_sigmas_type=final_sigmas_type,
        )
        self.timesteps = self._inner.timesteps
        BaseScheduler.__init__(self)

    def set_shift(self, shift: float) -> None:
        raise ValueError("EulerDiscreteScheduler does not support flow shift.")

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._inner.set_begin_index(begin_index)

    @property
    def begin_index(self) -> int | None:
        return self._inner.begin_index

    def set_timesteps(self, num_inference_steps: int, device=None, **kwargs):
        self._inner.set_timesteps(num_inference_steps, device=device, **kwargs)
        self.timesteps = self._inner.timesteps

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | None = None
    ) -> torch.Tensor:
        return self._inner.scale_model_input(sample, timestep)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        return self._inner.step(
            model_output,
            timestep,
            sample,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            generator=generator,
            return_dict=return_dict,
        )

    @property
    def sigmas(self):
        return self._inner.sigmas

    @property
    def init_noise_sigma(self):
        return self._inner.init_noise_sigma

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self._inner.add_noise(original_samples, noise, timesteps)


EntryClass = EulerDiscreteScheduler
