# SPDX-License-Identifier: Apache-2.0
#
# DPM-Solver++ multistep scheduler wrapper for SANA.
#
# SANA uses DPM-Solver++ (Lu et al., 2022) as its noise scheduler, which
# is a high-order ODE solver that converges in fewer steps than DDIM.
# With solver_order=2 and 20 steps, SANA achieves high-quality results.
#
# This wrapper delegates all numerical work to diffusers' implementation
# and only adapts the interface for sglang's denoising stage.

import torch
from diffusers import (
    DPMSolverMultistepScheduler as DiffusersDPMSolverMultistepScheduler,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler


class DPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """DPM-Solver++ multistep scheduler wrapper for sglang's BaseScheduler interface."""

    order = 1
    num_train_timesteps = 1000

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "scaled_linear",
        trained_betas=None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        use_karras_sigmas: bool = False,
        use_lu_lambdas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        use_flow_sigmas: bool = False,
        final_sigmas_type: str = "zero",
        lambda_min_clipped: float = -float("inf"),
        variance_type: str | None = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        flow_shift: float | None = None,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        self._inner = DiffusersDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            solver_order=solver_order,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
            algorithm_type=algorithm_type,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            euler_at_final=euler_at_final,
            use_karras_sigmas=use_karras_sigmas,
            use_lu_lambdas=use_lu_lambdas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            use_flow_sigmas=use_flow_sigmas,
            flow_shift=flow_shift,
            final_sigmas_type=final_sigmas_type,
            lambda_min_clipped=lambda_min_clipped,
            variance_type=variance_type,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        self.timesteps = self._inner.timesteps
        self.order = solver_order
        self._flow_shift = flow_shift
        self._begin_index: int | None = None
        BaseScheduler.__init__(self)

    def set_shift(self, shift: float) -> None:
        self._flow_shift = shift

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

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
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        return self._inner.step(
            model_output,
            timestep,
            sample,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=return_dict,
        )

    @property
    def sigmas(self):
        return getattr(self._inner, "sigmas", None)

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


EntryClass = DPMSolverMultistepScheduler
