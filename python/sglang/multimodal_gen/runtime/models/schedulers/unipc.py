# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.

import numpy as np
import torch


def compute_unipc_bh_coefficients(
    h: torch.Tensor,
    rks: torch.Tensor,
    order: int,
    predict_x0: bool,
    solver_type: str,
):
    """Build UniPC coefficients, leaving tensor assembly to each scheduler."""
    R = []
    b = []

    hh = -h if predict_x0 else h
    h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
    h_phi_k = h_phi_1 / hh - 1
    factorial_i = 1

    if solver_type == "bh1":
        B_h = hh
    elif solver_type == "bh2":
        B_h = torch.expm1(hh)
    else:
        raise NotImplementedError()

    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i

    return R, b, h_phi_1, B_h


def unipc_predictor_step(
    scheduler, model_output, model_output_convert, timestep, sample
):
    """Update UniPC history and order, then predict; callers own correction and indexing."""
    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
        scheduler.timestep_list[i] = scheduler.timestep_list[i + 1]
    scheduler.model_outputs[-1] = model_output_convert
    scheduler.timestep_list[-1] = timestep

    if scheduler.config.lower_order_final:
        this_order = min(
            scheduler.config.solver_order,
            len(scheduler.timesteps) - scheduler.step_index,
        )
    else:
        this_order = scheduler.config.solver_order
    scheduler.this_order = min(this_order, scheduler.lower_order_nums + 1)
    assert scheduler.this_order > 0

    scheduler.last_sample = sample
    # an external solver-p consumes the original, unconverted model output
    prev_sample = scheduler.multistep_uni_p_bh_update(
        model_output=model_output,
        sample=sample,
        order=scheduler.this_order,
    )
    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1
    return prev_sample


class UniPCSchedulerMixin:
    """Shared sample transforms and step state; subclasses own schedules and solvers."""

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return sample

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply per-sample dynamic thresholding before restoring the input dtype."""
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample.to(dtype)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64 timesteps
            schedule_timesteps = self.timesteps.to(
                original_samples.device, dtype=torch.float32
            )
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # training looks up each timestep; img2img and inpainting use the step state
        if self.begin_index is None:
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps) for t in timesteps
            ]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        return alpha_t * original_samples + sigma_t * noise
