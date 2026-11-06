# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team.
# Copyright 2025 TSAIL Team and The HuggingFace Team.
# Adapted from diffusers.schedulers.scheduling_euler_discrete.

import math

import numpy as np
import torch
from diffusers.utils import is_scipy_available

if is_scipy_available():
    import scipy.stats


class SigmaScheduleMixin:
    """Alternative sigma schedules for schedulers with a Diffusers config."""

    def _sigma_bounds(self, in_sigmas):
        sigma_min = self.config.get("sigma_min")
        sigma_max = self.config.get("sigma_max")
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()
        return sigma_min, sigma_max

    def _convert_to_karras(
        self, in_sigmas: torch.Tensor, num_inference_steps: int
    ) -> np.ndarray:
        """Construct the Karras et al. (2022) noise schedule."""
        sigma_min, sigma_max = self._sigma_bounds(in_sigmas)
        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def _convert_to_exponential(
        self, in_sigmas: torch.Tensor, num_inference_steps: int
    ) -> np.ndarray:
        """Construct an exponential noise schedule."""
        sigma_min, sigma_max = self._sigma_bounds(in_sigmas)
        return np.exp(
            np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps)
        )

    def _convert_to_beta(
        self,
        in_sigmas: torch.Tensor,
        num_inference_steps: int,
        alpha: float = 0.6,
        beta: float = 0.6,
    ) -> np.ndarray:
        """Construct the Beta Sampling (Lee et al., 2024) noise schedule."""
        sigma_min, sigma_max = self._sigma_bounds(in_sigmas)
        return np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
