# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """LTX-2 flow-match Euler scheduler.

    LTX-2 follows the native SGLang flow-match Euler implementation, with two
    compatibility overrides from the previous pipeline-local scheduler.
    """

    config_name = "scheduler_config.json"

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        if sigmas is not None and timesteps is None and mu is None:
            sigmas_tensor = torch.tensor(sigmas, dtype=torch.float32, device=device)
            timesteps_tensor = sigmas_tensor * self.config.num_train_timesteps
            sigmas_tensor = torch.cat(
                [sigmas_tensor, torch.zeros(1, device=sigmas_tensor.device)]
            )
            self.num_inference_steps = len(timesteps_tensor)
            self.timesteps = timesteps_tensor
            self.sigmas = sigmas_tensor
            self._step_index = None
            self._begin_index = None
            return

        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

    def _time_shift_exponential(
        self, mu: float, sigma: float, t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def __len__(self) -> int:
        return self.config.num_train_timesteps


EntryClass = LTX2FlowMatchScheduler
