# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class LTX2FlowMatchSchedulerOutput(BaseOutput):
    """Output class for ``LTX2FlowMatchScheduler.step``."""

    prev_sample: torch.FloatTensor


class LTX2FlowMatchScheduler(
    BaseScheduler, ConfigMixin, SchedulerMixin, SchedulerRLMixin
):
    """LTX-2 flow-match Euler scheduler.

    This mirrors the native flow-match Euler scheduler while keeping the LTX-2
    custom sigma path and fp32 ndarray time-shift behavior that previously lived
    in ``ltx_2_pipeline.py``.
    """

    config_name = "scheduler_config.json"
    _compatibles: list[Any] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float | None = 0.5,
        max_shift: float | None = 1.15,
        base_image_seq_len: int | None = 256,
        max_image_seq_len: int | None = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool | None = False,
        use_exponential_sigmas: bool | None = False,
        use_beta_sigmas: bool | None = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ):
        if (
            sum(
                [
                    self.config.use_beta_sigmas,
                    self.config.use_exponential_sigmas,
                    self.config.use_karras_sigmas,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, "
                "`config.use_exponential_sigmas`, `config.use_karras_sigmas` "
                "can be used."
            )
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError(
                "`time_shift_type` must either be 'exponential' or 'linear'."
            )

        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.num_train_timesteps = num_train_timesteps

        self._step_index: int | None = None
        self._begin_index: int | None = None

        self._shift = shift

        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        BaseScheduler.__init__(self)

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        self._shift = shift

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        noise: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Forward process in flow matching."""
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            assert isinstance(timestep, torch.Tensor)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            assert isinstance(timestep, torch.Tensor)
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
        return sigma * self.config.num_train_timesteps

    def time_shift(
        self, mu: float, sigma: float, t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        elif self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)
        else:
            raise ValueError(f"Unknown time_shift_type: {self.config.time_shift_type}")

    def stretch_shift_to_terminal(
        self, t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        if scale_factor == 0:
            return t
        return 1 - (one_minus_z / scale_factor)

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

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                "`mu` must be passed when `use_dynamic_shifting` is set to be `True`"
            )

        if (
            sigmas is not None
            and timesteps is not None
            and len(sigmas) != len(timesteps)
        ):
            raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as "
                    "num_inference_steps, if `num_inference_steps` is provided"
                )
        else:
            if sigmas is not None:
                num_inference_steps = len(sigmas)
            elif timesteps is not None:
                num_inference_steps = len(timesteps)
            else:
                raise ValueError(
                    "Either num_inference_steps, sigmas, or timesteps must be provided"
                )

        self.num_inference_steps = num_inference_steps

        is_timesteps_provided = timesteps is not None

        timesteps_array: np.ndarray | None = None
        if is_timesteps_provided:
            assert timesteps is not None
            timesteps_array = np.array(timesteps).astype(np.float32)

        sigmas_array: np.ndarray
        if sigmas is None:
            if timesteps_array is None:
                timesteps_array = np.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    num_inference_steps,
                )
            sigmas_array = timesteps_array / self.config.num_train_timesteps
        else:
            sigmas_array = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas_array)

        if self.config.use_dynamic_shifting:
            assert mu is not None, "mu cannot be None when use_dynamic_shifting is True"
            sigmas_array = self.time_shift(mu, 1.0, sigmas_array)
        else:
            sigmas_array = (
                self.shift * sigmas_array / (1 + (self.shift - 1) * sigmas_array)
            )

        if self.config.shift_terminal:
            sigmas_array = np.array(
                self.stretch_shift_to_terminal(sigmas_array), dtype=np.float32
            )

        if self.config.use_karras_sigmas:
            sigmas_array = self._convert_to_karras(
                in_sigmas=sigmas_array, num_inference_steps=num_inference_steps
            )
        elif self.config.use_exponential_sigmas:
            sigmas_array = self._convert_to_exponential(
                in_sigmas=sigmas_array, num_inference_steps=num_inference_steps
            )
        elif self.config.use_beta_sigmas:
            sigmas_array = self._convert_to_beta(
                in_sigmas=sigmas_array, num_inference_steps=num_inference_steps
            )

        sigmas_tensor = torch.from_numpy(np.asarray(sigmas_array)).to(
            dtype=torch.float32, device=device
        )
        if not is_timesteps_provided:
            timesteps_tensor = sigmas_tensor * self.config.num_train_timesteps
        else:
            assert timesteps_array is not None
            timesteps_tensor = torch.from_numpy(timesteps_array).to(
                dtype=torch.float32, device=device
            )

        if self.config.invert_sigmas:
            sigmas_tensor = 1.0 - sigmas_tensor
            timesteps_tensor = sigmas_tensor * self.config.num_train_timesteps
            sigmas_tensor = torch.cat(
                [sigmas_tensor, torch.ones(1, device=sigmas_tensor.device)]
            )
        else:
            sigmas_tensor = torch.cat(
                [sigmas_tensor, torch.zeros(1, device=sigmas_tensor.device)]
            )

        self.timesteps = timesteps_tensor
        self.sigmas = sigmas_tensor
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self,
        timestep: float | torch.FloatTensor,
        schedule_timesteps: torch.Tensor | None = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep: float | torch.FloatTensor) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int | torch.Tensor,
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        per_token_timesteps: torch.Tensor | None = None,
        batch=None,
        return_dict: bool = True,
    ) -> LTX2FlowMatchSchedulerOutput | tuple[torch.FloatTensor, ...]:
        if isinstance(timestep, int | torch.IntTensor | torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as "
                    "timesteps to `LTX2FlowMatchScheduler.step()` is not supported. "
                    "Make sure to pass one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            assert self.step_index is not None, "step_index should not be None"
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        if batch is not None and batch.rollout:
            if not self.already_prepared_rollout(batch):
                raise RuntimeError("Rollout not prepared before step")
            prev_sample = self.flow_sde_sampling(
                batch, model_output, sample, current_sigma, next_sigma, generator
            )
        else:
            if self.config.stochastic_sampling:
                x0 = sample - current_sigma * model_output
                noise = torch.randn_like(sample)
                prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
            else:
                prev_sample = sample + dt * model_output

        assert self._step_index is not None, "_step_index should not be None"
        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if isinstance(prev_sample, torch.Tensor | float) and not return_dict:
            return (prev_sample,)

        return LTX2FlowMatchSchedulerOutput(prev_sample=prev_sample)

    def _convert_to_karras(
        self, in_sigmas: torch.Tensor | np.ndarray, num_inference_steps: int
    ) -> np.ndarray:
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def _convert_to_exponential(
        self, in_sigmas: torch.Tensor | np.ndarray, num_inference_steps: int
    ) -> np.ndarray:
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        return np.exp(
            np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps)
        )

    def _convert_to_beta(
        self,
        in_sigmas: torch.Tensor | np.ndarray,
        num_inference_steps: int,
        alpha: float = 0.6,
        beta: float = 0.6,
    ) -> np.ndarray:
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        return np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )

    def _time_shift_exponential(
        self, mu: float, sigma: float, t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(
        self, mu: float, sigma: float, t: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        return mu / (mu + (1 / t - 1) ** sigma)

    def add_noise(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.IntTensor,
    ) -> torch.Tensor:
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
            assert timestep.numel() == clean_latent.shape[0]
        elif timestep.ndim == 1:
            if timestep.shape[0] == 1:
                timestep = timestep.expand(clean_latent.shape[0])
            else:
                assert timestep.numel() == clean_latent.shape[0]
        else:
            raise ValueError(f"[add_noise] Invalid timestep shape: {timestep.shape}")

        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * clean_latent + sigma * noise
        return sample.type_as(noise)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | None = None
    ) -> torch.Tensor:
        return sample

    def __len__(self) -> int:
        return self.config.num_train_timesteps


EntryClass = LTX2FlowMatchScheduler
