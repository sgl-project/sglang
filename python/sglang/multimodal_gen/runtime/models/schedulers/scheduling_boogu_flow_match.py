# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def _lin_mu(
    seq_len: int,
    base_shift: float,
    max_shift: float,
    base_image_seq_len: int,
    max_image_seq_len: int,
) -> float:
    slope = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    return slope * seq_len + base_shift - slope * base_image_seq_len


def _time_shift_v1(t: np.ndarray, mu: float, sigma: float = 1.0) -> np.ndarray:
    eps = 1e-8
    t1 = np.clip(1.0 - t, eps, 1.0 - eps)
    num = math.exp(mu)
    return (1.0 - num / (num + np.power(1.0 / t1 - 1.0, sigma))).astype(np.float32)


def _time_shift_v2(t: np.ndarray, m: float) -> np.ndarray:
    return (t / (m - m * t + t)).astype(np.float32)


class BooguFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    _compatibles: list[str] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        do_shift: bool = True,
        dynamic_time_shift: bool = True,
        time_shift_version: str = "v1",
        seq_len: int | None = None,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        time_shift_v2_half_scaling_factor: float = 60.0,
    ):
        self.timesteps = torch.linspace(
            0, 1, num_train_timesteps + 1, dtype=torch.float32
        )[:-1]
        self._timesteps = self.timesteps
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.time_shift_v2_scaling_factor = time_shift_v2_half_scaling_factor * 2

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def _shifted_schedule(
        self, num_inference_steps: int, num_tokens: int | None
    ) -> np.ndarray:
        t = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]
        if not self.config.do_shift:
            return t

        version = self.config.time_shift_version
        if self.config.dynamic_time_shift:
            if num_tokens is None or num_tokens <= 0:
                return t
            if version == "v1":
                mu = _lin_mu(
                    seq_len=max(1, int(num_tokens) // 4),
                    base_shift=self.config.base_shift,
                    max_shift=self.config.max_shift,
                    base_image_seq_len=self.config.base_image_seq_len,
                    max_image_seq_len=self.config.max_image_seq_len,
                )
                return _time_shift_v1(t, mu)
            if version == "v2":
                return _time_shift_v2(
                    t, math.sqrt(num_tokens) / self.time_shift_v2_scaling_factor
                )
            return t

        seq_len = self.config.seq_len
        if seq_len is None or seq_len <= 0:
            return t
        if version == "v1":
            mu = _lin_mu(
                seq_len=int(seq_len),
                base_shift=self.config.base_shift,
                max_shift=self.config.max_shift,
                base_image_seq_len=self.config.base_image_seq_len,
                max_image_seq_len=self.config.max_image_seq_len,
            )
            return _time_shift_v1(t, mu)
        if version == "v2":
            return _time_shift_v2(
                t, math.sqrt(seq_len) / self.time_shift_v2_scaling_factor
            )
        return t

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[float] | None = None,
        num_tokens: int | None = None,
    ) -> None:
        if timesteps is None:
            self.num_inference_steps = num_inference_steps
            schedule = self._shifted_schedule(num_inference_steps, num_tokens)
        else:
            self.num_inference_steps = len(timesteps)
            schedule = np.asarray(timesteps, dtype=np.float32)

        self.timesteps = torch.from_numpy(schedule).to(
            dtype=torch.float32, device=device
        )
        self._timesteps = torch.cat(
            [self.timesteps, torch.ones(1, device=self.timesteps.device)]
        )
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self._timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        return indices[1 if len(indices) > 1 else 0].item()

    def _init_step_index(self, timestep) -> None:
        if self._begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self._timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        if self._step_index is None:
            self._init_step_index(timestep)

        t = self._timesteps[self._step_index]
        t_next = self._timesteps[self._step_index + 1]
        prev_sample = sample.to(torch.float32) + (t_next - t) * model_output
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return prev_sample

    def scale_model_input(
        self, sample: torch.Tensor, timestep: float | None = None
    ) -> torch.Tensor:
        return sample

    def __len__(self) -> int:
        return self.config.num_train_timesteps


EntryClass = BooguFlowMatchScheduler
