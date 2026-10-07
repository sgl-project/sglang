# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusers scheduler for distilled Kandinsky 6 PiFlow checkpoints."""

from __future__ import annotations

import torch
from diffusers.configuration_utils import register_to_config

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)


class DXPolicy:
    def __init__(  # noqa: PLR0913
        self,
        denoising_output: torch.Tensor,
        x_t_src: torch.Tensor,
        sigma_t_src: torch.Tensor,
        segment_size: float | torch.Tensor = 1.0,
        shift: float = 1.0,
        mode: str = "grid",
        eps: float = 1e-4,
    ) -> None:
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps
        if mode not in ("grid", "polynomial"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

        self.sigma_t_src = sigma_t_src.reshape(
            *sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1])
        )
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        segment = segment_size
        if isinstance(segment, torch.Tensor) and segment.dim() < self.raw_t_src.dim():
            segment = segment.reshape(
                *segment.size(), *((self.raw_t_src.dim() - segment.dim()) * [1])
            )
        self.raw_t_dst = (self.raw_t_src - segment).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)
        self.denoising_output_x_0 = self._u_to_x_0(
            denoising_output, self.x_t_src, self.sigma_t_src
        )

    @staticmethod
    def _interpolate(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        if n < 2:  # noqa: PLR2004
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        indices = torch.stack([t0, t1], dim=1)
        values = torch.gather(x, dim=1, index=indices.expand(-1, -1, *x.shape[2:]))
        return (t1 - t) * values[:, 0] + (t - t0) * values[:, 1]

    def _unwarp_t(self, sigma_t: torch.Tensor) -> torch.Tensor:
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _u_to_x_0(
        denoising_output: torch.Tensor,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        return x_t.unsqueeze(1) - sigma_t.unsqueeze(1) * denoising_output

    def pi(self, x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        if self.mode == "grid":
            x_0 = self._interpolate(
                self.denoising_output_x_0,
                (raw_t - self.raw_t_dst) / self.segment_size,
            )
        else:
            p_order = self.denoising_output_x_0.size(1)
            diff_t = self.raw_t_src - raw_t
            basis = torch.stack([diff_t**i for i in range(p_order)], dim=1)
            x_0 = torch.sum(basis * self.denoising_output_x_0, dim=1)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)


def shift_timesteps(t: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * t / (1 + (shift - 1) * t)


def policy_rollout_fm(  # noqa: PLR0913
    x_t_start: torch.Tensor,
    sigma_t_start: torch.Tensor,
    raw_t_start: torch.Tensor,
    raw_t_end: torch.Tensor,
    total_substeps: int,
    policy: DXPolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_batches = x_t_start.size(0)
    ndim = x_t_start.dim()
    shape = (num_batches, *((ndim - 1) * [1]))
    raw_t_start = raw_t_start.reshape(shape)
    raw_t_end = raw_t_end.reshape(shape)
    sigma_t = sigma_t_start.reshape(shape)

    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps
    max_num_substeps = num_substeps.max()

    raw_t = raw_t_start
    x_t = x_t_start
    for substep_id in range(max_num_substeps.item()):
        velocity = policy.pi(x_t, sigma_t)
        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + velocity * (sigma_t_minus - sigma_t)

        active_mask = num_substeps > substep_id
        x_t = torch.where(active_mask, x_t_minus, x_t)
        sigma_t = torch.where(active_mask, sigma_t_minus, sigma_t)
        raw_t = torch.where(active_mask, raw_t_minus, raw_t)

    return x_t, sigma_t, sigma_t.flatten() * 1_000


class PiflowScheduler(FlowMatchEulerDiscreteScheduler):
    is_piflow = True

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        n_grid: int = 10,
        nfe: int | None = None,
        eps: float = 1e-6,
        final_step_size_scale: float = 0.5,
        num_policy_substeps: int = 128,
    ) -> None:
        if n_grid < 2:
            raise ValueError(f"PiflowScheduler requires n_grid >= 2, got {n_grid}")
        if eps <= 0:
            raise ValueError(f"PiflowScheduler requires eps > 0, got {eps}")
        if not 0 < final_step_size_scale <= 1:
            raise ValueError("PiflowScheduler requires 0 < final_step_size_scale <= 1")
        if num_policy_substeps < 1:
            raise ValueError("PiflowScheduler requires num_policy_substeps >= 1")
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
        self.n_grid = int(n_grid)
        self.eps = float(eps)
        self.final_step_size_scale = float(final_step_size_scale)
        self.num_policy_substeps = int(num_policy_substeps)
        self._piflow_raw_timesteps = torch.empty(0)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        if sigmas is not None or mu is not None or timesteps is not None:
            raise ValueError(
                "PiflowScheduler only supports its configured distilled timestep schedule"
            )
        if num_inference_steps is None or num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be positive, got {num_inference_steps}"
            )
        one_minus_final = 1.0 - self.final_step_size_scale
        segment = 1.0 / (num_inference_steps - one_minus_final)
        raw = (
            1.0
            - torch.arange(num_inference_steps, dtype=torch.float32, device=device)
            * segment
        )
        sigmas = shift_timesteps(raw, float(self.config.shift))
        self.num_inference_steps = int(num_inference_steps)
        self._piflow_raw_timesteps = raw
        self.timesteps = sigmas * self.config.num_train_timesteps
        self.sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        self._step_index = None
        self._begin_index = None

    def _to_grid(
        self, model_output: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        if (
            model_output.ndim != sample.ndim
            or model_output.shape[:-1] != sample.shape[:-1]
        ):
            raise ValueError(
                "Piflow model output must match sample shape except for the output channels: "
                f"got {tuple(model_output.shape)} for sample {tuple(sample.shape)}"
            )
        if model_output.shape[-1] % self.n_grid != 0:
            raise ValueError(
                f"Piflow model output channels {model_output.shape[-1]} are not divisible by n_grid={self.n_grid}"
            )
        output_dim = model_output.shape[-1] // self.n_grid
        if output_dim != sample.shape[-1]:
            raise ValueError(
                "Piflow model output channels do not match the sample: "
                f"expected {sample.shape[-1] * self.n_grid}, got {model_output.shape[-1]}"
            )
        return model_output.reshape(
            *model_output.shape[:-1], self.n_grid, output_dim
        ).movedim(-2, 1)

    def _policy_step(
        self, model_output: torch.Tensor, sample: torch.Tensor, step_index: int
    ) -> torch.Tensor:
        model_output = self._to_grid(model_output, sample)
        raw_src = self._piflow_raw_timesteps[step_index].to(device=sample.device)
        raw_dst = (
            self._piflow_raw_timesteps[step_index + 1]
            if step_index + 1 < self._piflow_raw_timesteps.numel()
            else self._piflow_raw_timesteps.new_full((), self.eps)
        ).to(device=sample.device)
        sigma_src = self.sigmas[step_index].to(device=sample.device)
        token_shape = (sample.shape[0], *((sample.ndim - 1) * [1]))
        sigma = sigma_src.expand(sample.shape[0]).reshape(token_shape)
        segment = (raw_src - raw_dst).expand(sample.shape[0])
        policy = DXPolicy(
            model_output,
            sample,
            sigma,
            segment,
            shift=float(self.config.shift),
            mode="grid",
            eps=self.eps,
        )
        updated, _, _ = policy_rollout_fm(
            sample,
            sigma,
            raw_src.expand(sample.shape[0]),
            raw_dst.expand(sample.shape[0]),
            self.num_policy_substeps,
            policy,
        )
        return updated.to(dtype=sample.dtype)

    def _step_index_for(self, timestep: torch.Tensor | float) -> int:
        if self.step_index is None:
            if self.begin_index is not None:
                self._step_index = self.begin_index
            else:
                schedule_timesteps = self.timesteps
                if not isinstance(schedule_timesteps, torch.Tensor):
                    schedule_timesteps = torch.as_tensor(
                        schedule_timesteps, dtype=torch.float32
                    )
                timestep = torch.as_tensor(
                    timestep,
                    device=schedule_timesteps.device,
                    dtype=schedule_timesteps.dtype,
                )
                indices = torch.nonzero(schedule_timesteps == timestep).flatten()
                if not indices.numel():
                    raise ValueError(
                        f"timestep {timestep.item()} is not in the Piflow schedule"
                    )
                position = 1 if indices.numel() > 1 else 0
                self._step_index = int(indices[position].item())
        if self.step_index is None or self.step_index >= self.num_inference_steps:
            raise RuntimeError(
                "PiflowScheduler.step called after the schedule was exhausted"
            )
        return int(self.step_index)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple:
        if isinstance(timestep, int) or isinstance(
            timestep, (torch.IntTensor, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps to PiflowScheduler.step() is not supported; "
                "pass a value from scheduler.timesteps instead"
            )
        step_index = self._step_index_for(timestep)
        # PiFlow's policy rollout performs its update in float32. Keep that
        # precision across outer steps, matching the native sampler.
        updated = self._policy_step(model_output, sample.to(torch.float32), step_index)
        self._step_index += 1
        if return_dict:
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=updated)
        return (updated,)


EntryClass = PiflowScheduler
