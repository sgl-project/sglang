# SPDX-License-Identifier: Apache-2.0
# Adapted from Helios diffusers scheduler:
# https://github.com/BestWishYsh/Helios
"""
Helios scheduler implementing flow-matching with UniPC/Euler solvers.

For Phase 1 T2V (stages=1), this simplifies to standard flow-matching
with dynamic shifting and UniPC multistep solver.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HeliosSchedulerOutput:
    prev_sample: torch.FloatTensor
    model_outputs: torch.FloatTensor | None = None
    last_sample: torch.FloatTensor | None = None
    this_order: int | None = None


class HeliosSchedulerConfig:
    """Mimics diffusers config interface for scheduler parameters."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


class HeliosScheduler:
    """
    Helios multi-stage scheduler supporting Euler, UniPC, and DMD solvers.

    For Phase 1 T2V with stages=1, this is a standard flow-matching scheduler
    with optional time shifting and UniPC multistep updates.
    """

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        stages: int = 1,
        stage_range: list | None = None,
        gamma: float = 1 / 3,
        thresholding: bool = False,
        prediction_type: str = "flow_prediction",
        solver_order: int = 2,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list[int] | None = None,
        use_flow_sigmas: bool = True,
        scheduler_type: str = "unipc",
        use_dynamic_shifting: bool = False,
        time_shift_type: str = "linear",
        **kwargs,
    ):
        if stage_range is None:
            # Evenly divide [0, 1] into 3 stages for pyramid SR
            stage_range = [0, 1 / 3, 2 / 3, 1]
        if disable_corrector is None:
            disable_corrector = []

        self.config = HeliosSchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            stages=stages,
            stage_range=stage_range,
            gamma=gamma,
            thresholding=thresholding,
            prediction_type=prediction_type,
            solver_order=solver_order,
            predict_x0=predict_x0,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            disable_corrector=disable_corrector,
            use_flow_sigmas=use_flow_sigmas,
            scheduler_type=scheduler_type,
            use_dynamic_shifting=use_dynamic_shifting,
            time_shift_type=time_shift_type,
        )

        self.timestep_ratios = {}
        self.timesteps_per_stage = {}
        self.sigmas_per_stage = {}
        self.start_sigmas = {}
        self.end_sigmas = {}
        self.ori_start_sigmas = {}

        self.init_sigmas_for_each_stage()
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.gamma = gamma

        if solver_type not in ["bh1", "bh2"]:
            raise NotImplementedError(f"{solver_type} is not implemented")

        self.predict_x0 = predict_x0
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = None
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        self.num_inference_steps = None

    def init_sigmas(self):
        num_train_timesteps = self.config.num_train_timesteps
        shift = self.config.shift

        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps + 1)
        sigmas = 1.0 - alphas
        sigmas = np.flip(shift * sigmas / (1 + (shift - 1) * sigmas))[:-1].copy()
        sigmas = torch.from_numpy(sigmas)
        timesteps = (sigmas * num_train_timesteps).clone()

        self._step_index = None
        self._begin_index = None
        self.timesteps = timesteps
        self.sigmas = sigmas.to("cpu")

    def init_sigmas_for_each_stage(self):
        self.init_sigmas()

        stage_distance = []
        stages = self.config.stages
        training_steps = self.config.num_train_timesteps
        stage_range = self.config.stage_range

        for i_s in range(stages):
            start_indice = int(stage_range[i_s] * training_steps)
            start_indice = max(start_indice, 0)
            end_indice = int(stage_range[i_s + 1] * training_steps)
            end_indice = min(end_indice, training_steps)
            start_sigma = self.sigmas[start_indice].item()
            end_sigma = (
                self.sigmas[end_indice].item() if end_indice < training_steps else 0.0
            )
            self.ori_start_sigmas[i_s] = start_sigma

            if i_s != 0:
                ori_sigma = 1 - start_sigma
                gamma = self.config.gamma
                corrected_sigma = (
                    1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                ) * ori_sigma
                start_sigma = 1 - corrected_sigma

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[i_s] = start_sigma
            self.end_sigmas[i_s] = end_sigma

        tot_distance = sum(stage_distance)
        for i_s in range(stages):
            if i_s == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:i_s]) / tot_distance
            if i_s == stages - 1:
                # Use value just below 1.0 to avoid out-of-bounds indexing
                end_ratio = 1.0 - 1e-16
            else:
                end_ratio = sum(stage_distance[: i_s + 1]) / tot_distance
            self.timestep_ratios[i_s] = (start_ratio, end_ratio)

        for i_s in range(stages):
            timestep_ratio = self.timestep_ratios[i_s]
            # Clamp to max valid timestep (num_train_timesteps - 1)
            timestep_max = min(
                self.timesteps[int(timestep_ratio[0] * training_steps)], 999
            )
            timestep_min = self.timesteps[
                min(int(timestep_ratio[1] * training_steps), training_steps - 1)
            ]
            timesteps = np.linspace(timestep_max, timestep_min, training_steps + 1)
            self.timesteps_per_stage[i_s] = (
                timesteps[:-1]
                if isinstance(timesteps, torch.Tensor)
                else torch.from_numpy(timesteps[:-1])
            )
            # Sigma range [0.999, 0]: start just below 1.0 to avoid singularity
            stage_sigmas = np.linspace(0.999, 0, training_steps + 1)
            self.sigmas_per_stage[i_s] = torch.from_numpy(stage_sigmas[:-1])

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def time_shift(self, mu, sigma, t):
        if self.config.time_shift_type == "exponential":
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
        elif self.config.time_shift_type == "linear":
            return mu / (mu + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int,
        stage_index: int | None = None,
        device: str | torch.device = None,
        sigmas=None,
        mu=None,
        is_amplify_first_chunk: bool = False,
    ):
        if self.config.scheduler_type == "dmd":
            if is_amplify_first_chunk:
                num_inference_steps = num_inference_steps * 2 + 1
            else:
                num_inference_steps = num_inference_steps + 1

        self.num_inference_steps = num_inference_steps
        self.init_sigmas()

        if self.config.stages == 1:
            if sigmas is None:
                sigmas = np.linspace(
                    1,
                    1 / self.config.num_train_timesteps,
                    num_inference_steps + 1,
                )[:-1].astype(np.float32)
                if self.config.shift != 1.0:
                    assert not self.config.use_dynamic_shifting
                    sigmas = self.time_shift(self.config.shift, 1.0, sigmas)
            timesteps = (sigmas * self.config.num_train_timesteps).copy()
            sigmas = torch.from_numpy(sigmas)
        else:
            stage_timesteps = self.timesteps_per_stage[stage_index]
            timesteps = np.linspace(
                stage_timesteps[0].item(),
                stage_timesteps[-1].item(),
                num_inference_steps,
            )
            stage_sigmas = self.sigmas_per_stage[stage_index]
            ratios = np.linspace(
                stage_sigmas[0].item(), stage_sigmas[-1].item(), num_inference_steps
            )
            sigmas = torch.from_numpy(ratios)

        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1)]).to(device=device)

        self._step_index = None
        self.reset_scheduler_history()

        if self.config.scheduler_type == "dmd":
            self.timesteps = self.timesteps[:-1]
            self.sigmas = torch.cat([self.sigmas[:-2], self.sigmas[-1:]])

        if self.config.use_dynamic_shifting:
            assert self.config.shift == 1.0
            self.sigmas = self.time_shift(mu, 1.0, self.sigmas)
            if self.config.stages == 1:
                self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
            else:
                self.timesteps = self.timesteps_per_stage[
                    stage_index
                ].min() + self.sigmas[:-1] * (
                    self.timesteps_per_stage[stage_index].max()
                    - self.timesteps_per_stage[stage_index].min()
                )

    # ---------------------------------- Euler ----------------------------------
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step_euler(
        self,
        model_output: torch.FloatTensor,
        timestep=None,
        sample: torch.FloatTensor = None,
        return_dict: bool = True,
        **kwargs,
    ) -> HeliosSchedulerOutput | tuple:
        if self.step_index is None:
            self._step_index = 0

        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return HeliosSchedulerOutput(prev_sample=prev_sample)

    # ---------------------------------- UniPC ----------------------------------
    def _sigma_to_alpha_sigma_t(self, sigma):
        if self.config.use_flow_sigmas:
            alpha_t = 1 - sigma
            sigma_t = torch.clamp(sigma, min=1e-8)
        else:
            alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
            sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def convert_model_output(self, model_output, sample=None, sigma=None, **kwargs):
        flag = False
        if sigma is None:
            flag = True
            sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "flow_prediction":
                if flag:
                    sigma_t = self.sigmas[self.step_index]
                else:
                    sigma_t = sigma
                x0_pred = sample - sigma_t * model_output
            elif self.config.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type {self.config.prediction_type} not supported"
                )
            return x0_pred
        else:
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                return (sample - alpha_t * model_output) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                return alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type {self.config.prediction_type} not supported"
                )

    def multistep_uni_p_bh_update(
        self, model_output, sample=None, order=None, sigma=None, sigma_next=None
    ):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample

        if sigma_next is None and sigma is None:
            sigma_t, sigma_s0 = (
                self.sigmas[self.step_index + 1],
                self.sigmas[self.step_index],
            )
        else:
            sigma_t, sigma_s0 = sigma_next, sigma
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = (
                torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            )
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = (
                torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            )
            x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output,
        last_sample=None,
        this_sample=None,
        order=None,
        sigma_before=None,
        sigma=None,
    ):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        if sigma_before is None and sigma is None:
            sigma_t, sigma_s0 = (
                self.sigmas[self.step_index],
                self.sigmas[self.step_index - 1],
            )
        else:
            sigma_t, sigma_s0 = sigma, sigma_before
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = (
                torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
                if D1s is not None
                else 0
            )
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = (
                torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
                if D1s is not None
                else 0
            )
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t.to(x.dtype)

    def step_unipc(
        self,
        model_output,
        timestep=None,
        sample=None,
        return_dict: bool = True,
        **kwargs,
    ) -> HeliosSchedulerOutput | tuple:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', run 'set_timesteps' first"
            )

        if self.step_index is None:
            self._step_index = 0

        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(
                self.config.solver_order, len(self.timesteps) - self.step_index
            )
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return HeliosSchedulerOutput(prev_sample=prev_sample)

    # ---------------------------------- DMD ----------------------------------
    def add_noise(self, original_samples, noise, timestep, sigmas, timesteps):
        sigmas = sigmas.to(noise.device)
        timesteps = timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def convert_flow_pred_to_x0(self, flow_pred, xt, timestep, sigmas, timesteps):
        original_dtype = flow_pred.dtype
        device = flow_pred.device
        flow_pred, xt, sigmas, timesteps = (
            x.double().to(device) for x in (flow_pred, xt, sigmas, timesteps)
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def step_dmd(
        self,
        model_output: torch.FloatTensor,
        timestep=None,
        sample: torch.FloatTensor = None,
        return_dict: bool = True,
        cur_sampling_step: int = 0,
        dmd_noisy_tensor: torch.FloatTensor | None = None,
        dmd_sigmas: torch.FloatTensor | None = None,
        dmd_timesteps: torch.FloatTensor | None = None,
        all_timesteps: torch.FloatTensor | None = None,
        **kwargs,
    ) -> HeliosSchedulerOutput | tuple:
        pred_image_or_video = self.convert_flow_pred_to_x0(
            flow_pred=model_output,
            xt=sample,
            timestep=torch.full(
                (model_output.shape[0],),
                timestep,
                dtype=torch.long,
                device=model_output.device,
            ),
            sigmas=dmd_sigmas,
            timesteps=dmd_timesteps,
        )
        if cur_sampling_step < len(all_timesteps) - 1:
            prev_sample = self.add_noise(
                pred_image_or_video,
                dmd_noisy_tensor,
                torch.full(
                    (model_output.shape[0],),
                    all_timesteps[cur_sampling_step + 1],
                    dtype=torch.long,
                    device=model_output.device,
                ),
                sigmas=dmd_sigmas,
                timesteps=dmd_timesteps,
            )
        else:
            prev_sample = pred_image_or_video

        if not return_dict:
            return (prev_sample,)
        return HeliosSchedulerOutput(prev_sample=prev_sample)

    # ---------------------------------- Main step ----------------------------------
    def step(
        self,
        model_output,
        timestep=None,
        sample=None,
        return_dict: bool = True,
        **kwargs,
    ) -> HeliosSchedulerOutput | tuple:
        if self.config.scheduler_type == "euler":
            return self.step_euler(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict,
            )
        elif self.config.scheduler_type == "unipc":
            return self.step_unipc(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict,
            )
        elif self.config.scheduler_type == "dmd":
            return self.step_dmd(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"Scheduler type '{self.config.scheduler_type}' not implemented"
            )

    def reset_scheduler_history(self):
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.disable_corrector = self.config.disable_corrector
        self.solver_p = None
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

    def set_shift(self, shift: float):
        """Update the shift parameter (called by SchedulerLoader after loading)."""
        self.config.shift = shift
        self.shift = shift

    def __len__(self):
        return self.config.num_train_timesteps


# Alias for Helios-Distilled which uses "HeliosDMDScheduler" in scheduler_config.json
HeliosDMDScheduler = HeliosScheduler

EntryClass = [HeliosScheduler, "HeliosDMDScheduler"]
