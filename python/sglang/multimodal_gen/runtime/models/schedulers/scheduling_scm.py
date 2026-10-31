# SPDX-License-Identifier: Apache-2.0
"""TrigFlow sCM sampling (SANA-Sprint) for the shared denoising loop."""

import torch
from diffusers import SCMScheduler as DiffusersSCMScheduler
from diffusers.schedulers.scheduling_scm import SCMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


class SCMScheduler(DiffusersSCMScheduler):
    """Diffusers' ``SCMScheduler`` with SANA-Sprint's per-step model I/O folded in.

    Diffusers' ``SanaSprintPipeline`` rescales the DiT input, converts the DiT's
    flow prediction to TrigFlow and divides the result by ``sigma_data`` inline in
    its loop. Here ``scale_model_input`` and ``step`` do that work, so the shared
    ``DenoisingStage`` loop runs the sampler unchanged. ``timesteps`` holds the
    sCM timesteps the DiT is conditioned on; the TrigFlow angles stay internal.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "trigflow",
        sigma_data: float = 0.5,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            sigma_data=sigma_data,
        )
        if prediction_type != "trigflow":
            raise ValueError(f"Unsupported SCM prediction_type: {prediction_type!r}")
        # Latent preparation scales the initial N(0, I) noise by this.
        self.init_noise_sigma = sigma_data
        self.angles: torch.Tensor | None = None

    def set_shift(self, shift: float) -> None:
        raise ValueError("SCM sampling has no flow shift")

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        timesteps: list[float] | None = None,
        max_timesteps: float = 1.57080,
        intermediate_timesteps: float | None = None,
    ) -> None:
        if timesteps is not None:
            # Explicit TrigFlow angles replace the max/intermediate schedule.
            super().set_timesteps(
                num_inference_steps,
                timesteps=list(timesteps),
                device=device,
                max_timesteps=None,
                intermediate_timesteps=None,
            )
        else:
            # Diffusers' 2-step default; Diffusers rejects an intermediate angle otherwise.
            if intermediate_timesteps is None and num_inference_steps == 2:
                intermediate_timesteps = 1.3
            super().set_timesteps(
                num_inference_steps,
                device=device,
                max_timesteps=max_timesteps,
                intermediate_timesteps=intermediate_timesteps,
            )
        # num_inference_steps + 1 angles, ending at 0 (the clean sample).
        self.angles = self.timesteps
        angles = self.angles[:-1]
        self.timesteps = torch.sin(angles) / (torch.cos(angles) + torch.sin(angles))

    def _scm_timestep(self, timestep) -> torch.Tensor:
        if self.step_index is None:
            self._init_step_index(timestep)
        return self.timesteps[self.step_index]

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        scm_t = self._scm_timestep(timestep).to(sample.device)
        return sample / self.config.sigma_data * torch.sqrt(scm_t**2 + (1 - scm_t) ** 2)

    def step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
    ):
        """One sCM step. ``pred_original_sample`` stays in the ``sigma_data``-scaled
        TrigFlow space on every step; only the final ``prev_sample`` is divided by
        ``sigma_data`` for VAE decoding."""
        if self.num_inference_steps is None:
            raise ValueError("Run `set_timesteps` before `step`")
        sigma_data = self.config.sigma_data
        scm_t = self._scm_timestep(timestep).to(sample.device)
        s = self.angles[self.step_index].to(sample.device)
        t = self.angles[self.step_index + 1].to(sample.device)

        norm = torch.sqrt(scm_t**2 + (1 - scm_t) ** 2)
        model_input = sample / sigma_data * norm
        # A 0-dim scm_t does not promote a bf16 DiT output; upcast explicitly.
        trigflow_pred = (
            (1 - 2 * scm_t) * model_input
            + (1 - 2 * scm_t + 2 * scm_t**2) * model_output.float()
        ) / norm
        trigflow_pred = trigflow_pred * sigma_data

        pred_x0 = torch.cos(s) * sample - torch.sin(s) * trigflow_pred
        self._step_index += 1
        if self._step_index == len(self.timesteps):
            # Leave the sigma_data-scaled TrigFlow space for VAE decoding.
            prev_sample = pred_x0 / sigma_data
        else:
            noise = (
                randn_tensor(
                    model_output.shape, device=model_output.device, generator=generator
                )
                * sigma_data
            )
            prev_sample = torch.cos(t) * pred_x0 + torch.sin(t) * noise

        if not return_dict:
            return (prev_sample, pred_x0)
        return SCMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)


EntryClass = SCMScheduler
