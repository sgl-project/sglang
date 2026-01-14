# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0
"""
Pass-through scheduler for ComfyUI integration.

This scheduler does not modify latents - it simply returns the input sample unchanged.
The actual denoising logic is handled by ComfyUI.
"""

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ComfyUIPassThroughSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor`): The input sample unchanged (pass-through).
    """

    prev_sample: torch.FloatTensor


class ComfyUIPassThroughScheduler(BaseScheduler, ConfigMixin, SchedulerMixin):
    """
    Pass-through scheduler for ComfyUI integration.

    This scheduler does not modify latents. It is used when the denoising logic
    is handled externally by ComfyUI. The scheduler simply returns the input
    sample unchanged, allowing ComfyUI to manage the denoising process.

    Usage:
        - num_inference_steps is always 1 (each step is handled separately)
        - timesteps are provided externally by ComfyUI
        - step() returns the input sample unchanged
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps=1000,
        *args,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        # Initialize timesteps as empty - will be set externally
        self.timesteps = torch.tensor([], dtype=torch.long)
        self.shift = 0.0
        self._step_index = 0  # Track current step index
        self._begin_index: int | None = None  # For compatibility with DenoisingStage

    def set_timesteps(
        self,
        num_inference_steps=1,  # Always 1 for ComfyUI
        timesteps=None,  # Can be provided externally
        device=None,
        **kwargs,
    ):
        """
        Set timesteps. For ComfyUI, timesteps are provided externally.

        Args:
            num_inference_steps: Ignored (always 1 for ComfyUI)
            timesteps: External timesteps provided by ComfyUI
            device: Device to place timesteps on
        """
        if timesteps is not None:
            # Use externally provided timesteps
            if isinstance(timesteps, torch.Tensor):
                self.timesteps = timesteps
            else:
                self.timesteps = torch.tensor(timesteps, dtype=torch.long)
            if device is not None:
                self.timesteps = self.timesteps.to(device)
        else:
            # Create a single timestep if none provided
            if device is None:
                device = torch.device("cpu")
            self.timesteps = torch.tensor([0], dtype=torch.long, device=device)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor | int,
        sample: torch.FloatTensor,
        return_dict: bool = False,
        **kwargs,
    ) -> tuple | ComfyUIPassThroughSchedulerOutput:
        """
        Pass-through step: returns the input sample unchanged.

        This scheduler does not modify latents. The actual denoising is handled
        by ComfyUI, so we simply return the input sample as-is.

        Args:
            model_output: Predicted noise (ignored, but kept for API compatibility)
            timestep: Current timestep (ignored, but kept for API compatibility)
            sample: Input latents (returned unchanged)
            return_dict: Whether to return a dict or tuple

        Returns:
            The input sample unchanged (prev_sample = sample)
        """
        # Increment step index for tracking
        self._step_index += 1

        # Simply return the input sample unchanged
        prev_sample = sample

        if not return_dict:
            return (prev_sample,)

        return ComfyUIPassThroughSchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | None = None
    ) -> torch.Tensor:
        """
        Scale model input. For pass-through scheduler, returns input unchanged.

        Args:
            sample: Input sample
            timestep: Timestep (ignored)

        Returns:
            Input sample unchanged
        """
        return sample

    def set_shift(self, shift: float) -> None:
        """
        Set shift parameter (no-op for pass-through scheduler).

        Args:
            shift: Shift value (ignored)
        """
        self.shift = shift

    def set_begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index: The begin index for the scheduler.
        """
        self._begin_index = begin_index

    @property
    def begin_index(self) -> int | None:
        """
        The index for the first timestep.
        """
        return self._begin_index

    @property
    def step_index(self) -> int:
        """
        The index counter for current timestep.
        """
        return self._step_index

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples. For pass-through scheduler, returns original samples.

        Args:
            original_samples: Original clean samples
            noise: Noise to add (ignored)
            timestep: Timestep (ignored)

        Returns:
            Original samples unchanged
        """
        return original_samples


EntryClass = ComfyUIPassThroughScheduler
