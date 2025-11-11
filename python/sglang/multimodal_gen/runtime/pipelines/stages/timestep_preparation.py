# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Timestep preparation stages for diffusion pipelines.

This module contains implementations of timestep preparation stages for diffusion pipelines.
"""

import inspect
from typing import Any, Callable, Tuple

import numpy as np

from sglang.multimodal_gen.configs.pipelines import FluxPipelineConfig
from sglang.multimodal_gen.configs.pipelines.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class TimestepPreparationStage(PipelineStage):
    """
    Stage for preparing timesteps for the diffusion process.

    This stage handles the preparation of the timestep sequence that will be used
    during the diffusion process.
    """

    def __init__(
        self,
        scheduler,
        prepare_extra_set_timesteps_kwargs: list[
            Callable[[Req, ServerArgs], Tuple[str, Any]]
        ] = [],
    ) -> None:
        self.scheduler = scheduler
        self.prepare_extra_set_timesteps_kwargs = prepare_extra_set_timesteps_kwargs

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Prepare timesteps for the diffusion process.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with prepared timesteps.
        """
        scheduler = self.scheduler
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas
        n_tokens = batch.n_tokens

        is_flux = (
            isinstance(server_args.pipeline_config, FluxPipelineConfig)
            or isinstance(server_args.pipeline_config, QwenImagePipelineConfig)
            or isinstance(server_args.pipeline_config, QwenImageEditPipelineConfig)
        )
        if is_flux:
            sigmas = (
                np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                if sigmas is None
                else sigmas
            )

        # Prepare extra kwargs for set_timesteps
        extra_set_timesteps_kwargs = {}
        if (
            n_tokens is not None
            and "n_tokens" in inspect.signature(scheduler.set_timesteps).parameters
        ):
            extra_set_timesteps_kwargs["n_tokens"] = n_tokens

        for callee in self.prepare_extra_set_timesteps_kwargs:
            key, value = callee(batch, server_args)
            assert isinstance(key, str)
            extra_set_timesteps_kwargs[key] = value

        # Handle custom timesteps or sigmas
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )

        if timesteps is not None:
            accepts_timesteps = (
                "timesteps" in inspect.signature(scheduler.set_timesteps).parameters
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(
                timesteps=timesteps, device=device, **extra_set_timesteps_kwargs
            )
            timesteps = scheduler.timesteps
        elif sigmas is not None:
            accept_sigmas = (
                "sigmas" in inspect.signature(scheduler.set_timesteps).parameters
            )
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(
                sigmas=sigmas, device=device, **extra_set_timesteps_kwargs
            )
            timesteps = scheduler.timesteps
        else:
            scheduler.set_timesteps(
                num_inference_steps, device=device, **extra_set_timesteps_kwargs
            )
            timesteps = scheduler.timesteps

        # Update batch with prepared timesteps
        batch.timesteps = timesteps
        self.log_debug(f"timesteps: {timesteps}")
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify timestep preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("timesteps", batch.timesteps, V.none_or_tensor)
        result.add_check("sigmas", batch.sigmas, V.none_or_list)
        result.add_check("n_tokens", batch.n_tokens, V.none_or_positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify timestep preparation stage outputs."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        return result
