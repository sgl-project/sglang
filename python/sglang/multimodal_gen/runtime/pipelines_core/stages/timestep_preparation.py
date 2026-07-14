# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Timestep preparation stages for diffusion pipelines.

This module contains implementations of timestep preparation stages for diffusion pipelines.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TimestepPreparationFingerprint:
    num_inference_steps: int
    timesteps: Any
    sigmas: Any
    n_tokens: int | None
    height: int | None
    width: int | None
    num_frames: int | None


class TimestepPreparationStage(PipelineStage):
    """
    Stage for preparing timesteps for the diffusion process.

    This stage handles the preparation of the timestep sequence that will be used
    during the diffusion process.
    """

    deduplicated_tensor_tree_output_fields = ("timesteps", "sigmas")
    deduplicated_deepcopy_output_fields = ("scheduler",)
    deduplicated_extra_tensor_tree_output_keys = ("mu",)

    def __init__(
        self,
        scheduler,
        prepare_extra_set_timesteps_kwargs: (
            list[Callable[[Req, ServerArgs], Tuple[str, Any]]] | None
        ) = None,
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.prepare_extra_set_timesteps_kwargs = list(
            prepare_extra_set_timesteps_kwargs or []
        )

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



        Returns:
            The batch with prepared timesteps.
        """
        if batch.scheduler is not None and batch.timesteps is not None:
            return batch

        scheduler_template = self.scheduler
        if batch.rollout:
            # The rollout SDE/log-prob path may need an RL-capable scheduler
            # the pipeline does not serve with; the serving->rollout mapping
            # is owned by post_training and the resolved scheduler by this
            # request, so a serving-only engine never initializes one.
            from sglang.multimodal_gen.runtime.post_training.rollout_scheduler_registry import (
                resolve_rollout_scheduler,
            )

            scheduler_template = resolve_rollout_scheduler(self.scheduler)
        scheduler = get_or_create_request_scheduler(batch, scheduler_template)
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas
        n_tokens = batch.n_tokens

        sigmas = server_args.pipeline_config.prepare_sigmas(sigmas, num_inference_steps)
        batch.sigmas = sigmas

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
            if key == "mu":
                batch.extra["mu"] = value

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
        batch.scheduler = scheduler
        if not batch.is_warmup:
            self.log_debug("timesteps: %s", timesteps)
        return batch

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> TimestepPreparationFingerprint:
        return TimestepPreparationFingerprint(
            num_inference_steps=batch.num_inference_steps,
            timesteps=self.freeze_for_dedup(batch.timesteps),
            sigmas=self.freeze_for_dedup(batch.sigmas),
            n_tokens=batch.n_tokens,
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
        )

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
        if (
            batch.is_warmup
            and isinstance(batch.timesteps, torch.Tensor)
            and torch.isnan(batch.timesteps).any()
        ):
            # diffusers flow-match scheduler can emit NaN for one-step warmup
            batch.timesteps = torch.ones(
                (1,), dtype=torch.float32, device=get_local_torch_device()
            )

        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        return result


class DMDTimestepPreparationStage(PipelineStage):
    """Prepare distilled DMD timesteps from pipeline config."""

    deduplicated_tensor_tree_output_fields = ("timesteps",)
    deduplicated_deepcopy_output_fields = ("scheduler",)

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if batch.scheduler is not None and batch.timesteps is not None:
            return batch

        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        num_train_timesteps = getattr(scheduler, "num_train_timesteps", None)
        if num_train_timesteps is None:
            num_train_timesteps = scheduler.config.num_train_timesteps
        num_train_timesteps = int(num_train_timesteps)
        scheduler.set_timesteps(num_train_timesteps)

        timesteps = torch.tensor(
            server_args.pipeline_config.dmd_denoising_steps, dtype=torch.long
        ).cpu()
        if server_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat(
                (scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            timesteps = scheduler_timesteps[num_train_timesteps - timesteps]

        batch.timesteps = timesteps.to(get_local_torch_device())
        batch.scheduler = scheduler
        if not batch.is_warmup:
            self.log_debug("DMD timesteps: %s", batch.timesteps)
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dmd_denoising_steps",
            server_args.pipeline_config.dmd_denoising_steps,
            V.list_not_empty,
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("scheduler", batch.scheduler, V.not_none)
        return result
