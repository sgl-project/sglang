# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for all pipeline executors.
"""

import contextlib
import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch

from sglang.multimodal_gen.runtime.distributed import get_world_rank
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler

if TYPE_CHECKING:
    # Only for type checkers; avoids runtime circular import
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage

logger = init_logger(__name__)


class Timer(StageProfiler):
    """
    A wrapper around StageProfiler to maintain backward compatibility.
    It forces simple logging behavior (log start/end) regardless of env vars.
    """

    def __init__(self, name="Stage"):
        super().__init__(
            stage_name=name, logger=logger, metrics=None, log_stage_start_end=True
        )


def _split_stages_by_per_output(
    stages: List["PipelineStage"],
) -> tuple[List["PipelineStage"], List["PipelineStage"]]:
    """Split stages into shared and per-output groups.

    Once the first per-output stage is hit, all subsequent stages
    are treated as per-output to preserve execution order.
    """
    shared: List["PipelineStage"] = []
    per_output: List["PipelineStage"] = []
    hit = False
    for stage in stages:
        if not hit and not stage.requires_per_output_execution:
            shared.append(stage)
        else:
            hit = True
            per_output.append(stage)
    return shared, per_output


def _prepare_single_output_batch(batch: Req, output_index: int) -> Req:
    """Create a shallow copy of batch configured for a single output."""
    sub_batch = copy.copy(batch)

    if isinstance(batch.generator, list) and len(batch.generator) > 1:
        sub_batch.generator = [batch.generator[output_index]]
    if batch.seeds is not None and len(batch.seeds) > 1:
        sub_batch.seeds = [batch.seeds[output_index]]

    sub_batch.latents = None
    sub_batch.num_outputs_per_prompt = 1
    sub_batch._current_output_index = output_index
    return sub_batch


def _merge_output_batches(output_batches: List[OutputBatch]) -> OutputBatch:
    """Merge multiple single-output OutputBatch objects into one."""
    if len(output_batches) == 1:
        return output_batches[0]

    outputs = []
    for ob in output_batches:
        if ob.output is not None:
            if isinstance(ob.output, (list, tuple)):
                outputs.extend(ob.output)
            elif isinstance(ob.output, torch.Tensor):
                outputs.append(ob.output)

    # Cat into (N, ...) tensor so enumerate() yields per-sample slices.
    if outputs and all(isinstance(o, torch.Tensor) for o in outputs):
        try:
            outputs = torch.cat(outputs, dim=0)
        except Exception:
            pass

    # Bool check safe for both list and tensor.
    has_outputs = (
        outputs.numel() > 0 if isinstance(outputs, torch.Tensor) else bool(outputs)
    )

    # Merge trajectory data across outputs.
    # trajectory_timesteps are identical for every output – take the first.
    merged_trajectory_timesteps = None
    for ob in output_batches:
        if ob.trajectory_timesteps is not None:
            merged_trajectory_timesteps = ob.trajectory_timesteps
            break

    # trajectory_latents: concatenate along batch dim (dim 0).
    merged_trajectory_latents = None
    traj_latent_parts = [
        ob.trajectory_latents
        for ob in output_batches
        if ob.trajectory_latents is not None
    ]
    if traj_latent_parts:
        merged_trajectory_latents = torch.cat(traj_latent_parts, dim=0)

    # trajectory_decoded: list[Tensor] per timestep, each shaped [B_i, ...].
    # Concatenate along batch dim so the merged list keeps the same timesteps.
    merged_trajectory_decoded = None
    traj_decoded_parts = [
        ob.trajectory_decoded
        for ob in output_batches
        if ob.trajectory_decoded is not None
    ]
    if traj_decoded_parts:
        merged_trajectory_decoded = [
            torch.cat([part[t] for part in traj_decoded_parts], dim=0)
            for t in range(len(traj_decoded_parts[0]))
        ]

    return OutputBatch(
        output=outputs if has_outputs else None,
        trajectory_timesteps=merged_trajectory_timesteps,
        trajectory_latents=merged_trajectory_latents,
        trajectory_decoded=merged_trajectory_decoded,
        metrics=output_batches[0].metrics,
        noise_pred=None,
    )


class PipelineExecutor(ABC):
    """
    Abstract base class for all pipeline executors.

    Executors orchestrate the execution of pipeline, with managing the parallel and communications required by stages

    """

    def __init__(self, server_args):
        self.server_args = server_args

    def execute_with_profiling(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:

        with self.profile_execution(batch, dump_rank=0):
            batch = self.execute(stages, batch, server_args)

        return batch

    @abstractmethod
    def execute(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Execute the pipeline stages.

        Args:
            stages: A list of pipeline stages to execute.
            batch: The batch to process.
            server_args: The server arguments.

        Returns:
            The processed batch.
        """
        raise NotImplementedError

    @contextlib.contextmanager
    def profile_execution(self, batch: Req, dump_rank: int = 0):
        """
        Context manager for profiling execution.
        """
        do_profile = batch.profile and not batch.is_warmup
        if not do_profile:
            # fast forward
            yield
            return

        request_id = batch.request_id
        rank = get_world_rank()

        profiler = SGLDiffusionProfiler(
            request_id=request_id,
            rank=rank,
            full_profile=batch.profile_all_stages,
            num_steps=batch.num_profiled_timesteps,
            num_inference_steps=batch.num_inference_steps,
        )
        try:
            yield
        finally:
            profiler.stop(dump_rank=dump_rank)
