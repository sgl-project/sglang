# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for all pipeline executors.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

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


class PipelineExecutor(ABC):
    """
    Abstract base class for all pipeline executors.

    Executors orchestrate the execution of pipeline, with managing the parallel and communications required by stages

    """

    def __init__(self, server_args):
        self.server_args = server_args
        self.component_residency_manager = None

    def begin_component_residency_request(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        self.component_residency_manager.begin_request(stages, batch, server_args)

    def before_stage(
        self,
        stage: "PipelineStage",
        stage_index: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        stage.set_component_residency_manager(self.component_residency_manager)
        self.component_residency_manager.before_stage(
            stage, stage_index, batch, server_args
        )

    def after_stage(self, stage_index: int) -> None:
        self.component_residency_manager.after_stage(stage_index)

    def finish_component_residency_request(self) -> None:
        self.component_residency_manager.finish_request()

    def execute_with_profiling(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:

        with self.profile_execution(batch, dump_rank=0):
            batch = self.execute(stages, batch, server_args)

        return batch

    def execute_group_with_profiling(
        self,
        stages: List["PipelineStage"],
        batches: list[Req],
        server_args: ServerArgs,
    ):
        """Execute a grouped request under the same profiler as a single request."""
        with self.profile_execution(batches[0], dump_rank=0):
            batches = self.execute_group(stages, batches, server_args)
        return batches

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

    def execute_group(
        self,
        stages: List["PipelineStage"],
        batches: list[Req],
        server_args: ServerArgs,
    ):
        """Execute all pipeline stages over a group of independent requests.

        Executors own cross-rank scheduling, while stages own whether duplicate
        work can be removed. The base executor simply calls
        ``stage.run_grouped_requests`` for each stage in order.
        """
        for stage in stages:
            batches = stage.run_grouped_requests(batches, server_args)
        return batches

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
