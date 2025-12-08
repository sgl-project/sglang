# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for all pipeline executors.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
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
        super().__init__(stage_name=name, timings=None, simple_log=True, logger=logger)


class PipelineExecutor(ABC):
    """
    Abstract base class for all pipeline executors.

    Executors orchestrate the execution of pipeline, with managing the parallel and communications required by stages

    """

    def __init__(self, server_args):
        self.server_args = server_args

    @abstractmethod
    def execute(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
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
    def profile_execution(self, batch: Req, check_rank: int = 0, dump_rank: int = 0):
        """
        Context manager for profiling execution.
        """
        do_profile = batch.profile

        if not do_profile:
            yield
            return

        request_id = batch.request_id
        profiler = SGLDiffusionProfiler(
            request_id=request_id,
            rank=check_rank,
            full_profile=batch.profile_all_stages,
            num_steps=batch.num_profiled_timesteps,
            num_inference_steps=batch.num_inference_steps,
        )
        try:
            yield
        finally:
            should_export = check_rank == 0
            profiler.stop(export_trace=should_export, dump_rank=dump_rank)
