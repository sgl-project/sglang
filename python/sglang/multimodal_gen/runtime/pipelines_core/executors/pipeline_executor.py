# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for all pipeline executors.
"""

import contextlib
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

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


class SGLDiffusionProfiler:
    """
    A wrapper around torch.profiler to simplify usage in pipelines.
    Supports both full profiling and scheduled profiling.
    """

    def __init__(
        self,
        request_id: str | None = None,
        rank: int = 0,
        full_profile: bool = True,
        num_steps: int | None = None,
        log_dir: str = "./logs",
    ):
        self.request_id = request_id or "profile_trace"
        self.rank = rank
        self.full_profile = full_profile
        self.log_dir = log_dir

        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            pass

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        if self.full_profile:
            self.profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
            )
        else:
            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=1,
                    active=num_steps if num_steps else 2,
                    repeat=5,
                ),
                on_trace_ready=lambda _: torch.profiler.tensorboard_trace_handler(
                    self.log_dir
                ),
                record_shapes=True,
                with_stack=True,
            )

    def start(self):
        logger.info("Starting Profiler...")
        self.profiler.start()

    def step(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.profiler.step()

    def stop(self, export_trace: bool = True, dump_rank: int | None = None):
        logger.info("Stopping Profiler...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.profiler.stop()

        if export_trace and self.full_profile:
            self.export_trace(dump_rank)

    def export_trace(self, dump_rank: int | None = None):
        if dump_rank is None:
            dump_rank = self.rank

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            trace_path = os.path.abspath(
                os.path.join(
                    self.log_dir,
                    f"{self.request_id}-global-rank{dump_rank}.trace.json.gz",
                )
            )
            logger.info(f"Saving profiler traces to: {trace_path}")
            self.profiler.export_chrome_trace(trace_path)
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When used as context manager, we usually assume full profiling
        # and want to export at the end
        self.stop(export_trace=True)


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

        Args:
            batch: The request batch.
            check_rank: The rank used to check if we should export the trace (usually local rank or cfg rank).
                        If 0, trace is exported.
            dump_rank: The rank used in the output filename (usually world rank).
        """
        do_full_stages_profile = bool(batch.profile and batch.full_stages)

        if not do_full_stages_profile:
            yield
            return

        request_id = batch.request_id
        profiler = SGLDiffusionProfiler(
            request_id=request_id,
            rank=check_rank,
            full_profile=True,
        )

        profiler.start()
        try:
            yield
        finally:
            should_export = check_rank == 0
            profiler.stop(export_trace=should_export, dump_rank=dump_rank)
