# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for all pipeline executors.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List

import torch

from sglang.multimodal_gen.runtime.distributed import get_world_rank
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range
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
        batch: Any,
        server_args: ServerArgs,
    ) -> None:
        self.component_residency_manager.begin_request(stages, batch, server_args)

    def before_stage(
        self,
        stage: "PipelineStage",
        stage_index: int,
        batch: Any,
        server_args: ServerArgs,
    ) -> None:
        stage.set_component_residency_manager(self.component_residency_manager)
        self.component_residency_manager.before_stage(
            stage, stage_index, batch, server_args
        )

    def finish_component_residency_request(self) -> None:
        self.component_residency_manager.finish_request()

    @contextlib.contextmanager
    def _component_residency_request(
        self,
        stages: List["PipelineStage"],
        payload: Any,
        server_args: ServerArgs,
    ):
        self.begin_component_residency_request(stages, payload, server_args)
        try:
            yield
        finally:
            self.finish_component_residency_request()

    @staticmethod
    def _is_warmup_payload(payload: Any) -> bool:
        if isinstance(payload, list):
            return bool(payload) and all(
                getattr(item, "is_warmup", False) for item in payload
            )
        return getattr(payload, "is_warmup", False)

    def _should_use_stage_nvtx(self, payload: Any, server_args: ServerArgs) -> bool:
        return server_args.enable_layerwise_nvtx_marker and not self._is_warmup_payload(
            payload
        )

    def _run_stage_with_executor_hooks(
        self,
        stage: "PipelineStage",
        stage_index: int,
        payload: Any,
        server_args: ServerArgs,
        run_stage: Callable[["PipelineStage", Any], Any],
        use_nvtx: bool,
    ) -> Any:
        stage_name = stage._component_stage_name()
        self.before_stage(stage, stage_index, payload, server_args)
        with maybe_nvtx_range(f"stage_{stage_name}", use_nvtx):
            payload = self.run_stage_with_context(
                stage, payload, server_args, run_stage
            )
        return payload

    @staticmethod
    def _step_stage_profiler() -> None:
        profiler = SGLDiffusionProfiler.get_instance()
        if profiler:
            profiler.step_stage()

    def execute_with_profiling(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:

        with self.profile_execution(batch, dump_rank=0):
            with current_platform.inference_mode():
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
            with current_platform.inference_mode():
                batches = self.execute_group(stages, batches, server_args)
        return batches

    @staticmethod
    @contextlib.contextmanager
    def _stage_execution_context(stage: "PipelineStage", server_args: ServerArgs):
        if PipelineExecutor._stage_needs_version_counters(stage, server_args):
            # fsdp and cpu-offload hooks need tensor version counters
            with torch.inference_mode(False), torch.no_grad():
                yield
            return
        yield

    @staticmethod
    def _stage_needs_version_counters(
        stage: "PipelineStage", server_args: ServerArgs
    ) -> bool:
        if server_args.use_fsdp_inference:
            return True

        stage_name = stage._active_component_stage_name()
        for use in stage.component_uses(server_args, stage_name):
            component_name = use.component_name
            if server_args.dit_cpu_offload and component_name in (
                "transformer",
                "transformer_2",
                "video_dit",
                "audio_dit",
            ):
                return True
            if server_args.text_encoder_cpu_offload and component_name.startswith(
                "text_encoder"
            ):
                return True
            if server_args.image_encoder_cpu_offload and component_name in (
                "image_encoder",
                "condition_image_encoder",
            ):
                return True
            if server_args.vae_cpu_offload and component_name in (
                "vae",
                "video_vae",
                "audio_vae",
                "vocoder",
                "spatial_upsampler",
                "condition_image_encoder",
            ):
                return True
        return False

    def run_stage_with_context(
        self,
        stage: "PipelineStage",
        payload,
        server_args: ServerArgs,
        run_stage,
    ):
        with self._stage_execution_context(stage, server_args):
            return run_stage(stage, payload)

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
