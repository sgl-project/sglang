# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Synchronous pipeline executor implementation.
"""

from typing import Any, Callable, List

from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SyncExecutor(PipelineExecutor):
    """
    A simple synchronous executor that runs stages sequentially.
    """

    def _run_profile_all_stages(
        self,
        stages: List[PipelineStage],
        payload: Any,
        server_args: ServerArgs,
        run_stage: Callable[[PipelineStage, Any], Any],
    ) -> Any:
        """Execute all pipeline stages sequentially and step the profiler."""

        use_nvtx = self._should_use_stage_nvtx(payload, server_args)
        with self._component_residency_request(stages, payload, server_args):
            for stage_index, stage in enumerate(stages):
                payload = self._run_stage_with_executor_hooks(
                    stage,
                    stage_index,
                    payload,
                    server_args,
                    run_stage,
                    use_nvtx,
                )
                self._step_stage_profiler()
        return payload

    def run_profile_all_stages(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        return self._run_profile_all_stages(
            stages,
            batch,
            server_args,
            lambda stage, current: stage(current, server_args),
        )

    def execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Execute the pipeline stages sequentially.
        """

        batch = self.run_profile_all_stages(stages, batch, server_args)

        return batch

    def execute_group(
        self,
        stages: List[PipelineStage],
        batches: list[Req],
        server_args: ServerArgs,
    ):
        return self._run_profile_all_stages(
            stages,
            batches,
            server_args,
            lambda stage, current: stage.run_grouped_requests(current, server_args),
        )
