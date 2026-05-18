# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Synchronous pipeline executor implementation.
"""

from typing import Any, Callable, List

from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
    SGLDiffusionProfiler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range


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
        # Match the warmup gating used inside DenoisingStage so the executor
        # wrappers don't leak a stage_<Name> range during server-side warmup.
        # ``payload`` can be a single Req or a list of Req (execute_group);
        # treat the group as warmup only if all entries agree.
        if isinstance(payload, list):
            is_warmup = bool(payload) and all(
                getattr(p, "is_warmup", False) for p in payload
            )
        else:
            is_warmup = getattr(payload, "is_warmup", False)
        use_nvtx = server_args.enable_layerwise_nvtx_marker and not is_warmup
        self.begin_component_residency_request(stages, payload, server_args)
        try:
            for stage_index, stage in enumerate(stages):
                stage_name = stage.__class__.__name__
                self.before_stage(stage, stage_index, payload, server_args)
                with maybe_nvtx_range(f"stage_{stage_name}", use_nvtx):
                    payload = self.run_stage_with_context(
                        stage, payload, server_args, run_stage
                    )
                self.after_stage(stage_index)
                profiler = SGLDiffusionProfiler.get_instance()
                if profiler:
                    profiler.step_stage()
        finally:
            self.finish_component_residency_request()
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
