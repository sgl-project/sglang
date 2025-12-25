# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Synchronous pipeline executor implementation.
"""
from typing import List

from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
    SGLDiffusionProfiler,
    Timer,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SyncExecutor(PipelineExecutor):
    """
    A simple synchronous executor that runs stages sequentially.
    """

    def run_profile_all_stages(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Execute all pipeline stages sequentially.
        """
        import threading

        offload_threads = []

        for i, stage in enumerate(stages):
            # 1. Load current stage (if not already loaded/prefetched)
            # This is fast if already loaded
            stage.load_model(server_args)

            # 2. Prefetch next stage
            prefetch_thread = None
            if i + 1 < len(stages):
                next_stage = stages[i + 1]
                # Start preloading next stage in background
                prefetch_thread = threading.Thread(
                    target=next_stage.load_model, args=(server_args,)
                )
                prefetch_thread.start()

            # 3. Execute current stage
            with Timer(stage.__class__.__name__):
                batch = stage(batch, server_args)

            # 4. Wait for prefetch to complete before moving to next stage
            # This ensures next stage is ready when loop continues,
            # and prevents too many concurrent load operations.
            if prefetch_thread:
                prefetch_thread.join()

            # 5. Async offload current stage
            t = threading.Thread(target=stage.offload_model, args=(server_args,))
            t.start()
            offload_threads.append(t)

            profiler = SGLDiffusionProfiler.get_instance()
            if profiler:
                profiler.step_stage()

        # Ensure all offloads are completed
        for t in offload_threads:
            t.join()

        return batch

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
