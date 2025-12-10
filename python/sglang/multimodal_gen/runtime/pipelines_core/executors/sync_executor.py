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
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
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
    ) -> Req:
        """
        Execute all pipeline stages sequentially.
        """
        for stage in stages:
            with Timer(stage.__class__.__name__):
                batch = stage(batch, server_args)

            profiler = SGLDiffusionProfiler.get_instance()
            if profiler:
                profiler.step_stage()
        return batch

    def execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute the pipeline stages sequentially.
        """

        with self.profile_execution(batch, check_rank=0, dump_rank=0):
            batch = self.run_profile_all_stages(stages, batch, server_args)

        return batch
