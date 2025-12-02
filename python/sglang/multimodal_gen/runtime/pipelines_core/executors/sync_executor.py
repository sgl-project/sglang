# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Synchronous pipeline executor implementation.
"""
from typing import List
import os
import torch
import torch.profiler

from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
    Timer,
    logger,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SyncExecutor(PipelineExecutor):
    """
    A simple synchronous executor that runs stages sequentially.
    """

    def run_all_stages(
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
        logger.info("Running pipeline stages sequentially with SyncExecutor.")

        # New profiling flag: profile all stages when both profile and full_stages are set
        do_full_stages_profile = bool(getattr(batch, "profile", False) and getattr(batch, "full_stages", False))

        if do_full_stages_profile:
            try:
                os.makedirs("./logs", exist_ok=True)
            except Exception:
                pass
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            with torch.profiler.profile(
                activities=activities, record_shapes=True, with_stack=True
            ) as prof:
                batch = self.run_all_stages(stages, batch, server_args)
            request_id = getattr(batch, "request_id", "full_stages")
            rank = 0
            try:
                os.makedirs("./logs", exist_ok=True)
            except Exception:
                pass
            trace_path = os.path.abspath(
                f"./logs/{request_id}-global-rank{rank}.trace.json.gz"
            )
            logger.info("Saving stages profiler trace to: %s", trace_path)
            prof.export_chrome_trace(trace_path)
        else:
            batch = self.run_all_stages(stages, batch, server_args)

        return batch
