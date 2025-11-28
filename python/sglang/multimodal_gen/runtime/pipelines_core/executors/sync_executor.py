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

        do_global_profile = bool(getattr(batch, "global_profile", False))
        global_full = bool(getattr(batch, "global_profile_full", False))

        def _run_all():
            nonlocal batch
            for stage in stages:
                with Timer(stage.__class__.__name__):
                    batch = stage(batch, server_args)
            return batch

        if do_global_profile:
            try:
                os.makedirs("./logs", exist_ok=True)
            except Exception:
                pass
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            if global_full:
                with torch.profiler.profile(
                    activities=activities, record_shapes=True, with_stack=True
                ) as prof:
                    batch = _run_all()
                request_id = getattr(batch, "request_id", "global_profile")
                rank = 0
                try:
                    os.makedirs("./logs", exist_ok=True)
                except Exception:
                    pass
                trace_path = os.path.abspath(
                    f"./logs/{request_id}-global-rank{rank}.trace.json.gz"
                )
                logger.info("Saving global profiler trace to: %s", trace_path)
                prof.export_chrome_trace(trace_path)
            else:
                with torch.profiler.profile(
                    activities=activities,
                    schedule=torch.profiler.schedule(
                        skip_first=0, wait=0, warmup=0, active=len(stages), repeat=1
                    ),
                    record_shapes=True,
                    with_stack=True,
                ) as prof:
                    for stage in stages:
                        with Timer(stage.__class__.__name__):
                            batch = stage(batch, server_args)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        prof.step()
                request_id = getattr(batch, "request_id", "global_profile")
                rank = 0
                try:
                    os.makedirs("./logs", exist_ok=True)
                except Exception:
                    pass
                trace_path = os.path.abspath(
                    f"./logs/{request_id}-global.stages-rank{rank}.trace.json.gz"
                )
                logger.info("Saving global (stages) profiler trace to: %s", trace_path)
                prof.export_chrome_trace(trace_path)
        else:
            batch = _run_all()

        return batch
