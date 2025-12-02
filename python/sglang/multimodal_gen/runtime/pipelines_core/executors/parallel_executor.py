# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import List
import os

import torch
import torch.profiler

from sglang.multimodal_gen.runtime.distributed import get_sp_group
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
    Timer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ParallelExecutor(PipelineExecutor):
    """
    The correctness of the execution relies on the parallelism_type declared by stages

    """

    def collect_from_main(self, batches: list[Req]):

        # TODO: fix this condition
        if self.server_args.sp_degree != 1:
            sp_group = get_sp_group()
            batches = broadcast_pyobj(
                batches,
                sp_group.rank,
                sp_group.cpu_group,
                src=sp_group.ranks[0],
            )

        if self.server_args.enable_cfg_parallel:
            batches = broadcast_pyobj(
                batches,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

    def run_all_stages(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute all pipeline stages respecting their declared parallelism type.
        """
        rank = get_classifier_free_guidance_rank()
        cfg_group = get_cfg_group()

        # TODO: decide when to gather on main when CFG_PARALLEL -> MAIN_RANK_ONLY
        for stage in stages:
            with Timer(stage.__class__.__name__):
                paradigm = stage.parallelism_type

                if paradigm == StageParallelismType.MAIN_RANK_ONLY:
                    if rank == 0:
                        # Only main rank executes, others just wait
                        batch = stage(batch, server_args)
                    torch.distributed.barrier()

                elif paradigm == StageParallelismType.CFG_PARALLEL:
                    obj_list = [batch] if rank == 0 else []
                    broadcasted_list = broadcast_pyobj(
                        obj_list, rank=rank, dist_group=cfg_group.cpu_group, src=0
                    )
                    if rank != 0:
                        batch = broadcasted_list[0]
                    batch = stage(batch, server_args)

                    torch.distributed.barrier()

                elif paradigm == StageParallelismType.REPLICATED:
                    batch = stage(batch, server_args)
        return batch

    def execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        rank = get_classifier_free_guidance_rank()

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
                activities=activities,
                record_shapes=True,
                with_stack=True,
            ) as prof:
                batch = self.run_all_stages(stages, batch, server_args)
            if rank == 0:
                try:
                    os.makedirs("./logs", exist_ok=True)
                except Exception:
                    pass
                request_id = getattr(batch, "request_id", "full_stages")
                world_rank = get_world_rank()
                trace_path = os.path.abspath(
                    f"./logs/{request_id}-global-rank{world_rank}.trace.json.gz"
                )
                logger.info("Saving stages profiler trace to: %s", trace_path)
                prof.export_chrome_trace(trace_path)
        else:
            batch = self.run_all_stages(stages, batch, server_args)

        return batch
