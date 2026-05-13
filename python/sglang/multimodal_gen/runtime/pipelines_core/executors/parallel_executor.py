# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Any, Callable, List

import torch

from sglang.multimodal_gen.runtime.distributed import get_sp_group
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
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

    def _execute_stages(
        self,
        stages: List[PipelineStage],
        batch: Any,
        server_args: ServerArgs,
        run_stage: Callable[[PipelineStage, Any], Any],
    ) -> Any:
        """Execute stages while respecting their declared parallelism type."""
        if server_args.enable_cfg_parallel:
            rank = get_classifier_free_guidance_rank()
        else:
            rank = get_world_rank()
        cfg_group = get_cfg_group()

        self.begin_component_residency_request(stages, batch, server_args)
        try:
            # TODO: decide when to gather on main when CFG_PARALLEL -> MAIN_RANK_ONLY
            for stage_index, stage in enumerate(stages):
                paradigm = stage.parallelism_type

                if paradigm == StageParallelismType.MAIN_RANK_ONLY:
                    if rank == 0:
                        # Only main rank executes, others just wait
                        self.before_stage(stage, stage_index, batch, server_args)
                        batch = stage(batch, server_args)
                        self.after_stage(stage_index)
                    torch.distributed.barrier()

                elif paradigm == StageParallelismType.CFG_PARALLEL:
                    obj_list = [batch] if rank == 0 else []
                    # `dist.broadcast(src=...)` expects a global rank for process groups.
                    broadcasted_list = broadcast_pyobj(
                        obj_list,
                        rank=get_world_rank(),
                        dist_group=cfg_group.cpu_group,
                        src=cfg_group.ranks[0],
                    )
                    if rank != 0:
                        batch = broadcasted_list[0]
                    self.before_stage(stage, stage_index, batch, server_args)
                    batch = stage(batch, server_args)
                    self.after_stage(stage_index)

                    torch.distributed.barrier()

                elif paradigm == StageParallelismType.REPLICATED:
                    self.before_stage(stage, stage_index, batch, server_args)
                    batch = stage(batch, server_args)
                    self.after_stage(stage_index)
        finally:
            self.finish_component_residency_request()
        return batch

    def execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        return self._execute_stages(
            stages,
            batch,
            server_args,
            lambda stage, current: stage(current, server_args),
        )

    def execute_group(
        self,
        stages: List[PipelineStage],
        batches: list[Req],
        server_args: ServerArgs,
    ):
        return self._execute_stages(
            stages,
            batches,
            server_args,
            lambda stage, current: stage.run_grouped_requests(current, server_args),
        )
