# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import List

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
    _merge_output_batches,
    _prepare_single_output_batch,
    _split_stages_by_per_output,
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

    def _execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Execute all pipeline stages respecting their declared parallelism type.
        """
        if server_args.enable_cfg_parallel:
            rank = get_classifier_free_guidance_rank()
        else:
            rank = get_world_rank()
        cfg_group = get_cfg_group()

        # TODO: decide when to gather on main when CFG_PARALLEL -> MAIN_RANK_ONLY
        for stage in stages:
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
    ) -> OutputBatch:
        num_outputs = batch.num_outputs_per_prompt

        if num_outputs <= 1:
            return self._execute(stages, batch, server_args)

        shared_stages, per_output_stages = _split_stages_by_per_output(stages)
        if not per_output_stages:
            return self._execute(stages, batch, server_args)

        logger.info("Multi-output mode: %d outputs requested.", num_outputs)

        # Phase 1: shared stages (e.g. text encoding) run once
        batch = self._execute(shared_stages, batch, server_args)

        # Phase 2: per-output stages loop N times with batch_size=1
        output_batches: List[OutputBatch] = []
        for i in range(num_outputs):
            sub_batch = _prepare_single_output_batch(batch, i)
            result = self._execute(per_output_stages, sub_batch, server_args)

            if isinstance(result, OutputBatch):
                output_batches.append(result)
            else:
                output_batches.append(
                    OutputBatch(
                        output=getattr(result, "output", None)
                        or getattr(result, "latents", None),
                        trajectory_timesteps=getattr(
                            result, "trajectory_timesteps", None
                        ),
                        trajectory_latents=getattr(result, "trajectory_latents", None),
                        trajectory_decoded=getattr(result, "trajectory_decoded", None),
                        metrics=getattr(result, "metrics", None),
                        noise_pred=getattr(result, "noise_pred", None),
                    )
                )

        # Phase 3: merge
        return _merge_output_batches(output_batches)
