# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Any, Callable, List

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_world_group,
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
        group = get_world_group()

        # When CFG parallel is enabled, consolidate ALL Gloo communication
        # (broadcasts and barriers) into the CFG group's process groups.
        #
        # On 4+ GPUs with DP>1, the WORLD group spans ALL ranks (e.g. [0,1,2,3])
        # while each CFG group only contains 2 ranks (e.g. [0,1] and [2,3]).
        # Broadcasting on the WORLD group with `src=0` causes both rank 0 and
        # rank 2 (both CFG rank 0) to act as senders, producing conflicting
        # data on the same Gloo collective — a preamble/data size mismatch.
        #
        # Using the CFG group ensures each DP replica broadcasts only within
        # its own 2-rank CFG pair.
        if server_args.enable_cfg_parallel:
            barrier_group = cfg_group.device_group
            cpu_group_for_broadcast = cfg_group.cpu_group
            broadcast_src_rank = cfg_group.ranks[0]
        else:
            barrier_group = group.device_group
            cpu_group_for_broadcast = group.cpu_group
            broadcast_src_rank = 0

        use_nvtx = self._should_use_stage_nvtx(batch, server_args)

        with self._component_residency_request(stages, batch, server_args):
            # TODO: decide when to gather on main when CFG_PARALLEL -> MAIN_RANK_ONLY
            for stage_index, stage in enumerate(stages):
                paradigm = stage.parallelism_type

                if paradigm == StageParallelismType.MAIN_RANK_ONLY:
                    if rank == 0:
                        # Only main rank executes, others just wait
                        batch = self._run_stage_with_executor_hooks(
                            stage,
                            stage_index,
                            batch,
                            server_args,
                            run_stage,
                            use_nvtx,
                        )
                    torch.distributed.barrier(group=barrier_group)

                elif paradigm == StageParallelismType.CFG_PARALLEL:
                    local_batch = batch
                    local_batch_fields = stage.cfg_parallel_local_batch_fields(
                        batch, server_args
                    )
                    # filter local batch fields from batch
                    if rank == 0 and local_batch_fields:
                        local_field_values = {
                            name: getattr(batch, name) for name in local_batch_fields
                        }
                        for name in local_batch_fields:
                            setattr(batch, name, None)
                    else:
                        local_field_values = {}

                    obj_list = [batch] if rank == 0 else []
                    try:
                        # `dist.broadcast(src=...)` expects a global rank for process groups.
                        broadcasted_list = broadcast_pyobj(
                            obj_list,
                            rank=get_world_rank(),
                            dist_group=cfg_group.cpu_group,
                            src=cfg_group.ranks[0],
                        )
                    finally:
                        if rank == 0:
                            # resume local batch fields on rank 0
                            for name, value in local_field_values.items():
                                setattr(batch, name, value)
                    if rank != 0:
                        batch = broadcasted_list[0]
                        for name in local_batch_fields:
                            setattr(batch, name, getattr(local_batch, name))
                    batch = self._run_stage_with_executor_hooks(
                        stage,
                        stage_index,
                        batch,
                        server_args,
                        run_stage,
                        use_nvtx,
                    )

                    torch.distributed.barrier(group=barrier_group)

                elif paradigm == StageParallelismType.REPLICATED:
                    batch = self._run_stage_with_executor_hooks(
                        stage,
                        stage_index,
                        batch,
                        server_args,
                        run_stage,
                        use_nvtx,
                    )
                elif paradigm == StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS:
                    obj_list = []
                    if rank == 0:
                        # Only main rank executes, others just wait
                        try:
                            batch = self._run_stage_with_executor_hooks(
                                stage,
                                stage_index,
                                batch,
                                server_args,
                                run_stage,
                                use_nvtx,
                            )
                            obj_list = [True, batch]
                        except Exception as e:
                            obj_list = [False, e]

                    # Send batch to other ranks.
                    # Use the CFG group (not WORLD) so each DP replica broadcasts
                    # only within its own CFG pair. With 4+ GPUs, using the WORLD
                    # group causes rank 0 and rank 2 (both CFG rank 0) to conflict
                    # as simultaneous senders on the same Gloo collective.
                    broadcasted_list = broadcast_pyobj(
                        obj_list,
                        rank=get_world_rank(),
                        dist_group=cpu_group_for_broadcast,
                        src=broadcast_src_rank,
                    )
                    if rank != 0:
                        success, batch = broadcasted_list[0], broadcasted_list[1]
                    else:
                        success = obj_list[0]

                    if not success:
                        raise RuntimeError(f"Error on rank 0") from batch

                    torch.distributed.barrier(group=barrier_group)
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

