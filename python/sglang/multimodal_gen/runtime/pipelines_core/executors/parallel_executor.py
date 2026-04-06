# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import List

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_group,
)
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


def _move_value_to_local_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu":
            return value
        return value.to(device=device)
    if isinstance(value, torch.Generator):
        if value.device.type == "cpu":
            return value
        return torch.Generator(device=device).manual_seed(value.initial_seed())
    if isinstance(value, list):
        return [_move_value_to_local_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_local_device(item, device) for item in value)
    if isinstance(value, dict):
        return {
            key: _move_value_to_local_device(item, device)
            for key, item in value.items()
        }
    return value


def _relocate_batch_to_local_device(batch: Req) -> Req:
    device = get_local_torch_device()
    for field_name in batch.__dataclass_fields__:
        value = getattr(batch, field_name)
        setattr(batch, field_name, _move_value_to_local_device(value, device))
    return batch


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
        world_rank = get_world_rank()
        cfg_group = get_cfg_group()

        # TODO: decide when to gather on main when CFG_PARALLEL -> MAIN_RANK_ONLY
        for stage in stages:
            paradigm = stage.parallelism_type

            if paradigm == StageParallelismType.MAIN_RANK_ONLY:
                if world_rank == 0:
                    # Only world rank 0 executes the stage and then shares the
                    # updated batch with the rest of the distributed workers.
                    batch = stage(batch, server_args)
                if torch.distributed.is_initialized():
                    world_group = get_world_group()
                    broadcasted_list = broadcast_pyobj(
                        [batch] if world_group.rank == 0 else [],
                        rank=world_group.rank,
                        dist_group=world_group.cpu_group,
                        src=world_group.ranks[0],
                    )
                    if world_group.rank != 0 and broadcasted_list:
                        batch = _relocate_batch_to_local_device(broadcasted_list[0])

            elif paradigm == StageParallelismType.CFG_PARALLEL:
                obj_list = [batch] if rank == 0 else []
                broadcasted_list = broadcast_pyobj(
                    obj_list, rank=rank, dist_group=cfg_group.cpu_group, src=0
                )
                if rank != 0:
                    batch = _relocate_batch_to_local_device(broadcasted_list[0])
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
        batch = self._execute(stages, batch, server_args)
        return batch
