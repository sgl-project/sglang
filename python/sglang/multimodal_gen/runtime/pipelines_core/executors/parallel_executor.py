# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Any, Callable, List

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_world_group,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.distributed.utils import broadcast_pyobj
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
        # Keep the sender's position in the random stream, not just its seed.
        generator = torch.Generator(device=device)
        generator.set_state(value.get_state())
        return generator
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
    """Move a broadcast batch's device tensors and generators to this rank's device.

    Unpickling restores them on the sender's device index.
    """
    device = get_local_torch_device()
    for field_name in batch.__dataclass_fields__:
        value = getattr(batch, field_name)
        setattr(batch, field_name, _move_value_to_local_device(value, device))
    return batch


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
                    torch.distributed.barrier()

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

                    torch.distributed.barrier()

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

                    # Send batch to other ranks
                    broadcasted_list = broadcast_pyobj(
                        obj_list, rank=rank, dist_group=group.cpu_group, src=0
                    )
                    success, broadcasted_batch = broadcasted_list

                    if not success:
                        if isinstance(broadcasted_batch, BaseException):
                            raise RuntimeError("Error on rank 0") from broadcasted_batch
                        raise RuntimeError(f"Error on rank 0: {broadcasted_batch}")

                    if rank != 0:
                        batch = broadcasted_batch
                        if stage.relocate_broadcast_batch:
                            batch = _relocate_batch_to_local_device(batch)

                    torch.distributed.barrier()
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
