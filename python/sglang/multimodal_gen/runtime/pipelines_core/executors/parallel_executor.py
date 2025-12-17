# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import List

import torch

from sglang.multimodal_gen.runtime.distributed import get_sp_group
from sglang.multimodal_gen.runtime.distributed.communication_backend import get_backend
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

    def _execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute all pipeline stages respecting their declared parallelism type.
        """
        rank = get_classifier_free_guidance_rank()
        world_rank = get_world_rank()
        cfg_group = get_cfg_group()

        # Disaggregation Logic
        do_disaggregation = (
            server_args.num_gpus > 1 and (server_args.num_gpus - 1) % 2 == 0
        )

        is_non_dit = False
        is_dit = False
        if do_disaggregation:
            # Assume Rank 0 is Non-DiT
            is_non_dit = world_rank == 0
            is_dit = world_rank > 0
            backend = get_backend()

        # TODO: decide when to gather on main when CFG_PARALLEL -> MAIN_RANK_ONLY
        for stage in stages:
            stage_name = stage.__class__.__name__

            # --- Disaggregation Interception ---
            if do_disaggregation:
                # Text Encoding (Non-DiT -> DiT)
                if "Text" in stage_name or "Encoding" in stage_name:
                    if "Image" in stage_name and "Encoding" in stage_name:
                        # ImageEncoding might be on Non-DiT too? Assuming yes.
                        pass

                    if is_non_dit:
                        with Timer(stage_name):
                            batch = stage(batch, server_args)
                        # After encoding, send to DiT leader (Rank 1)
                        # We send after the *last* encoding stage.
                        # But simpler: send after *every* encoding stage?
                        # Or just send before Denoising?
                        # Let's simply send the batch object. Overhead is pickle.
                        backend.send_object(batch, dst=1)
                    else:
                        # DiT ranks wait for data
                        batch = backend.recv_object(src=0)
                    continue

                # Denoising (DiT -> Non-DiT)
                if "Denoising" in stage_name or "Transformer" in stage_name:
                    if is_dit:
                        with Timer(stage_name):
                            batch = stage(batch, server_args)
                        # After denoising, send back to Non-DiT (Rank 0)
                        if world_rank == 1:  # DiT Leader
                            backend.send_object(batch, dst=0)
                    else:
                        # Non-DiT rank waits for data
                        batch = backend.recv_object(src=1)
                    continue

                # Decoding (Non-DiT)
                if "Decoding" in stage_name or "VAE" in stage_name:
                    if is_non_dit:
                        with Timer(stage_name):
                            batch = stage(batch, server_args)
                    else:
                        pass  # DiT doesn't participate
                    continue

            # --- End Disaggregation Interception ---

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

        if batch.profile and batch.profile_all_stages:
            world_rank = get_world_rank()
        else:
            world_rank = 0

        with self.profile_execution(batch, check_rank=rank, dump_rank=world_rank):
            batch = self._execute(stages, batch, server_args)

        return batch
