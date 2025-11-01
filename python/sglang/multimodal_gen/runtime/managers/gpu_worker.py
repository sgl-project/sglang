# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
import os
from typing import List

import torch
from setproctitle import setproctitle

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_tp_group,
)
from sglang.multimodal_gen.runtime.pipelines import build_pipeline
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import OutputBatch, Req
from sglang.multimodal_gen.runtime.server_args import PortArgs, ServerArgs
from sglang.multimodal_gen.runtime.utils.common import set_cuda_arch
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
    suppress_other_loggers,
)

logger = init_logger(__name__)

# ANSI color codes
CYAN = "\033[1;36m"
RESET = "\033[0;0m"


class GPUWorker:
    """
    A worker that executes the model on a single GPU.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        master_port: int,
        server_args: ServerArgs,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.master_port = master_port
        # FIXME: should we use tcp as distribute init method?
        self.server_args = server_args
        self.pipeline = None

        self.init_device_and_model()
        self.sp_group = get_sp_group()
        self.sp_cpu_group = self.sp_group.cpu_group
        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group

        self.cfg_group = get_cfg_group()
        self.cfg_cpu_group = self.cfg_group.cpu_group

    def init_device_and_model(self) -> None:
        """Initialize the device and load the model."""
        setproctitle(f"sgl_diffusion::scheduler:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)
        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.server_args.num_gpus)
        # Initialize the distributed environment
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=self.server_args.tp_size,
            enable_cfg_parallel=self.server_args.enable_cfg_parallel,
            ulysses_degree=self.server_args.ulysses_degree,
            ring_degree=self.server_args.ring_degree,
            sp_size=self.server_args.sp_degree,
            dp_size=self.server_args.dp_size,
        )

        self.pipeline = build_pipeline(self.server_args)

        logger.info(
            f"Worker {self.rank}: Initialized device, model, and distributed environment."
        )

    def execute_forward(self, batch: List[Req], server_args: ServerArgs) -> OutputBatch:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        # TODO: dealing with first req for now
        req = batch[0]
        output_batch = self.pipeline.forward(req, server_args)
        if req.perf_logger:
            req.perf_logger.log_total_duration("total_inference_time")
        return output_batch

    def set_lora_adapter(
        self, lora_nickname: str, lora_path: str | None = None
    ) -> None:
        """
        Set the LoRA adapter for the pipeline.
        """
        assert self.pipeline is not None
        self.pipeline.set_lora_adapter(lora_nickname, lora_path)

    def merge_lora_weights(self) -> None:
        """
        Merge LoRA weights.
        """
        assert self.pipeline is not None
        self.pipeline.merge_lora_weights()

    def unmerge_lora_weights(self) -> None:
        """
        Unmerge LoRA weights.
        """
        assert self.pipeline is not None
        self.pipeline.unmerge_lora_weights()


def run_scheduler_process(
    local_rank: int,
    rank: int,
    master_port: int,
    server_args: ServerArgs,
    pipe_writer: mp.connection.Connection,
    # For all workers: pipe to receive tasks from rank 0
    task_pipe_r: mp.connection.Connection,
    # For slave workers: pipe to send results back to rank 0
    result_pipe_w: mp.connection.Connection | None,
    # For rank 0 worker only: pipes to send tasks to slaves
    task_pipes_to_slaves: list[mp.connection.Connection] | None = None,
    # For rank 0 worker only: pipes to receive results from slaves
    result_pipes_from_slaves: list[mp.connection.Connection] | None = None,
) -> None:
    """
    The entry point for the worker process.
    Rank 0 acts as the master, handling ZMQ requests and coordinating slaves.
    Ranks > 0 act as slaves, waiting for tasks from the master.
    """
    configure_logger(server_args)
    suppress_other_loggers()
    set_cuda_arch()

    port_args = PortArgs.from_server_args(server_args)

    # start the scheduler event loop
    assert task_pipes_to_slaves is not None
    assert result_pipes_from_slaves is not None
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

    scheduler = Scheduler(
        server_args,
        gpu_id=rank,
        port_args=port_args,
        task_pipes_to_slaves=task_pipes_to_slaves,
        result_pipes_from_slaves=result_pipes_from_slaves,
    )
    logger.info(f"Worker {rank}: Scheduler loop started.")
    pipe_writer.send(
        {
            "status": "ready",
        }
    )
    scheduler.event_loop()
    logger.info(f"Worker {rank}: Shutdown complete.")
