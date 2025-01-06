import logging
import os
import signal
import threading
from typing import Optional

import psutil
import setproctitle
import torch
from sglang.srt.managers.scheduler.core import SchedulerCore
from sglang.srt.managers.scheduler.communication import SchedulerCommunication
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    setproctitle.setproctitle("sglang::scheduler")

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configue the logger
    if dp_rank is None:
        configure_logger(server_args, prefix=f" TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" DP{dp_rank} TP{tp_rank}")
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    parent_process = psutil.Process().parent()

    # Create a scheduler and run the event loop
    try:
        core = SchedulerCore(
            server_args=server_args, port_args=port_args,
            gpu_id=gpu_id, tp_rank=tp_rank, dp_rank=dp_rank,
        )
        communication = SchedulerCommunication(
            core=core, server_args=server_args, port_args=port_args, tp_rank=tp_rank,
        )
        core.callback = communication

        pipe_writer.send(
            {"status": "ready", "max_total_num_tokens": core.max_total_num_tokens}
        )
        if core.enable_overlap:
            core.event_loop_overlap()
        else:
            core.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
