"""
Scheduler process management for gRPC server.

This module handles launching and managing scheduler processes for the gRPC server,
including tensor parallelism, pipeline parallelism, and data parallelism configurations.
"""

import logging
import multiprocessing as mp
import signal
from typing import Dict, List, Optional, Tuple

from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, prepare_model_and_tokenizer
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)


def run_scheduler_with_signal_handling(*args, **kwargs):
    """
    Wrapper for run_scheduler_process that ignores SIGINT.

    The scheduler process should not handle Ctrl+C - it should only terminate
    when the parent gRPC server exits (via kill_itself_when_parent_died).

    Args:
        *args: Positional arguments for run_scheduler_process
        **kwargs: Keyword arguments for run_scheduler_process
    """
    # Ignore SIGINT in this subprocess - let the parent handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Now run the actual scheduler process
    run_scheduler_process(*args, **kwargs)


def launch_scheduler_process_only(
    server_args: ServerArgs,
    port_args: Optional[PortArgs] = None,
) -> Tuple[Dict, PortArgs, List[mp.Process]]:
    """
    Launch only the scheduler process(es) without tokenizer/detokenizer.

    This function handles all scheduler startup logic including:
    - Tensor parallelism (tp_size)
    - Pipeline parallelism (pp_size)
    - Data parallelism (dp_size)
    - Multi-node distributed setup

    Args:
        server_args: Server configuration
        port_args: Port configuration (created if None)

    Returns:
        Tuple of (scheduler_info, port_args, scheduler_processes):
        - scheduler_info: Dict with model metadata and configuration
        - port_args: Port configuration used for IPC
        - scheduler_processes: List of launched scheduler Process objects

    Raises:
        RuntimeError: If any scheduler process fails to initialize
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()

    # Fix CUDA multiprocessing issues - must be called before any CUDA operations
    mp.set_start_method("spawn", force=True)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # Prepare model and tokenizer paths
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []

    if server_args.dp_size == 1:
        # Single data parallel group - launch TP/PP schedulers
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        # Calculate TP/PP distribution across nodes
        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        # Launch scheduler for each TP/PP rank combination
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)

                # Calculate GPU ID for this rank
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )

                # Calculate MoE expert parallel rank
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

                # Create scheduler process
                proc = mp.Process(
                    target=run_scheduler_with_signal_handling,
                    args=(
                        server_args,
                        port_args,
                        gpu_id,
                        tp_rank,
                        moe_ep_rank,
                        pp_rank,
                        None,  # dp_rank
                        writer,
                    ),
                )

                with memory_saver_adapter.configure_subprocess():
                    proc.start()

                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Data parallelism - launch data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]

        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    # TODO(CatherineSue): handle cases for multi-node

    # Wait for all scheduler processes to be ready
    scheduler_infos = []
    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise RuntimeError(f"Failed to initialize scheduler rank {i}")

        if data.get("status") != "ready":
            raise RuntimeError(
                f"Scheduler rank {i} initialization failed: {data.get('error', 'Unknown error')}"
            )
        scheduler_infos.append(data)

    logger.info(
        f"All {len(scheduler_procs)} scheduler process(es) initialized successfully"
    )

    # Return the first scheduler's info (they should all be the same)
    return scheduler_infos[0], port_args, scheduler_procs
