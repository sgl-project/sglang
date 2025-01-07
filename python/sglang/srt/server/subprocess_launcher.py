import logging
import multiprocessing as mp
import os
import signal
from typing import Optional

import zmq

from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import load_chat_template_for_openai_api
from sglang.srt.server_args import EngineFragmentArgs, PortArgs, ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    configure_logger,
    create_zmq_ipc_name,
    get_zmq_socket,
    kill_process_tree,
    maybe_set_triton_cache_manager,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)

logger = logging.getLogger(__name__)


class SubprocessLauncher:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """

    def __init__(self, server_args: ServerArgs):
        # Configure global environment
        configure_logger(server_args)
        server_args.check_server_args()
        _set_envs_and_config(server_args)

        # Allocate ports for inter-process communications
        port_args = PortArgs.init_new(server_args)
        self.fragment_args = EngineFragmentArgs.init_new(server_args, port_args)
        logger.info(f"{server_args=}")

        # If using model from www.modelscope.cn, first download the model.
        server_args.model_path, server_args.tokenizer_path = (
            prepare_model_and_tokenizer(
                server_args.model_path, server_args.tokenizer_path
            )
        )

        ready_receivers, scheduler_procs = _start_scheduler_or_dp_controller_processes(
            port_args, server_args, self.fragment_args
        )

        # Launch detokenizer process
        detoken_proc = mp.Process(
            target=run_detokenizer_process,
            args=(
                server_args,
                port_args,
            ),
        )
        detoken_proc.start()

        # Launch tokenizer process
        tokenizer_manager = TokenizerManager(server_args, port_args)
        if server_args.chat_template:
            load_chat_template_for_openai_api(
                tokenizer_manager, server_args.chat_template
            )

        self._ready_receivers = ready_receivers
        self._scheduler_procs = scheduler_procs
        self._tokenizer_manager = tokenizer_manager

    def wait(self):
        # Wait for model to finish loading
        scheduler_infos = []
        for i in range(len(self._ready_receivers)):
            try:
                data = self._ready_receivers[i].recv_pyobj()
            except EOFError as e:
                logger.exception(e)
                logger.error(
                    f"Rank {i} scheduler is dead. Please check if there are relevant logs."
                )
                self._scheduler_procs[i].join()
                logger.error(f"Exit code: {self._scheduler_procs[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError(
                    "Initialization failed. Please see the error messages above."
                )
            scheduler_infos.append(data)

        # Assume all schedulers have same scheduler_info
        scheduler_info = scheduler_infos[0]

        return self._tokenizer_manager, scheduler_info


def _start_scheduler_or_dp_controller_processes(port_args, server_args, fragment_args):
    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        scheduler_procs = []
        scheduler_ready_receivers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            proc, ready_receiver = _start_scheduler_process(
                port_args, server_args, fragment_args, tp_rank, tp_size_per_node
            )
            scheduler_procs.append(proc)
            scheduler_ready_receivers.append(ready_receiver)

        if server_args.node_rank >= 1:
            # For other nodes, they do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            for proc in scheduler_procs:
                proc.join()

        return scheduler_ready_receivers, scheduler_procs
    else:
        # Launch the data parallel controller
        ready_ipc_name = create_zmq_ipc_name()
        ready_receiver = get_zmq_socket(zmq.Context(1), zmq.PULL, ready_ipc_name)
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, ready_ipc_name),
        )
        proc.start()
        return [ready_receiver], [proc]


def _start_scheduler_process(
    port_args,
    server_args,
    fragment_args: Optional[EngineFragmentArgs],
    tp_rank: int,
    tp_size_per_node: int,
):
    if server_args.fragment:
        ready_ipc_name = fragment_args.scheduler_ready_ipc_names[tp_rank]
        ready_receiver = get_zmq_socket(zmq.Context(1), zmq.PULL, ready_ipc_name)
        proc = None
    else:
        ready_ipc_name = create_zmq_ipc_name()
        ready_receiver = get_zmq_socket(zmq.Context(1), zmq.PULL, ready_ipc_name)
        gpu_id = server_args.base_gpu_id + tp_rank % tp_size_per_node
        proc = mp.Process(
            target=run_scheduler_process,
            args=(server_args, port_args, gpu_id, tp_rank, None, ready_ipc_name),
        )
        proc.start()
    return proc, ready_receiver


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer",
            "0.1.6",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )

    # Register the signal handler.
    # The child processes will send SIGQUIT to this process when any error happens
    # This process then clean up the whole process tree
    def sigquit_handler(signum, frame):
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)
