import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import signal
from typing import List, Tuple

import torch
from fastapi import Request
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import (
    load_chat_template_for_openai_api,
)
from sglang.srt.server.engine_base import EngineBase
from sglang.srt.server_args import PortArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    kill_process_tree,
)
from sglang.srt.utils import (
    assert_pkg_version,
    configure_logger,
    maybe_set_triton_cache_manager,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.version import __version__

logger = logging.getLogger(__name__)


class Engine(EngineBase):
    """
    SRT Engine without an HTTP server layer.

    This class provides a direct inference engine without the need for an HTTP server. It is designed for use cases where
    launching the HTTP server adds unnecessary complexity or overhead,
    """

    def __init__(self, log_level: str = "error", *args, **kwargs):
        """See the arguments in server_args.py::ServerArgs"""

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args)
        self.tokenizer_manager = tokenizer_manager
        self.scheduler_info = scheduler_info

    def shutdown(self):
        kill_process_tree(os.getpid(), include_parent=False)

    def get_tokenizer(self):
        if self.tokenizer_manager is None:
            raise ReferenceError("Tokenizer Manager is not initialized.")
        else:
            return self.tokenizer_manager.tokenizer

    def start_profile(self):
        self.tokenizer_manager.start_profile()

    def stop_profile(self):
        self.tokenizer_manager.stop_profile()

    def get_server_info(self):
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),  # server args
            **self.scheduler_info,
            "version": __version__,
        }

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ):
        """Initialize parameter update group."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.init_weights_update_group(obj, None)
        )

    def update_weights_from_distributed(self, name, dtype, shape):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromDistributedReqInput(
            name=name,
            dtype=dtype,
            shape=shape,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(self, named_tensors: List[Tuple[str, torch.Tensor]]):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=MultiprocessingSerializer.serialize(named_tensors)
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def get_weights_by_name(self, name, truncate_size=100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.get_weights_by_name(obj, None))

    async def _generate_request_impl(self, obj: GenerateReqInput, request: Request):
        return self.tokenizer_manager.generate_request(obj, request)

    def _create_abort_task_impl(self, obj: GenerateReqInput):
        return self.tokenizer_manager.create_abort_task(obj)


def _launch_subprocesses(
    server_args: ServerArgs,
):
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """

    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    port_args = PortArgs.init_new(server_args)
    logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        scheduler_procs = []
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = server_args.base_gpu_id + tp_rank % tp_size_per_node
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, port_args, gpu_id, tp_rank, None, writer),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        if server_args.node_rank >= 1:
            # For other nodes, they do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            for proc in scheduler_procs:
                proc.join()
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()

    # Launch detokenizer process
    launch_detoken_proc()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)
    if server_args.chat_template:
        load_chat_template_for_openai_api(tokenizer_manager, server_args.chat_template)

    # Wait for model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError as e:
            logger.exception(e)
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have same scheduler_info
    scheduler_info = scheduler_infos[0]

    return tokenizer_manager, scheduler_info


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