# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""

import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import threading
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import zmq
import zmq.asyncio
from PIL.Image import Image

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import torch
import uvloop

from sglang.srt.code_completion_parser import load_completion_template_for_openai_api
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import load_chat_template_for_openai_api
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    configure_logger,
    get_zmq_socket,
    kill_process_tree,
    launch_dummy_health_check_server,
    maybe_set_triton_cache_manager,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class Engine:
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # Launch subprocesses
        tokenizer_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args,
            port_args=port_args,
        )

        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.scheduler_info = scheduler_info

        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, port_args.rpc_ipc_name, True
        )

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        stream: bool = False,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, image_data=image_data)
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        return ret

    def shutdown(self):
        """Shutdown the engine"""
        kill_process_tree(os.getpid(), include_parent=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def start_profile(self, activities: Optional[List[str]] = None):
        if activities is None:
            activities = ["CPU", "GPU"]
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tokenizer_manager.start_profile(activities=activities))

    def stop_profile(self):
        self.tokenizer_manager.stop_profile()

    def get_server_info(self):
        loop = asyncio.get_event_loop()
        internal_states = loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()
        )

        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            **internal_states,
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

    def update_weights_from_distributed(self, name: str, dtype, shape):
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

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be true
        to avoid duplicated operations such as clearing cache."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
    ):
        """Update the weights from disk inplace without re-launching the engine.

        This method allows updating the model weights from disk without restarting
        the engine. It can be used to load a different model or update weights with
        new training.
        """
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
        )

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_disk(obj, None)
        )

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.get_weights_by_name(obj, None)
        )

    def release_memory_occupation(self):
        """Release GPU occupation temporarily."""
        obj = ReleaseMemoryOccupationReqInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self):
        """Resume GPU occupation."""
        obj = ResumeMemoryOccupationReqInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)
        )

    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

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
            "flashinfer_python",
            "0.2.3",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )

    def sigchld_handler(signum, frame):
        pid, exitcode = os.waitpid(0, os.WNOHANG)
        if exitcode != 0:
            logger.warning(
                "Child process unexpectedly failed with an exit code %d. pid=%d",
                exitcode,
                pid,
            )

    signal.signal(signal.SIGCHLD, sigchld_handler)

    # Register the signal handler.
    # The child processes will send SIGQUIT to this process when any error happens
    # This process then clean up the whole process tree
    def sigquit_handler(signum, frame):
        logger.error(
            "Received sigquit from a child process. It usually means the child failed."
        )
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None
) -> Tuple[TokenizerManager, Dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, port_args, gpu_id, tp_rank, None, writer),
            )
            with memory_saver_adapter.configure_subprocess():
                proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None

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
            tokenizer_manager, server_args.chat_template, server_args.model_path
        )

    if server_args.completion_template:
        load_completion_template_for_openai_api(server_args.completion_template)

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
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

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, scheduler_info
