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
import random
import signal
import threading
import time
from typing import AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import torch
import uvloop
import zmq

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    MultimodalDataInputFormat,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    parse_remote_instance_transfer_engine_info_from_scheduler_infos,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.tracing.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    configure_logger,
    get_bool_env_var,
    get_zmq_socket,
    is_cuda,
    kill_process_tree,
    launch_dummy_health_check_server,
    maybe_reindex_device_id,
    numa_utils,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_is_cuda = is_cuda()


def init_tokenizer_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
    TokenizerManagerClass: Optional[TokenizerManager] = None,
) -> Tuple[TokenizerManager, TemplateManager]:
    # Launch tokenizer process
    TokenizerManagerClass = TokenizerManagerClass or TokenizerManager
    tokenizer_manager = TokenizerManagerClass(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    return tokenizer_manager, template_manager


class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """

    # Some fields to allow people to override the server args
    # and launch processes for their private forks.
    server_args_class: ServerArgs = ServerArgs
    init_tokenizer_manager_func: Callable = staticmethod(init_tokenizer_manager)
    run_scheduler_process_func: Callable = staticmethod(run_scheduler_process)
    run_detokenizer_process_func: Callable = staticmethod(run_detokenizer_process)

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """

        # Parse server_args
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = self.server_args_class(**kwargs)
        self.server_args = server_args
        logger.info(f"{server_args=}")

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Launch subprocesses
        tokenizer_manager, template_manager, scheduler_infos, port_args = (
            _launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=self.init_tokenizer_manager_func,
                run_scheduler_process_func=self.run_scheduler_process_func,
                run_detokenizer_process_func=self.run_detokenizer_process_func,
            )
        )
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.scheduler_info = scheduler_infos[0]
        self.port_args = port_args
        self.remote_instance_transfer_engine_info = (
            parse_remote_instance_transfer_engine_info_from_scheduler_infos(
                scheduler_infos
            )
        )

        # Initialize ZMQ sockets
        context = zmq.Context(2)
        if self.server_args.node_rank == 0:
            self.send_to_rpc = get_zmq_socket(
                context, zmq.DEALER, self.port_args.rpc_ipc_name, True
            )
        else:
            self.send_to_rpc = None

        # Enable tracing
        if server_args.enable_trace:
            process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
            thread_label = "Tokenizer"
            if server_args.disaggregation_mode == "prefill":
                thread_label = "Prefill Tokenizer"
            elif server_args.disaggregation_mode == "decode":
                thread_label = "Decode Tokenizer"
            trace_set_thread_info(thread_label)

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

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
        # - List of preprocessed outputs from a Huggingface processor, each as a dict containing `format`: 'processor_output' and other data
        # - List of precomputed image embeddings, each as a dict containing field `format`: 'precomputed_embedding' and `feature`: the precomputed embedding
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        if self.server_args.enable_dp_attention:
            if data_parallel_rank is None:
                logger.debug("data_parallel_rank not provided, using default dispatch")
            elif data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            elif data_parallel_rank >= self.server_args.dp_size:
                raise ValueError(
                    f"data_parallel_rank must be less than dp_size: {self.server_args.dp_size}"
                )

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = self.loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = self.loop.run_until_complete(generator.__anext__())
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
        # - List of preprocessed outputs from a Huggingface processor, each as a dict containing `format`: 'processor_output' and other data
        # - List of precomputed image embeddings, each as a dict containing field `format`: 'precomputed_embedding' and `feature`: the precomputed embedding
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        if self.server_args.enable_dp_attention:
            if data_parallel_rank is None:
                logger.debug("data_parallel_rank not provided, using default dispatch")
            elif data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            elif data_parallel_rank >= self.server_args.dp_size:
                raise ValueError(
                    f"data_parallel_rank must be in range [0, {self.server_args.dp_size-1}]"
                )

        logger.debug(f"data_parallel_rank: {data_parallel_rank}")
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            return_hidden_states=return_hidden_states,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        dimensions: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    async def async_encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        dimensions: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,
        rid: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        Asynchronous version of encode method.

        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    def rerank(
        self,
        prompt: Union[List[List[str]]],
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret

    def shutdown(self):
        """Shutdown the engine"""
        kill_process_tree(os.getpid(), include_parent=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def flush_cache(self):
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def start_profile(self, **kwargs):
        self.loop.run_until_complete(self.tokenizer_manager.start_profile(**kwargs))

    def stop_profile(self):
        self.loop.run_until_complete(self.tokenizer_manager.stop_profile())

    def start_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.start_expert_distribution_record()
        )

    def stop_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.stop_expert_distribution_record()
        )

    def dump_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.dump_expert_distribution_record()
        )

    def get_server_info(self):
        internal_states = self.loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()
        )
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            "internal_states": internal_states,
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
        return self.loop.run_until_complete(
            self.tokenizer_manager.init_weights_update_group(obj, None)
        )

    def destroy_weights_update_group(
        self,
        group_name: str,
    ):
        """Destroy parameter update group."""
        obj = DestroyWeightsUpdateGroupReqInput(
            group_name=group_name,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.destroy_weights_update_group(obj, None)
        )

    def update_weights_from_distributed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_update_group",
        flush_cache: bool = True,
        load_format: Optional[str] = None,
    ):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        if load_format == "flattened_bucket":
            serialized_named_tensors = named_tensors
        else:
            serialized_named_tensors = [
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
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

        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_disk(obj, None)
        )

    def update_weights_from_ipc(
        self,
        zmq_handles: Dict[str, str],
        flush_cache: bool = True,
    ):
        """Update weights from IPC for checkpoint-engine integration."""
        obj = UpdateWeightsFromIPCReqInput(
            zmq_handles=zmq_handles,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_ipc(obj, None)
        )

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        return self.loop.run_until_complete(
            self.tokenizer_manager.get_weights_by_name(obj, None)
        )

    def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        """Load a new LoRA adapter without re-launching the engine."""

        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,
        )

        return self.loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter(obj, None)
        )

    def unload_lora_adapter(self, lora_name: str):
        """Unload a LoRA adapter without re-launching the engine."""

        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)

        return self.loop.run_until_complete(
            self.tokenizer_manager.unload_lora_adapter(obj, None)
        )

    def release_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        return self.loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)
        )

    def freeze_gc(self):
        """
        To maintain a high performance server with low latency, we want to reduce the
        stalls caused by the garbage collector scanning through a large number of objects.

        It is usually helpful to start the server and warm it up with real requests to
        initialize many of the long-lived objects that do not need to be garbage collected.

        After sufficient warmup, we can call this function to freeze the garbage collector
        so that all objects created before this point are considered out of scope for garbage
        collection.
        """

        self.loop.run_until_complete(self.tokenizer_manager.freeze_gc())

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

    def score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair. For example:
        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223] # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"
        The api would then return the probabilities of the model producing "Yes" and "No" as the next token.
        The output would look like:
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]


        Args:
            query: The query text or pre-tokenized query token IDs. Must be provided.
            items: The item text(s) or pre-tokenized item token IDs. Must be provided.
            label_token_ids: List of token IDs to compute probabilities for. If None, no token probabilities will be computed.
            apply_softmax: Whether to normalize probabilities using softmax.
            item_first: If True, prepend items to query. Otherwise append items to query.

        Returns:
            List of dictionaries mapping token IDs to their probabilities for each item.
            Each dictionary in the list corresponds to one item input.

        Raises:
            ValueError: If query is not provided, or if items is not provided,
                      or if token IDs are out of vocabulary, or if logprobs are not available for the specified tokens.
        """
        return self.loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                request=None,
            )
        )

    async def async_score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """
        Asynchronous version of score method.

        See score() for detailed documentation.
        """
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            request=None,
        )


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    if "NCCL_CUMEM_ENABLE" not in os.environ or server_args.enable_symm_mem:
        os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    if (
        "NCCL_NVLS_ENABLE" not in os.environ
        or server_args.enable_nccl_nvls
        or server_args.enable_symm_mem
    ):
        os.environ["NCCL_NVLS_ENABLE"] = str(
            int(server_args.enable_nccl_nvls or server_args.enable_symm_mem)
        )
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        # flashinfer uses this environment variable for various kernels from MoE to quant kernels
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    if os.environ.get("CUTE_DSL_LOG_LEVEL") is None:
        # Default to warning level, to avoid too many logs
        os.environ["CUTE_DSL_LOG_LEVEL"] = "30"

    if os.environ.get("CUTE_DSL_LOG_TO_CONSOLE") is None:
        # Need to set log to console, otherwise the log level won't take effect
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"

    # Can also be passed as argument
    os.environ["SGLANG_RUN_ID"] = (
        f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"
    )

    # Set prometheus env vars
    set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        if server_args.attention_backend == "flashinfer":
            assert_pkg_version(
                "flashinfer_python",
                "0.5.3",
                "Please uninstall the old version and "
                "reinstall the latest version by following the instructions "
                "at https://docs.flashinfer.ai/installation.html.",
            )
        if _is_cuda:
            assert_pkg_version(
                "sgl-kernel",
                "0.3.20",
                "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
            )

    if server_args.custom_sigquit_handler is None:
        # Register the signal handler.
        # The child processes will send SIGQUIT to this process when any error happens
        # This process then clean up the whole process tree
        # Note: This sigquit handler is used in the launch phase, and may be replaced by
        # the running_phase_sigquit_handler in the tokenizer manager after the grpc server is launched.
        def launch_phase_sigquit_handler(signum, frame):
            logger.error(
                "Received sigquit from a child process. It usually means the child failed."
            )
            kill_process_tree(os.getpid())

        signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)
    else:
        # Allow users to register a custom SIGQUIT handler for things like crash dump
        logger.error(
            f"Using custom SIGQUIT handler: {server_args.custom_sigquit_handler}"
        )
        signal.signal(signal.SIGQUIT, server_args.custom_sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _wait_for_scheduler_ready(
    scheduler_pipe_readers: List,
    scheduler_procs: List,
) -> List[Dict]:
    """Wait for the model to finish loading and return scheduler infos."""
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
    return scheduler_infos


def _launch_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    run_scheduler_process_func: Callable,
):
    scheduler_procs = []

    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

                with maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=run_scheduler_process_func,
                        args=(
                            server_args,
                            port_args,
                            gpu_id,
                            tp_rank,
                            moe_ep_rank,
                            pp_rank,
                            None,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                        server_args, gpu_id
                    ):
                        proc.start()

                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            kwargs=dict(
                server_args=server_args,
                port_args=port_args,
                pipe_writer=writer,
                run_scheduler_process_func=run_scheduler_process_func,
            ),
        )
        proc.start()
        scheduler_procs.append(proc)

    return scheduler_procs, scheduler_pipe_readers


def _launch_subprocesses(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable,
    run_scheduler_process_func: Callable,
    run_detokenizer_process_func: Callable,
    port_args: Optional[PortArgs] = None,
) -> Tuple[TokenizerManager, TemplateManager, Tuple[Dict], PortArgs]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    _set_envs_and_config(server_args)
    server_args.check_server_args()

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
    logger.info(f"{server_args=}")

    # Launch scheduler processes
    scheduler_procs, scheduler_pipe_readers = _launch_scheduler_processes(
        server_args=server_args,
        port_args=port_args,
        run_scheduler_process_func=run_scheduler_process_func,
    )

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        scheduler_infos = _wait_for_scheduler_ready(
            scheduler_pipe_readers, scheduler_procs
        )

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, scheduler_infos, port_args

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None, scheduler_infos, port_args

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process_func,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Init tokenizer manager first, as the bootstrap server is initialized here
    if server_args.tokenizer_worker_num == 1:
        tokenizer_manager, template_manager = init_tokenizer_manager_func(
            server_args, port_args
        )
    else:
        # Launch multi-tokenizer router
        tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
        template_manager = None

    # Wait for the model to finish loading
    scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)

    # Get back some info from scheduler to tokenizer_manager
    tokenizer_manager.max_req_input_len = scheduler_infos[0]["max_req_input_len"]

    return tokenizer_manager, template_manager, scheduler_infos, port_args
