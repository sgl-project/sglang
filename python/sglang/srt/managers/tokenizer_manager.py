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
"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import copy
import dataclasses
import logging
import os
import pickle
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Any, Awaitable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import fastapi
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.srt.aio_rwlock import RWLock
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.image_processor import (
    get_dummy_image_processor,
    get_image_processor,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    FlushCacheReq,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.metrics.collector import TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    dataclass_to_string_truncated,
    get_zmq_socket,
    kill_process_tree,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event
    obj: Any

    # For metrics
    created_time: float
    first_token_time: Optional[float] = None

    # For streaming output
    last_output_offset: int = 0


class TokenizerManager:
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args

        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.log_requests = server_args.log_requests
        self.log_requests_level = 0

        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )

        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id

        # Create image processor placeholder
        self.image_processor = get_dummy_image_processor()

        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # We want to parallelize the image pre-processing so we create an executor for it
                self.image_processor = get_image_processor(
                    self.model_config.hf_config, server_args, self.processor
                )
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

        # Store states
        self.no_create_loop = False
        self.rid_to_state: Dict[str, ReqState] = {}
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: List[Tuple] = []

        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None
        )
        self.asyncio_tasks = set()

        # For session info
        self.session_futures = {}  # session_id -> asyncio event

        # Others
        self.gracefully_exit = False
        self.init_weights_update_group_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_distributed_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_weights_from_tensor_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.get_weights_by_name_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.release_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        # Set after scheduler is initialized
        self.max_req_input_len = None

        # Metrics
        if self.enable_metrics:
            self.metrics_collector = TokenizerMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    # TODO: Add lora name/path in the future,
                },
            )

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut),
                    self._handle_batch_output,
                ),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (
                    InitWeightsUpdateGroupReqOutput,
                    self.init_weights_update_group_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromDistributedReqOutput,
                    self.update_weights_from_distributed_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromTensorReqOutput,
                    self.update_weights_from_tensor_communicator.handle_recv,
                ),
                (
                    GetWeightsByNameReqOutput,
                    self.get_weights_by_name_communicator.handle_recv,
                ),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
            ]
        )

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = time.time()

        self.auto_create_handle_loop()

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        obj.normalize_batch_and_arguments()

        if self.log_requests:
            max_length = 2048 if self.log_requests_level == 0 else 1 << 30
            logger.info(
                f"Receive: obj={dataclass_to_string_truncated(obj, max_length)}"
            )

        async with self.model_update_lock.reader_lock:
            is_single = obj.is_single
            if is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(obj, tokenized_obj, created_time)
                async for response in self._wait_one_response(obj, request):
                    yield response
            else:
                async for response in self._handle_batch_request(
                    obj, request, created_time
                ):
                    yield response

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )
            input_ids = self.tokenizer.encode(input_text)

        if self.is_generation:
            # TODO: also support getting embeddings for multimodal models
            image_inputs: Dict = await self.image_processor.process_images_async(
                obj.image_data, input_text or input_ids, obj, self.max_req_input_len
            )
            if image_inputs and "input_ids" in image_inputs:
                input_ids = image_inputs["input_ids"]
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        input_token_num = len(input_ids) if input_ids is not None else 0
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        if (
            obj.sampling_params.get("max_new_tokens") is not None
            and obj.sampling_params.get("max_new_tokens") + input_token_num
            >= self.context_len
        ):
            raise ValueError(
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of "
                f"{obj.sampling_params.get('max_new_tokens') + input_token_num} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{obj.sampling_params.get('max_new_tokens')} tokens for the "
                f"completion. Please reduce the number of tokens in the input "
                f"messages or the completion to fit within the limit."
            )

        # Parse sampling parameters
        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Build return object
        if isinstance(obj, GenerateReqInput):
            tokenized_obj = TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                obj.stream,
                lora_path=obj.lora_path,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                sampling_params,
            )

        return tokenized_obj

    def _send_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
        created_time: Optional[float] = None,
    ):
        event = asyncio.Event()
        state = ReqState([], False, event, obj, created_time=created_time)
        self.rid_to_state[obj.rid] = state
        self.send_to_scheduler.send_pyobj(tokenized_obj)

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]

        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(obj.rid)
                    raise ValueError(f"Abort request {obj.rid}")
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length = 2048 if self.log_requests_level == 0 else 1 << 30
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length)}, out={dataclass_to_string_truncated(out, max_length)}"
                    logger.info(msg)
                del self.rid_to_state[obj.rid]

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        raise ValueError(finish_reason["message"])

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    self.abort_request(obj.rid)
                    raise ValueError(f"Abort request {obj.rid}")

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
        created_time: Optional[float] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            # Send all requests
            for i in range(batch_size):
                tmp_obj = obj[i]
                tokenized_obj = await self._tokenize_one_request(tmp_obj)
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                generators.append(self._wait_one_response(tmp_obj, request))
                rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    def flush_cache(self):
        req = FlushCacheReq()
        self.send_to_scheduler.send_pyobj(req)

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        self.send_to_scheduler.send_pyobj(req)

    def start_profile(self):
        req = ProfileReq.START_PROFILE
        self.send_to_scheduler.send_pyobj(req)

    def stop_profile(self):
        req = ProfileReq.STOP_PROFILE
        self.send_to_scheduler.send_pyobj(req)

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        if True:
            # Hold the lock if it is not async. This means that weight sync
            # cannot run while requests are in progress.
            async with self.model_update_lock.writer_lock:
                return await self._wait_for_model_update_from_disk(obj)

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        self.send_to_scheduler.send_pyobj(obj)
        self.model_update_result = asyncio.Future()
        if self.server_args.dp_size == 1:
            result = await self.model_update_result
            if result.success:
                self.served_model_name = obj.model_path
                self.server_args.model_path = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            return result.success, result.message
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp = []
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self.server_args.model_path = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            return all_success, all_message

    async def init_weights_update_group(
        self,
        obj: InitWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.init_weights_update_group_communicator(obj))[0]
        return result.success, result.message

    async def update_weights_from_distributed(
        self,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be for update weights from distributed"

        # This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            result = (await self.update_weights_from_distributed_communicator(obj))[0]
            return result.success, result.message

    async def update_weights_from_tensor(
        self,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be for update weights from distributed"

        # This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            result = (await self.update_weights_from_tensor_communicator(obj))[0]
            return result.success, result.message

    async def get_weights_by_name(
        self, obj: GetWeightsByNameReqInput, request: Optional[fastapi.Request] = None
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if self.server_args.dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def open_session(
        self, obj: OpenSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        self.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(
        self, obj: CloseSessionReqInput, request: Optional[fastapi.Request] = None
    ):
        await self.send_to_scheduler.send_pyobj(obj)

    def configure_logging(self, obj: ConfigureLoggingReq):
        if obj.log_requests is not None:
            self.log_requests = obj.log_requests
        if obj.log_requests_level is not None:
            self.log_requests_level = obj.log_requests_level
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        logging.info(f"Config logging: {obj=}")

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(1)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        if self.no_create_loop:
            return

        self.no_create_loop = True
        loop = asyncio.get_event_loop()
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )

        # We cannot add signal handler when the tokenizer manager is not in
        # the main thread due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = SignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.signal_handler)
        else:
            logger.warning(
                "Signal handler is not added because the tokenizer manager is "
                "not in the main thread. This disables graceful shutdown of the "
                "tokenizer manager when SIGTERM is received."
            )
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            logger.info(
                f"Gracefully exiting... remaining number of requests {remain_num_req}"
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            self._result_dispatcher(recv_obj)

    def _handle_batch_output(
        self, recv_obj: Union[BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut]
    ):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                continue

            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
            }

            if getattr(state.obj, "return_logprob", False):
                self.convert_logprob_style(
                    meta_info,
                    state.obj.top_logprobs_num,
                    state.obj.return_text_in_logprobs,
                    recv_obj,
                    i,
                )

            if self.server_args.speculative_algorithm:
                meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]

            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if isinstance(recv_obj, BatchStrOut):
                out_dict = {
                    "text": recv_obj.output_strs[i],
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchTokenIDOut):
                out_dict = {
                    "token_ids": recv_obj.output_ids[i],
                    "meta_info": meta_info,
                }
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.out_list.append(out_dict)
            state.finished = recv_obj.finished_reasons[i] is not None
            state.event.set()

            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)

    def convert_logprob_style(
        self,
        meta_info: dict,
        top_logprobs_num: int,
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ):
        meta_info["input_token_logprobs"] = self.detokenize_logprob_tokens(
            recv_obj.input_token_logprobs_val[recv_obj_index],
            recv_obj.input_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
            recv_obj.output_token_logprobs_val[recv_obj_index],
            recv_obj.output_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )

        if top_logprobs_num > 0:
            meta_info["input_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                recv_obj.input_top_logprobs_val[recv_obj_index],
                recv_obj.input_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                recv_obj.output_top_logprobs_val[recv_obj_index],
                recv_obj.output_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret

    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOut, i: int):
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        if state.first_token_time is None:
            state.first_token_time = time.time()
            self.metrics_collector.observe_time_to_first_token(
                state.first_token_time - state.created_time
            )
        else:
            if completion_tokens >= 2:
                # Compute time_per_output_token for the streaming case
                self.metrics_collector.observe_time_per_output_token(
                    (time.time() - state.first_token_time) / (completion_tokens - 1)
                )

        if state.finished:
            self.metrics_collector.observe_one_finished_request(
                recv_obj.prompt_tokens[i], completion_tokens
            )
            self.metrics_collector.observe_e2e_request_latency(
                time.time() - state.created_time
            )
            # Compute time_per_output_token for the non-streaming case
            if (
                hasattr(state.obj, "stream")
                and not state.obj.stream
                and completion_tokens >= 1
            ):
                self.metrics_collector.observe_time_per_output_token(
                    (time.time() - state.created_time) / completion_tokens
                )

    def dump_requests(self, state: ReqState, out_dict: dict):
        self.dump_request_list.append(
            (state.obj, out_dict, state.created_time, time.time())
        )

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            logger.info(f"Dump {len(self.dump_request_list)} requests to {filename}")

            to_dump = self.dump_request_list
            self.dump_request_list = []

            def background_task():
                os.makedirs(self.dump_requests_folder, exist_ok=True)
                with open(filename, "wb") as f:
                    pickle.dump(to_dump, f)

            # Schedule the task to run in the background without awaiting it
            asyncio.create_task(asyncio.to_thread(background_task))

    def _handle_open_session_req_output(self, recv_obj):
        self.session_futures[recv_obj.session_id].set_result(
            recv_obj.session_id if recv_obj.success else None
        )

    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        if self.server_args.dp_size == 1:
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp.append(recv_obj)
            # set future if the all results are recevied
            if len(self.model_update_tmp) == self.server_args.dp_size:
                self.model_update_result.set_result(self.model_update_tmp)


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


class SignalHandler:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def signal_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True


T = TypeVar("T")


class _Communicator(Generic[T]):
    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._result_future: Optional[asyncio.Future] = None
        self._result_values: Optional[List[T]] = None

    async def __call__(self, obj):
        self._sender.send_pyobj(obj)
        self._result_future = asyncio.Future()
        self._result_values = []
        await self._result_future
        result_values = self._result_values
        self._result_future = self._result_values = None
        return result_values

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_future.set_result(None)
