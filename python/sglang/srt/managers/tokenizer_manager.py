"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import dataclasses
import json
import logging
import os
import signal
import sys
from typing import Dict, List, Optional, Tuple, Union

import fastapi
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_processor,
    get_tokenizer,
)
from sglang.srt.managers.image_processor import (
    get_dummy_image_processor,
    get_image_processor,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    EmbeddingReqInput,
    FlushCacheReq,
    GenerateReqInput,
    GetMemPoolSizeReq,
    GetMemPoolSizeReqOutput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    get_zmq_socket,
    is_generation_model,
    is_multimodal_model,
    kill_child_process,
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event


class TokenizerManager:
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args

        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name
        )

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_override_args=json.loads(server_args.json_model_override_args),
        )
        self.is_generation = is_generation_model(
            self.hf_config.architectures, self.server_args.is_embedding
        )
        self.context_len = server_args.context_length or get_context_length(
            self.hf_config
        )

        # Create image processor placeholder
        self.image_processor = get_dummy_image_processor()

        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(self.hf_config.architectures):
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # We want to parallelize the image pre-processing so we create an executor for it
                self.image_processor = get_image_processor(
                    self.hf_config, server_args, self.processor
                )
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )

        # Store states
        self.to_create_loop = True
        self.rid_to_state: Dict[str, ReqState] = {}

        # For update model weights
        self.model_update_lock = asyncio.Lock()
        self.model_update_result = None

        # Others
        self.gracefully_exit = False

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        if self.to_create_loop:
            self.create_handle_loop()

        while self.model_update_lock.locked():
            await asyncio.sleep(0.001)

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        obj.set_missing_values()
        is_single = obj.is_single
        if is_single:
            async for response in self._handle_single_request(obj, request):
                yield response
        else:
            async for response in self._handle_batch_request(obj, request):
                yield response

    async def _handle_single_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # Tokenize
        rid = obj.rid
        input_text = obj.text
        if obj.input_ids is None:
            input_ids = self.tokenizer.encode(input_text)
        else:
            input_ids = obj.input_ids

        if self.is_generation:
            image_inputs = await self.image_processor.process_images_async(
                obj.image_data, input_text or input_ids, obj
            )
            if image_inputs and "input_ids" in image_inputs:
                input_ids = image_inputs["input_ids"]
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num

        if len(input_ids) >= self.context_len:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Parse sampling parameters
        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Send to the controller
        if isinstance(obj, GenerateReqInput):
            tokenized_obj = TokenizedGenerateReqInput(
                rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                obj.stream,
                obj.lora_path
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                rid,
                input_text,
                input_ids,
                sampling_params,
            )

        self.send_to_scheduler.send_pyobj(tokenized_obj)

        # Recv results
        event = asyncio.Event()
        state = ReqState([], False, event)
        self.rid_to_state[rid] = state
        async for response in self._wait_for_response(state, obj, rid, request):
            yield response

    async def _wait_for_response(
        self,
        state: ReqState,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        rid: str,
        request: Optional[fastapi.Request] = None,
    ):
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(rid)
                    raise ValueError(f"Abort request {rid}")
                continue

            if isinstance(obj, GenerateReqInput):
                out = self.convert_logprob_style(
                    state.out_list[-1],
                    obj.return_logprob,
                    obj.top_logprobs_num,
                    obj.return_text_in_logprobs,
                )
            else:  # isinstance(obj, (EmbeddingReqInput,))
                out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.server_args.log_requests:
                    # Log requests
                    logger.info(f"in={obj}, out={out}")
                del self.rid_to_state[rid]
                yield out
                break

            state.event.clear()
            yield out

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

    async def get_memory_pool_size(self):
        if self.to_create_loop:
            self.create_handle_loop()

        req = GetMemPoolSizeReq()

        self.send_to_scheduler.send_pyobj(req)
        self.mem_pool_size = asyncio.Future()

        # FIXME: Each request should have its own future instead of using `self.mem_pool_size`.
        if self.server_args.dp_size == 1:
            res = await self.mem_pool_size
            return res.size
        else: # self.server_args.dp_size > 1
            self.mem_pool_size_tmp = []
            res = await self.mem_pool_size
            ret = [r.size for r in res]
            return ret

    async def update_weights(
        self, obj: UpdateWeightReqInput, request: Optional[fastapi.Request] = None
    ):
        if self.to_create_loop:
            self.create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format

        if not self.model_update_lock.locked():
        
            async with self.model_update_lock:
                # wait for the previous generation requests to finish
                while len(self.rid_to_state) > 0:
                    await asyncio.sleep(0.001)
                self.send_to_scheduler.send_pyobj(obj)
                self.model_update_result = asyncio.Future()

                if self.server_args.dp_size == 1:
                    result = await self.model_update_result
                    if result.success:
                        self.server_args.model_path = obj.model_path
                        self.server_args.load_format = obj.load_format
                        self.model_path = obj.model_path
                    return result.success, result.message
                else: # self.server_args.dp_size > 1
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

        else:
            return False, "Another update is in progress. Please try again later."

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

    def create_handle_loop(self):
        if not self.to_create_loop:
            return

        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        loop.create_task(self.handle_loop())

        signal_handler = SignalHandler(self)
        loop.add_signal_handler(signal.SIGTERM, signal_handler.signal_handler)
        loop.create_task(self.sigterm_watchdog())

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(60)

        # drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            logger.info(
                f"Gracefully exiting... remaining number of requests {remain_num_req}"
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        kill_child_process(include_self=True)
        sys.exit(-1)

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj: Union[
                BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut, UpdateWeightReqOutput
            ] = await self.recv_from_detokenizer.recv_pyobj()

            if isinstance(recv_obj, UpdateWeightReqOutput):
                if self.server_args.dp_size == 1:
                    self.model_update_result.set_result(recv_obj)
                else: # self.server_args.dp_size > 1
                    self.model_update_tmp.append(recv_obj)
                    # set future if the all results are recevied
                    if len(self.model_update_tmp) == self.server_args.dp_size:
                        self.model_update_result.set_result(self.model_update_tmp)
                continue
            elif isinstance(recv_obj, GetMemPoolSizeReqOutput):
                if self.server_args.dp_size == 1:
                    self.mem_pool_size.set_result(recv_obj)
                else: # self.sever_args.dp_size > 1
                    self.mem_pool_size_tmp.append(recv_obj)
                    # set future if the all results are received
                    if len(self.mem_pool_size_tmp) == self.server_args.dp_size:
                        self.mem_pool_size.set_result(self.mem_pool_size_tmp)
                continue

            assert isinstance(
                recv_obj, (BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut)
            ), f"Unexpected obj received: {type(recv_obj)}"

            for i, rid in enumerate(recv_obj.rids):
                state = self.rid_to_state.get(rid, None)
                if state is None:
                    continue

                recv_obj.meta_info[i]["id"] = rid
                if isinstance(recv_obj, BatchStrOut):
                    out_dict = {
                        "text": recv_obj.output_strs[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                elif isinstance(recv_obj, BatchTokenIDOut):
                    out_dict = {
                        "token_ids": recv_obj.output_ids[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                else:
                    assert isinstance(recv_obj, BatchEmbeddingOut)
                    out_dict = {
                        "embedding": recv_obj.embeddings[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                state.out_list.append(out_dict)
                state.finished = recv_obj.finished_reason[i] is not None
                state.event.set()

    def convert_logprob_style(
        self,
        ret: dict,
        return_logprob: bool,
        top_logprobs_num: int,
        return_text_in_logprobs: bool,
    ):
        if return_logprob:
            ret["meta_info"]["input_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["input_token_logprobs"], return_text_in_logprobs
            )
            ret["meta_info"]["output_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["output_token_logprobs"], return_text_in_logprobs
            )

            if top_logprobs_num > 0:
                ret["meta_info"]["input_top_logprobs"] = (
                    self.detokenize_top_logprobs_tokens(
                        ret["meta_info"]["input_top_logprobs"],
                        return_text_in_logprobs,
                    )
                )
                ret["meta_info"]["output_top_logprobs"] = (
                    self.detokenize_top_logprobs_tokens(
                        ret["meta_info"]["output_top_logprobs"], return_text_in_logprobs
                    )
                )
        return ret

    def detokenize_logprob_tokens(
        self, token_logprobs: List[Tuple[float, int]], decode_to_text: bool
    ):
        # TODO(lianmin): This should run on DetokenizerManager
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

        assert self.tokenizer is not None
        token_ids = [tid for _, tid in token_logprobs]
        token_texts = self.tokenizer.batch_decode(token_ids)
        return [
            (logprob, token_id, token_text)
            for (logprob, token_id), token_text in zip(token_logprobs, token_texts)
        ]

    def detokenize_top_logprobs_tokens(self, top_logprobs, decode_to_text: bool):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        for i, token_top_logprobs in enumerate(top_logprobs):
            if token_top_logprobs:
                top_logprobs[i] = self.detokenize_logprob_tokens(
                    token_top_logprobs, decode_to_text
                )
        return top_logprobs


class SignalHandler:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def signal_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True
