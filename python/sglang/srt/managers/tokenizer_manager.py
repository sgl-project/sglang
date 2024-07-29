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
import concurrent.futures
import dataclasses
import logging
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import numpy as np
import transformers
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
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchStrOut,
    BatchTokenIDOut,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import is_multimodal_model, load_image
from sglang.utils import get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    out_list: List
    finished: bool
    event: asyncio.Event


class TokenizerManager:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args: dict = None,
    ):
        self.server_args = server_args

        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = context.socket(zmq.PULL)
        self.recv_from_detokenizer.bind(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{port_args.controller_port}")

        self.model_path = server_args.model_path
        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_overide_args=model_overide_args,
        )

        if server_args.context_length is not None:
            self.context_len = server_args.context_length
        else:
            self.context_len = get_context_length(self.hf_config)

        if is_multimodal_model(self.model_path):
            self.processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
            self.tokenizer = self.processor.tokenizer
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.executor = concurrent.futures.ProcessPoolExecutor(
                initializer=init_global_processor,
                mp_context=mp.get_context("fork"),
                initargs=(server_args,),
            )
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )

        self.to_create_loop = True
        self.rid_to_state: Dict[str, ReqState] = {}

    async def get_pixel_values(self, image_data):
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints if aspect_ratio == "anyres" else None
        )
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                get_pixel_values,
                image_data,
                aspect_ratio,
                grid_pinpoints,
            )
        else:
            return get_pixel_values(
                image_data, aspect_ratio, grid_pinpoints, self.processor
            )

    async def generate_request(self, obj: GenerateReqInput, request=None):
        if self.to_create_loop:
            self.create_handle_loop()

        obj.post_init()
        is_single = obj.is_single

        if is_single:
            async for response in self._handle_single_request(obj, request):
                yield response
        else:
            if obj.stream:
                raise ValueError("Do not support stream for batch mode.")

            async for response in self._handle_batch_request(obj, request):
                yield response

    async def _handle_single_request(
        self, obj, request, index=None, is_cache_for_prefill=False
    ):
        if not is_cache_for_prefill:
            not_use_index = not (index is not None)
            rid = obj.rid if not_use_index else obj.rid[index]
            input_text = obj.text if not_use_index else obj.text[index]
            input_ids = (
                self.tokenizer.encode(input_text)
                if obj.input_ids is None
                else obj.input_ids
            )
            if not not_use_index and obj.input_ids:
                input_ids = obj.input_ids[index]

            self._validate_input_length(input_ids)

            sampling_params = self._get_sampling_params(
                obj.sampling_params if not_use_index else obj.sampling_params[index]
            )
            pixel_values, image_hash, image_size = await self._get_pixel_values(
                obj.image_data if not_use_index else obj.image_data[index]
            )
            return_logprob = (
                obj.return_logprob if not_use_index else obj.return_logprob[index]
            )
            logprob_start_len = (
                obj.logprob_start_len if not_use_index else obj.logprob_start_len[index]
            )
            top_logprobs_num = (
                obj.top_logprobs_num if not_use_index else obj.top_logprobs_num[index]
            )
        else:
            if isinstance(obj.text, list):
                input_text = obj.text[index]
                rid = obj.rid[index]
            else:
                input_text = obj.text
                rid = obj.rid[0]
            input_ids = self.tokenizer.encode(input_text)
            sampling_params = SamplingParams(**obj.sampling_params[0])
            sampling_params.max_new_tokens = 0
            pixel_values, image_hash, image_size = await self._get_pixel_values(
                obj.image_data[0]
            )
            return_logprob = obj.return_logprob[0]
            logprob_start_len = obj.logprob_start_len[0]
            top_logprobs_num = obj.top_logprobs_num[0]

        tokenized_obj = TokenizedGenerateReqInput(
            rid,
            input_text,
            input_ids,
            pixel_values,
            image_hash,
            image_size,
            sampling_params,
            return_logprob,
            logprob_start_len,
            top_logprobs_num,
            obj.stream,
        )
        self.send_to_router.send_pyobj(tokenized_obj)

        event = asyncio.Event()
        state = ReqState([], False, event)
        self.rid_to_state[rid] = state
        if not is_cache_for_prefill:
            async for response in self._wait_for_response(
                event, state, obj, rid, request
            ):
                yield response
        else:
            await self._wait_for_cache_prefill_response(event, state, obj, rid, request)
            yield input_ids

    async def _handle_batch_request(self, obj: GenerateReqInput, request):
        batch_size = obj.batch_size
        parallel_sample_num = obj.parallel_sample_num

        if parallel_sample_num != 1:
            # Send prefill requests to cache the common input
            parallel_sample_num += 1
            input_id_result = [] if obj.input_ids is None else None
            for i in range(batch_size):
                async for input_id in self._handle_single_request(
                    obj, request, index=i, is_cache_for_prefill=True
                ):
                    if input_id_result is not None:
                        input_id_result.append(input_id)
                    pass
            if len(input_id_result) > 1 and input_id_result is not None:
                obj.input_ids = input_id_result
            elif input_id_result is not None:
                obj.input_ids = input_id_result[0]
        # First send out all requests
        for i in range(batch_size):
            for j in range(parallel_sample_num):
                if j == 0 and parallel_sample_num != 1:
                    continue
                index = i * parallel_sample_num + j
                if parallel_sample_num != 1:
                    # Here when using parallel sampling we should consider prefill stage so the index is :  j + i * (parallel_sample_num-1) + batch_size - 1
                    index += batch_size - 1 - i
                rid = obj.rid[index]
                if parallel_sample_num == 1:
                    ## select operation
                    if obj.input_ids is None:
                        input_text = obj.text[i]
                        input_ids = self.tokenizer.encode(obj.text[i])
                    else:
                        input_text = None
                        input_ids = obj.input_ids[i]
                else:
                    if batch_size == 1:
                        input_text = obj.text
                        input_ids = obj.input_ids
                    else:
                        input_text = obj.text[i]
                        input_ids = obj.input_ids[i]
                sampling_params = self._get_sampling_params(obj.sampling_params[index])
                pixel_values, image_hash, image_size = await self._get_pixel_values(
                    obj.image_data[index]
                )

                tokenized_obj = TokenizedGenerateReqInput(
                    rid,
                    input_text,
                    input_ids,
                    pixel_values,
                    image_hash,
                    image_size,
                    sampling_params,
                    obj.return_logprob[index],
                    obj.logprob_start_len[index],
                    obj.top_logprobs_num[index],
                    obj.stream,
                )
                self.send_to_router.send_pyobj(tokenized_obj)

                event = asyncio.Event()
                state = ReqState([], False, event)
                self.rid_to_state[rid] = state

        # Then wait for all responses
        output_list = []
        for i in range(batch_size):
            for j in range(parallel_sample_num):
                if j == 0 and parallel_sample_num != 1:
                    continue
                index = i * parallel_sample_num + j
                if parallel_sample_num != 1:
                    index += batch_size - 1 - i
                rid = obj.rid[index]
                state = self.rid_to_state[rid]

                while True:
                    try:
                        await asyncio.wait_for(state.event.wait(), timeout=4)
                        break
                    except asyncio.TimeoutError:
                        if request is not None and await request.is_disconnected():
                            for rid in obj.rid:
                                self.abort_request(rid)
                            raise ValueError(f"Abort request {rid}")
                        continue
                output_list.append(
                    self.convert_logprob_style(
                        state.out_list[-1],
                        obj.return_logprob[index],
                        obj.top_logprobs_num[index],
                        obj.return_text_in_logprobs,
                    )
                )
                assert state.finished
                del self.rid_to_state[rid]

        yield output_list

    def _validate_input_length(self, input_ids: List[int]):
        if len(input_ids) >= self.context_len:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

    def _get_sampling_params(self, sampling_params_data: dict):
        sampling_params = SamplingParams(**sampling_params_data)
        if sampling_params.max_new_tokens != 0:
            sampling_params.normalize(self.tokenizer)
            sampling_params.verify()
        return sampling_params

    async def _get_pixel_values(self, image_data):
        if isinstance(image_data, list) and len(image_data) > 0:
            return await self.get_pixel_values(image_data[0])
        elif isinstance(image_data, str):
            return await self.get_pixel_values(image_data)
        else:
            return None, None, None

    async def _wait_for_response(
        self,
        event: asyncio.Event,
        state: ReqState,
        obj: GenerateReqInput,
        rid: str,
        request,
    ):
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(rid)
                    raise ValueError(f"Abort request {rid}")
                continue

            out = self.convert_logprob_style(
                state.out_list[-1],
                obj.return_logprob,
                obj.top_logprobs_num,
                obj.return_text_in_logprobs,
            )

            if self.server_args.log_requests and state.finished:
                logger.info(f"in={obj.text}, out={out}")

            state.out_list = []
            if state.finished:
                del self.rid_to_state[rid]
                yield out
                break

            event.clear()
            yield out

    async def _wait_for_cache_prefill_response(
        self,
        event: asyncio.Event,
        state: ReqState,
        obj: GenerateReqInput,
        rid: str,
        request,
    ):
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
                break
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    for rid in obj.rid:
                        self.abort_request(rid)
                    raise ValueError(f"Abort request {rid}")
                continue

        assert state.finished
        del self.rid_to_state[rid]

    def flush_cache(self):
        req = FlushCacheReq()
        self.send_to_router.send_pyobj(req)

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        self.send_to_router.send_pyobj(req)

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(3)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rids:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def create_handle_loop(self):
        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        loop.create_task(self.handle_loop())

    async def handle_loop(self):
        while True:
            recv_obj: BatchTokenIDOut = await self.recv_from_detokenizer.recv_pyobj()
            assert isinstance(recv_obj, BatchStrOut)

            for i, rid in enumerate(recv_obj.rids):
                state = self.rid_to_state.get(rid, None)
                if state is None:
                    continue

                recv_obj.meta_info[i]["id"] = rid
                out_dict = {
                    "text": recv_obj.output_strs[i],
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
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

        token_ids = [tid for _, tid in token_logprobs]
        token_texts = self.tokenizer.batch_decode(token_ids)
        return [
            (logprob, token_id, token_text)
            for (logprob, token_id), token_text, in zip(token_logprobs, token_texts)
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


global global_processor


def init_global_processor(server_args: ServerArgs):
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )


def get_pixel_values(
    image_data, image_aspect_ratio=None, image_grid_pinpoints=None, processor=None
):
    try:
        processor = processor or global_processor
        image, image_size = load_image(image_data)
        if image_size is not None:
            image_hash = hash(image_data)
            pixel_values = processor.image_processor(image)["pixel_values"]
            for _ in range(len(pixel_values)):
                pixel_values[_] = pixel_values[_].astype(np.float16)
            pixel_values = np.stack(pixel_values, axis=0)
            return pixel_values, image_hash, image_size
        else:
            image_hash = hash(image_data)
            if image_aspect_ratio == "pad":
                image = expand2square(
                    image,
                    tuple(int(x * 255) for x in processor.image_processor.image_mean),
                )
                pixel_values = processor.image_processor(image)["pixel_values"][0]
            elif image_aspect_ratio == "anyres":
                pixel_values = process_anyres_image(
                    image, processor.image_processor, image_grid_pinpoints
                )
            else:
                pixel_values = processor.image_processor(image)["pixel_values"][0]
            pixel_values = pixel_values.astype(np.float16)
            return pixel_values, image_hash, image.size
    except Exception:
        print("Exception in TokenizerManager:\n" + get_exception_traceback())
