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
from typing import Dict, List, Optional, Tuple, Union

import fastapi
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
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    EmbeddingReqInput,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import is_generation_model, is_multimodal_model, load_image
from sglang.utils import get_exception_traceback

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
        model_override_args: dict = None,
    ):
        self.server_args = server_args

        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = context.socket(zmq.PULL)
        self.recv_from_detokenizer.bind(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        self.send_to_controller = context.socket(zmq.PUSH)
        self.send_to_controller.connect(f"tcp://127.0.0.1:{port_args.controller_port}")

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_override_args=model_override_args,
        )
        self.is_generation = is_generation_model(
            self.hf_config.architectures, self.server_args.is_embedding
        )
        self.context_len = server_args.context_length or get_context_length(
            self.hf_config
        )

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

                # We want to parallelize the image pre-processing so we
                # create an executor for it
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

        # Store states
        self.to_create_loop = True
        self.rid_to_state: Dict[str, ReqState] = {}

        # For update model weights
        self.model_update_lock = asyncio.Lock()
        self.model_update_result = None

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        if self.to_create_loop:
            self.create_handle_loop()

        while self.model_update_lock.locked():
            await asyncio.sleep(0.001)

        obj.post_init()
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
        index: Optional[int] = None,
        is_cache_for_prefill: Optional[bool] = False,
    ):
        if not is_cache_for_prefill:  # The normal case with a single prompt
            not_use_index = index is None

            rid = obj.rid if not_use_index else obj.rid[index]
            input_text = obj.text if not_use_index else obj.text[index]
            if obj.input_ids is None:
                assert self.tokenizer is not None
                input_ids = self.tokenizer.encode(input_text)
            else:
                input_ids = obj.input_ids if not_use_index else obj.input_ids[index]

            self._validate_input_length(input_ids)

            sampling_params = self._get_sampling_params(
                obj.sampling_params if not_use_index else obj.sampling_params[index]
            )

            if self.is_generation:
                pixel_values, image_hashes, image_sizes = await self._get_pixel_values(
                    obj.image_data if not_use_index else obj.image_data[index]
                )
                return_logprob = (
                    obj.return_logprob if not_use_index else obj.return_logprob[index]
                )
                logprob_start_len = (
                    obj.logprob_start_len
                    if not_use_index
                    else obj.logprob_start_len[index]
                )
                if return_logprob and logprob_start_len == -1:
                    logprob_start_len = len(input_ids) - 1
                top_logprobs_num = (
                    obj.top_logprobs_num
                    if not_use_index
                    else obj.top_logprobs_num[index]
                )
        else:  # A prefill request to cache the common prompt for parallel sampling
            assert self.is_generation
            if obj.text is not None:
                if isinstance(obj.text, list):
                    input_text = obj.text[index]
                    rid = obj.rid[index]
                else:
                    input_text = obj.text
                    rid = obj.rid[0]
                if self.tokenizer is not None:
                    input_ids = self.tokenizer.encode(input_text)
                else:
                    assert obj.input_ids is not None
                    input_ids = obj.input_ids
                    if isinstance(obj.input_ids, list) and isinstance(
                        obj.input_ids[0], list
                    ):
                        # when obj["input_ids"] is List[List[int]]
                        input_ids = obj.input_ids[index]
                        rid = obj.rid[index]
                    else:
                        input_ids = obj.input_ids
                        rid = obj.rid[0]
            else:
                input_text = None
                if isinstance(obj.input_ids, list) and isinstance(
                    obj.input_ids[0], list
                ):
                    # when obj["input_ids"] is List[List[int]]
                    input_ids = obj.input_ids[index]
                    rid = obj.rid[index]
                else:
                    input_ids = obj.input_ids
                    rid = obj.rid[0]

            sampling_params = SamplingParams(**obj.sampling_params[0])
            sampling_params.max_new_tokens = 0
            pixel_values, image_hashes, image_sizes = await self._get_pixel_values(
                obj.image_data[0]
            )
            return_logprob = obj.return_logprob[0]
            logprob_start_len = obj.logprob_start_len[0]
            top_logprobs_num = obj.top_logprobs_num[0]

        # Send to the controller
        if self.is_generation:
            if return_logprob and logprob_start_len == -1:
                logprob_start_len = len(input_ids) - 1
            tokenized_obj = TokenizedGenerateReqInput(
                rid,
                input_text,
                input_ids,
                pixel_values,
                image_hashes,
                image_sizes,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                obj.stream,
            )
        else:  # is embedding
            tokenized_obj = TokenizedEmbeddingReqInput(
                rid,
                input_text,
                input_ids,
                sampling_params,
            )
        self.send_to_controller.send_pyobj(tokenized_obj)

        # Recv results
        event = asyncio.Event()
        state = ReqState([], False, event)
        self.rid_to_state[rid] = state
        if not is_cache_for_prefill:
            async for response in self._wait_for_response(state, obj, rid, request):
                yield response
        else:
            assert self.is_generation
            await self._wait_for_cache_prefill_response(state, obj, rid, request)
            yield input_ids

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        batch_size = obj.batch_size
        if self.is_generation:
            parallel_sample_num = obj.parallel_sample_num

            if parallel_sample_num != 1:
                # Send prefill requests to cache the common prefix
                parallel_sample_num += 1
                input_id_result = [] if obj.input_ids is None else None
                for i in range(batch_size):
                    async for input_id in self._handle_single_request(
                        obj, request, index=i, is_cache_for_prefill=True
                    ):
                        if input_id_result is not None:
                            input_id_result.append(input_id)
                if input_id_result is not None and len(input_id_result) > 1:
                    obj.input_ids = input_id_result
                elif input_id_result is not None:
                    obj.input_ids = input_id_result[0]
        else:
            parallel_sample_num = 1

        # First send out all requests
        generators = []
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
                    assert obj.input_ids is not None
                    if batch_size == 1:
                        input_text = None
                        input_ids = obj.input_ids
                    else:
                        input_text = None
                        input_ids = obj.input_ids[i]
                sampling_params = self._get_sampling_params(obj.sampling_params[index])

                if self.is_generation:
                    if obj.return_logprob[index] and obj.logprob_start_len[index] == -1:
                        obj.logprob_start_len[index] = len(input_ids) - 1
                    pixel_values, image_hashes, image_sizes = (
                        await self._get_pixel_values(obj.image_data[index])
                    )

                    tokenized_obj = TokenizedGenerateReqInput(
                        rid,
                        input_text,
                        input_ids,
                        pixel_values,
                        image_hashes,
                        image_sizes,
                        sampling_params,
                        obj.return_logprob[index],
                        obj.logprob_start_len[index],
                        obj.top_logprobs_num[index],
                        obj.stream,
                    )
                else:
                    tokenized_obj = TokenizedEmbeddingReqInput(
                        rid,
                        input_text,
                        input_ids,
                        sampling_params,
                    )
                self.send_to_controller.send_pyobj(tokenized_obj)

                event = asyncio.Event()
                state = ReqState([], False, event)
                self.rid_to_state[rid] = state

                generators.append(
                    self._wait_for_response(
                        state,
                        obj,
                        rid,
                        request,
                        index=index,
                        response_index=len(generators),
                    )
                )

        # Then process the responses based on streaming option
        is_stream = hasattr(obj, "stream") and obj.stream

        tasks = [asyncio.create_task(gen.__anext__()) for gen in generators]
        output_list = [None] * len(tasks)

        # Recv results
        while tasks:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                cur_index = tasks.index(task)

                try:
                    result = task.result()

                    if is_stream:
                        yield result
                    else:
                        output_list[result["index"]] = result

                    tasks[cur_index] = asyncio.create_task(
                        generators[cur_index].__anext__()
                    )
                except StopAsyncIteration:
                    del generators[cur_index]
                    del tasks[cur_index]

        if not is_stream:
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

    async def _wait_for_response(
        self,
        state: ReqState,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        rid: str,
        request: Optional[fastapi.Request] = None,
        index: Optional[int] = None,
        response_index: int = 0,
    ):
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    for rid in [obj.rid] if obj.is_single else obj.rid:
                        self.abort_request(rid)
                    raise ValueError(f"Abort request {rid}")
                continue

            if self.is_generation:
                out = self.convert_logprob_style(
                    state.out_list[-1],
                    obj.return_logprob if index is None else obj.return_logprob[index],
                    (
                        obj.top_logprobs_num
                        if index is None
                        else obj.top_logprobs_num[index]
                    ),
                    obj.return_text_in_logprobs,
                )
            else:  # isinstance(obj, EmbeddingReqInput)
                out = state.out_list[-1]

            out["index"] = response_index

            # Log requests
            if self.server_args.log_requests and state.finished:
                logger.info(f"in={obj}, out={out}")

            state.out_list = []
            if state.finished:
                del self.rid_to_state[rid]
                yield out
                break

            state.event.clear()
            yield out

    async def _wait_for_cache_prefill_response(
        self,
        state: ReqState,
        obj: GenerateReqInput,
        rid: str,
        request: Optional[fastapi.Request] = None,
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
        self.send_to_controller.send_pyobj(req)

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        self.send_to_controller.send_pyobj(req)

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
                    await asyncio.sleep(0)
                self.send_to_controller.send_pyobj(obj)
                self.model_update_result = asyncio.Future()
                result = await self.model_update_result
                if result.success:
                    self.server_args.model_path = obj.model_path
                    self.server_args.load_format = obj.load_format
                    self.model_path = obj.model_path
            return result.success, result.message
        else:
            return False, "Another update is in progress. Please try again later."

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(3)
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

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj: Union[
                BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut, UpdateWeightReqOutput
            ] = await self.recv_from_detokenizer.recv_pyobj()

            if isinstance(recv_obj, UpdateWeightReqOutput):
                self.model_update_result.set_result(recv_obj)
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
                    read_start = 0 if i == 0 else recv_obj.read_offsets[i - 1]
                    out_dict = {
                        "token_ids": recv_obj.decode_ids[
                            read_start : recv_obj.read_offsets[i]
                        ],
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
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

        assert self.tokenizer is not None
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

    async def _get_pixel_values(self, image_data: List[Union[str, bytes]]):
        if not image_data:
            return None, None, None

        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints
            if hasattr(self.hf_config, "image_grid_pinpoints")
            and "anyres" in aspect_ratio
            else None
        )

        if isinstance(image_data, list) and len(image_data) > 0:
            # Multiple images
            if len(image_data) > 1:
                aspect_ratio = "pad"  # LLaVA OneVision Handling: more than one image --> interleaved image mode or video mode. We do not use anyres
                pixel_values, image_hashes, image_sizes = [], [], []
                for img_data in image_data:
                    pixel_v, image_h, image_s = await self._process_single_image(
                        img_data, aspect_ratio, grid_pinpoints
                    )
                    pixel_values.append(pixel_v)
                    image_hashes.append(image_h)
                    image_sizes.append(image_s)

                if isinstance(pixel_values[0], np.ndarray):
                    pixel_values = np.stack(pixel_values, axis=0)
            else:
                # A single image
                pixel_values, image_hash, image_size = await self._process_single_image(
                    image_data[0], aspect_ratio, grid_pinpoints
                )
                image_hashes = [image_hash]
                image_sizes = [image_size]
        elif isinstance(image_data, str):
            # A single image
            pixel_values, image_hash, image_size = await self._process_single_image(
                image_data, aspect_ratio, grid_pinpoints
            )
            image_hashes = [image_hash]
            image_sizes = [image_size]
        else:
            raise ValueError(f"Invalid image data: {image_data}")

        return pixel_values, image_hashes, image_sizes

    async def _process_single_image(
        self, image_data: Union[bytes, str], aspect_ratio: str, grid_pinpoints: str
    ):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                _process_single_image_task,
                image_data,
                aspect_ratio,
                grid_pinpoints,
            )
        else:
            return _process_single_image_task(
                image_data, aspect_ratio, grid_pinpoints, self.processor
            )


global global_processor


def init_global_processor(server_args: ServerArgs):
    """Init the global processor for multi modal models."""
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )


def _process_single_image_task(
    image_data: Union[str, bytes],
    image_aspect_ratio: Optional[str] = None,
    image_grid_pinpoints: Optional[str] = None,
    processor=None,
):
    try:
        processor = processor or global_processor
        image, image_size = load_image(image_data)
        if image_size is not None:
            # It is a video with multiple images
            image_hash = hash(image_data)
            pixel_values = processor.image_processor(image)["pixel_values"]
            for _ in range(len(pixel_values)):
                pixel_values[_] = pixel_values[_].astype(np.float16)
            pixel_values = np.stack(pixel_values, axis=0)
            return pixel_values, image_hash, image_size
        else:
            # It is an image
            image_hash = hash(image_data)
            if image_aspect_ratio == "pad":
                image = expand2square(
                    image,
                    tuple(int(x * 255) for x in processor.image_processor.image_mean),
                )
                pixel_values = processor.image_processor(image.convert("RGB"))[
                    "pixel_values"
                ][0]
            elif image_aspect_ratio == "anyres" or (
                image_aspect_ratio is not None and "anyres_max" in image_aspect_ratio
            ):
                pixel_values = process_anyres_image(
                    image, processor.image_processor, image_grid_pinpoints
                )
            else:
                pixel_values = processor.image_processor(image)["pixel_values"][0]

            if isinstance(pixel_values, np.ndarray):
                pixel_values = pixel_values.astype(np.float16)

            return pixel_values, image_hash, image.size
    except Exception:
        logger.error("Exception in TokenizerManager:\n" + get_exception_traceback())
