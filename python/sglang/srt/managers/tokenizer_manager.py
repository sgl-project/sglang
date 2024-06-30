"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import concurrent.futures
import dataclasses
import logging
import multiprocessing as mp
import os
from typing import Dict, List

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
        self.send_to_router.connect(f"tcp://127.0.0.1:{port_args.router_port}")

        self.model_path = server_args.model_path
        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_overide_args=model_overide_args,
        )
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
            rid = obj.rid

            if obj.input_ids is None:
                input_ids = self.tokenizer.encode(obj.text)
            else:
                input_ids = obj.input_ids

            if len(input_ids) >= self.context_len:
                raise ValueError(
                    f"The input ({len(input_ids)} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens)."
                )

            sampling_params = SamplingParams(**obj.sampling_params)
            if sampling_params.max_new_tokens != 0:
                sampling_params.normalize(self.tokenizer)
                sampling_params.verify()

            if isinstance(obj.image_data, list) and len(obj.image_data) > 0:
                pixel_values, image_hash, image_size = await self.get_pixel_values(
                    obj.image_data[0]
                )
            elif isinstance(obj.image_data, str):
                pixel_values, image_hash, image_size = await self.get_pixel_values(
                    obj.image_data
                )
            else:
                pixel_values, image_hash, image_size = None, None, None
            tokenized_obj = TokenizedGenerateReqInput(
                rid=rid,
                input_text=obj.text,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_hash=image_hash,
                image_size=image_size,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                top_logprobs_num=obj.top_logprobs_num,
                stream=obj.stream,
            )
            self.send_to_router.send_pyobj(tokenized_obj)

            event = asyncio.Event()
            state = ReqState([], False, event)
            self.rid_to_state[rid] = state

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
        else:
            if obj.stream:
                raise ValueError("Do not support stream for batch mode.")

            if obj.input_ids is None:
                bs = len(obj.text)
            else:
                bs = len(obj.input_ids)

            for i in range(bs):
                rid = obj.rid[i]

                if obj.input_ids is None:
                    input_text = obj.text[i]
                    input_ids = self.tokenizer.encode(obj.text[i])
                else:
                    input_text = None
                    input_ids = obj.input_ids[i]

                sampling_params = SamplingParams(**obj.sampling_params[i])
                if sampling_params.max_new_tokens != 0:
                    sampling_params.normalize(self.tokenizer)
                    sampling_params.verify()
                if obj.image_data[i] is None:
                    pixel_values, image_hash, image_size = None, None, None
                else:
                    pixel_values, image_hash, image_size = await self.get_pixel_values(
                        obj.image_data[i]
                    )
                tokenized_obj = TokenizedGenerateReqInput(
                    rid=rid,
                    input_text=input_text,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_hash=image_hash,
                    image_size=image_size,
                    sampling_params=sampling_params,
                    return_logprob=obj.return_logprob[i],
                    logprob_start_len=obj.logprob_start_len[i],
                    top_logprobs_num=obj.top_logprobs_num[i],
                    stream=obj.stream,
                )
                self.send_to_router.send_pyobj(tokenized_obj)

                event = asyncio.Event()
                state = ReqState([], False, event)
                self.rid_to_state[rid] = state

            output_list = []
            for i in range(bs):
                rid = obj.rid[i]
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
                        obj.return_logprob[i],
                        obj.top_logprobs_num[i],
                        obj.return_text_in_logprobs,
                    )
                )
                assert state.finished
                del self.rid_to_state[rid]

            yield output_list

    def flush_cache(self):
        req = FlushCacheReq()
        self.send_to_router.send_pyobj(req)

    def abort_request(self, rid):
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
                    "text": recv_obj.output_str[i],
                    "meta_info": recv_obj.meta_info[i],
                }
                state.out_list.append(out_dict)
                state.finished = recv_obj.finished_reason[i] is not None
                state.event.set()

    def convert_logprob_style(
        self, ret, return_logprob, top_logprobs_num, return_text_in_logprobs
    ):
        if return_logprob:
            ret["meta_info"]["prefill_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["prefill_token_logprobs"], return_text_in_logprobs
            )
            ret["meta_info"]["decode_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["decode_token_logprobs"], return_text_in_logprobs
            )
        if top_logprobs_num > 0:
            ret["meta_info"][
                "prefill_top_logprobs"
            ] = self.detokenize_top_logprobs_tokens(
                ret["meta_info"]["prefill_top_logprobs"], return_text_in_logprobs
            )
            ret["meta_info"][
                "decode_top_logprobs"
            ] = self.detokenize_top_logprobs_tokens(
                ret["meta_info"]["decode_top_logprobs"], return_text_in_logprobs
            )
        return ret

    def detokenize_logprob_tokens(self, token_logprobs, decode_to_text):
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

        token_ids = [tid for _, tid in token_logprobs]
        token_texts = self.tokenizer.batch_decode(token_ids)
        return [
            (logprob, token_id, token_text)
            for (logprob, token_id), token_text, in zip(token_logprobs, token_texts)
        ]

    def detokenize_top_logprobs_tokens(self, top_logprobs, decode_to_text):
        for i, t in enumerate(top_logprobs):
            if t:
                top_logprobs[i] = self.detokenize_logprob_tokens(t, decode_to_text)
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
        if image_size != None:
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
