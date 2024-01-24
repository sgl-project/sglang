import asyncio
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
from typing import List

import numpy as np
import transformers
import uvloop
import zmq
import zmq.asyncio
from sglang.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_processor,
    get_tokenizer,
)
from sglang.srt.managers.io_struct import (
    BatchStrOut,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback, is_multimodal_model, load_image

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclasses.dataclass
class ReqState:
    out_list: List
    finished: bool
    event: asyncio.Event
    lock: asyncio.Lock


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
        image = load_image(image_data)
        image_hash = hash(image_data)
        if image_aspect_ratio == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_processor.image_mean)
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


class TokenizerManager:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = context.socket(zmq.PULL)
        self.recv_from_detokenizer.bind(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{port_args.router_port}")

        self.model_path = server_args.model_path
        self.hf_config = get_config(
            self.model_path, trust_remote_code=server_args.trust_remote_code
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
        self.rid_to_state = {}  # Dict[str -> ReqState]

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

    async def generate_request(self, obj: GenerateReqInput):
        if self.to_create_loop:
            await self.create_handle_loop()

        is_single = isinstance(obj.text, str)

        if is_single:
            rid = obj.rid
            input_ids = self.tokenizer.encode(obj.text)
            sampling_params = SamplingParams(**obj.sampling_params)
            if sampling_params.max_new_tokens != 0:
                sampling_params.normalize(self.tokenizer)
                sampling_params.verify()
            if obj.image_data is None:
                pixel_values, image_hash, image_size = None, None, None
            else:
                pixel_values, image_hash, image_size = await self.get_pixel_values(
                    obj.image_data
                )
            tokenized_obj = TokenizedGenerateReqInput(
                rid=rid,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_hash=image_hash,
                image_size=image_size,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                stream=obj.stream,
            )
            self.send_to_router.send_pyobj(tokenized_obj)

            lock = asyncio.Lock()
            event = asyncio.Event()
            state = ReqState([], False, event, lock)
            self.rid_to_state[rid] = state

            while True:
                await event.wait()
                yield state.out_list[-1]
                state.out_list = []
                if state.finished:
                    del self.rid_to_state[rid]
                    break
                event.clear()
        else:
            assert obj.stream is False
            bs = len(obj.text)
            for i in range(bs):
                rid = obj.rid[i]
                input_ids = self.tokenizer.encode(obj.text[i])
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
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_hash=image_hash,
                    image_size=image_size,
                    sampling_params=sampling_params,
                    return_logprob=obj.return_logprob[i],
                    logprob_start_len=obj.logprob_start_len[i],
                    stream=obj.stream,
                )
                self.send_to_router.send_pyobj(tokenized_obj)

                lock = asyncio.Lock()
                event = asyncio.Event()
                state = ReqState([], False, event, lock)
                self.rid_to_state[rid] = state

            output_list = []
            for i in range(bs):
                rid = obj.rid[i]
                state = self.rid_to_state[rid]
                await state.event.wait()
                output_list.append(state.out_list[-1])
                assert state.finished
                del self.rid_to_state[rid]

            yield output_list

    async def create_handle_loop(self):
        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        loop.create_task(self.handle_loop())

    async def handle_loop(self):
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()

            if isinstance(recv_obj, BatchStrOut):
                for i, rid in enumerate(recv_obj.rids):
                    recv_obj.meta_info[i]["id"] = rid
                    out_dict = {
                        "text": recv_obj.output_str[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                    state = self.rid_to_state[rid]
                    state.out_list.append(out_dict)
                    state.finished = recv_obj.finished[i]
                    state.event.set()
            else:
                raise ValueError(f"Invalid object: {recv_obj}")
