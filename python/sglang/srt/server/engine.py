import asyncio
import atexit
import dataclasses
import json
import logging
import os
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

import orjson
import torch
from fastapi import Request
from starlette.responses import StreamingResponse

from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput, ReleaseGPUOccupationReqInput, ResumeGPUOccupationReqInput,
)
from sglang.srt.orchestration import std
from sglang.srt.server.engine_base import EngineBase
from sglang.srt.server.utils import create_error_response
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.version import __version__

logger = logging.getLogger(__name__)


class Engine(EngineBase):
    """
    SRT Engine without an HTTP server layer.

    This class provides a direct inference engine without the need for an HTTP server. It is designed for use cases where
    launching the HTTP server adds unnecessary complexity or overhead,
    """

    def __init__(self, log_level: str = "error", *args, server_args=None, **kwargs):
        """See the arguments in server_args.py::ServerArgs"""

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        server_args = server_args or ServerArgs(*args, log_level=log_level, **kwargs)
        entrypoint, scheduler_info = std.launcher.launch(server_args=server_args)
        self.entrypoint = entrypoint
        self.scheduler_info = scheduler_info

    def _generate_impl(self, obj: GenerateReqInput):
        # get the current event loop
        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(self._generate_raw(obj, None))

        if obj.stream is True:

            def generator_wrapper():
                offset = 0
                loop = asyncio.get_event_loop()
                generator = ret.body_iterator
                while True:
                    chunk = loop.run_until_complete(generator.__anext__())
                    if chunk.startswith(_STREAM_END_SYMBOL):
                        break
                    else:
                        data = json.loads(chunk[len(_STREAM_CHUNK_START_SYMBOL) :])
                        data["text"] = data["text"][offset:]
                        offset += len(data["text"])
                        yield data

            # we cannot yield in the scope of generate() because python does not allow yield + return in the same function
            # however, it allows to wrap the generator as a subfunction and return
            return generator_wrapper()
        else:
            return ret

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Dict] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        stream: bool = False,
    ):
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            lora_path=lora_path,
            stream=stream,
        )

        ret = await self._generate_raw(obj, None)

        if stream is True:
            generator = ret.body_iterator

            async def generator_wrapper():
                offset = 0
                while True:
                    chunk = await generator.__anext__()
                    if chunk.startswith(_STREAM_END_SYMBOL):
                        break
                    else:
                        data = json.loads(chunk[len(_STREAM_CHUNK_START_SYMBOL) :])
                        data["text"] = data["text"][offset:]
                        offset += len(data["text"])
                        yield data

            return generator_wrapper()
        else:
            return ret

    async def _generate_raw(self, obj: GenerateReqInput, request: Request):
        if obj.stream:

            async def stream_results() -> AsyncIterator[bytes]:
                try:
                    async for out in self.entrypoint.generate_request(obj, request):
                        yield b"data: " + orjson.dumps(
                            out, option=orjson.OPT_NON_STR_KEYS
                        ) + b"\n\n"
                except ValueError as e:
                    out = {"error": {"message": str(e)}}
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                stream_results(),
                media_type="text/event-stream",
                background=self.entrypoint.create_abort_task(obj),
            )
        else:
            try:
                ret = await self.entrypoint.generate_request(obj, request).__anext__()
                return ret
            except ValueError as e:
                logger.error(f"Error: {e}")
                # TODO: maybe we should not return such ORJSONResponse for engine API,
                # but for backward compatibility we do so
                return create_error_response(e)

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        obj = EmbeddingReqInput(text=prompt)

        # get the current event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._encode_raw(obj, None))

    async def _encode_raw(self, obj: EmbeddingReqInput, request: Request):
        try:
            ret = await self.entrypoint.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            # TODO: maybe we should not return such ORJSONResponse for engine API,
            # but for backward compatibility we do so
            return create_error_response(e)

    def shutdown(self):
        kill_process_tree(os.getpid(), include_parent=False)

    def get_tokenizer(self):
        if self.entrypoint is None:
            raise ReferenceError("Tokenizer Manager is not initialized.")
        else:
            return self.entrypoint.tokenizer

    def start_profile(self):
        self.entrypoint.start_profile()

    def stop_profile(self):
        self.entrypoint.stop_profile()

    def get_server_info(self):
        return {
            **dataclasses.asdict(self.entrypoint.server_args),  # server args
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
            self.entrypoint.init_weights_update_group(obj, None)
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
            self.entrypoint.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
    ):
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=MultiprocessingSerializer.serialize(named_tensors),
            load_format=load_format,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.entrypoint.update_weights_from_tensor(obj, None)
        )

    def get_weights_by_name(self, name, truncate_size=100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.entrypoint.get_weights_by_name(obj, None))

    def release_gpu_occupation(self):
        """Release GPU occupation temporarily"""
        obj = ReleaseGPUOccupationReqInput()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.entrypoint.release_gpu_occupation(obj, None))

    def resume_gpu_occupation(self):
        """Resume GPU occupation"""
        obj = ResumeGPUOccupationReqInput()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.entrypoint.resume_gpu_occupation(obj, None))

_STREAM_END_SYMBOL = b"data: [DONE]"
_STREAM_CHUNK_START_SYMBOL = b"data:"
