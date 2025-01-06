import asyncio
import json
import logging
from abc import ABC
from typing import Dict, List, Optional, Union, AsyncIterator

import orjson
from fastapi import Request
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
)
from sglang.srt.server.utils import create_error_response
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class EngineBase(ABC):
    """Common API and logic for both Engine and EngineFragment"""

    # TODO refactor these later
    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
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

        # get the current event loop
        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(self._generate_raw(obj, None))

        if stream is True:
            def generator_wrapper():
                offset = 0
                loop = asyncio.get_event_loop()
                generator = ret.body_iterator
                while True:
                    chunk = loop.run_until_complete(generator.__anext__())
                    if chunk.startswith(_STREAM_END_SYMBOL):
                        break
                    else:
                        data = json.loads(chunk[len(_STREAM_CHUNK_START_SYMBOL):])
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
                        data = json.loads(chunk[len(_STREAM_CHUNK_START_SYMBOL):])
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
                    async for out in self._generate_request_impl(obj, request):
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
                background=self._create_abort_task_impl(obj),
            )
        else:
            try:
                ret = await self._generate_request_impl(obj, request).__anext__()
                return ret
            except ValueError as e:
                logger.error(f"Error: {e}")
                # TODO: maybe we should not return such ORJSONResponse for engine API,
                # but for backward compatibility we do so
                return create_error_response(e)

    async def _generate_request_impl(self, obj: GenerateReqInput, request: Request):
        raise NotImplementedError

    def _create_abort_task_impl(self, obj: GenerateReqInput):
        raise NotImplementedError


# TODO refactor
_STREAM_END_SYMBOL = b"data: [DONE]"
_STREAM_CHUNK_START_SYMBOL = b"data:"
