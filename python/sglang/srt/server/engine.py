import asyncio
import atexit
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from http import HTTPStatus
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

import aiohttp
import orjson
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.metrics.func_timer import enable_func_timer, time_func_latency
from sglang.srt.openai_api.adapter import (
    load_chat_template_for_openai_api,
    v1_batches,
    v1_cancel_batch,
    v1_chat_completions,
    v1_completions,
    v1_delete_file,
    v1_embeddings,
    v1_files_create,
    v1_retrieve_batch,
    v1_retrieve_file,
    v1_retrieve_file_content,
)
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    add_api_key_middleware,
    add_prometheus_middleware,
    assert_pkg_version,
    configure_logger,
    delete_directory,
    is_port_available,
    kill_process_tree,
    maybe_set_triton_cache_manager,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
    set_uvicorn_logging_configs,
)
from sglang.utils import get_exception_traceback
from sglang.version import __version__

logger = logging.getLogger(__name__)



class Engine:
    """
    SRT Engine without an HTTP server layer.

    This class provides a direct inference engine without the need for an HTTP server. It is designed for use cases where
    launching the HTTP server adds unnecessary complexity or overhead,
    """

    def __init__(self, log_level: str = "error", *args, **kwargs):
        """See the arguments in server_args.py::ServerArgs"""

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        launch_engine(server_args=server_args)

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
        ret = loop.run_until_complete(generate_request(obj, None))

        if stream is True:

            def generator_wrapper():
                offset = 0
                loop = asyncio.get_event_loop()
                generator = ret.body_iterator
                while True:
                    chunk = loop.run_until_complete(generator.__anext__())

                    if chunk.startswith(STREAM_END_SYMBOL):
                        break
                    else:
                        data = json.loads(chunk[len(STREAM_CHUNK_START_SYMBOL) :])
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

        ret = await generate_request(obj, None)

        if stream is True:
            generator = ret.body_iterator

            async def generator_wrapper():
                offset = 0

                while True:
                    chunk = await generator.__anext__()

                    if chunk.startswith(STREAM_END_SYMBOL):
                        break
                    else:
                        data = json.loads(chunk[len(STREAM_CHUNK_START_SYMBOL) :])
                        data["text"] = data["text"][offset:]
                        offset += len(data["text"])
                        yield data

            return generator_wrapper()
        else:
            return ret

    def shutdown(self):
        kill_process_tree(os.getpid(), include_parent=False)

    def get_tokenizer(self):
        global tokenizer_manager

        if tokenizer_manager is None:
            raise ReferenceError("Tokenizer Manager is not initialized.")
        else:
            return tokenizer_manager.tokenizer

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        obj = EmbeddingReqInput(text=prompt)

        # get the current event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(encode_request(obj, None))

    def start_profile(self):
        tokenizer_manager.start_profile()

    def stop_profile(self):
        tokenizer_manager.stop_profile()

    def get_server_info(self):
        return {
            **dataclasses.asdict(tokenizer_manager.server_args),  # server args
            **scheduler_info,
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
            tokenizer_manager.init_weights_update_group(obj, None)
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
            tokenizer_manager.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(self, named_tensors: List[Tuple[str, torch.Tensor]]):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=MultiprocessingSerializer.serialize(named_tensors)
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def get_weights_by_name(self, name, truncate_size=100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(tokenizer_manager.get_weights_by_name(obj, None))

    def release_memory_occupation(self):
        """Release GPU occupation temporarily"""
        obj = ReleaseMemoryOccupationReqInput()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(tokenizer_manager.release_memory_occupation(obj, None))

    def resume_memory_occupation(self):
        """Resume GPU occupation"""
        obj = ResumeMemoryOccupationReqInput()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(tokenizer_manager.resume_memory_occupation(obj, None))
