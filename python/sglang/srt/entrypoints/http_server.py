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

This file implements HTTP APIs for the inference engine via fastapi.
"""

import asyncio
import dataclasses
import json
import logging
import multiprocessing as multiprocessing
import os
import tempfile
import threading
import time
from http import HTTPStatus
from typing import AsyncIterator, Callable, Dict, Optional
from multiprocessing import shared_memory, Lock, Value, Manager
import ctypes
import sys
# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

from contextlib import asynccontextmanager

import numpy as np
import orjson
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.disaggregation.utils import (
    FakeBootstrapHost,
    register_disaggregation_server,
)
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    ParseFunctionCallReq,
    ProfileReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SeparateReasoningReqInput,
    SetInternalStateReq,
    SlowDownReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
    VertexGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.metrics.func_timer import enable_func_timer
from sglang.srt.openai_api.adapter import (
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
    v1_score,
)
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    add_prometheus_middleware,
    delete_directory,
    get_bool_env_var,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sglang.srt.warmup import execute_warmups
from sglang.utils import get_exception_traceback
from sglang.version import __version__
from sglang.srt.openai_api.adapter import load_chat_template_for_openai_api
from sglang.srt.code_completion_parser import load_completion_template_for_openai_api

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state

def serialize_port_args(port_args: PortArgs) -> dict:
    """将 PortArgs 序列化为可共享的字典"""
    return {
        "tokenizer_ipc_name": port_args.tokenizer_ipc_name,
        "scheduler_input_ipc_name": port_args.scheduler_input_ipc_name,
        "detokenizer_ipc_name": port_args.detokenizer_ipc_name,
        "nccl_port": port_args.nccl_port,
        "rpc_ipc_name": port_args.rpc_ipc_name,
    }

def deserialize_port_args(data: dict) -> PortArgs:
    """从共享的字典反序列化为 PortArgs"""
    return PortArgs(**data)

def serialize_server_args(server_args: ServerArgs) -> dict:
    """将 ServerArgs 序列化为可共享的字典"""
    return dataclasses.asdict(server_args)

def deserialize_server_args(data: dict) -> ServerArgs:
    """从共享的字典反序列化为 ServerArgs"""
    return ServerArgs(**data)

def serialize_scheduler_info(scheduler_info: Dict) -> dict:
    """将 scheduler_info 序列化为可共享的字典"""
    return scheduler_info

def deserialize_scheduler_info(data: dict) -> Dict:
    """从共享的字典反序列化为 scheduler_info"""
    return data

def write_to_shared_memory(data: dict, name: str) -> shared_memory.SharedMemory:
    """将数据写入共享内存"""
    serialized = json.dumps(data).encode('utf-8')
    size = len(serialized)
    try:
        # 尝试打开已存在的共享内存
        shm = shared_memory.SharedMemory(name=name)
        # 如果大小不够，关闭并重新创建
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileNotFoundError:
        # 如果不存在，创建新的共享内存
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    
    shm.buf[:size] = serialized
    return shm

def read_from_shared_memory(name: str) -> dict:
    """从共享内存读取数据"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = json.loads(bytes(shm.buf).decode('utf-8'))
        shm.close()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found")

def get_main_process_id() -> int:
    """获取主进程的 ID"""
    return multiprocessing.current_process()._parent_pid

def serialize_tokenizer_mapping(mapping: Dict[int, str]) -> dict:
    """将 tokenizer_ipc_name 映射序列化为可共享的字典"""
    return mapping

def deserialize_tokenizer_mapping(data: dict) -> Dict[int, str]:
    """从共享的字典反序列化为 tokenizer_ipc_name 映射"""
    return data

def create_shared_lock() -> Lock:
    """创建一个共享的锁"""
    return Lock()

def update_tokenizer_mapping(worker_id: int, ipc_name: str, shm_name: str, lock_value: Value) -> tuple[shared_memory.SharedMemory, int]:
    """更新或创建 tokenizer_ipc_name 映射"""
    with lock_value.get_lock():  # 使用共享锁确保进程间同步
        try:
            # 尝试读取现有的映射
            existing_data = read_from_shared_memory(shm_name)
            mapping = deserialize_tokenizer_mapping(existing_data)
        except FileNotFoundError:
            # 如果不存在，创建新的映射
            mapping = {}
        
        # 更新映射
        mapping[worker_id] = ipc_name
        print(f"worker_id:{worker_id}, mapping:{mapping}")
        
        # 写入共享内存
        shm = write_to_shared_memory(
            serialize_tokenizer_mapping(mapping),
            shm_name,
        )
        return shm, len(mapping)

@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    server_args = getattr(fast_api_app, "server_args", None)
    if server_args is not None:
        if server_args.warmups is not None:
            await execute_warmups(
                server_args.warmups.split(","), _global_state.tokenizer_manager
            )
            logger.info("Warmup ended")

        warmup_thread = getattr(fast_api_app, "warmup_thread", None)
        if warmup_thread is not None:
            warmup_thread.start()
    else:
        pid = os.getpid()
        main_pid = get_main_process_id()
        print(f"current worker_id: {pid}, main processID: {main_pid}")
        
        # 从共享内存读取 port_args、server_args 和 scheduler_info
        max_retries = 5
        retry_delay = 1  # 秒
        
        for retry in range(max_retries):
            try:
                port_args_data = read_from_shared_memory(f"port_args_{main_pid}")
                server_args_data = read_from_shared_memory(f"server_args_{main_pid}")
                scheduler_info_data = read_from_shared_memory(f"scheduler_info_{main_pid}")
                lock_data = read_from_shared_memory(f"mapping_lock_{main_pid}")
                break
            except FileNotFoundError as e:
                if retry < max_retries - 1:
                    print(f"等待共享内存就绪，重试 {retry + 1}/{max_retries}")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"无法访问共享内存: {e}")
        
        lock_value = Value(ctypes.c_int, lock_data["lock_value"])
        
        port_args = deserialize_port_args(port_args_data)
        server_args = deserialize_server_args(server_args_data)
        scheduler_info = deserialize_scheduler_info(scheduler_info_data)
        
        port_args.tokenizer_ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        
        # 使用共享锁更新 tokenizer_ipc_name 映射
        tokenizer_mapping_shm, mapping_len = update_tokenizer_mapping(
            pid,
            port_args.tokenizer_ipc_name,
            f"tokenizer_mapping_{main_pid}",
            lock_value
        )
        
        # Launch tokenizer process
        tokenizer_manager = TokenizerManager(server_args, port_args)
        if server_args.chat_template:
            load_chat_template_for_openai_api(
                tokenizer_manager, server_args.chat_template, server_args.model_path
            )
        if server_args.completion_template:
            load_completion_template_for_openai_api(server_args.completion_template)
        tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
        set_global_state(
            _GlobalState(
                tokenizer_manager=tokenizer_manager,
                scheduler_info=scheduler_info,
            )
        )
        
        print(f"mapping_length={mapping_len},worker_num={server_args.worker_num}")
        # 检查是否所有 worker 都已注册
        
        if server_args.warmups is not None:
            logger.info("Warmup started")
            await execute_warmups(
                server_args.warmups.split(","), _global_state.tokenizer_manager
            )
            logger.info("Warmup ended")
        # wait detokenizer manager register zmq
        time.sleep(12)
        logger.info("warmup_thread start")
        print(f"warmup_thread start p")
        
        # 为每个 worker 创建自己的 warmup 线程
        warmup_thread = threading.Thread(
            target=_wait_and_warmup,
            args=(
                server_args,
                None,  # pipe_finish_writer 在 worker 中不需要
                _global_state.tokenizer_manager.image_token_id,
                None,  # launch_callback 在 worker 中不需要
            ),
        )
        print(f"warmup_thread start warmup_thread={warmup_thread}")
        warmup_thread.start()
        
        print(f"worker {pid} started")
    try:
        yield
    finally:
        if server_args.worker_num > 1:
            print(f"worker {pid} ending")
            # 清理共享内存
            try:
                if "warmup_thread" in locals() and warmup_thread.is_alive():
                    warmup_thread.join()
                with lock_value.get_lock():  # 使用共享锁确保清理时的进程间同步
                    tokenizer_mapping_shm.close()
                    # 检查是否还有其他 worker 在使用这个映射
                    try:
                        mapping = deserialize_tokenizer_mapping(read_from_shared_memory(f"tokenizer_mapping_{main_pid}"))
                        if len(mapping) <= 1:  # 如果只有当前 worker 在使用
                            tokenizer_mapping_shm.unlink()
                    except FileNotFoundError:
                        pass
            except Exception as e:
                print(f"清理共享内存时出错: {e}")
            print(f"worker {pid} ended")


# Fast API
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))


##### Native API endpoints #####


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """Check the health of the inference server by generating one token."""

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    rid = f"HEALTH_CHECK_{time.time()}"

    if _global_state.tokenizer_manager.is_image_gen:
        raise NotImplementedError()
    elif _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    tic = time.perf_counter()
    task = asyncio.create_task(gen())
    while time.perf_counter() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.health_check_failed = False
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.health_check_failed = True
    return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,
        "is_generation": _global_state.tokenizer_manager.is_generation,
    }
    return result


@app.get("/get_server_info")
async def get_server_info():
    internal_states = await _global_state.tokenizer_manager.get_internal_state()
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        **_global_state.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
    }


@app.get("/get_load")
async def get_load():
    return await _global_state.tokenizer_manager.get_load()


@app.api_route("/set_internal_state", methods=["POST", "PUT"])
async def set_internal_state(obj: SetInternalStateReq, request: Request):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route("/generate", methods=["POST", "PUT"])
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return ret
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)


@app.api_route("/generate_from_file", methods=["POST"])
async def generate_from_file_request(file: UploadFile, request: Request):
    """Handle a generate request, this is purely to work with input_embeds."""
    content = await file.read()
    input_embeds = json.loads(content.decode("utf-8"))

    obj = GenerateReqInput(
        input_embeds=input_embeds,
        sampling_params={
            "repetition_penalty": 1.2,
            "temperature": 0.2,
            "max_new_tokens": 512,
        },
    )

    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        logger.error(f"Error: {e}")
        return _create_error_response(e)


@app.api_route("/encode", methods=["POST", "PUT"])
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/classify", methods=["POST", "PUT"])
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache():
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    """Start profiling."""
    if obj is None:
        obj = ProfileReqInput()

    await _global_state.tokenizer_manager.start_profile(
        output_dir=obj.output_dir,
        num_steps=obj.num_steps,
        activities=obj.activities,
        with_stack=obj.with_stack,
        record_shapes=obj.record_shapes,
        profile_by_stage=obj.profile_by_stage,
    )
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
async def start_expert_distribution_record_async():
    """Start recording the expert distribution. Clear the previous record if any."""
    await _global_state.tokenizer_manager.start_expert_distribution_record()
    return Response(
        content="Start recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
async def stop_expert_distribution_record_async():
    """Stop recording the expert distribution."""
    await _global_state.tokenizer_manager.stop_expert_distribution_record()
    return Response(
        content="Stop recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
async def dump_expert_distribution_record_async():
    """Dump expert distribution record."""
    await _global_state.tokenizer_manager.dump_expert_distribution_record()
    return Response(
        content="Dump expert distribution record.\n",
        status_code=200,
    )


@app.post("/update_weights_from_disk")
async def update_weights_from_disk(obj: UpdateWeightFromDiskReqInput, request: Request):
    """Update the weights from disk inplace without re-launching the server."""
    success, message, num_paused_requests = (
        await _global_state.tokenizer_manager.update_weights_from_disk(obj, request)
    )
    content = {
        "success": success,
        "message": message,
        "num_paused_requests": num_paused_requests,
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.post("/init_weights_update_group")
async def init_weights_update_group(
    obj: InitWeightsUpdateGroupReqInput, request: Request
):
    """Initialize the parameter update group."""
    success, message = await _global_state.tokenizer_manager.init_weights_update_group(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weights_from_tensor")
async def update_weights_from_tensor(
    obj: UpdateWeightsFromTensorReqInput, request: Request
):
    """Update the weights from tensor inplace without re-launching the server.
    Notes:
    1. Ensure that the model is on the correct device (e.g., GPU) before calling this endpoint. If the model is moved to the CPU unexpectedly, it may cause performance issues or runtime errors.
    2. HTTP will transmit only the metadata of the tensor, while the tensor itself will be directly copied to the model.
    3. Any binary data in the named tensors should be base64 encoded.
    """

    success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
        obj, request
    )
    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(
    obj: UpdateWeightsFromDistributedReqInput, request: Request
):
    """Update model parameter from distributed online."""
    success, message = (
        await _global_state.tokenizer_manager.update_weights_from_distributed(
            obj, request
        )
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
async def get_weights_by_name(obj: GetWeightsByNameReqInput, request: Request):
    """Get model parameter by name."""
    try:
        ret = await _global_state.tokenizer_manager.get_weights_by_name(obj, request)
        if ret is None:
            return _create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
async def release_memory_occupation(
    obj: ReleaseMemoryOccupationReqInput, request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
async def resume_memory_occupation(
    obj: ResumeMemoryOccupationReqInput, request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/slow_down", methods=["GET", "POST"])
async def slow_down(obj: SlowDownReqInput, request: Request):
    """Slow down the system deliberately. Only for testing. Example scenario:
    when we want to test performance of D in large-scale PD disaggregation and have no enough nodes for P,
    we can use this to slow down D to let it have enough running sequences, and then disable slowdown
    to let it run in full batch size.
    """
    try:
        await _global_state.tokenizer_manager.slow_down(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput, request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj, request)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/configure_logging", methods=["GET", "POST"])
async def configure_logging(obj: ConfigureLoggingReq, request: Request):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)
    return Response(status_code=200)


@app.post("/abort_request")
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(rid=obj.rid)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.post("/parse_function_call")
async def parse_function_call_request(obj: ParseFunctionCallReq, request: Request):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = FunctionCallParser(tools=obj.tools, tool_call_parser=obj.tool_call_parser)

    # 2) Call the non-stream parsing method (non-stream)
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # Convert pydantic objects to dictionaries
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/separate_reasoning")
async def separate_reasoning_request(obj: SeparateReasoningReqInput, request: Request):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = ReasoningParser(model_type=obj.reasoning_parser)

    # 2) Call the non-stream parsing method (non-stream)
    reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "reasoning_text": reasoning_text,
        "text": normal_text,
    }

    return ORJSONResponse(content=response_data, status_code=200)


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    return await v1_completions(_global_state.tokenizer_manager, raw_request)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(_global_state.tokenizer_manager, raw_request)


@app.post("/v1/embeddings", response_class=ORJSONResponse)
async def openai_v1_embeddings(raw_request: Request):
    response = await v1_embeddings(_global_state.tokenizer_manager, raw_request)
    return response


@app.get("/v1/models", response_class=ORJSONResponse)
def available_models():
    """Show available models."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=_global_state.tokenizer_manager.model_config.context_len,
            )
        )
    return ModelList(data=model_cards)


@app.post("/v1/files")
async def openai_v1_files(file: UploadFile = File(...), purpose: str = Form("batch")):
    return await v1_files_create(
        file, purpose, _global_state.tokenizer_manager.server_args.file_storage_path
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/delete
    return await v1_delete_file(file_id)


@app.post("/v1/batches")
async def openai_v1_batches(raw_request: Request):
    return await v1_batches(_global_state.tokenizer_manager, raw_request)


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batches(batch_id: str):
    # https://platform.openai.com/docs/api-reference/batch/cancel
    return await v1_cancel_batch(_global_state.tokenizer_manager, batch_id)


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    return await v1_retrieve_batch(batch_id)


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve
    return await v1_retrieve_file(file_id)


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve-contents
    return await v1_retrieve_file_content(file_id)


## SageMaker API
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.post("/invocations")
async def sagemaker_chat_completions(raw_request: Request):
    return await v1_chat_completions(_global_state.tokenizer_manager, raw_request)


## Vertex AI API
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
async def vertex_generate(vertex_req: VertexGenerateReqInput, raw_request: Request):
    if not vertex_req.instances:
        return []
    inputs = {}
    for input_key in ("text", "input_ids", "input_embeds"):
        if vertex_req.instances[0].get(input_key):
            inputs[input_key] = [
                instance.get(input_key) for instance in vertex_req.instances
            ]
            break
    image_data = [
        instance.get("image_data")
        for instance in vertex_req.instances
        if instance.get("image_data") is not None
    ] or None
    req = GenerateReqInput(
        **inputs,
        image_data=image_data,
        **(vertex_req.parameters or {}),
    )
    ret = await generate_request(req, raw_request)
    if isinstance(ret, Response):
        return ret
    return ORJSONResponse({"predictions": ret})


@app.post("/v1/score")
async def v1_score_request(raw_request: Request):
    """Endpoint for the decoder-only scoring API. See Engine.score() for detailed documentation."""
    return await v1_score(_global_state.tokenizer_manager, raw_request)


def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    port_args = PortArgs.init_new(server_args)
    tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args, port_args=port_args)
    if server_args.worker_num > 1 :
        # get main process ID
        main_pid = get_main_process_id()
        current_pid = os.getpid()
        logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")
        
        # 创建共享的锁值
        lock_value = Value(ctypes.c_int, 0)
        
        # 将 port_args 和 server_args 写入共享内存
        port_args_shm = write_to_shared_memory(
            serialize_port_args(port_args), 
            f"port_args_{os.getpid()}"
        )
        server_args_shm = write_to_shared_memory(
            serialize_server_args(server_args), 
            f"server_args_{os.getpid()}"
        )
        
        # 将锁值的地址写入共享内存
        lock_shm = write_to_shared_memory(
            {"lock_value": lock_value.value},
            f"mapping_lock_{os.getpid()}"
        )
        
        
        
        # 将 scheduler_info 写入共享内存
        scheduler_info_shm = write_to_shared_memory(
            serialize_scheduler_info(scheduler_info),
            f"scheduler_info_{os.getpid()}"
        )
        port_args_shm.close()
        server_args_shm.close()
        scheduler_info_shm.close()
        lock_shm.close()
    else:
        warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            _global_state.tokenizer_manager.image_token_id,
            launch_callback,
        ),
    )
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            _global_state.tokenizer_manager.image_token_id,
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        if server_args.worker_num > 1:
            uvicorn.run(
            "sglang.srt.entrypoints.http_server:app",
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
            workers=server_args.worker_num,
        )
        else:
            uvicorn.run(
                "app",
                host=server_args.host,
                port=server_args.port,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=5,
                loop="uvloop",
            )
    finally:
        if server_args.worker_num > 1:
            port_args_shm.unlink()
            server_args_shm.unlink()
            scheduler_info_shm.unlink()
            lock_shm.unlink()
        else:
            warmup_thread.join()


def _wait_and_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
    image_token_text: str,
    launch_callback: Optional[Callable[[], None]] = None,
):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    model_info = res.json()

    # Send a warmup request
    request_name = "/generate" if model_info["is_generation"] else "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    else:
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    # Debug dumping
    if server_args.debug_tensor_dump_input_file:
        json_data.pop("text", None)
        json_data["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        json_data["sampling_params"]["max_new_tokens"] = 0

    try:
        if server_args.disaggregation_mode == "null":
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res}"
        else:
            logger.info(f"Start of prefill warmup ...")
            json_data = {
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "bootstrap_host": [FakeBootstrapHost] * server_args.dp_size,
                # This is a hack to ensure fake transfer is enabled during prefill warmup
                # ensure each dp rank has a unique bootstrap_room during prefill warmup
                "bootstrap_room": [
                    i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                    for i in range(server_args.dp_size)
                ],
                "input_ids": [[0, 1, 2, 3]] * server_args.dp_size,
            }
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=1800,  # because of deep gemm precache is very long if not precache.
            )
            logger.info(
                f"End of prefill warmup with status {res.status_code}, resp: {res.json()}"
            )

    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    # Debug print
    # logger.info(f"{res.json()=}")

    logger.info("The server is fired up and ready to roll!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)

    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if server_args.pdlb_url is not None:
        register_disaggregation_server(
            server_args.disaggregation_mode,
            server_args.port,
            server_args.disaggregation_bootstrap_port,
            server_args.pdlb_url,
        )

    if launch_callback is not None:
        launch_callback()
