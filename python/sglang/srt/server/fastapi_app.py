import dataclasses
import logging
from http import HTTPStatus
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    ReleaseGPUOccupationReqInput,
    ResumeGPUOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.metrics.func_timer import time_func_latency
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
)
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server.engine import Engine
from sglang.srt.server.utils import create_error_response
from sglang.version import __version__

logger = logging.getLogger(__name__)

# Fast API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


##### Global States #####


@dataclasses.dataclass
class _GlobalState:
    engine: Engine


_global_state: Optional[_GlobalState] = None


def setup_global_state(engine: Engine):
    global _global_state
    _global_state = _GlobalState(engine=engine)


##### Native API endpoints #####


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """Check the health of the inference server by generating one token."""

    sampling_params = {"max_new_tokens": 1, "temperature": 0.7}

    if _global_state.engine.entrypoint.is_generation:
        gri = GenerateReqInput(input_ids=[0], sampling_params=sampling_params)
    else:
        gri = EmbeddingReqInput(input_ids=[0], sampling_params=sampling_params)

    try:
        async for _ in _global_state.engine.entrypoint.generate_request(gri, request):
            break
        return Response(status_code=200)
    except Exception as e:
        logger.exception(e)
        return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": _global_state.engine.entrypoint.model_path,
        "tokenizer_path": _global_state.engine.entrypoint.server_args.tokenizer_path,
        "is_generation": _global_state.engine.entrypoint.is_generation,
    }
    return result


@app.get("/get_server_info")
async def get_server_info():
    return {
        **dataclasses.asdict(
            _global_state.engine.entrypoint.server_args
        ),  # server args
        **_global_state.engine.scheduler_info,
        "version": __version__,
    }


@app.post("/flush_cache")
async def flush_cache():
    """Flush the radix cache."""
    _global_state.engine.entrypoint.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


@app.get("/start_profile")
@app.post("/start_profile")
async def start_profile_async():
    """Start profiling."""
    _global_state.engine.entrypoint.start_profile()
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.get("/stop_profile")
@app.post("/stop_profile")
async def stop_profile_async():
    """Stop profiling."""
    _global_state.engine.entrypoint.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.post("/update_weights_from_disk")
@time_func_latency
async def update_weights_from_disk(obj: UpdateWeightFromDiskReqInput, request: Request):
    """Update the weights from disk in-place without re-launching the server."""
    success, message = await _global_state.engine.entrypoint.update_weights_from_disk(
        obj, request
    )
    content = {"success": success, "message": message}
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
    success, message = await _global_state.engine.entrypoint.init_weights_update_group(
        obj, request
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(
    obj: UpdateWeightsFromDistributedReqInput, request: Request
):
    """Update model parameter from distributed online."""
    success, message = (
        await _global_state.engine.entrypoint.update_weights_from_distributed(
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
        ret = await _global_state.engine.entrypoint.get_weights_by_name(obj, request)
        if ret is None:
            return create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return create_error_response(e)


@app.api_route("/release_gpu_occupation", methods=["GET", "POST"])
async def release_gpu_occupation(obj: ReleaseGPUOccupationReqInput, request: Request):
    """Release GPU occupation temporarily"""
    try:
        await _global_state.engine.entrypoint.release_gpu_occupation(obj, request)
    except Exception as e:
        return create_error_response(e)


@app.api_route("/resume_gpu_occupation", methods=["GET", "POST"])
async def resume_gpu_occupation(obj: ResumeGPUOccupationReqInput, request: Request):
    """Resume GPU occupation"""
    try:
        await _global_state.engine.entrypoint.resume_gpu_occupation(obj, request)
    except Exception as e:
        return create_error_response(e)


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput, request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.engine.entrypoint.open_session(obj, request)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session"""
    try:
        await _global_state.engine.entrypoint.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return create_error_response(e)


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route("/generate", methods=["POST", "PUT"])
@time_func_latency
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    return await _global_state.engine._generate_raw(obj, request)


@app.api_route("/encode", methods=["POST", "PUT"])
@time_func_latency
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    return await _global_state.engine._encode_raw(obj, request)


@app.api_route("/classify", methods=["POST", "PUT"])
@time_func_latency
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.engine.entrypoint.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return create_error_response(e)


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions")
@time_func_latency
async def openai_v1_completions(raw_request: Request):
    return await v1_completions(_global_state.engine.entrypoint, raw_request)


@app.post("/v1/chat/completions")
@time_func_latency
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(_global_state.engine.entrypoint, raw_request)


@app.post("/v1/embeddings", response_class=ORJSONResponse)
@time_func_latency
async def openai_v1_embeddings(raw_request: Request):
    response = await v1_embeddings(_global_state.engine.entrypoint, raw_request)
    return response


@app.get("/v1/models", response_class=ORJSONResponse)
def available_models():
    """Show available models."""
    served_model_names = [_global_state.engine.entrypoint.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(ModelCard(id=served_model_name, root=served_model_name))
    return ModelList(data=model_cards)


@app.post("/v1/files")
async def openai_v1_files(file: UploadFile = File(...), purpose: str = Form("batch")):
    return await v1_files_create(
        file,
        purpose,
        _global_state.engine.entrypoint.server_args.file_storage_pth,
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/delete
    return await v1_delete_file(file_id)


@app.post("/v1/batches")
async def openai_v1_batches(raw_request: Request):
    return await v1_batches(_global_state.engine.entrypoint, raw_request)


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batches(batch_id: str):
    # https://platform.openai.com/docs/api-reference/batch/cancel
    return await v1_cancel_batch(_global_state.engine.entrypoint, batch_id)


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
