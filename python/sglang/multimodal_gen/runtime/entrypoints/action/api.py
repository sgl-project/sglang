# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_generation_response,
    action_metadata,
    action_raw_response,
    infer_action,
    pack_msgpack,
    unpack_msgpack,
)
from sglang.multimodal_gen.runtime.entrypoints.action.ws_utils import (
    run_action_msgpack_ws,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    flatten_extra_params,
    save_image_to_path,
    temp_dir_if_disabled,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.utils.json_response import orjson_response

router = APIRouter(prefix="/v1/actions", tags=["actions"])

_ACTION_VIDEO_EXTENSIONS = {
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

_ACTION_FORM_NON_PARAMETER_FIELDS = {
    "extra_body",
    "extra_params",
    "id",
    "image_reference",
    "input_reference",
    "prompt",
    "reference_url",
    "request_id",
    "task",
    "video_reference",
}


def _wants_msgpack(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").lower()
    accept = request.headers.get("accept", "").lower()
    return "msgpack" in content_type or "msgpack" in accept


def _response_format(payload: dict) -> str:
    runtime = payload.get("runtime") or {}
    response_format = str(runtime.get("response_format", "envelope")).lower()
    if response_format not in ("envelope", "raw"):
        raise ValueError("runtime.response_format must be 'envelope' or 'raw'")
    return response_format


def _prefer_numpy_output(payload: dict) -> None:
    runtime = payload.setdefault("runtime", {})
    runtime.setdefault("output_format", "numpy")


def _parse_form_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not value.strip():
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def _parse_extra_params(value: Any, field_name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return flatten_extra_params(dict(parsed))


def _is_form_upload(value: Any) -> bool:
    return callable(getattr(value, "read", None)) and hasattr(value, "filename")


def _is_probably_video_upload(value: Any) -> bool:
    content_type = (getattr(value, "content_type", "") or "").lower()
    if content_type.startswith("video/"):
        return True
    filename = getattr(value, "filename", None)
    if not filename:
        return False
    filename = str(filename).split("?", 1)[0].split("#", 1)[0]
    return os.path.splitext(filename)[1].lower() in _ACTION_VIDEO_EXTENSIONS


async def _save_action_upload(
    upload: Any,
    request_id: str,
    uploads_dir: str,
    *,
    fallback_name: str,
) -> str:
    filename = getattr(upload, "filename", None) or fallback_name
    target_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
    return await save_image_to_path(upload, target_path)


async def _multipart_action_payload(request: Request, uploads_dir: str) -> dict:
    raw_form = await request.form()
    request_id = (
        raw_form.get("request_id") or raw_form.get("id") or generate_request_id()
    )

    parameters: dict[str, Any] = {}
    for key, value in raw_form.multi_items():
        if key in _ACTION_FORM_NON_PARAMETER_FIELDS or _is_form_upload(value):
            continue
        parameters[key] = _parse_form_value(value)

    parameters.update(_parse_extra_params(raw_form.get("extra_body"), "extra_body"))
    parameters.update(_parse_extra_params(raw_form.get("extra_params"), "extra_params"))
    flatten_extra_params(parameters)

    action_mode = str(parameters.get("action_mode", "policy")).strip().lower()
    observation: dict[str, Any] = {}

    upload_fields = (
        ("video_reference", "video"),
        ("image_reference", "image"),
        ("input_reference", None),
    )
    for field_name, forced_observation_key in upload_fields:
        upload = raw_form.get(field_name)
        if not _is_form_upload(upload):
            continue
        saved_path = await _save_action_upload(
            upload,
            str(request_id),
            uploads_dir,
            fallback_name=field_name,
        )
        observation_key = forced_observation_key
        if observation_key is None:
            observation_key = (
                "video"
                if action_mode == "inverse_dynamics"
                or _is_probably_video_upload(upload)
                else "input_reference"
            )
        observation[observation_key] = saved_path
        break

    for form_key, observation_key in (
        ("video_path", "video"),
        ("video_url", "video"),
        ("reference_url", "video" if action_mode == "inverse_dynamics" else "image"),
        (
            "input_reference",
            "video" if action_mode == "inverse_dynamics" else "input_reference",
        ),
    ):
        value = raw_form.get(form_key)
        if isinstance(value, str) and value and observation_key not in observation:
            observation[observation_key] = value

    return {
        "request_id": request_id,
        "input": {
            "task": raw_form.get("prompt") or raw_form.get("task") or "",
            "observation": observation,
        },
        "parameters": parameters,
    }


@router.post("/generations")
async def create_action_generation(request: Request):
    server_args: ServerArgs = request.app.state.server_args
    content_type = request.headers.get("content-type", "").lower()
    is_multipart = "multipart/form-data" in content_type
    input_context = (
        temp_dir_if_disabled(getattr(server_args, "input_save_path", None))
        if is_multipart
        else nullcontext(None)
    )
    try:
        with input_context as uploads_dir:
            if is_multipart:
                payload = await _multipart_action_payload(request, uploads_dir)
            elif "msgpack" in content_type:
                payload = unpack_msgpack(await request.body())
            else:
                payload = await request.json()
            wants_msgpack = _wants_msgpack(request)
            if wants_msgpack:
                _prefer_numpy_output(payload)
            output = await infer_action(payload, server_args)
            if _response_format(payload) == "raw":
                response = action_raw_response(output, preserve_numpy=wants_msgpack)
            else:
                response = action_generation_response(
                    output,
                    server_args,
                    preserve_numpy=wants_msgpack,
                )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if wants_msgpack:
        return Response(
            content=pack_msgpack(response), media_type="application/msgpack"
        )
    return orjson_response(response)


@router.get("/metadata")
async def action_metadata_endpoint(request: Request):
    return orjson_response(action_metadata(request.app.state.server_args))


@router.websocket("/realtime")
async def action_realtime_ws(websocket: WebSocket):
    server_args: ServerArgs = websocket.app.state.server_args
    await run_action_msgpack_ws(
        websocket,
        server_args,
        prepare_payload=_prefer_numpy_output,
        build_response=lambda output: action_generation_response(
            output,
            server_args,
            preserve_numpy=True,
        ),
    )
