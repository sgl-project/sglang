# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import dataclasses
import io
import time
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from sglang.multimodal_gen.configs.sample.vla import VLASamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def pack_numpy_payload(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_numpy_payload(obj):
    ndarray_marker = obj.get("__ndarray__") or obj.get(b"__ndarray__")
    npgeneric_marker = obj.get("__npgeneric__") or obj.get(b"__npgeneric__")
    data = obj.get("data", obj.get(b"data"))
    dtype = obj.get("dtype", obj.get(b"dtype"))
    shape = obj.get("shape", obj.get(b"shape"))
    if ndarray_marker:
        return np.ndarray(
            buffer=data,
            dtype=np.dtype(dtype),
            shape=shape,
        )
    if npgeneric_marker:
        return np.dtype(dtype).type(data)
    return obj


def pack_msgpack(payload: Any) -> bytes:
    import msgpack

    return msgpack.packb(payload, default=pack_numpy_payload, use_bin_type=True)


def unpack_msgpack(payload: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(payload, object_hook=unpack_numpy_payload, raw=False)


def _decode_b64_image(payload: dict[str, Any]) -> Image.Image:
    data = payload.get("b64_json") or payload.get("base64")
    if not data:
        raise ValueError("image payload requires b64_json")
    if isinstance(data, str) and "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _decode_tensor_payload(payload: dict[str, Any]) -> Any:
    values = payload.get("values")
    if values is None:
        values = payload.get("data")
    if values is None:
        return payload
    dtype = payload.get("dtype")
    array = np.asarray(values, dtype=np.dtype(dtype) if dtype else None)
    shape = payload.get("shape")
    if shape is not None:
        array = array.reshape(tuple(shape))
    return array


def _normalize_image_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "b64_json" in value or "base64" in value:
        return _decode_b64_image(value)
    if "values" in value or "data" in value:
        return _decode_tensor_payload(value)
    return value


def _normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(observation)
    images = normalized.get("images")
    if isinstance(images, dict):
        normalized["images"] = {
            name: _normalize_image_value(value) for name, value in images.items()
        }
    state = normalized.get("state")
    if isinstance(state, dict):
        normalized["state"] = _decode_tensor_payload(state)
    observation_state = normalized.get("observation.state")
    if isinstance(observation_state, dict):
        normalized["observation.state"] = _decode_tensor_payload(observation_state)
    noise = normalized.get("noise")
    if isinstance(noise, dict):
        normalized["noise"] = _decode_tensor_payload(noise)
    observation_noise = normalized.get("observation.noise")
    if isinstance(observation_noise, dict):
        normalized["observation.noise"] = _decode_tensor_payload(observation_noise)
    return normalized


def images_from_observation(
    observation: dict[str, Any],
    pipeline_config: Any,
) -> dict[str, Any]:
    if isinstance(observation.get("images"), dict):
        images = dict(observation["images"])
    else:
        images = {}
    for key in pipeline_config.image_keys:
        if key in observation:
            images[key] = observation[key]
        full_key = f"observation.images.{key}"
        if full_key in observation:
            images[key] = observation[full_key]
    return {name: _normalize_image_value(value) for name, value in images.items()}


def action_metadata(server_args: ServerArgs) -> dict[str, Any]:
    pipeline_config = server_args.pipeline_config
    policy_family = getattr(
        pipeline_config,
        "policy_family",
        type(pipeline_config).__name__.removesuffix("PipelineConfig").lower(),
    )
    return {
        "object": "action.metadata",
        "model": server_args.model_id or server_args.model_path,
        "model_path": server_args.model_path,
        "policy_family": policy_family,
        "input": {
            "image_keys": list(pipeline_config.image_keys),
            "image_size": list(pipeline_config.image_size),
            "state_dim": pipeline_config.state_dim,
        },
        "output": {
            "action_type": "continuous",
            "action_horizon": pipeline_config.action_horizon,
            "action_dim": pipeline_config.output_action_dim,
            "padded_action_dim": pipeline_config.action_dim,
            "dtype": "float32",
        },
        "runtime": {
            "materialize_dtype": pipeline_config.materialize_dtype,
            "enable_autocast": pipeline_config.enable_autocast,
        },
        "defaults": {
            "num_inference_steps": pipeline_config.default_num_inference_steps,
            "prefix_cache": (
                "auto" if pipeline_config.enable_global_prefix_cache else False
            ),
            "cuda_graph": "auto" if pipeline_config.enable_action_cuda_graph else False,
        },
        "capabilities": {
            "exact_prefix_cache": True,
            "cuda_graph": pipeline_config.enable_action_cuda_graph,
            "realtime_websocket": True,
            "openpi_websocket": True,
            "batch_inputs": False,
            "multiple_candidates": False,
        },
    }


def _runtime_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.lower()
        if value == "auto":
            return default
        if value in ("true", "1", "yes"):
            return True
        if value in ("false", "0", "no"):
            return False
    return bool(value)


def _action_request_to_observation(payload: dict[str, Any]) -> dict[str, Any]:
    if "input" not in payload:
        return _normalize_observation(payload)

    input_payload = payload.get("input") or {}
    observation = dict(input_payload.get("observation") or {})
    if "task" in input_payload:
        observation["prompt"] = input_payload["task"]
    elif "prompt" in input_payload:
        observation["prompt"] = input_payload["prompt"]
    if "images" in input_payload:
        observation["images"] = input_payload["images"]
    if "state" in input_payload:
        observation["state"] = input_payload["state"]
    if "noise" in input_payload:
        observation["noise"] = input_payload["noise"]
    return _normalize_observation(observation)


@lru_cache(maxsize=32)
def _resolve_action_sampling_params_cls_cached(
    model_path: str,
    backend: str | None,
    model_id: str | None,
    pipeline_class_name: str | None,
) -> type[VLASamplingParams]:
    if pipeline_class_name:
        from sglang.multimodal_gen.registry import get_pipeline_config_classes

        config_classes = get_pipeline_config_classes(pipeline_class_name)
        if config_classes is not None:
            _, sampling_params_cls = config_classes
            if issubclass(sampling_params_cls, VLASamplingParams):
                return sampling_params_cls

    from sglang.multimodal_gen.registry import get_model_info

    model_info = get_model_info(
        model_path,
        backend=backend,
        model_id=model_id,
    )
    sampling_params_cls = model_info.sampling_param_cls
    if not issubclass(sampling_params_cls, VLASamplingParams):
        raise ValueError(
            f"Action endpoint requires VLASamplingParams, got {sampling_params_cls.__name__}"
        )
    return sampling_params_cls


def _resolve_action_sampling_params_cls(
    server_args: ServerArgs,
) -> type[VLASamplingParams]:
    return _resolve_action_sampling_params_cls_cached(
        server_args.model_path,
        getattr(server_args, "backend", None),
        getattr(server_args, "model_id", None),
        getattr(server_args, "pipeline_class_name", None),
    )


@lru_cache(maxsize=32)
def _sampling_params_field_names(
    sampling_params_cls: type[VLASamplingParams],
) -> frozenset[str]:
    return frozenset(field.name for field in dataclasses.fields(sampling_params_cls))


def build_action_sampling_params(
    payload: dict[str, Any],
    server_args: ServerArgs,
) -> VLASamplingParams:
    pipeline_config = server_args.pipeline_config
    observation = _action_request_to_observation(payload)
    parameters = dict(payload.get("parameters") or {})
    runtime = dict(payload.get("runtime") or {})
    if "return_timing" in payload and "return_timing" not in runtime:
        runtime["return_timing"] = payload["return_timing"]
    images = images_from_observation(observation, pipeline_config)
    state = observation.get("state")
    if state is None:
        state = observation.get("observation.state")
    noise = observation.get("noise")
    if noise is None:
        noise = observation.get("observation.noise")
    prompt = observation.get("prompt") or observation.get("task") or ""
    prefix_cache = runtime.get("prefix_cache")
    if prefix_cache is None:
        prefix_cache = observation.get("enable_prefix_cache")
    if prefix_cache is None:
        prefix_cache = observation.get("enable_pi_prefix_cache")
    cuda_graph = runtime.get("cuda_graph")
    if cuda_graph is None:
        cuda_graph = observation.get("enable_cuda_graph")
    if cuda_graph is None:
        cuda_graph = observation.get("enable_pi_cuda_graph")
    output_format = str(
        runtime.get(
            "output_format",
            parameters.get(
                "output_format",
                observation.get("output_format", "list"),
            ),
        )
    ).lower()
    if output_format not in ("list", "numpy"):
        raise ValueError("output_format must be 'list' or 'numpy'")

    sampling_params_cls = _resolve_action_sampling_params_cls(server_args)
    sampling_kwargs = {
        "prompt": prompt,
        "images": images,
        "image_masks": observation.get("image_masks"),
        "camera_order": observation.get("camera_order"),
        "state": state,
        "noise": noise,
        "observation": observation,
        "action_horizon": int(
            parameters.get(
                "action_horizon",
                observation.get("action_horizon", pipeline_config.action_horizon),
            )
        ),
        "action_dim": int(
            parameters.get(
                "action_dim",
                observation.get("action_dim", pipeline_config.action_dim),
            )
        ),
        "num_inference_steps": int(
            parameters.get(
                "num_inference_steps",
                observation.get(
                    "num_inference_steps",
                    pipeline_config.default_num_inference_steps,
                ),
            )
        ),
        "output_format": output_format,
        "return_timing": _runtime_bool(runtime.get("return_timing"), True),
        "enable_prefix_cache": _runtime_bool(prefix_cache, True),
        "enable_cuda_graph": _runtime_bool(cuda_graph, True),
    }
    supported_fields = _sampling_params_field_names(sampling_params_cls)
    sp = sampling_params_cls(
        **{
            name: value
            for name, value in sampling_kwargs.items()
            if name in supported_fields
        }
    )
    sp._adjust(server_args)
    return sp


async def infer_action(
    payload: dict[str, Any],
    server_args: ServerArgs,
) -> dict[str, Any]:
    sp = build_action_sampling_params(payload, server_args)
    req = prepare_request(server_args, sp)
    response = await async_scheduler_client.forward(req)
    if getattr(response, "error", None):
        raise RuntimeError(response.error)
    if response.output is None:
        raise RuntimeError("action policy returned no output")
    return response.output[0]


def action_generation_response(
    output: dict[str, Any],
    server_args: ServerArgs,
    *,
    preserve_numpy: bool = False,
) -> dict[str, Any]:
    actions = output["actions"]
    if isinstance(actions, np.ndarray):
        action_shape = list(actions.shape)
        action_values = actions if preserve_numpy else actions.tolist()
    else:
        horizon = len(actions) if isinstance(actions, list) else 0
        action_dim = len(actions[0]) if horizon and isinstance(actions[0], list) else 0
        action_shape = [horizon, action_dim]
        action_values = actions
    response = {
        "id": f"act_{uuid.uuid4().hex}",
        "object": "action.generation",
        "created": int(time.time()),
        "model": server_args.model_id or server_args.model_path,
        "data": [
            {
                "index": 0,
                "input_index": 0,
                "candidate_index": 0,
                "action": {
                    "type": "continuous",
                    "dtype": "float32",
                    "shape": action_shape,
                    "values": action_values,
                },
            }
        ],
        "usage": {
            "action_horizon": action_shape[0] if action_shape else 0,
            "action_dim": action_shape[1] if len(action_shape) > 1 else 0,
            "denoise_steps": output.get("parameters", {}).get(
                "num_inference_steps",
                server_args.pipeline_config.default_num_inference_steps,
            ),
            "prefix_cache_hit": bool(output.get("cache", {}).get("hit", False)),
        },
    }
    if "timings" in output:
        response["timings"] = output["timings"]
    if "cache" in output:
        response["cache"] = output["cache"]
    return response


def action_raw_response(
    output: dict[str, Any],
    *,
    preserve_numpy: bool = False,
) -> dict[str, Any]:
    response = dict(output)
    actions = response.get("actions")
    if isinstance(actions, np.ndarray) and not preserve_numpy:
        response["actions"] = actions.tolist()
    return response
