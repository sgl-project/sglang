# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 wiring for the generic omni orchestrator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sglang.omni.backends.colocated import ColocatedPipelineBackend
from sglang.omni.backends.srt import SRTARBackend
from sglang.omni.coordinator import OmniCoordinator
from sglang.omni.protocol import OmniRequest

_MODE_ALIASES = {
    "text_to_image": "t2i",
    "txt2img": "t2i",
    "image_edit": "edit",
    "img2img": "edit",
    "interleaved": "interleave",
    "chat": "vlm",
    "vlm_chat": "vlm",
}
_SUPPORTED_MODES = {"t2i", "edit", "interleave", "vlm"}
_REQUEST_METADATA_FIELDS = {
    "mode",
    "task",
    "max_new_tokens",
    "max_length",
    "max_interleave_images",
    "max_interleave_text_segments",
    "think",
    "think_max_new_tokens",
}


class SenseNovaU1OmniPlugin:
    """Normalize SenseNova U1 request aliases into native sampling params."""

    model_name = "sensenova-u1"

    def normalize_request(self, request: OmniRequest) -> OmniRequest:
        sampling_payload, request_metadata = split_request_metadata(
            request.sampling_params
        )
        metadata = dict(request.metadata)
        metadata.update(request_metadata)
        mode = normalize_mode(metadata.pop("task", metadata.pop("mode", request.mode)))
        sampling_params = build_sampling_params(
            sampling_payload,
            mode=mode,
            think=_resolve_bool(metadata.pop("think", request.think), name="think"),
        )
        max_images = int(metadata.pop("max_interleave_images", request.max_images))
        max_text_segments = int(
            metadata.pop("max_interleave_text_segments", request.max_text_segments)
        )
        think_max_new_tokens = metadata.pop(
            "think_max_new_tokens",
            request.think_max_new_tokens,
        )
        metadata["mode"] = mode
        return replace(
            request,
            model=request.model or self.model_name,
            mode=mode,
            sampling_params=sampling_params,
            max_images=max_images,
            max_text_segments=max_text_segments,
            think=bool(getattr(sampling_params, "think_mode", request.think)),
            think_max_new_tokens=think_max_new_tokens,
            metadata=metadata,
        )


def build_sensenova_u1_orchestrator(
    *,
    srt_bridge: Any,
    generation_backend: Any | None = None,
    server_args: Any | None = None,
) -> OmniCoordinator:
    if generation_backend is None:
        generation_backend = _build_default_generation_backend(server_args)
    plugin = SenseNovaU1OmniPlugin()
    coordinator = OmniCoordinator(
        ar_backend=SRTARBackend(srt_bridge),
        generation_backend=generation_backend,
        request_adapter=plugin.normalize_request,
        metadata={"model": plugin.model_name},
    )
    return coordinator


def build_sensenova_u1_orchestrator_from_scheduler(
    *,
    scheduler: Any,
    srt_request_executor: Any | None = None,
    srt_u_decode_max_new_tokens: int | None = None,
    generation_backend: Any | None = None,
    server_args: Any | None = None,
) -> OmniCoordinator:
    from sglang.srt.omni_session.sensenova_u1 import build_sensenova_u1_middle_bridge

    return build_sensenova_u1_orchestrator(
        srt_bridge=build_sensenova_u1_middle_bridge(
            scheduler=scheduler,
            srt_request_executor=srt_request_executor,
            srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
        ),
        generation_backend=generation_backend,
        server_args=server_args,
    )


def normalize_mode(mode: Any | None) -> str:
    if mode is None:
        return "interleave"
    normalized = str(mode).strip().lower().replace("-", "_")
    normalized = _MODE_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported SenseNova-U1 omni mode {mode!r}; "
            f"expected one of {sorted(_SUPPORTED_MODES)}"
        )
    return normalized


def build_sampling_params(
    payload: Any | None,
    *,
    mode: str,
    think: bool,
) -> Any:
    if payload is None:
        sampling_params = _build_sensenova_sampling_dataclass({})
    elif not isinstance(payload, dict):
        sampling_params = payload
    else:
        sampling_params = _build_sensenova_sampling_dataclass(payload)

    setattr(sampling_params, "ug_generation_mode", mode)
    setattr(sampling_params, "think_mode", bool(think))
    return sampling_params


def split_request_metadata(payload: Any | None) -> tuple[Any | None, dict[str, Any]]:
    if not isinstance(payload, dict):
        return payload, {}
    sampling_payload = dict(payload)
    metadata = {}
    for key in tuple(sampling_payload):
        if key in _REQUEST_METADATA_FIELDS:
            metadata[key] = sampling_payload.pop(key)
    return sampling_payload, metadata


def _resolve_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a bool, got {value!r}")


def _build_sensenova_sampling_dataclass(payload: dict[str, Any]) -> Any:
    from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
        build_sensenova_u1_sampling_params,
    )

    aliases = {
        "num_steps": "num_inference_steps",
        "steps": "num_inference_steps",
        "guidance_scale": "cfg_text_scale",
        "text_cfg_scale": "cfg_text_scale",
        "img_cfg_scale": "cfg_img_scale",
        "image_cfg_scale": "cfg_img_scale",
    }
    values = {aliases.get(key, key): value for key, value in payload.items()}
    return build_sensenova_u1_sampling_params(values)


def _build_default_generation_backend(srt_server_args: Any | None) -> Any:
    from sglang.multimodal_gen.runtime.pipelines.sensenova_u1 import (
        SenseNovaU1Pipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
        SyncExecutor,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs as DiffusionServerArgs,
        set_global_server_args,
    )

    model_path = _resolve_model_path(srt_server_args)
    pipeline_server_args = DiffusionServerArgs.from_kwargs(
        model_path=model_path,
        pipeline_class_name="SenseNovaU1Pipeline",
    )
    set_global_server_args(pipeline_server_args)
    # u1 G is colocated with srt because each denoise step needs live srt session/KV
    pipeline = SenseNovaU1Pipeline(
        model_path,
        pipeline_server_args,
        executor=SyncExecutor(server_args=pipeline_server_args),
    )
    return ColocatedPipelineBackend(
        pipeline=pipeline,
        server_args=pipeline_server_args,
        context_ops_extra_key="sensenova_u1_context_ops",
        output_extra_key="sensenova_u1_generated_segment",
    )


def _resolve_model_path(server_args: Any | None) -> str:
    if server_args is not None:
        model_path = getattr(server_args, "model_path", None)
        if model_path:
            return model_path
    return SenseNovaU1OmniPlugin.model_name
