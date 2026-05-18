# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 wiring for the generic omni coordinator."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from sglang.omni.backends.ar.srt import SRTARBackend
from sglang.omni.backends.mm_gen.pipeline_forward_backend import (
    LazyDirectPipelineForwardBackend,
)
from sglang.omni.core.coordinator import OmniCoordinator
from sglang.omni.core.protocol import MultimodalGenerationBackend, OmniRequest

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
        SenseNovaU1SamplingParams,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs as DiffusionServerArgs,
    )
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.omni_session.session_adapter import SRTBackedOmniSessionAdapter
    from sglang.srt.omni_session.srt_executor import OmniSRTSchedulerExecutor
    from sglang.srt.server_args import ServerArgs as SRTServerArgs

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
    "max_text_segments_after_media",
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
        max_text_segments_after_media = metadata.pop(
            "max_text_segments_after_media",
            request.max_text_segments_after_media,
        )
        if max_text_segments_after_media is None and mode == "interleave":
            max_text_segments_after_media = 1
        metadata["mode"] = mode
        return replace(
            request,
            model=request.model or self.model_name,
            mode=mode,
            sampling_params=sampling_params,
            max_images=max_images,
            max_text_segments=max_text_segments,
            max_text_segments_after_media=max_text_segments_after_media,
            think=sampling_params.think_mode,
            think_max_new_tokens=think_max_new_tokens,
            metadata=metadata,
        )


def build_sensenova_u1_coordinator(
    *,
    srt_session_adapter: "SRTBackedOmniSessionAdapter",
    mm_generation_backend: MultimodalGenerationBackend | None = None,
    server_args: "SRTServerArgs | None" = None,
) -> OmniCoordinator:
    if mm_generation_backend is None:
        mm_generation_backend = _build_default_generation_backend(server_args)
    plugin = SenseNovaU1OmniPlugin()
    coordinator = OmniCoordinator(
        ar_backend=SRTARBackend(srt_session_adapter),
        mm_generation_backend=mm_generation_backend,
        request_adapter=plugin.normalize_request,
        metadata={"model": plugin.model_name},
        max_concurrent_generations=_resolve_omni_max_concurrent_generations(
            server_args
        ),
    )
    return coordinator


def build_sensenova_u1_coordinator_from_scheduler(
    *,
    scheduler: "Scheduler",
    srt_request_executor: "OmniSRTSchedulerExecutor | None" = None,
    srt_ar_decode_max_new_tokens: int | None = None,
    generation_backend: MultimodalGenerationBackend | None = None,
    server_args: "SRTServerArgs | None" = None,
) -> OmniCoordinator:
    from sglang.omni.model_adapters.sensenova_u1.session_adapter import (
        build_sensenova_u1_srt_session_adapter,
    )

    return build_sensenova_u1_coordinator(
        srt_session_adapter=build_sensenova_u1_srt_session_adapter(
            scheduler=scheduler,
            srt_request_executor=srt_request_executor,
            srt_ar_decode_max_new_tokens=srt_ar_decode_max_new_tokens,
        ),
        mm_generation_backend=generation_backend,
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
    payload: dict[str, Any] | None,
    *,
    mode: str,
    think: bool,
) -> "SenseNovaU1SamplingParams":
    if payload is None:
        sampling_params = _build_sensenova_sampling_dataclass({})
    else:
        sampling_params = _build_sensenova_sampling_dataclass(payload)

    sampling_params.omni_generation_mode = mode
    sampling_params.think_mode = bool(think)
    return sampling_params


def split_request_metadata(
    payload: Any | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if payload is None:
        return None, {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"SenseNova U1 sampling_params must be an object, got {payload!r}"
        )
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


def _build_sensenova_sampling_dataclass(
    payload: dict[str, Any],
) -> "SenseNovaU1SamplingParams":
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


def _build_default_generation_backend(
    srt_server_args: "SRTServerArgs | None",
) -> LazyDirectPipelineForwardBackend:
    from sglang.multimodal_gen.runtime.pipelines.sensenova_u1 import (
        SenseNovaU1Pipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
        SyncExecutor,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs as DiffusionServerArgs,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        set_global_server_args,
    )

    diffusion_server_kwargs = _build_diffusion_server_kwargs(srt_server_args)
    pipeline_server_args = DiffusionServerArgs.from_kwargs(**diffusion_server_kwargs)
    _validate_diffusion_server_args(pipeline_server_args)
    set_global_server_args(pipeline_server_args)

    # U1 pixel-flow stays same-process because denoising reads live SRT KV handles.
    def build_pipeline() -> SenseNovaU1Pipeline:
        return SenseNovaU1Pipeline(
            pipeline_server_args.model_path,
            pipeline_server_args,
            executor=SyncExecutor(server_args=pipeline_server_args),
        )

    return LazyDirectPipelineForwardBackend(
        pipeline_factory=build_pipeline,
        server_args=pipeline_server_args,
    )


def _resolve_omni_max_concurrent_generations(
    srt_server_args: "SRTServerArgs | None",
) -> int:
    value = (
        1
        if srt_server_args is None
        else srt_server_args.omni_max_concurrent_generations
    )
    if value is None:
        value = 1
    value = int(value)
    if value <= 0:
        raise ValueError(
            "omni_max_concurrent_generations must be positive, got " f"{value}"
        )
    return value


def _build_diffusion_server_kwargs(
    srt_server_args: "SRTServerArgs | None",
) -> dict[str, Any]:
    kwargs = _parse_diffusion_server_args(
        None if srt_server_args is None else srt_server_args.diffusion_server_args
    )
    kwargs.setdefault("model_path", _resolve_model_path(srt_server_args))
    kwargs["pipeline_class_name"] = "SenseNovaU1Pipeline"
    return kwargs


def _parse_diffusion_server_args(value: Any | None) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(
            f"diffusion_server_args must be a string or dict, got {value!r}"
        )

    text = value.strip()
    if not text:
        return {}
    if text.startswith("@"):
        return _load_diffusion_server_args_file(text[1:])
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("diffusion_server_args JSON must be an object")
        return payload
    if os.path.exists(text):
        return _load_diffusion_server_args_file(text)
    return _parse_diffusion_server_cli_args(text)


def _parse_diffusion_server_cli_args(value: str) -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
        add_multimodal_gen_serve_args,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs as DiffusionServerArgs,
    )

    argv = shlex.split(value)
    parser = argparse.ArgumentParser(add_help=False)
    add_multimodal_gen_serve_args(parser)
    raw_args, unknown_args = parser.parse_known_args(argv)

    dynamic_paths, remaining = DiffusionServerArgs._extract_component_paths(
        unknown_args
    )
    dynamic_attention_backends, remaining = (
        DiffusionServerArgs._extract_component_attention_backends(remaining)
    )
    if remaining:
        raise ValueError("Unrecognized diffusion_server_args: " + " ".join(remaining))

    provided_args = _provided_cli_args(raw_args, argv)
    config_file = provided_args.get("config")
    if config_file:
        config_args = DiffusionServerArgs.load_config_file(config_file)
        provided_args = {**config_args, **provided_args}

    if dynamic_paths:
        component_paths = dict(provided_args.get("component_paths") or {})
        component_paths.update(dynamic_paths)
        provided_args["component_paths"] = component_paths
    if dynamic_attention_backends:
        component_attention_backends = (
            DiffusionServerArgs._parse_component_attention_backend_map(
                provided_args.get("component_attention_backends")
            )
        )
        component_attention_backends.update(dynamic_attention_backends)
        provided_args["component_attention_backends"] = component_attention_backends
    return provided_args


def _provided_cli_args(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    provided_names = set()
    for arg in argv:
        if arg.startswith("--"):
            provided_names.add(arg.split("=", 1)[0].replace("-", "_").lstrip("_"))
    return {key: value for key, value in vars(args).items() if key in provided_names}


def _load_diffusion_server_args_file(path: str) -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs as DiffusionServerArgs,
    )

    payload = DiffusionServerArgs.load_config_file(path)
    if not isinstance(payload, dict):
        raise ValueError("diffusion_server_args file must contain an object")
    return payload


def _validate_diffusion_server_args(server_args: "DiffusionServerArgs") -> None:
    validate_runtime = getattr(server_args.pipeline_config, "validate_runtime", None)
    if not callable(validate_runtime):
        return

    from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

    unsupported = validate_runtime(
        num_gpus=server_args.num_gpus,
        enable_cfg_parallel=server_args.enable_cfg_parallel,
        disagg_mode=server_args.disagg_role != RoleType.MONOLITHIC,
    )
    if unsupported:
        raise ValueError(
            "SenseNova U1 same-process diffusion engine does not support "
            + ", ".join(unsupported)
        )


def _resolve_model_path(server_args: "SRTServerArgs | None") -> str:
    if server_args is not None:
        model_path = server_args.model_path
        if model_path:
            return model_path
    return SenseNovaU1OmniPlugin.model_name
