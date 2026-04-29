# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.ug import (
    UGContextStage,
    UGDecodeStage,
    UGDenoiseStage,
    UGLatentStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.adapter import UGModelRunnerAdapter
from sglang.srt.ug.bagel import create_bagel_ug_model_adapter
from sglang.srt.ug.denoiser import SRTBackedUGDenoiserBridge, UGDenoiserBridge
from sglang.srt.ug.interleaved import (
    UGInterleavedRequest,
    UGInterleavedResponse,
    UGRuntimeStats,
)
from sglang.srt.ug.runtime import FakeUGModelRunner, UGSessionRuntime
from sglang.srt.ug.srt_executor import (
    UGSRTRequestBoundaryExecutor,
    UGSRTSchedulerExecutor,
)


class _UGRuntimeTreeCache:
    def __init__(self) -> None:
        self.released_sessions: list[str] = []

    def release_session(self, session_id: str) -> None:
        self.released_sessions.append(session_id)


def _build_srt_owned_ug_runtime(
    model_runner=None,
    *,
    scheduler=None,
    srt_request_executor=None,
    srt_u_decode_max_new_tokens: int = 0,
    srt_image_tokenization: Literal["multimodal", "text_placeholder"] = "multimodal",
) -> UGSessionRuntime:
    if srt_request_executor is None:
        srt_request_executor = _build_srt_request_executor(scheduler)
    session_controller = (
        srt_request_executor.session_controller
        if scheduler is not None
        else SessionController(_UGRuntimeTreeCache())
    )
    model_config = getattr(scheduler, "model_config", None)
    return UGSessionRuntime(
        model_runner=model_runner or FakeUGModelRunner(),
        session_controller=session_controller,
        srt_request_executor=srt_request_executor,
        tokenizer=getattr(scheduler, "tokenizer", None),
        vocab_size=getattr(model_config, "vocab_size", 32000),
        srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
        srt_image_tokenization=srt_image_tokenization,
    )


def _build_srt_request_executor(scheduler=None):
    if scheduler is not None:
        return UGSRTSchedulerExecutor(scheduler)
    return UGSRTRequestBoundaryExecutor()


def _load_ug_bridge(
    model_path: str,
    *,
    scheduler=None,
    srt_u_decode_max_new_tokens: int | None = None,
) -> UGDenoiserBridge:
    if srt_u_decode_max_new_tokens is None:
        srt_u_decode_max_new_tokens = 1 if scheduler is not None else 0
    srt_request_executor = _build_srt_request_executor(scheduler)
    model_path_lower = model_path.lower()
    if "fake-ug" in model_path_lower:
        return SRTBackedUGDenoiserBridge(
            _build_srt_owned_ug_runtime(
                scheduler=scheduler,
                srt_request_executor=srt_request_executor,
                srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
            )
        )
    if "bagel" in model_path_lower:
        native_srt_denoise_executor = None
        native_srt_u_context = False
        if scheduler is not None and "mock-bagel" not in model_path_lower:
            native_srt_denoise_executor = (
                srt_request_executor.create_bagel_native_srt_denoise_executor()
            )
            native_srt_u_context = True
        adapter = create_bagel_ug_model_adapter(
            model_path,
            native_srt_denoise_executor=native_srt_denoise_executor,
            native_srt_u_context=native_srt_u_context,
        )
        return SRTBackedUGDenoiserBridge(
            _build_srt_owned_ug_runtime(
                UGModelRunnerAdapter(adapter),
                scheduler=scheduler,
                srt_request_executor=srt_request_executor,
                srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
                srt_image_tokenization="multimodal",
            )
        )
    raise ValueError(f"Unsupported UG model path: {model_path}")


class UGPipeline(ComposedPipelineBase):
    pipeline_name = "UGPipeline"
    _required_config_modules: list[str] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules and "ug_bridge" in loaded_modules:
            return loaded_modules
        return {
            "ug_bridge": _load_ug_bridge(
                self.model_path,
                scheduler=getattr(server_args, "ug_srt_scheduler", None),
                srt_u_decode_max_new_tokens=getattr(
                    server_args,
                    "ug_srt_u_decode_max_new_tokens",
                    None,
                ),
            )
        }

    def create_pipeline_stages(self, server_args: ServerArgs):
        bridge = self.get_module("ug_bridge")
        self.add_stage(UGContextStage(bridge))
        self.add_stage(UGLatentStage(bridge))
        self.add_stage(UGDenoiseStage(bridge))
        self.add_stage(UGDecodeStage(bridge))

    def forward_interleaved(
        self,
        messages: UGInterleavedRequest | list[Any],
        sampling_params: UGSamplingParams | dict[str, Any] | None = None,
        server_args: ServerArgs | None = None,
        **sampling_kwargs: Any,
    ) -> UGInterleavedResponse:
        """Experimental UG interleaved API.

        This is intentionally Python-only and internal for now. It accepts a
        single interleaved request and returns ordered output segments without
        promising OpenAI-compatible request or response shapes.
        """

        server_args = server_args or self.server_args
        if server_args is None:
            raise ValueError("UG interleaved API requires server_args")
        request = _normalize_interleaved_request(
            messages, sampling_params, sampling_kwargs
        )
        batch = Req(
            sampling_params=request.sampling_params,
            extra={"ug_interleaved_messages": request.to_legacy_segments()},
        )
        try:
            result = self.forward(batch, server_args)
            contexts = result.extra.get("ug_contexts")
            stats = _collect_interleaved_runtime_stats(
                self.get_module("ug_bridge"), contexts
            )
            return UGInterleavedResponse.from_legacy_segments(
                list(result.extra["ug_output_segments"]),
                stats=stats,
                metadata=dict(request.metadata),
            )
        finally:
            contexts = batch.extra.get("ug_contexts")
            if contexts is not None:
                self.get_module("ug_bridge").release_contexts(contexts)

    def forward_interleaved_batch(
        self,
        requests: list[UGInterleavedRequest],
        server_args: ServerArgs | None = None,
    ) -> list[UGInterleavedResponse]:
        """Run multiple UG sessions through the experimental interleaved API.

        This intentionally provides session isolation before throughput
        batching. Each request gets its own SRT-owned UG session and release
        path, so it is safe for the server/API surface while the native batch
        optimizer remains a future step.
        """

        return [
            self.forward_interleaved(request, server_args=server_args)
            for request in requests
        ]


EntryClass = UGPipeline


def _normalize_interleaved_request(
    messages: UGInterleavedRequest | list[Any],
    sampling_params: UGSamplingParams | dict[str, Any] | None,
    sampling_kwargs: dict[str, Any],
) -> UGInterleavedRequest:
    if isinstance(messages, UGInterleavedRequest):
        if messages.sampling_params is not None and (
            sampling_params is not None or sampling_kwargs
        ):
            raise ValueError(
                "UG interleaved request already contains sampling_params; pass "
                "overrides by constructing a new UGInterleavedRequest"
            )
        return UGInterleavedRequest(
            messages=messages.messages,
            sampling_params=_normalize_interleaved_sampling_params(
                (
                    messages.sampling_params
                    if sampling_params is None
                    else sampling_params
                ),
                sampling_kwargs,
            ),
            metadata=dict(messages.metadata),
        )
    return UGInterleavedRequest.from_segments(
        messages,
        sampling_params=_normalize_interleaved_sampling_params(
            sampling_params, sampling_kwargs
        ),
    )


def _normalize_interleaved_sampling_params(
    sampling_params: UGSamplingParams | dict[str, Any] | None,
    sampling_kwargs: dict[str, Any],
) -> UGSamplingParams:
    if sampling_params is None:
        return UGSamplingParams(**sampling_kwargs)
    if isinstance(sampling_params, dict):
        values = dict(sampling_params)
        values.update(sampling_kwargs)
        return UGSamplingParams(**values)
    if sampling_kwargs:
        raise ValueError(
            "UG interleaved sampling keyword overrides require sampling_params "
            "to be omitted or passed as a dict"
        )
    return sampling_params


def _collect_interleaved_runtime_stats(
    bridge: UGDenoiserBridge,
    contexts: Any | None,
) -> UGRuntimeStats | None:
    if contexts is None or contexts.full.session is None:
        return None
    runtime = getattr(bridge, "runtime", None)
    if runtime is None:
        return None
    return UGRuntimeStats.from_debug_counters(
        runtime.get_debug_counters(contexts.full.session)
    )
