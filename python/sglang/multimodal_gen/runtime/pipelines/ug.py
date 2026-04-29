# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
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


EntryClass = UGPipeline
