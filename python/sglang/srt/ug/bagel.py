# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from sglang.srt.ug.adapter import (
    UGModelAdapterProtocol,
    UGModelAppendImageResult,
    UGModelPrefillResult,
)
from sglang.srt.ug.bagel_cache import BAGELSRTKVCacheFactory
from sglang.srt.ug.context import UGSRTRequestView
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGLatentDecodeRequest,
    UGLatentPrepareRequest,
    UGLatentPrepareResult,
    UGSegmentState,
    UGSRTPreparedInput,
    UGVelocityRequest,
)

_BAGEL_REQUIRED_CHECKPOINT_FILES = (
    "llm_config.json",
    "vit_config.json",
    "ae.safetensors",
    "ema.safetensors",
)
_BAGEL_REQUIRED_MODULES = (
    "accelerate",
    "data.data_utils",
    "data.transforms",
    "inferencer",
    "modeling.autoencoder",
    "modeling.bagel",
    "modeling.qwen2",
)
_BAGEL_NATIVE_SRT_REQUIRED_MODULES = (
    "data.data_utils",
    "data.transforms",
    "modeling.autoencoder",
    "modeling.bagel",
    "modeling.qwen2",
)
_BAGEL_SAME_DEVICE_MODULES = (
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
)
_BAGEL_GENERATION_INPUT_KEYS = (
    "packed_text_ids",
    "packed_text_indexes",
    "packed_vae_token_indexes",
    "packed_vae_position_ids",
    "packed_seqlens",
    "packed_position_ids",
    "packed_indexes",
    "key_values_lens",
    "packed_key_value_indexes",
)
_BAGEL_CFG_TEXT_INPUT_KEYS = (
    "cfg_packed_position_ids",
    "cfg_packed_query_indexes",
    "cfg_key_values_lens",
    "cfg_packed_key_value_indexes",
)
_BAGEL_CFG_IMG_INPUT_KEYS = _BAGEL_CFG_TEXT_INPUT_KEYS


class BAGELAdapterError(RuntimeError):
    """Raised when the BAGEL adapter cannot be constructed safely."""


class BAGELDenoiseStepError(RuntimeError):
    """Raised when a BAGEL single-step denoise call is malformed."""


class BAGELPreparedDenoise:
    """Official BAGEL denoise inputs prepared from a SRT-owned UG session."""

    def __init__(
        self,
        *,
        generation_input: dict[str, Any],
        cfg_text_generation_input: dict[str, Any],
        cfg_img_generation_input: dict[str, Any],
        past_key_values: Any,
        cfg_text_past_key_values: Any | None = None,
        cfg_img_past_key_values: Any | None = None,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_type: str = "parallel",
    ) -> None:
        self.generation_input = generation_input
        self.cfg_text_generation_input = cfg_text_generation_input
        self.cfg_img_generation_input = cfg_img_generation_input
        self.past_key_values = past_key_values
        self.cfg_text_past_key_values = cfg_text_past_key_values
        self.cfg_img_past_key_values = cfg_img_past_key_values
        self.cfg_text_scale = cfg_text_scale
        self.cfg_img_scale = cfg_img_scale
        self.cfg_interval = cfg_interval
        self.cfg_renorm_min = cfg_renorm_min
        self.cfg_renorm_type = cfg_renorm_type
        self.cfg_type = cfg_type


@dataclass
class BAGELNativeSRTPreparedDenoise:
    """Denoise inputs whose U context is owned by SRT KV cache."""

    generation_input: dict[str, Any]
    srt_kv_token_binding: Any | None = None
    cfg_text_scale: float = 1.0
    cfg_img_scale: float = 1.0
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    cfg_type: str = "parallel"


class BAGELNativeSRTDenoiseExecutor:
    """Calls SRT-native BAGEL gen forward instead of official `_forward_flow`."""

    def __init__(
        self,
        srt_model: Any,
        *,
        forward_batch_provider: Any | None = None,
    ) -> None:
        self.srt_model = srt_model
        self.forward_batch_provider = forward_batch_provider
        self.velocity_count = 0

    def predict_velocity(
        self,
        *,
        prepared: BAGELNativeSRTPreparedDenoise,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_prepared(prepared)
        self._validate_cfg(prepared, timestep)
        generation_input = prepared.generation_input
        predictor = getattr(self.srt_model, "predict_velocity_from_packed_gen", None)
        if not callable(predictor):
            raise BAGELAdapterError(
                "Native SRT BAGEL denoise requires "
                "predict_velocity_from_packed_gen on the SRT model"
            )
        forward_batch_context = self._build_forward_batch(
            prepared=prepared,
            latent_tokens=latent_tokens,
            timestep=timestep,
        )
        forward_batch = getattr(
            forward_batch_context,
            "forward_batch",
            forward_batch_context,
        )
        self.velocity_count += 1
        try:
            return predictor(
                latent_tokens=latent_tokens,
                timestep=timestep,
                packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
                packed_vae_position_ids=generation_input["packed_vae_position_ids"],
                packed_text_ids=generation_input["packed_text_ids"],
                packed_text_indexes=generation_input["packed_text_indexes"],
                packed_position_ids=generation_input["packed_position_ids"],
                packed_seqlens=generation_input["packed_seqlens"],
                forward_batch=forward_batch,
            )
        finally:
            release = getattr(forward_batch_context, "release", None)
            if callable(release):
                release()

    def _build_forward_batch(
        self,
        *,
        prepared: BAGELNativeSRTPreparedDenoise,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Any:
        if self.forward_batch_provider is None:
            return None
        return self.forward_batch_provider(
            prepared=prepared,
            latent_tokens=latent_tokens,
            timestep=timestep,
        )

    @staticmethod
    def _validate_prepared(prepared: BAGELNativeSRTPreparedDenoise) -> None:
        _require_keys(
            prepared.generation_input,
            _BAGEL_GENERATION_INPUT_KEYS,
            "native_srt_generation_input",
        )

    @staticmethod
    def _validate_cfg(
        prepared: BAGELNativeSRTPreparedDenoise,
        timestep: torch.Tensor,
    ) -> None:
        cfg_text_scale, cfg_img_scale = BAGELDenoiseStepRunner._effective_cfg_scales(
            prepared,
            timestep,
        )
        if cfg_text_scale > 1.0 or cfg_img_scale > 1.0:
            raise BAGELDenoiseStepError(
                "Native SRT BAGEL denoise currently supports cfg_text_scale <= 1 "
                "and cfg_img_scale <= 1 only"
            )


class BAGELDenoiseStepRunner:
    """Runs the single `_forward_flow` step extracted from BAGEL.generate_image."""

    def predict_velocity(
        self,
        *,
        model: Any,
        prepared: BAGELPreparedDenoise,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_prepared(prepared)
        timestep = self._expand_timestep(timestep, latent_tokens)
        cfg_text_scale, cfg_img_scale = self._effective_cfg_scales(prepared, timestep)
        generation_input = prepared.generation_input
        cfg_text_input = prepared.cfg_text_generation_input
        cfg_img_input = prepared.cfg_img_generation_input
        self._ensure_flow_runtime_flags(model)

        return model._forward_flow(
            x_t=latent_tokens,
            timestep=timestep,
            packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
            packed_vae_position_ids=generation_input["packed_vae_position_ids"],
            packed_text_ids=generation_input["packed_text_ids"],
            packed_text_indexes=generation_input["packed_text_indexes"],
            packed_position_ids=generation_input["packed_position_ids"],
            packed_indexes=generation_input["packed_indexes"],
            packed_seqlens=generation_input["packed_seqlens"],
            key_values_lens=generation_input["key_values_lens"],
            past_key_values=prepared.past_key_values,
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            cfg_renorm_min=prepared.cfg_renorm_min,
            cfg_renorm_type=prepared.cfg_renorm_type,
            cfg_text_scale=cfg_text_scale,
            cfg_text_packed_position_ids=cfg_text_input["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=cfg_text_input["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=cfg_text_input["cfg_key_values_lens"],
            cfg_text_past_key_values=prepared.cfg_text_past_key_values,
            cfg_text_packed_key_value_indexes=cfg_text_input[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_scale=cfg_img_scale,
            cfg_img_packed_position_ids=cfg_img_input["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=cfg_img_input["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=cfg_img_input["cfg_key_values_lens"],
            cfg_img_past_key_values=prepared.cfg_img_past_key_values,
            cfg_img_packed_key_value_indexes=cfg_img_input[
                "cfg_packed_key_value_indexes"
            ],
            cfg_type=prepared.cfg_type,
        )

    @staticmethod
    def build_timesteps(
        *,
        num_timesteps: int,
        timestep_shift: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_timesteps <= 1:
            raise BAGELDenoiseStepError(
                f"num_timesteps must be > 1, got {num_timesteps}"
            )
        timesteps = torch.linspace(1, 0, num_timesteps, device=device, dtype=dtype)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        return timesteps[:-1], dts

    @staticmethod
    def _effective_cfg_scales(
        prepared: BAGELPreparedDenoise,
        timestep: torch.Tensor,
    ) -> tuple[float, float]:
        t = float(timestep.flatten()[0].detach().cpu())
        start, end = prepared.cfg_interval
        if t > start and t <= end:
            return prepared.cfg_text_scale, prepared.cfg_img_scale
        return 1.0, 1.0

    @staticmethod
    def _expand_timestep(
        timestep: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        timestep = timestep.to(device=latent_tokens.device)
        if timestep.numel() == 1:
            return timestep.reshape(1).expand(latent_tokens.shape[0])
        if timestep.shape[0] != latent_tokens.shape[0]:
            raise BAGELDenoiseStepError(
                "BAGEL timestep must be scalar or match latent token batch size: "
                f"{tuple(timestep.shape)} vs {tuple(latent_tokens.shape)}"
            )
        return timestep

    @staticmethod
    def _validate_prepared(prepared: BAGELPreparedDenoise) -> None:
        _require_keys(
            prepared.generation_input,
            _BAGEL_GENERATION_INPUT_KEYS,
            "generation_input",
        )
        _require_keys(
            prepared.cfg_text_generation_input,
            _BAGEL_CFG_TEXT_INPUT_KEYS,
            "cfg_text_generation_input",
        )
        _require_keys(
            prepared.cfg_img_generation_input,
            _BAGEL_CFG_IMG_INPUT_KEYS,
            "cfg_img_generation_input",
        )

    @staticmethod
    def _ensure_flow_runtime_flags(model: Any) -> None:
        language_model = getattr(model, "language_model", None)
        inner_model = getattr(language_model, "model", None)
        if inner_model is not None and not hasattr(inner_model, "enable_taylorseer"):
            inner_model.enable_taylorseer = False


@dataclass
class BAGELSessionContext:
    gen_context: dict[str, Any]
    cfg_text_context: dict[str, Any]
    cfg_img_context: dict[str, Any]
    image_shape: tuple[int, int]
    prepared_denoise: BAGELPreparedDenoise | None = None
    decode_count: int = 0
    append_image_count: int = 0
    native_srt_u_context: bool = False
    native_srt_u_context_request_id: str | None = None
    native_srt_u_context_token_binding: Any | None = None
    srt_u_forward_results: dict[str, Any] = field(default_factory=dict)
    srt_u_forward_events: list[tuple[str, str]] = field(default_factory=list)
    srt_last_u_decode_output_ids: tuple[int, ...] = ()
    native_srt_pending_added_tokens: int = 0


class BAGELUForwardBridge:
    """Maps safe SRT request views onto official BAGEL U context updates."""

    def __init__(
        self,
        *,
        srt_u_forward_executor: "BAGELSRTUForwardExecutor | None" = None,
    ) -> None:
        self.srt_u_forward_executor = (
            srt_u_forward_executor or BAGELSRTUForwardExecutor()
        )

    def observe(
        self,
        backend: "BAGELInterleaveContextBackend",
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        state = backend._state_for(session.handle.session_id)
        state.srt_u_forward_events.append((request.state, request.request_id))
        result = self.srt_u_forward_executor.execute(
            backend,
            session=session,
            request=request,
            messages=messages,
        )
        if result is not None:
            state.srt_u_forward_results[request.request_id] = result


class BAGELSRTUForwardExecutor:
    """Executes BAGEL U context updates from SRT request views."""

    def execute(
        self,
        backend: "BAGELInterleaveContextBackend",
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> UGModelPrefillResult | UGModelAppendImageResult | None:
        backend._bind_srt_request_tokens(request)

        if request.state == UGSegmentState.U_PREFILL.value:
            return backend._apply_prefill_interleaved(
                session=session,
                messages=messages,
            )

        if request.state == UGSegmentState.APPEND_IMAGE.value:
            image = messages[0].content if messages else None
            return backend._apply_append_generated_image(
                session=session,
                image=image,
            )

        if request.state == UGSegmentState.U_DECODE.value:
            state = backend._state_for(session.handle.session_id)
            state.srt_last_u_decode_output_ids = request.output_ids
        return None


class BAGELNativeSRTUForwardExecutor(BAGELSRTUForwardExecutor):
    """Consumes U-prefill views produced by native SRT BAGEL forward.

    This executor deliberately does not call official BAGEL
    `update_context_text`. It marks the session as SRT-owned so the G denoise
    path can reuse SRT KV through a native generation ForwardBatch.
    """

    def execute(
        self,
        backend: "BAGELInterleaveContextBackend",
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> UGModelPrefillResult | UGModelAppendImageResult | None:
        backend._bind_srt_request_tokens(request)

        state = backend._state_for(session.handle.session_id)
        backend._sync_native_srt_context_from_request(
            state,
            request=request,
            messages=messages,
        )
        added_tokens = int(request.metadata.get("ug_srt_added_token_count", 0))
        state.native_srt_pending_added_tokens += max(0, added_tokens)
        is_final_segment = bool(request.metadata.get("ug_srt_is_final_segment", True))
        if request.state == UGSegmentState.U_PREFILL.value:
            state.decode_count = 0
            state.prepared_denoise = None
            state.native_srt_u_context = True
            state.native_srt_u_context_request_id = request.request_id
            state.native_srt_u_context_token_binding = request.metadata.get(
                "srt_kv_token_binding"
            )
            if is_final_segment:
                total_added = state.native_srt_pending_added_tokens
                state.native_srt_pending_added_tokens = 0
                return UGModelPrefillResult(added_tokens=total_added)
            return None

        if request.state == UGSegmentState.APPEND_IMAGE.value:
            state.prepared_denoise = None
            state.native_srt_u_context = True
            state.native_srt_u_context_request_id = request.request_id
            state.native_srt_u_context_token_binding = request.metadata.get(
                "srt_kv_token_binding"
            )
            if is_final_segment:
                total_added = state.native_srt_pending_added_tokens
                state.native_srt_pending_added_tokens = 0
                state.append_image_count += 1
                return UGModelAppendImageResult(added_tokens=total_added)
            return None

        if request.state == UGSegmentState.U_DECODE.value:
            state.srt_last_u_decode_output_ids = request.output_ids
            return None
        return None


class BAGELInterleaveContextBackend:
    """Wraps a BAGEL inferencer or native-SRT shell behind UG adapter methods."""

    def __init__(
        self,
        inferencer: Any,
        *,
        step_runner: BAGELDenoiseStepRunner | None = None,
        u_forward_bridge: BAGELUForwardBridge | None = None,
        srt_kv_cache_factory: BAGELSRTKVCacheFactory | None = None,
        native_srt_denoise_executor: BAGELNativeSRTDenoiseExecutor | None = None,
        default_image_shape: tuple[int, int] = (1024, 1024),
    ) -> None:
        self.inferencer = inferencer
        self.step_runner = step_runner or BAGELDenoiseStepRunner()
        self.u_forward_bridge = u_forward_bridge or BAGELUForwardBridge()
        self.srt_kv_cache_factory = srt_kv_cache_factory
        self.native_srt_denoise_executor = native_srt_denoise_executor
        self.default_image_shape = default_image_shape
        self.sessions: dict[str, BAGELSessionContext] = {}

    def observe_srt_u_forward(
        self,
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        self.u_forward_bridge.observe(
            self,
            session=session,
            request=request,
            messages=messages,
        )

    def prepare_srt_u_message_inputs(
        self,
        *,
        session,
        message: UGInterleavedMessage,
        state: UGSegmentState,
    ) -> list[UGSRTPreparedInput] | None:
        if message.type != "image":
            return None
        if not self._uses_native_srt_u_forward():
            return None
        return self._prepare_native_srt_image_inputs(
            session=session,
            image=message.content,
            state=state,
        )

    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        cached = self._pop_srt_forward_result(session, UGModelPrefillResult)
        if cached is not None:
            return cached
        return self._apply_prefill_interleaved(session=session, messages=messages)

    def _apply_prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        if self._uses_native_srt_u_forward():
            raise BAGELAdapterError(
                "BAGEL native SRT U forward requires an SRT-executed prefill "
                "result; refusing to fall back to official BAGEL "
                "update_context_text/update_context_image"
            )
        state = self._state_for(session.handle.session_id)
        state.decode_count = 0
        added_tokens = 0
        for message in messages:
            if message.type == "text":
                text = str(message.content)
                state.cfg_text_context = self._clone_context(
                    state.gen_context,
                    session_id=session.handle.session_id,
                    role="cfg_text",
                )
                with _bagel_autocast(self.inferencer.model):
                    state.gen_context = self.inferencer.update_context_text(
                        text,
                        state.gen_context,
                    )
                    state.cfg_img_context = self.inferencer.update_context_text(
                        text,
                        state.cfg_img_context,
                    )
                added_tokens += len(text.split())
            elif message.type == "image":
                image = self._prepare_image(message.content)
                with _bagel_autocast(self.inferencer.model):
                    state.gen_context = self.inferencer.update_context_image(
                        image,
                        state.gen_context,
                        vae=True,
                        vit=True,
                    )
                state.image_shape = self._image_shape(image)
                state.cfg_text_context = self._clone_context(
                    state.gen_context,
                    session_id=session.handle.session_id,
                    role="cfg_text",
                )
                added_tokens += 2
            else:
                raise ValueError(f"Unsupported BAGEL message type: {message.type}")
            state.prepared_denoise = None
        return UGModelPrefillResult(added_tokens=added_tokens)

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        state = self._state_for(session.handle.session_id)
        if state.decode_count == 0:
            state.decode_count += 1
            return UGDecodeResult(type="image_marker")
        if state.append_image_count > 0 and state.decode_count == 1:
            state.decode_count += 1
            if state.native_srt_u_context:
                output_ids = state.srt_last_u_decode_output_ids
                text = " ".join(str(token_id) for token_id in output_ids)
                return UGDecodeResult(type="text", text=text)
            with _bagel_autocast(self.inferencer.model):
                text = self.inferencer.gen_text(
                    state.gen_context,
                    do_sample=False,
                    temperature=0.3,
                    max_length=512,
                )
            return UGDecodeResult(type="text", text=text)
        state.decode_count += 1
        return UGDecodeResult(type="done")

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        state = self._state_for(session.handle.session_id)
        if state.prepared_denoise is None:
            state.prepared_denoise = self._prepare_denoise(
                state, request.sampling_params
            )
        latent_tokens = request.latent_tokens
        timestep = request.timestep
        runtime_model = (
            self._native_srt_model()
            if isinstance(state.prepared_denoise, BAGELNativeSRTPreparedDenoise)
            else self.inferencer.model
        )
        runtime_device = _bagel_runtime_device(runtime_model)
        if runtime_device is not None:
            latent_tokens = latent_tokens.to(runtime_device)
            timestep = timestep.to(runtime_device)

        with torch.autocast(
            device_type="cuda",
            enabled=latent_tokens.is_cuda,
            dtype=torch.bfloat16,
        ):
            if isinstance(state.prepared_denoise, BAGELNativeSRTPreparedDenoise):
                if self.native_srt_denoise_executor is None:
                    raise BAGELAdapterError(
                        "BAGEL native SRT U context requires a native SRT denoise "
                        "executor"
                    )
                velocity = self.native_srt_denoise_executor.predict_velocity(
                    prepared=state.prepared_denoise,
                    latent_tokens=latent_tokens,
                    timestep=timestep,
                )
            else:
                velocity = self.step_runner.predict_velocity(
                    model=self.inferencer.model,
                    prepared=state.prepared_denoise,
                    latent_tokens=latent_tokens,
                    timestep=timestep,
                )
        return velocity.to(request.latent_tokens.device)

    def prepare_latents_from_session(
        self, *, session, request: UGLatentPrepareRequest
    ) -> UGLatentPrepareResult | None:
        state = self._state_for(session.handle.session_id)
        if state.prepared_denoise is None:
            state.prepared_denoise = self._prepare_denoise(
                state,
                request.sampling_params,
                seed=request.seed,
            )

        generation_input = state.prepared_denoise.generation_input
        latent_tokens = generation_input.get("packed_init_noises")
        latent_position_ids = generation_input.get("packed_vae_position_ids")
        if latent_tokens is None or latent_position_ids is None:
            raise BAGELDenoiseStepError(
                "BAGEL prepare_vae_latent did not return packed_init_noises "
                "and packed_vae_position_ids"
            )

        image_shape = self._image_shape_from_params(
            request.sampling_params,
            state.image_shape,
        )
        latent_shape_model = (
            self._native_srt_model()
            if isinstance(state.prepared_denoise, BAGELNativeSRTPreparedDenoise)
            else self.inferencer.model
        )
        latent_shape = _bagel_latent_shape(
            latent_shape_model, image_shape, latent_tokens
        )
        return UGLatentPrepareResult(
            latent_tokens=latent_tokens,
            latent_position_ids=latent_position_ids,
            latent_shape=latent_shape,
        )

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        cached = self._pop_srt_forward_result(session, UGModelAppendImageResult)
        if cached is not None:
            return cached
        return self._apply_append_generated_image(session=session, image=image)

    def _apply_append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        if self._uses_native_srt_u_forward():
            raise BAGELAdapterError(
                "BAGEL native SRT U forward requires an SRT-executed append-image "
                "result; refusing to fall back to official BAGEL update_context_image"
            )
        state = self._state_for(session.handle.session_id)
        image = self._prepare_image(image)
        with _bagel_autocast(self.inferencer.model):
            state.gen_context = self.inferencer.update_context_image(
                image,
                state.gen_context,
                vae=True,
                vit=True,
            )
        state.image_shape = self._image_shape(image)
        state.cfg_text_context = self._clone_context(
            state.gen_context,
            session_id=session.handle.session_id,
            role="cfg_text",
        )
        state.append_image_count += 1
        state.prepared_denoise = None
        return UGModelAppendImageResult(added_tokens=2)

    def _pop_srt_forward_result(self, session, result_type):
        request_id = session.srt_last_request_id
        if request_id is None:
            return None
        state = self._state_for(session.handle.session_id)
        result = state.srt_u_forward_results.pop(request_id, None)
        if result is None:
            return None
        if not isinstance(result, result_type):
            raise BAGELAdapterError(
                "BAGEL SRT U forward result type mismatch for "
                f"{request_id}: expected {result_type.__name__}, got "
                f"{type(result).__name__}"
            )
        return result

    def decode_latents_to_image(
        self, *, session, request: UGLatentDecodeRequest
    ) -> Any | None:
        state = self._state_for(session.handle.session_id)
        image_shape = self._image_shape_from_params(
            request.sampling_params,
            state.image_shape,
        )
        latent_tokens = request.latent_tokens
        if latent_tokens.ndim == 3:
            latent_tokens = latent_tokens[0]

        if state.native_srt_u_context:
            decoder = getattr(self._native_srt_model(), "decode_bagel_image", None)
            if not callable(decoder):
                raise BAGELAdapterError(
                    "BAGEL native SRT image decode requires decode_bagel_image "
                    "on the SRT model"
                )
            return decoder(latent_tokens, image_shape)

        vae_model = getattr(self.inferencer, "vae_model", None)
        vae_device = _bagel_runtime_device(vae_model)
        if vae_device is not None:
            latent_tokens = latent_tokens.to(vae_device)

        vae_dtype = _bagel_runtime_dtype(vae_model)
        if vae_dtype is not None and latent_tokens.is_floating_point():
            latent_tokens = latent_tokens.to(dtype=vae_dtype)

        return self.inferencer.decode_image(latent_tokens, image_shape)

    def close_session(self, *, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        if self.srt_kv_cache_factory is not None:
            self.srt_kv_cache_factory.release_session(session_id)

    def _state_for(self, session_id: str) -> BAGELSessionContext:
        state = self.sessions.get(session_id)
        if state is not None:
            return state
        gen_context = self._init_context(session_id=session_id, role="full")
        state = BAGELSessionContext(
            gen_context=gen_context,
            cfg_text_context=self._clone_context(
                gen_context,
                session_id=session_id,
                role="cfg_text",
            ),
            cfg_img_context=self._clone_context(
                gen_context,
                session_id=session_id,
                role="cfg_img",
            ),
            image_shape=self.default_image_shape,
        )
        self.sessions[session_id] = state
        return state

    def _init_context(self, *, session_id: str, role: str) -> dict[str, Any]:
        context = self.inferencer.init_gen_context()
        if self.srt_kv_cache_factory is None:
            return context
        context = dict(context)
        context["past_key_values"] = self.srt_kv_cache_factory.create_cache(
            session_id=session_id,
            role=role,
            template_cache=context["past_key_values"],
        )
        return context

    def _clone_context(
        self,
        context: dict[str, Any],
        *,
        session_id: str,
        role: str,
    ) -> dict[str, Any]:
        if self.srt_kv_cache_factory is None:
            return _clone_context(context)
        cloned = {
            key: deepcopy(value)
            for key, value in context.items()
            if key != "past_key_values"
        }
        cloned["past_key_values"] = self.srt_kv_cache_factory.clone_cache(
            context["past_key_values"],
            session_id=session_id,
            role=role,
        )
        return cloned

    def _bind_srt_request_tokens(self, request: UGSRTRequestView) -> None:
        if self.srt_kv_cache_factory is None:
            return
        binding = request.metadata.get("srt_kv_token_binding")
        if binding is not None:
            self.srt_kv_cache_factory.bind_request_tokens(binding)

    def _prepare_denoise(
        self,
        state: BAGELSessionContext,
        sampling_params: Any | None,
        *,
        seed: int | None = None,
    ) -> BAGELPreparedDenoise | BAGELNativeSRTPreparedDenoise:
        if state.native_srt_u_context:
            return self._prepare_native_srt_denoise(
                state,
                sampling_params,
                seed=seed,
            )
        image_shape = self._image_shape_from_params(sampling_params, state.image_shape)
        model = self.inferencer.model
        with _bagel_seed_context(seed):
            generation_input = model.prepare_vae_latent(
                curr_kvlens=state.gen_context["kv_lens"],
                curr_rope=state.gen_context["ropes"],
                image_sizes=[image_shape],
                new_token_ids=self.inferencer.new_token_ids,
            )
        cfg_text_generation_input = model.prepare_vae_latent_cfg(
            curr_kvlens=state.cfg_text_context["kv_lens"],
            curr_rope=state.cfg_text_context["ropes"],
            image_sizes=[image_shape],
        )
        cfg_img_generation_input = model.prepare_vae_latent_cfg(
            curr_kvlens=state.cfg_img_context["kv_lens"],
            curr_rope=state.cfg_img_context["ropes"],
            image_sizes=[image_shape],
        )
        return BAGELPreparedDenoise(
            generation_input=generation_input,
            cfg_text_generation_input=cfg_text_generation_input,
            cfg_img_generation_input=cfg_img_generation_input,
            past_key_values=state.gen_context["past_key_values"],
            cfg_text_past_key_values=state.cfg_text_context["past_key_values"],
            cfg_img_past_key_values=state.cfg_img_context["past_key_values"],
            cfg_text_scale=float(getattr(sampling_params, "cfg_text_scale", 4.0)),
            cfg_img_scale=float(getattr(sampling_params, "cfg_img_scale", 1.5)),
            cfg_interval=tuple(getattr(sampling_params, "cfg_interval", (0.4, 1.0))),
            cfg_renorm_min=float(getattr(sampling_params, "cfg_renorm_min", 0.0)),
            cfg_renorm_type=getattr(sampling_params, "cfg_renorm_type", "global"),
        )

    def _prepare_native_srt_denoise(
        self,
        state: BAGELSessionContext,
        sampling_params: Any | None,
        *,
        seed: int | None = None,
    ) -> BAGELNativeSRTPreparedDenoise:
        if self.native_srt_denoise_executor is None:
            raise BAGELAdapterError(
                "BAGEL native SRT U context requires a native SRT denoise executor"
            )
        image_shape = self._image_shape_from_params(sampling_params, state.image_shape)
        binding = self._native_srt_token_binding(state)
        curr_kvlens, curr_rope = self._native_srt_curr_lengths(state)
        model = self._native_srt_model()
        with _bagel_seed_context(seed):
            generation_input = model.prepare_vae_latent(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                image_sizes=[image_shape],
                new_token_ids=self.inferencer.new_token_ids,
            )
        return BAGELNativeSRTPreparedDenoise(
            generation_input=generation_input,
            srt_kv_token_binding=binding,
            cfg_text_scale=float(getattr(sampling_params, "cfg_text_scale", 1.0)),
            cfg_img_scale=float(getattr(sampling_params, "cfg_img_scale", 1.0)),
            cfg_interval=tuple(getattr(sampling_params, "cfg_interval", (0.0, 1.0))),
            cfg_renorm_min=float(getattr(sampling_params, "cfg_renorm_min", 0.0)),
            cfg_renorm_type=getattr(sampling_params, "cfg_renorm_type", "global"),
        )

    def _prepare_native_srt_image_inputs(
        self,
        *,
        session,
        image: Any | None,
        state: UGSegmentState,
    ) -> list[UGSRTPreparedInput]:
        model = self._native_srt_model()
        if not hasattr(model, "prepare_vae_images") or not hasattr(
            model, "prepare_vit_images"
        ):
            raise BAGELAdapterError(
                "Native SRT BAGEL image U forward requires "
                "prepare_vae_images/prepare_vit_images on the SRT model"
            )

        session_state = self._state_for(session.handle.session_id)
        image = self._prepare_image(image)
        session_state.image_shape = self._image_shape(image)
        curr_kvlens, curr_rope = self._native_srt_curr_lengths(session_state)

        chunks: list[UGSRTPreparedInput] = []
        with _bagel_autocast(model):
            vae_input, curr_kvlens, curr_rope = model.prepare_vae_images(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                images=[image],
                transforms=self.inferencer.vae_transform,
                new_token_ids=self.inferencer.new_token_ids,
            )
            chunks.append(
                self._native_srt_prepared_input_from_packed_sequence(
                    generation_input=vae_input,
                    input_embeds=self._embed_native_srt_vae_image(vae_input),
                    message_image=image,
                    stage="vae",
                    state=state,
                    text_token_key="packed_text_indexes",
                    vae_token_key="packed_vae_token_indexes",
                    replace_token_key="packed_vae_token_indexes",
                )
            )

            vit_input, curr_kvlens, curr_rope = model.prepare_vit_images(
                curr_kvlens=curr_kvlens,
                curr_rope=curr_rope,
                images=[image],
                transforms=self.inferencer.vit_transform,
                new_token_ids=self.inferencer.new_token_ids,
            )
            chunks.append(
                self._native_srt_prepared_input_from_packed_sequence(
                    generation_input=vit_input,
                    input_embeds=self._embed_native_srt_vit_image(vit_input),
                    message_image=image,
                    stage="vit",
                    state=state,
                    text_token_key="packed_text_indexes",
                    vae_token_key=None,
                    replace_token_key="packed_vit_token_indexes",
                )
            )

        session_state.gen_context["kv_lens"] = curr_kvlens
        session_state.gen_context["ropes"] = curr_rope
        session_state.cfg_text_context = self._clone_context(
            session_state.gen_context,
            session_id=session.handle.session_id,
            role="cfg_text",
        )
        session_state.prepared_denoise = None
        return chunks

    def _native_srt_prepared_input_from_packed_sequence(
        self,
        *,
        generation_input: dict[str, Any],
        input_embeds: torch.Tensor,
        message_image: Any,
        stage: str,
        state: UGSegmentState,
        text_token_key: str,
        vae_token_key: str | None,
        replace_token_key: str,
    ) -> UGSRTPreparedInput:
        input_embeds = input_embeds.detach().to(dtype=torch.float32, device="cpu")
        seq_len = int(input_embeds.shape[0])
        input_ids = [0] * seq_len
        packed_text_ids = generation_input["packed_text_ids"].to("cpu").tolist()
        packed_text_indexes = generation_input[text_token_key].to("cpu").tolist()
        for token_id, index in zip(packed_text_ids, packed_text_indexes):
            input_ids[int(index)] = int(token_id)

        text_indices = [int(index) for index in packed_text_indexes]
        vae_indices = (
            [int(index) for index in generation_input[vae_token_key].to("cpu").tolist()]
            if vae_token_key is not None
            else None
        )
        replace_positions = [
            int(index)
            for index in generation_input[replace_token_key].to("cpu").tolist()
        ]
        replace_embeds = input_embeds[replace_positions].tolist()
        metadata = {
            "bagel_u_image_stage": stage,
            "ug_srt_added_token_count": seq_len,
            "ug_srt_bagel_rope_delta": 0,
        }
        return UGSRTPreparedInput(
            input_ids=input_ids,
            input_text=f"<bagel:{stage}:image>",
            messages=[UGInterleavedMessage(type="image", content=message_image)],
            replace_embeds=replace_embeds,
            replace_positions=replace_positions,
            position_ids=[
                int(position)
                for position in generation_input["packed_position_ids"]
                .to("cpu")
                .tolist()
            ],
            non_causal_query_attention=True,
            bagel_text_token_indices=text_indices,
            bagel_vae_token_indices=vae_indices,
            adapter_metadata=metadata,
        )

    def _embed_native_srt_vae_image(
        self, generation_input: dict[str, Any]
    ) -> torch.Tensor:
        embedder = getattr(self._native_srt_model(), "embed_bagel_vae_image", None)
        if not callable(embedder):
            raise BAGELAdapterError(
                "BAGEL native SRT image U forward requires embed_bagel_vae_image "
                "on the SRT model"
            )
        return embedder(generation_input)

    def _embed_native_srt_vit_image(
        self, generation_input: dict[str, Any]
    ) -> torch.Tensor:
        embedder = getattr(self._native_srt_model(), "embed_bagel_vit_image", None)
        if not callable(embedder):
            raise BAGELAdapterError(
                "BAGEL native SRT image U forward requires embed_bagel_vit_image "
                "on the SRT model"
            )
        return embedder(generation_input)

    def _native_srt_curr_lengths(
        self,
        state: BAGELSessionContext,
    ) -> tuple[list[int], list[int]]:
        binding = state.native_srt_u_context_token_binding
        if binding is not None:
            token_count = int(binding.token_count)
            ropes = state.gen_context.get("ropes") or [0]
            return [token_count], [int(ropes[0])]
        kv_lens = state.gen_context.get("kv_lens") or [0]
        ropes = state.gen_context.get("ropes") or [0]
        return [int(kv_lens[0])], [int(ropes[0])]

    def _sync_native_srt_context_from_request(
        self,
        state: BAGELSessionContext,
        *,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        binding = request.metadata.get("srt_kv_token_binding")
        if binding is not None:
            state.native_srt_u_context_token_binding = binding
            state.gen_context["kv_lens"] = [int(binding.token_count)]

        rope_delta = self._native_srt_rope_delta(request=request, messages=messages)
        if rope_delta:
            ropes = state.gen_context.get("ropes") or [0]
            state.gen_context["ropes"] = [int(ropes[0]) + rope_delta]

    @staticmethod
    def _native_srt_rope_delta(
        *,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> int:
        explicit = request.metadata.get("ug_srt_bagel_rope_delta")
        if explicit is not None:
            return int(explicit)
        if request.state == UGSegmentState.U_DECODE.value:
            return 1 if request.output_ids else 0
        if request.state in {
            UGSegmentState.U_PREFILL.value,
            UGSegmentState.APPEND_IMAGE.value,
        }:
            return len(messages)
        return 0

    def _uses_native_srt_u_forward(self) -> bool:
        return isinstance(
            self.u_forward_bridge.srt_u_forward_executor,
            BAGELNativeSRTUForwardExecutor,
        )

    def _native_srt_model(self) -> Any:
        if self.native_srt_denoise_executor is None:
            raise BAGELAdapterError(
                "BAGEL native SRT path requires a native SRT denoise executor"
            )
        srt_model = getattr(self.native_srt_denoise_executor, "srt_model", None)
        if srt_model is None:
            raise BAGELAdapterError(
                "BAGEL native SRT denoise executor does not expose an SRT model"
            )
        return srt_model

    @staticmethod
    def _native_srt_token_binding(state: BAGELSessionContext) -> Any:
        binding = state.native_srt_u_context_token_binding
        if binding is None:
            raise BAGELAdapterError(
                "BAGEL native SRT U context is missing SRT token binding"
            )
        token_count = int(binding.token_count)
        if token_count <= 0:
            raise BAGELAdapterError("BAGEL native SRT U context token binding is empty")
        return binding

    def _prepare_image(self, image: Any | None) -> Any | None:
        if image is None:
            return None
        transform = getattr(self.inferencer, "vae_transform", None)
        if transform is None:
            return image
        resize_transform = getattr(transform, "resize_transform", None)
        if resize_transform is None:
            return image
        return resize_transform(image)

    def _image_shape(self, image: Any | None) -> tuple[int, int]:
        size = getattr(image, "size", None)
        if isinstance(size, tuple) and len(size) == 2:
            width, height = size
            return int(height), int(width)
        return self.default_image_shape

    @staticmethod
    def _image_shape_from_params(
        sampling_params: Any | None,
        default: tuple[int, int],
    ) -> tuple[int, int]:
        if sampling_params is None:
            return default
        height = getattr(sampling_params, "height", None) or default[0]
        width = getattr(sampling_params, "width", None) or default[1]
        return int(height), int(width)


class BAGELBackendProtocol(Protocol):
    def prepare_srt_u_message_inputs(
        self,
        *,
        session,
        message: UGInterleavedMessage,
        state: UGSegmentState,
    ) -> list[UGSRTPreparedInput] | None: ...

    def observe_srt_u_forward(
        self,
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None: ...

    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult: ...

    def decode_next_segment(self, *, session) -> UGDecodeResult: ...

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor: ...

    def prepare_latents_from_session(
        self, *, session, request: UGLatentPrepareRequest
    ) -> UGLatentPrepareResult | None: ...

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult: ...

    def decode_latents_to_image(
        self, *, session, request: UGLatentDecodeRequest
    ) -> Any | None: ...

    def close_session(self, *, session_id: str) -> None: ...


class BAGELUGModelAdapter(UGModelAdapterProtocol):
    """BAGEL-facing UG adapter shell.

    The adapter can use either a deterministic mock backend for smoke tests or
    the official BAGEL modules behind the SRT-owned UG runtime. In the native
    SRT path, the shell keeps tokenizer/transforms only; SRT owns feature
    extractors, session requests, KV cache, and per-step G velocity execution.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: BAGELBackendProtocol | None = None,
        native_srt_denoise_executor: BAGELNativeSRTDenoiseExecutor | None = None,
        native_srt_u_context: bool | None = None,
    ) -> None:
        self.model_path = model_path
        if native_srt_u_context is None:
            native_srt_u_context = native_srt_denoise_executor is not None
        self.backend = backend or self._load_real_backend(
            model_path,
            native_srt_denoise_executor=native_srt_denoise_executor,
            native_srt_u_context=native_srt_u_context,
        )

    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        return self.backend.prefill_interleaved(session=session, messages=messages)

    def prepare_srt_u_message_inputs(
        self,
        *,
        session,
        message: UGInterleavedMessage,
        state: UGSegmentState,
    ) -> list[UGSRTPreparedInput] | None:
        prepare = getattr(self.backend, "prepare_srt_u_message_inputs", None)
        if not callable(prepare):
            return None
        return prepare(session=session, message=message, state=state)

    def observe_srt_u_forward(
        self,
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        observe = getattr(self.backend, "observe_srt_u_forward", None)
        if callable(observe):
            observe(session=session, request=request, messages=messages)

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        return self.backend.decode_next_segment(session=session)

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        return self.backend.predict_velocity_from_session(
            session=session,
            request=request,
        )

    def prepare_latents_from_session(
        self, *, session, request: UGLatentPrepareRequest
    ) -> UGLatentPrepareResult | None:
        return self.backend.prepare_latents_from_session(
            session=session,
            request=request,
        )

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        return self.backend.append_generated_image(session=session, image=image)

    def decode_latents_to_image(
        self, *, session, request: UGLatentDecodeRequest
    ) -> Any | None:
        return self.backend.decode_latents_to_image(
            session=session,
            request=request,
        )

    def close_session(self, *, session_id: str) -> None:
        self.backend.close_session(session_id=session_id)

    @staticmethod
    def _load_real_backend(
        model_path: str,
        *,
        native_srt_denoise_executor: BAGELNativeSRTDenoiseExecutor | None = None,
        native_srt_u_context: bool = False,
    ) -> BAGELBackendProtocol:
        checkpoint_dir = Path(model_path).expanduser()
        if not checkpoint_dir.exists():
            raise BAGELAdapterError(
                "BAGELUGModelAdapter requires a local BAGEL checkpoint directory. "
                "Download ByteDance-Seed/BAGEL-7B-MoT first, then pass the local "
                "directory path; use sglang-internal/mock-bagel for adapter smoke "
                "tests."
            )
        missing_files = [
            name
            for name in _BAGEL_REQUIRED_CHECKPOINT_FILES
            if not (checkpoint_dir / name).exists()
        ]
        if missing_files:
            raise BAGELAdapterError(
                "BAGEL checkpoint is missing required files: "
                f"{missing_files}. Expected a local ByteDance-Seed/BAGEL-7B-MoT "
                "checkout with the official config and weight files."
            )

        required_modules = (
            _BAGEL_NATIVE_SRT_REQUIRED_MODULES
            if native_srt_u_context
            else _BAGEL_REQUIRED_MODULES
        )
        missing_modules = [
            name for name in required_modules if _find_spec(name) is None
        ]
        if missing_modules:
            raise BAGELAdapterError(
                "BAGEL Python modules are not importable: "
                f"{missing_modules}. Add the official ByteDance-Seed/BAGEL repo "
                "to PYTHONPATH or vendor the required model code before enabling "
                "the real BAGEL backend."
            )

        if native_srt_u_context:
            inferencer = _build_native_srt_bagel_inferencer_shell(checkpoint_dir)
        else:
            inferencer = _build_official_bagel_inferencer(checkpoint_dir)
        u_forward_bridge = None
        if native_srt_u_context:
            u_forward_bridge = BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            )
        return BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=u_forward_bridge,
            native_srt_denoise_executor=native_srt_denoise_executor,
        )


class MockBAGELBackend:
    """Deterministic BAGEL-shaped backend for adapter and pipeline smoke tests."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.decode_counts: defaultdict[str, int] = defaultdict(int)
        self.closed_sessions: list[str] = []

    def prefill_interleaved(
        self, *, session, messages: list[UGInterleavedMessage]
    ) -> UGModelPrefillResult:
        self._record("prefill", session)
        token_count = 0
        for message in messages:
            if message.type == "text":
                token_count += len(str(message.content).split())
            elif message.type == "image":
                token_count += 2
            else:
                raise ValueError(f"Unsupported BAGEL message type: {message.type}")
        return UGModelPrefillResult(added_tokens=token_count)

    def prepare_srt_u_message_inputs(
        self,
        *,
        session,
        message: UGInterleavedMessage,
        state: UGSegmentState,
    ) -> list[UGSRTPreparedInput] | None:
        del session, message, state
        return None

    def observe_srt_u_forward(
        self,
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        del messages
        self.events.append((f"srt_{request.state}", session.handle.session_id))

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        self._record("decode", session)
        session_id = session.handle.session_id
        decode_count = self.decode_counts[session_id]
        self.decode_counts[session_id] += 1
        if decode_count == 0:
            return UGDecodeResult(type="image_marker")
        if decode_count == 1:
            return UGDecodeResult(type="text", text="bagel_mock_text_after_image")
        return UGDecodeResult(type="done")

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        self._record("velocity", session)
        scale = 2.0 + session.srt_request_count * 0.1
        return request.latent_tokens + scale * request.timestep.reshape(-1, 1, 1).to(
            request.latent_tokens
        )

    def prepare_latents_from_session(
        self, *, session, request: UGLatentPrepareRequest
    ) -> UGLatentPrepareResult | None:
        del request
        self._record("prepare_latents", session)
        return None

    def append_generated_image(
        self, *, session, image: Any | None
    ) -> UGModelAppendImageResult:
        del image
        self._record("append_image", session)
        return UGModelAppendImageResult(added_tokens=2)

    def decode_latents_to_image(
        self, *, session, request: UGLatentDecodeRequest
    ) -> Any | None:
        del request
        self._record("decode_latents", session)
        return None

    def close_session(self, *, session_id: str) -> None:
        self.events.append(("close", session_id))
        self.closed_sessions.append(session_id)

    def _record(self, event: str, session) -> None:
        self.events.append((event, session.handle.session_id))


def create_bagel_ug_model_adapter(
    model_path: str,
    *,
    native_srt_denoise_executor: BAGELNativeSRTDenoiseExecutor | None = None,
    native_srt_u_context: bool | None = None,
) -> BAGELUGModelAdapter:
    if "mock-bagel" in model_path.lower():
        return BAGELUGModelAdapter(model_path, backend=MockBAGELBackend())
    if native_srt_u_context is None:
        native_srt_u_context = native_srt_denoise_executor is not None
    return BAGELUGModelAdapter(
        model_path,
        native_srt_denoise_executor=native_srt_denoise_executor,
        native_srt_u_context=native_srt_u_context,
    )


def _require_keys(
    payload: dict[str, Any], required: tuple[str, ...], name: str
) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise BAGELDenoiseStepError(f"{name} is missing required keys: {missing}")


def _clone_context(context: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(context)


def _bagel_runtime_device(model: Any) -> torch.device | None:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return None
    return parameter.device


def _bagel_runtime_dtype(model: Any) -> torch.dtype | None:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return None
    return parameter.dtype


def _bagel_latent_shape(
    model: Any,
    image_shape: tuple[int, int],
    latent_tokens: torch.Tensor,
) -> tuple[int, int, int]:
    height, width = image_shape
    latent_downsample = int(getattr(model, "latent_downsample", 16))
    latent_height = height // latent_downsample
    latent_width = width // latent_downsample
    latent_dim = int(latent_tokens.shape[-1])
    return latent_height, latent_width, latent_dim


@contextmanager
def _bagel_seed_context(seed: int | None):
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


def _bagel_autocast(model: Any):
    device = _bagel_runtime_device(model)
    if device is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
    return nullcontext()


def _build_official_bagel_inferencer(
    checkpoint_dir: Path,
    *,
    loader_symbols: dict[str, Any] | None = None,
) -> Any:
    if torch.cuda.device_count() < 1:
        raise BAGELAdapterError(
            "Real BAGEL backend requires at least one CUDA device. "
            "Use sglang-internal/mock-bagel for CPU-only adapter tests."
        )

    _ensure_bagel_transformers_compat()
    symbols = loader_symbols or _import_bagel_loader_symbols()

    llm_config = symbols["Qwen2Config"].from_json_file(
        str(checkpoint_dir / "llm_config.json")
    )
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = getattr(llm_config, "eos_token_id", None)

    vit_config = symbols["SiglipVisionConfig"].from_json_file(
        str(checkpoint_dir / "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = symbols["load_ae"](
        local_path=str(checkpoint_dir / "ae.safetensors")
    )
    config = symbols["BagelConfig"](
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with symbols["init_empty_weights"]():
        language_model = symbols["Qwen2ForCausalLM"](llm_config)
        vit_model = symbols["SiglipVisionModel"](vit_config)
        model = symbols["Bagel"](language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config,
            meta=True,
        )

    tokenizer = symbols["Qwen2Tokenizer"].from_pretrained(str(checkpoint_dir))
    tokenizer, new_token_ids, _ = symbols["add_special_tokens"](tokenizer)

    vae_transform = symbols["ImageTransform"](1024, 512, 16)
    vit_transform = symbols["ImageTransform"](980, 224, 14)

    device_map = symbols["infer_auto_device_map"](
        model,
        max_memory={
            device_id: "80GiB" for device_id in range(torch.cuda.device_count())
        },
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    device_map = _pin_bagel_shared_modules(device_map)

    model = symbols["load_checkpoint_and_dispatch"](
        model,
        checkpoint=str(checkpoint_dir / "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder=str(checkpoint_dir / "offload"),
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    return symbols["InterleaveInferencer"](
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


class BAGELNativeSRTInferencerShell:
    """Tokenizer/transform shell for native SRT BAGEL execution.

    The shell intentionally does not load the official BAGEL model or VAE/VIT
    feature extractors. Native execution gets those modules from the SRT model
    instance loaded by ModelRunner.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        vae_transform: Any,
        vit_transform: Any,
        new_token_ids: dict[str, int],
    ) -> None:
        self.model = None
        self.vae_model = None
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def init_gen_context(self) -> dict[str, Any]:
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": None,
        }

    def update_context_text(self, *args, **kwargs):
        del args, kwargs
        raise BAGELAdapterError(
            "Native SRT BAGEL shell cannot update context through official BAGEL"
        )

    def update_context_image(self, *args, **kwargs):
        del args, kwargs
        raise BAGELAdapterError(
            "Native SRT BAGEL shell cannot append images through official BAGEL"
        )

    def decode_image(self, *args, **kwargs):
        del args, kwargs
        raise BAGELAdapterError(
            "Native SRT BAGEL shell cannot decode through official BAGEL"
        )

    def gen_text(self, *args, **kwargs):
        del args, kwargs
        raise BAGELAdapterError(
            "Native SRT BAGEL shell cannot decode text through official BAGEL"
        )


def _build_native_srt_bagel_inferencer_shell(
    checkpoint_dir: Path,
    *,
    loader_symbols: dict[str, Any] | None = None,
) -> BAGELNativeSRTInferencerShell:
    _ensure_bagel_transformers_compat()
    symbols = loader_symbols or _import_bagel_native_srt_shell_symbols()

    tokenizer = symbols["Qwen2Tokenizer"].from_pretrained(str(checkpoint_dir))
    tokenizer, new_token_ids, _ = symbols["add_special_tokens"](tokenizer)
    return BAGELNativeSRTInferencerShell(
        tokenizer=tokenizer,
        vae_transform=symbols["ImageTransform"](1024, 512, 16),
        vit_transform=symbols["ImageTransform"](980, 224, 14),
        new_token_ids=new_token_ids,
    )


def _import_bagel_native_srt_shell_symbols() -> dict[str, Any]:
    data_utils = _import_module("data.data_utils")
    transforms = _import_module("data.transforms")
    qwen2 = _import_module("modeling.qwen2")

    return {
        "add_special_tokens": _module_attr(data_utils, "add_special_tokens"),
        "ImageTransform": _module_attr(transforms, "ImageTransform"),
        "Qwen2Tokenizer": _module_attr(qwen2, "Qwen2Tokenizer"),
    }


def _pin_bagel_shared_modules(device_map: dict[str, Any]) -> dict[str, Any]:
    device_map = dict(device_map)
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(_BAGEL_SAME_DEVICE_MODULES[0], "cuda:0")
        for module_name in _BAGEL_SAME_DEVICE_MODULES:
            device_map[module_name] = first_device
        return device_map

    first_device = device_map.get(_BAGEL_SAME_DEVICE_MODULES[0])
    if first_device is None:
        return device_map
    for module_name in _BAGEL_SAME_DEVICE_MODULES:
        if module_name in device_map:
            device_map[module_name] = first_device
    return device_map


def _import_bagel_loader_symbols() -> dict[str, Any]:
    accelerate = _import_module("accelerate")
    data_utils = _import_module("data.data_utils")
    transforms = _import_module("data.transforms")
    inferencer_module = _import_module("inferencer")
    autoencoder = _import_module("modeling.autoencoder")
    bagel = _import_module("modeling.bagel")
    qwen2 = _import_module("modeling.qwen2")

    return {
        "infer_auto_device_map": _module_attr(accelerate, "infer_auto_device_map"),
        "load_checkpoint_and_dispatch": _module_attr(
            accelerate,
            "load_checkpoint_and_dispatch",
        ),
        "init_empty_weights": _module_attr(accelerate, "init_empty_weights"),
        "add_special_tokens": _module_attr(data_utils, "add_special_tokens"),
        "ImageTransform": _module_attr(transforms, "ImageTransform"),
        "InterleaveInferencer": _module_attr(
            inferencer_module,
            "InterleaveInferencer",
        ),
        "load_ae": _module_attr(autoencoder, "load_ae"),
        "BagelConfig": _module_attr(bagel, "BagelConfig"),
        "Bagel": _module_attr(bagel, "Bagel"),
        "Qwen2Config": _module_attr(bagel, "Qwen2Config"),
        "Qwen2ForCausalLM": _module_attr(bagel, "Qwen2ForCausalLM"),
        "SiglipVisionConfig": _module_attr(bagel, "SiglipVisionConfig"),
        "SiglipVisionModel": _module_attr(bagel, "SiglipVisionModel"),
        "Qwen2Tokenizer": _module_attr(qwen2, "Qwen2Tokenizer"),
    }


def _ensure_bagel_transformers_compat() -> None:
    try:
        rope_utils = importlib.import_module("transformers.modeling_rope_utils")
    except ImportError:
        return

    rope_init_functions = getattr(rope_utils, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(rope_init_functions, dict) or "default" in rope_init_functions:
        return

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
        del seq_len, kwargs
        base = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(
            config,
            "head_dim",
            getattr(config, "hidden_size") // getattr(config, "num_attention_heads"),
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        return inv_freq, 1.0

    rope_init_functions["default"] = _compute_default_rope_parameters


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise BAGELAdapterError(
            f"BAGEL Python module is not importable: {module_name}. "
            "Add the official ByteDance-Seed/BAGEL repo to PYTHONPATH before "
            "enabling the real BAGEL backend."
        ) from exc


def _module_attr(module: Any, attr_name: str) -> Any:
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise BAGELAdapterError(
            f"BAGEL Python module {module.__name__} is missing required symbol "
            f"{attr_name}."
        ) from exc


def _find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
