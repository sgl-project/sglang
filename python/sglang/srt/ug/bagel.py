# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import torch
from transformers import AutoTokenizer

from sglang.srt.ug.adapter import (
    UGModelAdapterProtocol,
    UGModelAppendImageResult,
    UGModelPrefillResult,
)
from sglang.srt.ug.bagel_cache import BAGELSRTKVCacheFactory
from sglang.srt.ug.bagel_preprocess import (
    BAGELImageTransform,
    add_bagel_special_tokens,
)
from sglang.srt.ug.context import UGSRTKVTokenBinding, UGSRTRequestView
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGLatentDecodeRequest,
    UGLatentPrepareRequest,
    UGLatentPrepareResult,
    UGSegmentState,
    UGSRTPreparedInput,
    UGVelocityRequest,
    UGVLMTextGenerationResult,
)
from sglang.srt.ug.sampling import get_bagel_effective_cfg_scales

_BAGEL_REQUIRED_CHECKPOINT_FILES = (
    "llm_config.json",
    "vit_config.json",
    "ae.safetensors",
    "ema.safetensors",
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


class BAGELAdapterError(RuntimeError):
    """Raised when the BAGEL adapter cannot be constructed safely."""


class BAGELDenoiseStepError(RuntimeError):
    """Raised when a BAGEL single-step denoise call is malformed."""


@dataclass
class BAGELNativeSRTPreparedDenoise:
    """Denoise inputs whose U context is owned by SRT KV cache."""

    generation_input: dict[str, Any]
    srt_kv_token_binding: Any | None = None
    cfg_text_generation_input: dict[str, Any] | None = None
    cfg_text_srt_kv_token_binding: Any | None = None
    cfg_img_generation_input: dict[str, Any] | None = None
    cfg_img_srt_kv_token_binding: Any | None = None
    cfg_text_scale: float = 1.0
    cfg_img_scale: float = 1.0
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    cfg_type: str = "parallel"


class BAGELNativeSRTDenoiseExecutor:
    """Calls SRT-native BAGEL gen forward through the ModelRunner path."""

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
        with torch.no_grad():
            cfg_text_scale, cfg_img_scale = _effective_cfg_scales(prepared, timestep)
            self.velocity_count += 1
            v_t = self._predict_velocity_branch(
                prepared=prepared,
                branch_name="full",
                generation_input=prepared.generation_input,
                srt_kv_token_binding=prepared.srt_kv_token_binding,
                latent_tokens=latent_tokens,
                timestep=timestep,
            )
            if cfg_text_scale <= 1.0:
                return v_t

            cfg_text_generation_input = self._cfg_branch_generation_input(
                prepared.generation_input,
                prepared.cfg_text_generation_input,
                branch_name="cfg_text",
            )
            cfg_text_v_t = self._predict_velocity_branch(
                prepared=prepared,
                branch_name="cfg_text",
                generation_input=cfg_text_generation_input,
                srt_kv_token_binding=prepared.cfg_text_srt_kv_token_binding,
                latent_tokens=latent_tokens,
                timestep=timestep,
            )

            cfg_img_v_t = None
            if cfg_img_scale > 1.0:
                cfg_img_generation_input = self._cfg_branch_generation_input(
                    prepared.generation_input,
                    prepared.cfg_img_generation_input,
                    branch_name="cfg_img",
                )
                cfg_img_v_t = self._predict_velocity_branch(
                    prepared=prepared,
                    branch_name="cfg_img",
                    generation_input=cfg_img_generation_input,
                    srt_kv_token_binding=prepared.cfg_img_srt_kv_token_binding,
                    latent_tokens=latent_tokens,
                    timestep=timestep,
                )

            return self._apply_cfg(
                v_t=v_t,
                cfg_text_v_t=cfg_text_v_t,
                cfg_img_v_t=cfg_img_v_t,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_renorm_min=prepared.cfg_renorm_min,
                cfg_renorm_type=prepared.cfg_renorm_type,
            )

    def _predict_velocity_branch(
        self,
        *,
        prepared: BAGELNativeSRTPreparedDenoise,
        branch_name: str,
        generation_input: dict[str, Any],
        srt_kv_token_binding: Any | None,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self.forward_batch_provider is not None and srt_kv_token_binding is None:
            raise BAGELDenoiseStepError(
                f"Native SRT BAGEL velocity branch {branch_name} requires "
                "an SRT token binding"
            )
        branch_prepared = replace(
            prepared,
            generation_input=generation_input,
            srt_kv_token_binding=srt_kv_token_binding,
        )
        self._validate_prepared(branch_prepared)
        predictor = getattr(self.srt_model, "predict_velocity_from_packed_gen", None)
        if not callable(predictor):
            raise BAGELAdapterError(
                "Native SRT BAGEL denoise requires "
                "predict_velocity_from_packed_gen on the SRT model"
            )
        forward_batch_context = self._build_forward_batch(
            prepared=branch_prepared,
            latent_tokens=latent_tokens,
            timestep=timestep,
        )
        forward_batch = getattr(
            forward_batch_context,
            "forward_batch",
            forward_batch_context,
        )
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

    @staticmethod
    def _cfg_branch_generation_input(
        base_generation_input: dict[str, Any],
        cfg_generation_input: dict[str, Any] | None,
        *,
        branch_name: str,
    ) -> dict[str, Any]:
        if cfg_generation_input is None:
            raise BAGELDenoiseStepError(
                f"Native SRT BAGEL CFG branch {branch_name} is missing "
                "prepared cfg generation input"
            )
        required = (
            "cfg_packed_position_ids",
            "cfg_key_values_lens",
            "cfg_packed_query_indexes",
            "cfg_packed_key_value_indexes",
        )
        _require_keys(cfg_generation_input, required, f"{branch_name}_generation_input")
        generation_input = dict(base_generation_input)
        generation_input["packed_position_ids"] = cfg_generation_input[
            "cfg_packed_position_ids"
        ]
        generation_input["key_values_lens"] = cfg_generation_input[
            "cfg_key_values_lens"
        ]
        generation_input["packed_indexes"] = cfg_generation_input[
            "cfg_packed_query_indexes"
        ]
        generation_input["packed_key_value_indexes"] = cfg_generation_input[
            "cfg_packed_key_value_indexes"
        ]
        return generation_input

    @staticmethod
    def _apply_cfg(
        *,
        v_t: torch.Tensor,
        cfg_text_v_t: torch.Tensor,
        cfg_img_v_t: torch.Tensor | None,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
    ) -> torch.Tensor:
        if cfg_renorm_type == "text_channel":
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
            scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(
                min=cfg_renorm_min,
                max=1.0,
            )
            v_t_text = v_t_text_ * scale
            if cfg_img_scale > 1.0:
                if cfg_img_v_t is None:
                    raise BAGELDenoiseStepError(
                        "Native SRT BAGEL CFG image branch velocity is missing"
                    )
                return cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
            return v_t_text

        v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
        if cfg_img_scale > 1.0:
            if cfg_img_v_t is None:
                raise BAGELDenoiseStepError(
                    "Native SRT BAGEL CFG image branch velocity is missing"
                )
            v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
        else:
            v_t_ = v_t_text_

        if cfg_renorm_type == "global":
            norm_v_t = torch.norm(v_t)
            norm_v_t_ = torch.norm(v_t_)
        elif cfg_renorm_type == "channel":
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
        else:
            raise BAGELDenoiseStepError(
                f"Unsupported BAGEL CFG renorm type: {cfg_renorm_type}"
            )
        scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(
            min=cfg_renorm_min,
            max=1.0,
        )
        return v_t_ * scale

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


@dataclass
class BAGELSessionContext:
    gen_context: dict[str, Any]
    cfg_text_context: dict[str, Any]
    cfg_img_context: dict[str, Any]
    image_shape: tuple[int, int]
    prepared_denoise: BAGELNativeSRTPreparedDenoise | None = None
    decode_count: int = 0
    append_image_count: int = 0
    native_srt_u_context: bool = False
    native_srt_u_context_request_id: str | None = None
    native_srt_u_context_token_binding: Any | None = None
    native_srt_cfg_text_token_binding: Any | None = None
    native_srt_cfg_img_token_binding: Any | None = None
    native_srt_cfg_text_token_count: int | None = None
    native_srt_cfg_img_token_count: int | None = None
    native_srt_cfg_img_requires_sidecar: bool = False
    srt_u_forward_results: dict[str, Any] = field(default_factory=dict)
    srt_u_forward_events: list[tuple[str, str]] = field(default_factory=list)
    srt_last_u_decode_output_ids: tuple[int, ...] = ()
    srt_last_u_decode_text: str = ""
    native_srt_pending_added_tokens: int = 0
    thinking_committed: bool = False


class BAGELUForwardBridge:
    """Maps safe SRT request views onto native BAGEL UG session state."""

    def __init__(
        self,
        *,
        srt_u_forward_executor: "BAGELSRTUForwardExecutor | None" = None,
    ) -> None:
        if srt_u_forward_executor is None:
            raise BAGELAdapterError(
                "BAGEL U forward bridge requires an explicit SRT-native executor"
            )
        self.srt_u_forward_executor = srt_u_forward_executor

    def observe(
        self,
        backend: "BAGELInterleaveContextBackend",
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        state_session_id = request.metadata.get(
            "ug_srt_owner_session_id",
            session.handle.session_id,
        )
        state = backend._state_for(str(state_session_id))
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
    """Compatibility base for BAGEL U-forward observers."""

    def execute(
        self,
        backend: "BAGELInterleaveContextBackend",
        *,
        session,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> UGModelPrefillResult | UGModelAppendImageResult | None:
        del backend, session, request, messages
        raise BAGELAdapterError(
            "BAGEL external Python U-forward fallback has been removed; "
            "use BAGELNativeSRTUForwardExecutor"
        )


class BAGELNativeSRTUForwardExecutor(BAGELSRTUForwardExecutor):
    """Consumes U-prefill views produced by native SRT BAGEL forward.

    It marks the session as SRT-owned so the G denoise path can reuse SRT KV
    through a native generation ForwardBatch.
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
        sidecar_role = request.metadata.get("ug_srt_sidecar_role")
        if sidecar_role is not None:
            backend._sync_native_srt_sidecar_from_request(request)
            return None

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
            state.thinking_committed = False
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
            state.srt_last_u_decode_text = str(
                request.metadata.get("srt_last_u_decode_text") or ""
            )
            return None
        return None


class BAGELInterleaveContextBackend:
    """Wraps the native-SRT BAGEL shell behind UG adapter methods."""

    def __init__(
        self,
        inferencer: Any,
        *,
        step_runner: Any | None = None,
        u_forward_bridge: BAGELUForwardBridge | None = None,
        srt_kv_cache_factory: BAGELSRTKVCacheFactory | None = None,
        native_srt_denoise_executor: BAGELNativeSRTDenoiseExecutor | None = None,
        default_image_shape: tuple[int, int] = (1024, 1024),
    ) -> None:
        del step_runner
        self.inferencer = inferencer
        self.u_forward_bridge = u_forward_bridge or BAGELUForwardBridge(
            srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
        )
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
        if not self._uses_native_srt_u_forward():
            return None
        if message.type == "text":
            return self._prepare_native_srt_text_inputs(
                session=session,
                text=str(message.content),
            )
        if message.type == "image":
            image, use_vae, use_vit = self._unpack_image_message(message.content)
            return self._prepare_native_srt_image_inputs(
                session=session,
                image=image,
                state=state,
                use_vae=use_vae,
                use_vit=use_vit,
            )
        return None

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
        del session, messages
        raise BAGELAdapterError(
            "BAGEL real backend requires SRT-executed U prefill; "
            "the external Python context-update fallback has been removed"
        )

    def decode_next_segment(self, *, session) -> UGDecodeResult:
        state = self._state_for(session.handle.session_id)
        if state.decode_count == 0:
            state.decode_count += 1
            return UGDecodeResult(type="image_marker")
        if state.append_image_count > 0 and state.decode_count == 1:
            state.decode_count += 1
            if state.native_srt_u_context:
                return UGDecodeResult(
                    type="text",
                    text=_strip_bagel_decoded_text(state.srt_last_u_decode_text),
                )
            raise BAGELAdapterError(
                "BAGEL text decode after generated image requires SRT-owned U "
                "context"
            )
        state.decode_count += 1
        return UGDecodeResult(type="done")

    def decode_next_segment_from_runtime(
        self, *, runtime: Any, session: Any
    ) -> UGDecodeResult:
        state = self._state_for(session.session_id)
        if state.decode_count == 0:
            if runtime.srt_u_decode_max_new_tokens > 0 and not state.thinking_committed:
                curr_kvlens, curr_rope = self._native_srt_curr_lengths(state)
                runtime.decode_text(
                    session,
                    max_new_tokens=runtime.srt_u_decode_max_new_tokens,
                    greedy=True,
                    model_state_updates=self._bagel_model_state_updates(
                        kv_lens=[
                            int(curr_kvlens[0]) + runtime.srt_u_decode_max_new_tokens
                        ],
                        ropes=[int(curr_rope[0]) + runtime.srt_u_decode_max_new_tokens],
                    ),
                )
            return self.decode_next_segment(session=SimpleNamespace(handle=session))
        if (
            state.native_srt_u_context
            and state.append_image_count > 0
            and state.decode_count == 1
        ):
            max_new_tokens = max(1, int(runtime.srt_u_decode_max_new_tokens))
            decoded = self.decode_vlm_text(
                runtime=runtime,
                session=session,
                max_new_tokens=max_new_tokens,
            )
            state.decode_count += 1
            state.srt_last_u_decode_text = decoded.text
            return UGDecodeResult(type="text", text=decoded.text)
        return self.decode_next_segment(session=SimpleNamespace(handle=session))

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise BAGELAdapterError(
                f"BAGEL VLM text decode requires max_new_tokens > 0, got {max_new_tokens}"
            )
        new_token_ids = self.inferencer.new_token_ids
        start_token_id = int(new_token_ids["bos_token_id"])
        end_token_id = int(new_token_ids["eos_token_id"])
        current_token = start_token_id
        current_handle = session
        generated: list[int] = []
        next_token_ids: list[int] = []
        position_ids: list[int] = []

        for step in range(max_new_tokens):
            state = self._state_for(current_handle.session_id)
            rope = int((state.gen_context.get("ropes") or [0])[0])
            curr_kvlens, _ = self._native_srt_curr_lengths(state)
            decoded = runtime.decode_text(
                current_handle,
                max_new_tokens=1,
                start_token_id=current_token,
                position_ids=[rope],
                drop_previous_output=step > 0,
                greedy=True,
                model_state_updates=self._bagel_model_state_updates(
                    kv_lens=[int(curr_kvlens[0]) + 1],
                    ropes=[rope + 1],
                ),
            )
            generated.append(int(current_token))
            position_ids.append(rope)
            current_handle = decoded.session
            if not decoded.output_ids:
                break
            next_token = int(decoded.output_ids[0])
            next_token_ids.append(next_token)
            if next_token == end_token_id:
                break
            current_token = next_token

        decode_token_ids = getattr(runtime, "_decode_token_ids", None)
        if callable(decode_token_ids):
            text = _strip_bagel_decoded_text(decode_token_ids(generated))
        else:
            text = _decode_bagel_token_ids(self.inferencer.tokenizer, generated)
        state = self._state_for(current_handle.session_id)
        state.thinking_committed = True
        state.srt_last_u_decode_text = text
        return UGVLMTextGenerationResult(
            session=current_handle,
            text=text,
            token_ids=tuple(generated),
            next_token_ids=tuple(next_token_ids),
            position_ids=tuple(position_ids),
        )

    def predict_velocity_from_session(
        self, *, session, request: UGVelocityRequest
    ) -> torch.Tensor:
        state = self._state_for_session(session)
        if state.prepared_denoise is None:
            state.prepared_denoise = self._prepare_denoise(
                state, request.sampling_params
            )
        latent_tokens = request.latent_tokens
        timestep = request.timestep
        if not isinstance(state.prepared_denoise, BAGELNativeSRTPreparedDenoise):
            raise BAGELAdapterError(
                "BAGEL G velocity requires native SRT prepared denoise inputs"
            )
        runtime_model = self._native_srt_model()
        runtime_device = _bagel_runtime_device(runtime_model)
        if runtime_device is not None:
            latent_tokens = latent_tokens.to(runtime_device)
            timestep = timestep.to(runtime_device)

        with torch.autocast(
            device_type="cuda",
            enabled=latent_tokens.is_cuda,
            dtype=torch.bfloat16,
        ):
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
        return velocity.to(request.latent_tokens.device)

    def prepare_latents_from_session(
        self, *, session, request: UGLatentPrepareRequest
    ) -> UGLatentPrepareResult | None:
        state = self._state_for_session(session)
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
        if not isinstance(state.prepared_denoise, BAGELNativeSRTPreparedDenoise):
            raise BAGELAdapterError(
                "BAGEL latent preparation requires native SRT prepared denoise inputs"
            )
        latent_shape_model = self._native_srt_model()
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
        del session, image
        raise BAGELAdapterError(
            "BAGEL real backend requires SRT-executed append-image U forward; "
            "the external Python context-update fallback has been removed"
        )

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

        if not state.native_srt_u_context:
            raise BAGELAdapterError("BAGEL image decode requires SRT-owned U context")
        decoder = getattr(self._native_srt_model(), "decode_bagel_image", None)
        if not callable(decoder):
            raise BAGELAdapterError(
                "BAGEL native SRT image decode requires decode_bagel_image "
                "on the SRT model"
            )
        return decoder(latent_tokens, image_shape)

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

    def _state_for_session(self, session: Any) -> BAGELSessionContext:
        state = self._state_for(session.handle.session_id)
        self._sync_bagel_state_from_metadata(
            state,
            getattr(session, "metadata", {}) or {},
        )
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
    ) -> BAGELNativeSRTPreparedDenoise:
        if not state.native_srt_u_context:
            raise BAGELAdapterError(
                "BAGEL G denoise requires SRT-owned U context before preparing "
                "latents"
            )
        return self._prepare_native_srt_denoise(
            state,
            sampling_params,
            seed=seed,
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
            cfg_text_generation_input = self._prepare_native_srt_cfg_latent(
                model,
                state.cfg_text_context,
                image_shape=image_shape,
            )
            cfg_img_generation_input = self._prepare_native_srt_cfg_latent(
                model,
                state.cfg_img_context,
                image_shape=image_shape,
            )
        return BAGELNativeSRTPreparedDenoise(
            generation_input=generation_input,
            srt_kv_token_binding=binding,
            cfg_text_generation_input=cfg_text_generation_input,
            cfg_text_srt_kv_token_binding=state.native_srt_cfg_text_token_binding,
            cfg_img_generation_input=cfg_img_generation_input,
            cfg_img_srt_kv_token_binding=state.native_srt_cfg_img_token_binding,
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
        use_vae: bool = True,
        use_vit: bool = True,
    ) -> list[UGSRTPreparedInput]:
        if not use_vae and not use_vit:
            raise BAGELAdapterError(
                "Native SRT BAGEL image U forward requires vae or vit modality"
            )
        model = self._native_srt_model()
        if not hasattr(model, "prepare_vae_images") or not hasattr(
            model, "prepare_vit_images"
        ):
            raise BAGELAdapterError(
                "Native SRT BAGEL image U forward requires "
                "prepare_vae_images/prepare_vit_images on the SRT model"
            )

        session_state = self._state_for_session(session)
        image = self._prepare_image(image)
        session_state.image_shape = self._image_shape(image)
        curr_kvlens, curr_rope = self._native_srt_curr_lengths(session_state)

        chunks: list[UGSRTPreparedInput] = []
        with torch.no_grad(), _bagel_autocast(model):
            if use_vae:
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
                        model_state_updates=self._bagel_model_state_updates(
                            kv_lens=curr_kvlens,
                            ropes=curr_rope,
                        ),
                    )
                )

            if use_vit:
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
                        model_state_updates=self._bagel_model_state_updates(
                            kv_lens=curr_kvlens,
                            ropes=curr_rope,
                        ),
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

    def _prepare_native_srt_text_inputs(
        self,
        *,
        session,
        text: str,
    ) -> list[UGSRTPreparedInput]:
        session_state = self._state_for_session(session)
        tokenizer = getattr(self.inferencer, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise BAGELAdapterError(
                "Native SRT BAGEL text U forward requires tokenizer.encode"
            )
        new_token_ids = self.inferencer.new_token_ids
        try:
            text_ids = list(encode(text))
        except TypeError:
            text_ids = list(encode(text, add_special_tokens=False))
        input_ids = [
            int(new_token_ids["bos_token_id"]),
            *[int(token_id) for token_id in text_ids],
            int(new_token_ids["eos_token_id"]),
        ]

        curr_kvlens, curr_rope = self._native_srt_curr_lengths(session_state)
        position_start = int(curr_rope[0])
        position_ids = list(range(position_start, position_start + len(input_ids)))
        pre_text_kvlen = int(curr_kvlens[0])
        session_state.cfg_text_context = self._clone_context(
            session_state.gen_context,
            session_id=session.handle.session_id,
            role="cfg_text",
        )
        session_state.native_srt_cfg_text_token_count = pre_text_kvlen
        session_state.native_srt_cfg_text_token_binding = None
        session_state.gen_context["kv_lens"] = [pre_text_kvlen + len(input_ids)]
        session_state.gen_context["ropes"] = [position_start + len(input_ids)]
        sidecar_inputs: list[UGSRTPreparedInput] = []
        text_message = UGInterleavedMessage(type="text", content=text)
        if pre_text_kvlen == 0:
            session_state.cfg_img_context = self._clone_context(
                session_state.gen_context,
                session_id=session.handle.session_id,
                role="cfg_img",
            )
            session_state.native_srt_cfg_img_token_count = int(
                session_state.gen_context["kv_lens"][0]
            )
            session_state.native_srt_cfg_img_requires_sidecar = False
        else:
            session_state.cfg_img_context = {
                "kv_lens": [len(input_ids)],
                "ropes": [len(input_ids)],
                "past_key_values": None,
            }
            session_state.native_srt_cfg_img_token_count = None
            session_state.native_srt_cfg_img_requires_sidecar = True
            sidecar_inputs.append(
                UGSRTPreparedInput(
                    input_ids=input_ids,
                    input_text=text,
                    messages=[text_message],
                    position_ids=list(range(len(input_ids))),
                    srt_sidecar_role="cfg_img",
                    srt_sidecar_session_id=f"{session.handle.session_id}:cfg_img",
                    adapter_metadata={
                        "bagel_u_text": True,
                        "ug_srt_added_token_count": len(input_ids),
                        "ug_srt_bagel_rope_delta": 0,
                        "ug_model_state_updates": self._bagel_model_state_updates(
                            kv_lens=session_state.gen_context["kv_lens"],
                            ropes=session_state.gen_context["ropes"],
                            cfg_text_kv_lens=session_state.cfg_text_context["kv_lens"],
                            cfg_text_ropes=session_state.cfg_text_context["ropes"],
                            cfg_text_token_count=pre_text_kvlen,
                            cfg_img_kv_lens=session_state.cfg_img_context["kv_lens"],
                            cfg_img_ropes=session_state.cfg_img_context["ropes"],
                            cfg_img_token_count=len(input_ids),
                            cfg_img_requires_sidecar=True,
                            cfg_img_sidecar_session_id=(
                                f"{session.handle.session_id}:cfg_img"
                            ),
                        ),
                    },
                )
            )
        session_state.native_srt_cfg_img_token_binding = None
        session_state.prepared_denoise = None

        main_input = UGSRTPreparedInput(
            input_ids=input_ids,
            input_text=text,
            messages=[text_message],
            position_ids=position_ids,
            adapter_metadata={
                "bagel_u_text": True,
                "ug_srt_added_token_count": len(input_ids),
                "ug_srt_bagel_rope_delta": 0,
                "ug_model_state_updates": self._bagel_model_state_updates(
                    kv_lens=session_state.gen_context["kv_lens"],
                    ropes=session_state.gen_context["ropes"],
                    cfg_text_kv_lens=session_state.cfg_text_context["kv_lens"],
                    cfg_text_ropes=session_state.cfg_text_context["ropes"],
                    cfg_text_token_count=pre_text_kvlen,
                    cfg_img_kv_lens=session_state.cfg_img_context["kv_lens"],
                    cfg_img_ropes=session_state.cfg_img_context["ropes"],
                    cfg_img_token_count=(
                        len(input_ids)
                        if session_state.native_srt_cfg_img_requires_sidecar
                        else session_state.native_srt_cfg_img_token_count
                    ),
                    cfg_img_requires_sidecar=(
                        session_state.native_srt_cfg_img_requires_sidecar
                    ),
                    cfg_img_sidecar_session_id=(
                        f"{session.handle.session_id}:cfg_img"
                        if session_state.native_srt_cfg_img_requires_sidecar
                        else None
                    ),
                ),
            },
        )
        return [
            main_input,
            *sidecar_inputs,
        ]

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
        model_state_updates: dict[str, Any] | None = None,
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
        if model_state_updates is not None:
            metadata["ug_model_state_updates"] = model_state_updates
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
            self._refresh_native_srt_cfg_token_bindings(state, binding)

        if self._sync_bagel_state_from_metadata(state, request.metadata):
            return

        rope_delta = self._native_srt_rope_delta(request=request, messages=messages)
        if rope_delta:
            ropes = state.gen_context.get("ropes") or [0]
            state.gen_context["ropes"] = [int(ropes[0]) + rope_delta]

    @classmethod
    def _sync_bagel_state_from_metadata(
        cls,
        state: BAGELSessionContext,
        metadata: dict[str, Any],
    ) -> bool:
        bagel_state = cls._bagel_model_state_from_metadata(metadata)
        if not bagel_state:
            return False
        kv_lens = bagel_state.get("logical_kv_lens")
        if kv_lens is not None:
            state.gen_context["kv_lens"] = [int(value) for value in kv_lens]
        ropes = bagel_state.get("logical_ropes")
        if ropes is not None:
            state.gen_context["ropes"] = [int(value) for value in ropes]
        cfg_text_kv_lens = bagel_state.get("cfg_text_logical_kv_lens")
        if cfg_text_kv_lens is not None:
            state.cfg_text_context["kv_lens"] = [
                int(value) for value in cfg_text_kv_lens
            ]
        cfg_text_ropes = bagel_state.get("cfg_text_logical_ropes")
        if cfg_text_ropes is not None:
            state.cfg_text_context["ropes"] = [int(value) for value in cfg_text_ropes]
        if "cfg_text_token_count" in bagel_state:
            state.native_srt_cfg_text_token_count = _optional_int(
                bagel_state.get("cfg_text_token_count")
            )
        cfg_img_kv_lens = bagel_state.get("cfg_img_logical_kv_lens")
        if cfg_img_kv_lens is not None:
            state.cfg_img_context["kv_lens"] = [int(value) for value in cfg_img_kv_lens]
        cfg_img_ropes = bagel_state.get("cfg_img_logical_ropes")
        if cfg_img_ropes is not None:
            state.cfg_img_context["ropes"] = [int(value) for value in cfg_img_ropes]
        if "cfg_img_token_count" in bagel_state:
            state.native_srt_cfg_img_token_count = _optional_int(
                bagel_state.get("cfg_img_token_count")
            )
        if "cfg_img_requires_sidecar" in bagel_state:
            state.native_srt_cfg_img_requires_sidecar = bool(
                bagel_state.get("cfg_img_requires_sidecar")
            )
        return any(
            value is not None
            for value in (
                kv_lens,
                ropes,
                cfg_text_kv_lens,
                cfg_text_ropes,
                cfg_img_kv_lens,
                cfg_img_ropes,
            )
        ) or any(
            key in bagel_state
            for key in (
                "cfg_text_token_count",
                "cfg_img_token_count",
                "cfg_img_requires_sidecar",
            )
        )

    @staticmethod
    def _bagel_model_state_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        model_state = metadata.get("ug_model_state") or {}
        if not isinstance(model_state, dict):
            return {}
        bagel_state = model_state.get("bagel") or {}
        if not isinstance(bagel_state, dict):
            return {}
        return bagel_state

    @staticmethod
    def _bagel_model_state_updates(
        *,
        kv_lens: list[int] | tuple[int, ...],
        ropes: list[int] | tuple[int, ...],
        cfg_text_kv_lens: list[int] | tuple[int, ...] | None = None,
        cfg_text_ropes: list[int] | tuple[int, ...] | None = None,
        cfg_text_token_count: int | None = None,
        cfg_img_kv_lens: list[int] | tuple[int, ...] | None = None,
        cfg_img_ropes: list[int] | tuple[int, ...] | None = None,
        cfg_img_token_count: int | None = None,
        cfg_img_requires_sidecar: bool | None = None,
        cfg_img_sidecar_session_id: str | None = None,
    ) -> dict[str, Any]:
        bagel_state: dict[str, Any] = {
            "logical_kv_lens": [int(value) for value in kv_lens],
            "logical_ropes": [int(value) for value in ropes],
        }
        if cfg_text_kv_lens is not None:
            bagel_state["cfg_text_logical_kv_lens"] = [
                int(value) for value in cfg_text_kv_lens
            ]
        if cfg_text_ropes is not None:
            bagel_state["cfg_text_logical_ropes"] = [
                int(value) for value in cfg_text_ropes
            ]
        if cfg_text_token_count is not None:
            bagel_state["cfg_text_token_count"] = int(cfg_text_token_count)
        if cfg_img_kv_lens is not None:
            bagel_state["cfg_img_logical_kv_lens"] = [
                int(value) for value in cfg_img_kv_lens
            ]
        if cfg_img_ropes is not None:
            bagel_state["cfg_img_logical_ropes"] = [
                int(value) for value in cfg_img_ropes
            ]
        if cfg_img_token_count is not None:
            bagel_state["cfg_img_token_count"] = int(cfg_img_token_count)
        if cfg_img_requires_sidecar is not None:
            bagel_state["cfg_img_requires_sidecar"] = bool(cfg_img_requires_sidecar)
        if cfg_img_sidecar_session_id is not None:
            bagel_state["cfg_img_sidecar_session_id"] = str(cfg_img_sidecar_session_id)
        return {"bagel": bagel_state}

    def _sync_native_srt_sidecar_from_request(
        self,
        request: UGSRTRequestView,
    ) -> None:
        owner_session_id = request.metadata.get("ug_srt_owner_session_id")
        sidecar_role = request.metadata.get("ug_srt_sidecar_role")
        binding = request.metadata.get("srt_kv_token_binding")
        if owner_session_id is None or sidecar_role is None or binding is None:
            return
        state = self._state_for(str(owner_session_id))
        if sidecar_role == "cfg_img":
            self._sync_bagel_state_from_metadata(state, request.metadata)
            state.native_srt_cfg_img_token_binding = binding
            state.native_srt_cfg_img_token_count = int(binding.token_count)
            state.native_srt_cfg_img_requires_sidecar = True

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

    @staticmethod
    def _prepare_native_srt_cfg_latent(
        model: Any,
        context: dict[str, Any],
        *,
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        prepare = getattr(model, "prepare_vae_latent_cfg", None)
        if not callable(prepare):
            raise BAGELAdapterError(
                "BAGEL native SRT CFG requires prepare_vae_latent_cfg on "
                "the SRT model"
            )
        kv_lens = context.get("kv_lens") or [0]
        ropes = context.get("ropes") or [0]
        return prepare(
            curr_kvlens=[int(kv_lens[0])],
            curr_rope=[int(ropes[0])],
            image_sizes=[image_shape],
        )

    def _refresh_native_srt_cfg_token_bindings(
        self,
        state: BAGELSessionContext,
        binding: UGSRTKVTokenBinding,
    ) -> None:
        if state.native_srt_cfg_text_token_count is not None:
            state.native_srt_cfg_text_token_binding = self._slice_srt_token_binding(
                binding,
                token_count=state.native_srt_cfg_text_token_count,
                role="cfg_text",
            )
        if (
            state.native_srt_cfg_img_token_count is not None
            and not state.native_srt_cfg_img_requires_sidecar
        ):
            state.native_srt_cfg_img_token_binding = self._slice_srt_token_binding(
                binding,
                token_count=state.native_srt_cfg_img_token_count,
                role="cfg_img",
            )

    @staticmethod
    def _slice_srt_token_binding(
        binding: UGSRTKVTokenBinding,
        *,
        token_count: int,
        role: str,
    ) -> UGSRTKVTokenBinding:
        token_count = int(token_count)
        if token_count < 0:
            raise BAGELAdapterError(
                f"BAGEL native SRT {role} token count is negative: {token_count}"
            )
        token_indices = binding.token_indices[:token_count]
        clone = getattr(token_indices, "clone", None)
        if callable(clone):
            token_indices = clone()
        return UGSRTKVTokenBinding(
            session_id=binding.session_id,
            request_id=f"{binding.request_id}:{role}",
            token_count=token_count,
            token_indices=token_indices,
        )

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

    @staticmethod
    def _unpack_image_message(content: Any) -> tuple[Any, bool, bool]:
        if not isinstance(content, dict):
            return content, True, True
        if "image" not in content:
            return content, True, True
        return (
            content.get("image"),
            bool(content.get("vae", True)),
            bool(content.get("vit", True)),
        )

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

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult: ...

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
    the SRT-native BAGEL backend. The real backend keeps only tokenizer and
    image transforms in this adapter; SRT owns feature extractors, session
    requests, KV cache, and per-step G velocity execution.
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

    def decode_next_segment_from_runtime(
        self, *, runtime: Any, session: Any
    ) -> UGDecodeResult:
        decode = getattr(self.backend, "decode_next_segment_from_runtime", None)
        if callable(decode):
            return decode(runtime=runtime, session=session)
        return self.backend.decode_next_segment(session=SimpleNamespace(handle=session))

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        return self.backend.decode_vlm_text(
            runtime=runtime,
            session=session,
            max_new_tokens=max_new_tokens,
        )

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
                "checkpoint directory with config and weight files."
            )

        if not native_srt_u_context or native_srt_denoise_executor is None:
            raise BAGELAdapterError(
                "BAGEL real backend is native-SRT only; pass "
                "native_srt_u_context=True and a BAGELNativeSRTDenoiseExecutor"
            )

        inferencer = _build_native_srt_bagel_inferencer_shell(checkpoint_dir)
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

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        del runtime
        self.events.append(("decode_vlm_text", session.session_id))
        token_ids = tuple(range(1, max(1, int(max_new_tokens)) + 1))
        return UGVLMTextGenerationResult(
            session=session,
            text="bagel_mock_vlm_text",
            token_ids=token_ids,
        )

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


def _effective_cfg_scales(
    prepared: BAGELNativeSRTPreparedDenoise,
    timestep: torch.Tensor,
) -> tuple[float, float]:
    return get_bagel_effective_cfg_scales(prepared, timestep)


def _clone_context(context: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(context)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _decode_bagel_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return _strip_bagel_decoded_text(str(tokenizer.decode(token_ids)))


def _strip_bagel_decoded_text(text: str) -> str:
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in text:
        text = text.split("<|im_start|>", 1)[1]
    return text


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


class BAGELNativeSRTInferencerShell:
    """Tokenizer/transform shell for native SRT BAGEL execution.

    The shell intentionally does not load a second BAGEL model or VAE/VIT
    feature extractor stack. Native execution gets those modules from the SRT
    model instance loaded by ModelRunner.
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
            "Native SRT BAGEL shell cannot update context outside SRT"
        )

    def update_context_image(self, *args, **kwargs):
        del args, kwargs
        raise BAGELAdapterError(
            "Native SRT BAGEL shell cannot append images outside SRT"
        )

    def decode_image(self, *args, **kwargs):
        del args, kwargs
        raise BAGELAdapterError("Native SRT BAGEL shell cannot decode outside SRT")

    def gen_text(self, *args, **kwargs):
        del args, kwargs
        raise BAGELAdapterError("Native SRT BAGEL shell cannot decode text outside SRT")


def _build_native_srt_bagel_inferencer_shell(
    checkpoint_dir: Path,
    *,
    tokenizer_loader: Any | None = None,
    image_transform_cls: type | None = None,
) -> BAGELNativeSRTInferencerShell:
    _ensure_bagel_transformers_compat()
    tokenizer_loader = tokenizer_loader or AutoTokenizer
    image_transform_cls = image_transform_cls or BAGELImageTransform

    tokenizer = tokenizer_loader.from_pretrained(
        str(checkpoint_dir),
        use_fast=False,
    )
    tokenizer, new_token_ids, _ = add_bagel_special_tokens(tokenizer)
    return BAGELNativeSRTInferencerShell(
        tokenizer=tokenizer,
        vae_transform=image_transform_cls(1024, 512, 16),
        vit_transform=image_transform_cls(980, 224, 14),
        new_token_ids=new_token_ids,
    )


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
