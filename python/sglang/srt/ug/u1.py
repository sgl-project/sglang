# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from sglang.srt.ug.adapter import UGModelAppendImageResult, UGModelPrefillResult
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.denoiser import (
    SRTBackedUGMiddleBridge,
    UGGSegmentExecutor,
)
from sglang.srt.ug.interleaved import UGGKind, UGGSegmentResult
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGSessionRuntime,
    UGSegmentState,
    UGSRTPreparedInput,
    UGVLMTextGenerationResult,
)

U1_IMG_START_TOKEN = "<img>"
U1_IMG_END_TOKEN = "</img>"
U1_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
U1_IMAGE_PLACEHOLDER = "<image>"
U1_IMAGENET_MEAN = (0.485, 0.456, 0.406)
U1_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class U1VLMBackendResult:
    text: str
    token_ids: tuple[int, ...] = ()
    next_token_ids: tuple[int, ...] = ()
    position_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class U1VLMBackend(Protocol):
    def generate_text(
        self,
        *,
        messages: list[UGInterleavedMessage],
        max_new_tokens: int,
    ) -> U1VLMBackendResult: ...


class U1SubprocessVLMBackend:
    """Opt-in external VLM runner for U1 parity.

    This backend keeps the official/compatibility implementation out of the
    SGLang runtime process. It is a parity bridge, not native SRT ModelRunner
    execution.
    """

    def __init__(
        self,
        *,
        python: str | Path,
        repo: str | Path,
        model_path: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_backend: str = "sdpa",
        timeout: int = 600,
        cuda_visible_devices: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.python = Path(python)
        self.repo = Path(repo)
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.attn_backend = attn_backend
        self.timeout = int(timeout)
        self.cuda_visible_devices = cuda_visible_devices
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def generate_text(
        self,
        *,
        messages: list[UGInterleavedMessage],
        max_new_tokens: int,
    ) -> U1VLMBackendResult:
        image_path = _first_u1_image_path(messages)
        question = _u1_question_text(messages)
        output_dir = self._make_output_dir()
        output_path = output_dir / "u1_vlm_candidate.txt"
        cmd = [
            str(self.python),
            str(self.repo / "examples/vqa/inference.py"),
            "--model_path",
            self.model_path,
            "--image",
            str(image_path),
            "--question",
            question,
            "--output",
            str(output_path),
            "--max_new_tokens",
            str(int(max_new_tokens)),
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--attn_backend",
            self.attn_backend,
        ]
        run_env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        completed = subprocess.run(
            cmd,
            cwd=self.repo,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "U1 VLM subprocess backend failed: "
                f"returncode={completed.returncode}, stderr={_tail(completed.stderr)}"
            )
        if not output_path.exists():
            raise RuntimeError("U1 VLM subprocess backend did not write output text")
        return U1VLMBackendResult(
            text=output_path.read_text(),
            metadata={
                "backend": "external_subprocess",
                "native_srt_model_runner": False,
                "command": cmd,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            },
        )

    def _make_output_dir(self) -> Path:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir
        return Path(tempfile.mkdtemp(prefix="u1-vlm-backend-"))


def is_sensenova_u1_ug_model(
    model_path: str | None,
    model_id: str | None = None,
) -> bool:
    identifier = f"{model_path or ''} {model_id or ''}".lower()
    return "sensenova-u1" in identifier or "sensenova_u1" in identifier


class U1UGModelAdapter:
    """SenseNova U1 UG adapter shell for the UG middle protocol.

    U1 uses pixel-flow G mechanics, so it intentionally does not expose BAGEL
    latent-flow methods such as velocity prediction or latent decode.
    """

    g_kind: UGGKind = "pixel_flow"

    bos_token_id = 1
    text_token_base = 1000
    image_token_id = 31001
    generated_image_token_id = 31003

    def __init__(
        self,
        *,
        vlm_backend: U1VLMBackend | None = None,
        native_tokenizer: Any | None = None,
    ) -> None:
        self.observed_u_forwards: list[dict[str, Any]] = []
        self._pending_segments_by_session: dict[str, list[dict[str, Any]]] = {}
        self._messages_by_session: dict[str, list[UGInterleavedMessage]] = {}
        self.vlm_backend = vlm_backend
        self.native_tokenizer = native_tokenizer

    def prepare_srt_u_interleaved_inputs(
        self,
        *,
        session: Any,
        messages: list[UGInterleavedMessage],
        state: Any,
    ) -> list[UGSRTPreparedInput] | None:
        if state != UGSegmentState.U_PREFILL or self.native_tokenizer is None:
            return None
        has_image = any(message.type == "image" for message in messages)
        has_text = any(message.type == "text" for message in messages)
        if not has_text:
            return None
        if not has_image:
            return [
                build_u1_native_t2i_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                )
            ]
        return [
            build_u1_native_vlm_prepared_input(
                tokenizer=self.native_tokenizer,
                messages=messages,
                session=session,
            )
        ]

    def prepare_srt_u_message_inputs(
        self,
        *,
        session: Any,
        message: Any,
        state: Any,
    ) -> list[UGSRTPreparedInput] | None:
        if message.type == "text":
            return [self._prepare_text_input(session=session, message=message)]
        if message.type == "image":
            if (
                state == UGSegmentState.APPEND_IMAGE
                and self.native_tokenizer is not None
            ):
                return [
                    build_u1_native_generated_image_commit_prepared_input(
                        tokenizer=self.native_tokenizer,
                        image=message.content,
                        session=session,
                    )
                ]
            return [
                self._prepare_image_input(
                    session=session,
                    message=message,
                    generated_image_commit=state == UGSegmentState.APPEND_IMAGE,
                )
            ]
        return None

    def observe_srt_u_forward(
        self,
        *,
        session: Any,
        request: Any,
        messages: list[Any],
    ) -> None:
        del session
        self.observed_u_forwards.append(
            {
                "request_id": request.request_id,
                "state": request.state,
                "origin_input_len": request.origin_input_len,
                "metadata": request.metadata,
                "message_types": [message.type for message in messages],
            }
        )

    def prefill_interleaved(
        self,
        *,
        session: Any,
        messages: list[Any],
    ) -> UGModelPrefillResult:
        try:
            self._remember_session_messages(session, messages)
            return UGModelPrefillResult(
                added_tokens=self._added_tokens_from_srt_session_view(session, messages)
            )
        finally:
            self._clear_pending_segments(session)

    def decode_next_segment(self, *, session: Any) -> Any:
        if self._has_generated_image_commit(session):
            return UGDecodeResult(type="text", text="u1_pixel_flow_text_after_image")
        return UGDecodeResult(type="image_marker")

    def decode_next_segment_from_runtime(self, *, runtime: Any, session: Any) -> Any:
        session_view = self._runtime_session_view(runtime=runtime, session=session)
        if not self._has_generated_image_commit(session_view):
            return UGDecodeResult(type="image_marker")
        if self.native_tokenizer is None or runtime is None:
            return UGDecodeResult(type="text", text="u1_pixel_flow_text_after_image")
        if getattr(runtime, "srt_request_executor", None) is None:
            return UGDecodeResult(type="text", text="u1_pixel_flow_text_after_image")
        max_new_tokens = max(
            1,
            int(getattr(runtime, "srt_u_decode_max_new_tokens", 0) or 0),
        )
        decoded = runtime.decode_text(
            session,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        return UGDecodeResult(type="text", text=decoded.text)

    @staticmethod
    def _runtime_session_view(*, runtime: Any, session: Any) -> Any:
        if getattr(session, "metadata", None) is not None:
            return session
        metadata: dict[str, Any] = {}
        get_debug_counters = getattr(runtime, "get_debug_counters", None)
        if callable(get_debug_counters):
            try:
                counters = get_debug_counters(session)
                metadata["ug_model_state"] = counters.get("ug_model_state", {})
                metadata["srt_last_u_decode_output_ids"] = counters.get(
                    "srt_last_u_decode_output_ids",
                    (),
                )
            except Exception:
                metadata = {}
        return SimpleNamespace(handle=session, metadata=metadata)

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> Any:
        if self.vlm_backend is None:
            if runtime is None:
                raise _not_wired()
            decoded = runtime.decode_text(
                session,
                max_new_tokens=max_new_tokens,
                greedy=True,
            )
            return UGVLMTextGenerationResult(
                session=decoded.session,
                text=decoded.text,
                token_ids=decoded.input_ids,
                next_token_ids=decoded.output_ids,
                position_ids=decoded.position_ids,
            )
        messages = self._messages_for_session(session)
        result = self.vlm_backend.generate_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        return UGVLMTextGenerationResult(
            session=session,
            text=result.text,
            token_ids=result.token_ids,
            next_token_ids=result.next_token_ids,
            position_ids=result.position_ids,
        )

    def append_generated_image(
        self,
        *,
        session: Any,
        image: Any | None,
    ) -> UGModelAppendImageResult:
        del image
        try:
            return UGModelAppendImageResult(
                added_tokens=self._added_tokens_from_srt_session_view(session, [])
            )
        finally:
            self._clear_pending_segments(session)

    def close_session(self, *, session_id: str) -> None:
        self._messages_by_session.pop(str(session_id), None)
        self._pending_segments_by_session.pop(str(session_id), None)

    def _prepare_text_input(
        self,
        *,
        session: Any,
        message: UGInterleavedMessage,
    ) -> UGSRTPreparedInput:
        text = str(message.content)
        text_token_ids = self._text_token_ids(text)
        input_ids = [self.bos_token_id] + text_token_ids
        token_indices = list(range(1, len(input_ids)))
        metadata = self._segment_metadata(
            session=session,
            segment_type="text",
            source="user_text",
            token_indices=token_indices,
            attention="causal",
            generated_image_commit=False,
        )
        return UGSRTPreparedInput(
            input_ids=input_ids,
            input_text=text,
            messages=[message],
            mot_text_token_indices=token_indices,
            adapter_metadata=metadata,
        )

    def _prepare_image_input(
        self,
        *,
        session: Any,
        message: UGInterleavedMessage,
        generated_image_commit: bool,
    ) -> UGSRTPreparedInput:
        image_token_id = (
            self.generated_image_token_id
            if generated_image_commit
            else self.image_token_id
        )
        input_ids = [self.bos_token_id, image_token_id, image_token_id + 1]
        token_indices = [1, 2]
        source = "generated_image" if generated_image_commit else "input_image"
        metadata = self._segment_metadata(
            session=session,
            segment_type="image",
            source=source,
            token_indices=token_indices,
            attention="hybrid",
            generated_image_commit=generated_image_commit,
        )
        return UGSRTPreparedInput(
            input_ids=input_ids,
            input_text=f"<u1:{source}>",
            messages=[message],
            non_causal_query_attention=True,
            mot_image_token_indices=token_indices,
            adapter_metadata=metadata,
        )

    def _segment_metadata(
        self,
        *,
        session: Any,
        segment_type: str,
        source: str,
        token_indices: list[int],
        attention: str,
        generated_image_commit: bool,
    ) -> dict[str, Any]:
        u1_segment = {
            "segment_type": segment_type,
            "source": source,
            "token_indices": list(token_indices),
            "attention_rows": [
                {
                    "kind": segment_type,
                    "attention": attention,
                    "start": min(token_indices) if token_indices else 0,
                    "end": (max(token_indices) + 1) if token_indices else 0,
                }
            ],
            "generated_image_commit": bool(generated_image_commit),
        }
        previous_segments = self._previous_u1_segments(session)
        u1_state = {
            "segments": previous_segments + [u1_segment],
            "last_segment_type": segment_type,
            "last_source": source,
            "last_generated_image_commit": bool(generated_image_commit),
        }
        self._remember_pending_segment(session, u1_segment)
        return {
            "u1": u1_segment,
            "ug_model_state_updates": {"u1": u1_state},
        }

    def _previous_u1_segments(self, session: Any) -> list[dict[str, Any]]:
        session_metadata = getattr(session, "metadata", {}) or {}
        model_state = session_metadata.get("ug_model_state") or {}
        u1_state = model_state.get("u1") or {}
        segments = [dict(segment) for segment in u1_state.get("segments", [])]
        session_key = self._session_key(session)
        if session_key is not None:
            segments.extend(
                dict(segment)
                for segment in self._pending_segments_by_session.get(session_key, [])
            )
        return segments

    def _has_generated_image_commit(self, session: Any) -> bool:
        return any(
            bool(segment.get("generated_image_commit"))
            for segment in self._previous_u1_segments(session)
        )

    def _remember_pending_segment(self, session: Any, segment: dict[str, Any]) -> None:
        session_key = self._session_key(session)
        if session_key is None:
            return
        self._pending_segments_by_session.setdefault(session_key, []).append(
            dict(segment)
        )

    def _clear_pending_segments(self, session: Any) -> None:
        session_key = self._session_key(session)
        if session_key is not None:
            self._pending_segments_by_session.pop(session_key, None)

    def _remember_session_messages(
        self,
        session: Any,
        messages: list[UGInterleavedMessage],
    ) -> None:
        session_key = self._session_key(session)
        if session_key is None:
            return
        stored = self._messages_by_session.setdefault(session_key, [])
        stored.extend(messages)

    def _messages_for_session(self, session: Any) -> list[UGInterleavedMessage]:
        session_key = getattr(session, "session_id", None)
        if session_key is None:
            session_key = self._session_key(session)
        if session_key is None:
            raise RuntimeError("U1 VLM decode requires a UG session id")
        messages = self._messages_by_session.get(str(session_key), [])
        if not messages:
            raise RuntimeError(
                f"U1 VLM decode has no messages for session {session_key}"
            )
        return list(messages)

    @staticmethod
    def _session_key(session: Any) -> str | None:
        handle = getattr(session, "handle", None)
        session_id = getattr(handle, "session_id", None)
        return str(session_id) if session_id is not None else None

    def _text_token_ids(self, text: str) -> list[int]:
        words = text.split() or [text]
        return [
            self.text_token_base + (sum(word.encode("utf-8")) % 1000) for word in words
        ]

    def _added_tokens_from_srt_session_view(
        self,
        session: Any,
        messages: list[Any],
    ) -> int:
        handle = getattr(session, "handle", None)
        previous_length = int(getattr(handle, "context_length", 0) or 0)
        srt_length = int(getattr(session, "srt_last_origin_input_len", 0) or 0)
        if srt_length > previous_length:
            return srt_length - previous_length
        return sum(self._message_token_count(message) for message in messages)

    def _message_token_count(self, message: Any) -> int:
        if message.type == "text":
            return 1 + len(self._text_token_ids(str(message.content)))
        if message.type == "image":
            return 3
        return 0


class U1SRTBackedUGMiddleBridge:
    """Pixel-flow U1 bridge shell backed by the common SRT UG session runtime."""

    g_kind: UGGKind = "pixel_flow"

    def __init__(self, runtime: UGSessionRuntime) -> None:
        self.runtime = runtime
        self._bridge = SRTBackedUGMiddleBridge(runtime)

    def prepare_u_context(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle:
        return self._bridge.prepare_u_context(
            prompt=prompt,
            image=image,
            think=think,
            think_max_new_tokens=think_max_new_tokens,
        )

    def prepare_u_context_from_messages(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle:
        return self._bridge.prepare_u_context_from_messages(
            messages=messages,
            think=think,
            think_max_new_tokens=think_max_new_tokens,
        )

    def run_g_segment(
        self,
        *,
        contexts: UGContextBundle,
        executor: UGGSegmentExecutor,
    ) -> Any:
        return self._bridge.run_g_segment(contexts=contexts, executor=executor)

    def run_native_pixel_flow_g_segment(
        self,
        *,
        contexts: UGContextBundle,
        batch: Any,
        server_args: Any,
    ) -> UGGSegmentResult | None:
        srt_executor = self.runtime.srt_request_executor
        create_executor = getattr(
            srt_executor,
            "create_u1_native_srt_pixel_flow_executor",
            None,
        )
        if not callable(create_executor):
            return None
        if contexts.full.session is None:
            raise ValueError("U1 native pixel-flow requires a SRT UG session")
        get_binding = getattr(srt_executor, "get_latest_ug_session_token_binding", None)
        if not callable(get_binding):
            raise RuntimeError(
                "U1 native pixel-flow requires latest SRT session token binding"
            )
        binding = get_binding(contexts.full.session.session_id)
        if binding is None:
            raise RuntimeError(
                "U1 native pixel-flow has no SRT KV token binding for session "
                f"{contexts.full.session.session_id}"
            )
        native_executor = create_executor()
        return native_executor.generate(
            contexts=contexts,
            batch=batch,
            server_args=server_args,
            srt_kv_token_binding=binding,
        )

    def commit_generated_segment(
        self,
        *,
        contexts: UGContextBundle,
        segment: Any,
    ) -> None:
        self._bridge.commit_generated_segment(contexts=contexts, segment=segment)

    def release(self, contexts: UGContextBundle) -> None:
        self._bridge.release(contexts)

    def continue_u_decode(self, *, contexts: UGContextBundle) -> UGDecodeResult:
        return self._bridge.continue_u_decode(contexts=contexts)

    def generate_vlm_text(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        return self._bridge.generate_vlm_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
        )


def build_u1_native_vlm_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    image = _first_u1_image_content(messages)
    question = _u1_question_text(messages)
    pixel_values, grid_hw = load_u1_native_image(image)
    input_ids, image_offsets, prompt = build_u1_vlm_input_ids_and_offsets(
        tokenizer=tokenizer,
        grid_hw=grid_hw,
        question=question,
    )

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values,
        model_specific_data={"image_grid_hws": grid_hw},
        offsets=image_offsets,
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        mm_inputs=mm_inputs,
        adapter_metadata={
            "u1": {
                "segment_type": "vlm",
                "source": "native_vlm_input",
                "image_grid_hw": [list(map(int, row)) for row in grid_hw.tolist()],
                "image_offsets": list(image_offsets),
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "vlm",
                    "last_source": "native_vlm_input",
                    "native_vlm_prompt": True,
                    "session_id": getattr(
                        getattr(session, "handle", None), "session_id", None
                    ),
                }
            },
        },
    )


def build_u1_native_t2i_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    prompt_text = _u1_question_text(messages)
    prompt = build_u1_t2i_prompt(prompt=prompt_text)
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError("U1 native T2I prompt produced no input ids")
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    if img_start_id not in input_ids:
        raise RuntimeError("U1 native T2I prompt did not contain <img> token")
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        adapter_metadata={
            "u1": {
                "segment_type": "t2i",
                "source": "native_t2i_prompt",
                "prompt_ends_with_image_marker": input_ids[-1] == img_start_id,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "t2i",
                    "last_source": "native_t2i_prompt",
                    "native_t2i_prompt": True,
                    "open_image_marker": input_ids[-1] == img_start_id,
                    "session_id": getattr(
                        getattr(session, "handle", None), "session_id", None
                    ),
                }
            },
        },
    )


def build_u1_native_generated_image_commit_prepared_input(
    *,
    tokenizer: Any,
    image: Any,
    session: Any | None = None,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
) -> UGSRTPreparedInput:
    pixel_values, grid_hw = load_u1_generated_image_for_commit(
        image,
        patch_size=patch_size,
    )
    merge_size = _u1_merge_size_from_downsample_ratio(downsample_ratio)
    grid_h = int(grid_hw[0, 0])
    grid_w = int(grid_hw[0, 1])
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(
            "U1 generated image patch grid must be divisible by merge size "
            f"{merge_size}, got {grid_h}x{grid_w}"
        )
    token_h = grid_h // merge_size
    token_w = grid_w // merge_size
    num_context_tokens = token_h * token_w
    if num_context_tokens <= 0:
        raise ValueError("U1 generated image commit requires image context tokens")

    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    img_end_id = tokenizer.convert_tokens_to_ids(U1_IMG_END_TOKEN)
    omit_start = _u1_session_has_open_image_marker(session, img_start_id)
    prefix_len = _u1_session_context_length(session)

    input_ids: list[int] = []
    position_ids: list[list[int]] = []
    if omit_start:
        context_t = prefix_len
    else:
        input_ids.append(int(img_start_id))
        position_ids.append([prefix_len, 0, 0])
        context_t = prefix_len + 1

    context_start = len(input_ids)
    input_ids.extend([int(context_id)] * num_context_tokens)
    for h_idx in range(token_h):
        for w_idx in range(token_w):
            position_ids.append([context_t, h_idx, w_idx])
    context_end = len(input_ids) - 1
    end_t = context_t + 1
    input_ids.append(int(img_end_id))
    position_ids.append([end_t, 0, 0])

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )
    import torch

    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values,
        model_specific_data={"image_grid_hws": grid_hw},
        offsets=[(context_start, context_end)],
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    mm_inputs.mrope_positions = torch.tensor(position_ids, dtype=torch.long).t()
    mm_inputs.mrope_position_delta = (
        mm_inputs.mrope_positions[:, -1:]
        .max(
            dim=0,
            keepdim=True,
        )
        .values
    )

    message = UGInterleavedMessage(type="image", content=image)
    metadata = _u1_generated_image_commit_metadata(
        session=session,
        token_indices=list(range(context_start, context_end + 1)),
        grid_hw=grid_hw,
        omit_start=omit_start,
    )
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text="<u1:generated_image_commit>",
        messages=[message],
        position_ids=position_ids,
        mm_inputs=mm_inputs,
        mot_image_token_indices=list(range(context_start, context_end + 1)),
        adapter_metadata=metadata,
    )


def build_u1_t2i_prompt(*, prompt: str) -> str:
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"{U1_IMG_START_TOKEN}"
    )


def build_u1_vlm_input_ids_and_offsets(
    *,
    tokenizer: Any,
    grid_hw: Any,
    question: str,
) -> tuple[list[int], list[tuple[int, int]], str]:
    prompt = build_u1_vlm_prompt(question=question)
    for i in range(int(grid_hw.shape[0])):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * 0.5**2)
        image_tokens = (
            U1_IMG_START_TOKEN
            + U1_IMG_CONTEXT_TOKEN * num_patch_token
            + U1_IMG_END_TOKEN
        )
        prompt = prompt.replace(U1_IMAGE_PLACEHOLDER, image_tokens, 1)

    input_ids = _u1_tokenize_to_ids(tokenizer, prompt)
    context_token_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    selected = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id == context_token_id
    ]
    if not selected:
        raise RuntimeError("U1 native VLM prompt did not contain image context tokens")
    return input_ids, [(selected[0], selected[-1])], prompt


def build_u1_vlm_prompt(*, question: str) -> str:
    return (
        f"<|im_start|>user\n{U1_IMAGE_PLACEHOLDER}\n{question}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


class U1NativeSRTPixelFlowExecutor:
    """Run SenseNova U1 pixel-flow G steps through SRT's ModelRunner/KV path."""

    def __init__(
        self,
        srt_model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.srt_model = srt_model
        self.forward_batch_provider = forward_batch_provider

    def generate(
        self,
        *,
        contexts: UGContextBundle,
        batch: Any,
        server_args: Any,
        srt_kv_token_binding: Any,
    ) -> UGGSegmentResult:
        del server_args
        import numpy as np
        import torch
        from PIL import Image

        if contexts.full.session is None:
            raise ValueError("U1 native pixel-flow requires contexts.full.session")
        sampling_params = batch.sampling_params
        cfg_text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
        cfg_img_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
        if cfg_text_scale > 1.0 or cfg_img_scale > 1.0:
            raise NotImplementedError(
                "U1 native SRT pixel-flow currently supports CFG scale <= 1.0; "
                "CFG side branches are a later parity step"
            )

        image_size = _u1_batch_image_size(batch)
        width, height = image_size
        patch_size = int(self.srt_model.patch_size)
        merge_size = int(1 / float(self.srt_model.downsample_ratio))
        divisor = patch_size * merge_size
        if width % divisor or height % divisor:
            raise ValueError(
                "U1 native pixel-flow image size must be divisible by "
                f"{divisor}, got {width}x{height}"
            )

        token_h = height // divisor
        token_w = width // divisor
        grid_h = height // patch_size
        grid_w = width // patch_size
        steps = int(getattr(sampling_params, "num_inference_steps", None) or 0)
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}")

        device = _u1_model_device(self.srt_model)
        dtype = _u1_model_dtype(self.srt_model)
        seed = int(getattr(batch, "seed", None) or 0)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise_scale = float(
            self.srt_model.noise_scale_for_image(grid_h=grid_h, grid_w=grid_w)
        )
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=device, dtype=torch.long)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = self.srt_model.apply_time_schedule(
            timesteps,
            image_seq_len=token_h * token_w,
            timestep_shift=float(getattr(sampling_params, "timestep_shift", 1.0)),
        )
        indexes_image = self.srt_model.build_t2i_image_indexes(
            token_h=token_h,
            token_w=token_w,
            text_len=int(getattr(srt_kv_token_binding, "token_count")),
            device=device,
        )
        generation_input = {
            "packed_seqlens": torch.tensor(
                [token_h * token_w], dtype=torch.int32, device=device
            ),
            "packed_position_ids": indexes_image,
        }
        prepared = SimpleNamespace(
            generation_input=generation_input,
            srt_kv_token_binding=srt_kv_token_binding,
        )

        for step_i in range(steps):
            timestep = timesteps[step_i]
            next_timestep = timesteps[step_i + 1]
            z = self.srt_model.patchify(image_prediction, patch_size * merge_size)
            image_input = self.srt_model.patchify(
                image_prediction,
                patch_size,
                channel_first=True,
            )
            image_embeds = self.srt_model.extract_feature(
                image_input.view(grid_h * grid_w, -1),
                gen_model=True,
                grid_hw=gen_grid_hw,
            ).view(1, token_h * token_w, -1)
            timestep_values = timestep.expand(token_h * token_w)
            timestep_embeddings = self.srt_model.fm_modules["timestep_embedder"](
                timestep_values
            ).view(1, token_h * token_w, -1)
            if getattr(self.srt_model, "add_noise_scale_embedding", False):
                noise_values = torch.full_like(
                    timestep_values,
                    noise_scale / float(self.srt_model.noise_scale_max_value),
                )
                timestep_embeddings = timestep_embeddings + self.srt_model.fm_modules[
                    "noise_scale_embedder"
                ](noise_values).view(1, token_h * token_w, -1)
            image_embeds = image_embeds + timestep_embeddings

            forward_batch_context = self.forward_batch_provider(
                prepared=prepared,
                latent_tokens=image_embeds,
                timestep=timestep,
            )
            forward_batch = getattr(
                forward_batch_context,
                "forward_batch",
                forward_batch_context,
            )
            try:
                v_pred = self.srt_model.predict_u1_pixel_flow_from_srt(
                    image_embeds=image_embeds,
                    indexes_image=indexes_image,
                    forward_batch=forward_batch,
                    timestep=timestep,
                    z=z,
                    image_size=image_size,
                )
            finally:
                release = getattr(forward_batch_context, "release", None)
                if callable(release):
                    release()

            z = z + (next_timestep - timestep) * v_pred
            image_prediction = self.srt_model.unpatchify(
                z,
                patch_size * merge_size,
                height,
                width,
            )

        array = (
            (image_prediction[0].float() * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        image = Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB")
        return UGGSegmentResult(
            type="image",
            image=image,
            metadata={
                "g_kind": "pixel_flow",
                "native_srt_pixel_flow": True,
                "temporary_g_kv": True,
                "timesteps": steps,
                "seed": seed,
                "width": width,
                "height": height,
                "grid": (token_h, token_w),
                "noise_scale": noise_scale,
            },
        )


def _u1_tokenize_to_ids(
    tokenizer: Any,
    prompt: str,
    *,
    add_special_tokens: bool | None = None,
) -> list[int]:
    kwargs = {"return_tensors": "pt"}
    if add_special_tokens is not None:
        kwargs["add_special_tokens"] = add_special_tokens
    try:
        tokenized = tokenizer(prompt, **kwargs)
    except TypeError:
        tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    if hasattr(input_ids, "tolist"):
        return input_ids[0].tolist()
    return list(input_ids[0])


def _u1_batch_image_size(batch: Any) -> tuple[int, int]:
    sampling_params = batch.sampling_params
    height = _u1_first_int(
        getattr(batch, "height", None),
        getattr(sampling_params, "height", None),
        default=1024,
    )
    width = _u1_first_int(
        getattr(batch, "width", None),
        getattr(sampling_params, "width", None),
        default=1024,
    )
    return width, height


def _u1_first_int(*values, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return int(default)


def _u1_model_device(srt_model: Any):
    import torch

    vision_model = getattr(srt_model, "vision_model", None)
    device = getattr(vision_model, "device", None)
    if device is not None:
        return device
    return next(srt_model.parameters()).device


def _u1_model_dtype(srt_model: Any):
    vision_model = getattr(srt_model, "vision_model", None)
    dtype = getattr(vision_model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(srt_model.parameters()).dtype


def load_u1_native_image(
    image: Any,
    *,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
):
    if isinstance(image, dict):
        pixel_values = image.get("pixel_values")
        grid_hw = image.get("grid_hw", image.get("image_grid_hws"))
        if pixel_values is not None and grid_hw is not None:
            import torch

            return (
                torch.as_tensor(pixel_values, dtype=torch.float32),
                torch.as_tensor(grid_hw, dtype=torch.long),
            )

    import numpy as np
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    resized = _u1_dynamic_preprocess_native_resolution(
        image,
        size_factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    pixel_values = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(U1_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(U1_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    return _u1_preprocess_pixel_values(pixel_values, patch_size=patch_size)


def load_u1_generated_image_for_commit(
    image: Any,
    *,
    patch_size: int = 16,
):
    if isinstance(image, dict):
        pixel_values = image.get("pixel_values")
        grid_hw = image.get("grid_hw", image.get("image_grid_hws"))
        if pixel_values is not None and grid_hw is not None:
            import torch

            return (
                torch.as_tensor(pixel_values, dtype=torch.float32),
                torch.as_tensor(grid_hw, dtype=torch.long),
            )

    import numpy as np
    import torch
    from PIL import Image

    if torch.is_tensor(image):
        pixel_values = image.detach().float().cpu()
        if pixel_values.ndim == 4:
            if int(pixel_values.shape[0]) != 1:
                raise ValueError(
                    "U1 generated image commit expects a single image tensor, "
                    f"got batch={int(pixel_values.shape[0])}"
                )
            pixel_values = pixel_values[0]
        if pixel_values.ndim != 3 or int(pixel_values.shape[0]) != 3:
            raise ValueError(
                "U1 generated image commit tensor must have shape [3,H,W] "
                f"or [1,3,H,W], got {tuple(pixel_values.shape)}"
            )
        if float(pixel_values.min()) < 0.0:
            pixel_values = pixel_values * 0.5 + 0.5
        pixel_values = pixel_values.clamp(0, 1)
    else:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1)

    height = int(pixel_values.shape[1])
    width = int(pixel_values.shape[2])
    if height % patch_size or width % patch_size:
        raise ValueError(
            "U1 generated image commit requires image size divisible by "
            f"patch_size={patch_size}, got {width}x{height}"
        )
    mean = torch.tensor(U1_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(U1_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    return _u1_preprocess_pixel_values(pixel_values, patch_size=patch_size)


def _u1_session_context_length(session: Any | None) -> int:
    handle = getattr(session, "handle", None)
    context_length = getattr(handle, "context_length", None)
    if context_length is not None:
        return int(context_length)
    return int(getattr(session, "srt_last_origin_input_len", 0) or 0)


def _u1_merge_size_from_downsample_ratio(downsample_ratio: float) -> int:
    if downsample_ratio <= 0:
        raise ValueError(f"downsample_ratio must be > 0, got {downsample_ratio}")
    merge_size = int(1 / downsample_ratio)
    if merge_size <= 0 or abs((1 / merge_size) - downsample_ratio) > 1e-6:
        raise ValueError(
            "U1 downsample_ratio must be the reciprocal of an integer, "
            f"got {downsample_ratio}"
        )
    return merge_size


def _u1_session_has_open_image_marker(
    session: Any | None,
    img_start_token_id: int,
) -> bool:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    u1_state = model_state.get("u1") or {}
    if bool(u1_state.get("open_image_marker")):
        return True
    last_output_ids = metadata.get("srt_last_u_decode_output_ids") or ()
    return bool(last_output_ids) and int(last_output_ids[-1]) == int(img_start_token_id)


def _u1_generated_image_commit_metadata(
    *,
    session: Any | None,
    token_indices: list[int],
    grid_hw: Any,
    omit_start: bool,
) -> dict[str, Any]:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    previous_state = model_state.get("u1") or {}
    previous_segments = [
        dict(segment) for segment in previous_state.get("segments", [])
    ]
    u1_segment = {
        "segment_type": "image",
        "source": "native_generated_image_commit",
        "token_indices": list(token_indices),
        "attention_rows": [
            {
                "kind": "image",
                "attention": "hybrid",
                "start": min(token_indices) if token_indices else 0,
                "end": (max(token_indices) + 1) if token_indices else 0,
            }
        ],
        "generated_image_commit": True,
        "native_generated_image_commit": True,
        "omit_image_start": bool(omit_start),
        "image_grid_hw": [list(map(int, row)) for row in grid_hw.tolist()],
    }
    u1_state = {
        "segments": previous_segments + [u1_segment],
        "last_segment_type": "image",
        "last_source": "native_generated_image_commit",
        "last_generated_image_commit": True,
        "native_generated_image_commit": True,
        "open_image_marker": False,
    }
    return {
        "u1": u1_segment,
        "ug_model_state_updates": {"u1": u1_state},
    }


def _u1_dynamic_preprocess_native_resolution(
    image: Any,
    *,
    size_factor: int,
    min_pixels: int,
    max_pixels: int,
):
    width, height = image.size
    resized_height, resized_width = _u1_smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


def _u1_smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _u1_preprocess_pixel_values(pixel_values: Any, *, patch_size: int):
    import torch

    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )
    grid_hw = torch.tensor([[grid_h, grid_w]], dtype=torch.long)
    return flatten_pixel_values.to(torch.float32), grid_hw


def _not_wired() -> NotImplementedError:
    return NotImplementedError(
        "SenseNova U1 UG backend is not wired yet. This shell only declares "
        "the pixel_flow capability; U path, G pixel-flow mechanics, and true "
        "weights are covered by later roadmap items."
    )


def _first_u1_image_path(messages: list[UGInterleavedMessage]) -> Path:
    content = _first_u1_image_content(messages)
    if isinstance(content, (str, Path)):
        return Path(content)
    save = getattr(content, "save", None)
    if callable(save):
        path = Path(tempfile.mkdtemp(prefix="u1-vlm-image-")) / "image.png"
        save(path)
        return path
    raise TypeError(
        "U1 VLM image message must be a path or PIL image, " f"got {type(content)}"
    )


def _first_u1_image_content(messages: list[UGInterleavedMessage]) -> Any:
    for message in messages:
        if message.type != "image":
            continue
        return message.content
    raise ValueError("U1 VLM text generation requires an image message")


def _u1_question_text(messages: list[UGInterleavedMessage]) -> str:
    parts = [str(message.content) for message in messages if message.type == "text"]
    question = "\n".join(part for part in parts if part)
    if not question:
        raise ValueError("U1 VLM text generation requires a text question")
    return question


def _tail(text: str | bytes | None, *, limit: int = 2000) -> str | None:
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]
