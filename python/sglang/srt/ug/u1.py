# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from sglang.srt.ug.adapter import UGModelAppendImageResult, UGModelPrefillResult
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.denoiser import (
    SRTBackedUGMiddleBridge,
    UGGSegmentExecutor,
)
from sglang.srt.ug.interleaved import UGGKind
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGSessionRuntime,
    UGSegmentState,
    UGSRTPreparedInput,
    UGVLMTextGenerationResult,
)


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

    def __init__(self) -> None:
        self.observed_u_forwards: list[dict[str, Any]] = []
        self._pending_segments_by_session: dict[str, list[dict[str, Any]]] = {}

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
            return UGModelPrefillResult(
                added_tokens=self._added_tokens_from_srt_session_view(session, messages)
            )
        finally:
            self._clear_pending_segments(session)

    def decode_next_segment(self, *, session: Any) -> Any:
        del session
        raise _not_wired()

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: Any,
        max_new_tokens: int,
    ) -> Any:
        del runtime, session, max_new_tokens
        raise _not_wired()

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
        del session_id

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


def _not_wired() -> NotImplementedError:
    return NotImplementedError(
        "SenseNova U1 UG backend is not wired yet. This shell only declares "
        "the pixel_flow capability; U path, G pixel-flow mechanics, and true "
        "weights are covered by later roadmap items."
    )
