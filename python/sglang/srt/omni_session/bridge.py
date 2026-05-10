# SPDX-License-Identifier: Apache-2.0
"""Middle bridge between generic omni orchestration and SRT session execution."""

from abc import ABC, abstractmethod
from typing import Any

from sglang.srt.omni_session.runtime_protocol import (
    OmniContextBundle,
    OmniContextHandle,
    OmniSessionHandle,
)
from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniInterleavedMessage,
    OmniSessionRuntime,
    OmniVLMTextGenerationResult,
)

DEFAULT_OMNI_TEXT_MAX_NEW_TOKENS = 128


class OmniSessionBridge(ABC):
    """bridge surface between generic omni orchestrator and SRT session runtime, translating ARBackend semantics into srt-session semantics

      e.g., U1SRTBackedOmniSessionBridge breaks down the generation process into:
       1. prefill
       2. decode until an image marker is met
       3. commit generated image into session

    """

    generation_kind: str = "generic"

    @abstractmethod
    def prepare_ar_context_from_messages(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
        session_id: str | None = None,
    ) -> OmniContextBundle: ...

    @abstractmethod
    def commit_generated_segment(
        self, *, contexts: OmniContextBundle, segment: Any
    ) -> None: ...

    @abstractmethod
    def release(self, contexts: OmniContextBundle) -> None: ...

    @abstractmethod
    def continue_ar_decode(
        self, *, contexts: OmniContextBundle
    ) -> OmniDecodeResult: ...

    @abstractmethod
    def generate_vlm_answer(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        """prefill image/text inputs and return a plain VLM text answer"""
        ...


class SRTBackedOmniSessionBridge(OmniSessionBridge):
    """omni middle bridge that keeps SRT as the session/KV owner.

    It asks the runtime to prefill/decode/commit AR-side chunks. Generation-side
    context access is exposed by `sglang.omni.backends.ar.srt`.
    """

    def __init__(
        self,
        runtime: OmniSessionRuntime,
        *,
        max_pre_image_decode_steps: int = 16,
    ) -> None:
        if max_pre_image_decode_steps <= 0:
            raise ValueError(
                "max_pre_image_decode_steps must be positive, got "
                f"{max_pre_image_decode_steps}"
            )
        self.runtime = runtime
        self.max_pre_image_decode_steps = max_pre_image_decode_steps

    def prepare_ar_context_from_messages(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
        session_id: str | None = None,
    ) -> OmniContextBundle:
        messages = normalize_omni_interleaved_messages(messages)
        session = self.runtime.prefill_interleaved(messages, session_id=session_id)
        pre_image_segments: list[dict[str, Any]] = []
        try:
            if think:
                thinking = self._decode_thinking_text(
                    session,
                    max_new_tokens=think_max_new_tokens,
                )
                session = thinking.session
                if thinking.text:
                    pre_image_segments.append(
                        {
                            "type": "text",
                            "text": thinking.text,
                            "metadata": {
                                "role": "think",
                                "token_ids": [
                                    int(token_id)
                                    for token_id in thinking.next_token_ids
                                ],
                            },
                        }
                    )
            for _ in range(self.max_pre_image_decode_steps):
                segment = self.runtime.decode_next_segment(session)
                if segment.type == "image_marker":
                    break
                if segment.type == "text":
                    pre_image_segments.append(
                        {
                            "type": "text",
                            "text": segment.text or "",
                            "metadata": {
                                "token_ids": [
                                    int(token_id) for token_id in segment.token_ids
                                ]
                            },
                        }
                    )
                    continue
                raise ValueError(
                    "omni middle bridge expected AR decode to request an image segment, "
                    f"got {segment.type}"
                )
            else:
                decoded_preview = "".join(
                    str(segment.get("text") or "")
                    for segment in pre_image_segments
                    if segment.get("type") == "text"
                )
                decoded_preview = decoded_preview[-240:]
                raise ValueError(
                    "omni middle bridge did not receive an image marker within "
                    f"{self.max_pre_image_decode_steps} AR decode steps"
                    f"; decoded_text_preview={decoded_preview!r}"
                )
        except Exception:
            self.runtime.close_session(session)
            raise

        text_tokens = sum(
            len(str(message.content).split())
            for message in messages
            if message.type == "text"
        )
        image_tokens = sum(2 for message in messages if message.type == "image")
        return OmniContextBundle(
            full=OmniContextHandle(
                session.anchor_request_id,
                session.context_length,
                session=session,
                metadata={"pre_image_segments": pre_image_segments},
            ),
            text_cfg=OmniContextHandle(
                f"{session.anchor_request_id}:text_cfg",
                image_tokens,
                session=session,
            ),
            image_cfg=OmniContextHandle(
                f"{session.anchor_request_id}:image_cfg",
                text_tokens,
                session=session,
            ),
        )

    def commit_generated_segment(
        self, *, contexts: OmniContextBundle, segment: Any
    ) -> None:
        if segment.type != "image":
            raise ValueError(f"omni commit expects image segment, got {segment.type}")
        image_for_commit = segment.commit_image
        if image_for_commit is None:
            image_for_commit = segment.image
        self._commit_generated_image(contexts=contexts, image=image_for_commit)

    def release(self, contexts: OmniContextBundle) -> None:
        if contexts.full.session is not None:
            self.runtime.close_session(contexts.full.session)

    def continue_ar_decode(self, *, contexts: OmniContextBundle) -> OmniDecodeResult:
        """continue interleaved AR decode from an SRT-owned context bundle"""
        if contexts.full.session is None:
            raise ValueError("SRT-backed omni contexts require a session handle")
        return self.runtime.decode_next_segment(contexts.full.session)

    def _decode_thinking_text(
        self,
        session: OmniSessionHandle,
        *,
        max_new_tokens: int | None,
    ) -> OmniVLMTextGenerationResult:
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_OMNI_TEXT_MAX_NEW_TOKENS
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError(
                f"omni think text generation requires max_new_tokens > 0, got {max_new_tokens}"
            )
        try:
            return self.runtime.model_policy.decode_vlm_text(
                runtime=self.runtime,
                session=session,
                max_new_tokens=max_new_tokens,
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"{self.runtime.model_policy.__class__.__name__} does not support "
                "omni think text generation"
            ) from exc

    def generate_vlm_answer(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        """run one-shot VLM QA: prefill inputs, decode text, and return the answer"""
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError(
                f"omni VLM text generation requires max_new_tokens > 0, got {max_new_tokens}"
            )

        # start the prefill and get the session
        session = self.runtime.prefill_interleaved(
            normalize_omni_interleaved_messages(messages)
        )
        try:
            return self.runtime.model_policy.decode_vlm_text(
                runtime=self.runtime,
                session=session,
                max_new_tokens=max_new_tokens,
            )
        except Exception:
            self.runtime.close_session(session)
            raise

    def _commit_generated_image(
        self, *, contexts: OmniContextBundle, image: Any | None
    ) -> None:
        if contexts.full.session is None:
            raise ValueError("SRT-backed omni contexts require a session handle")
        session = self.runtime.append_generated_image(
            contexts.full.session, image=image
        )
        contexts.full.request_id = session.anchor_request_id
        contexts.full.token_count = session.context_length
        contexts.full.session = session
        contexts.text_cfg.session = session
        contexts.image_cfg.session = session


def normalize_omni_interleaved_messages(
    messages: list[OmniInterleavedMessage | dict[str, Any]],
) -> list[OmniInterleavedMessage]:
    normalized: list[OmniInterleavedMessage] = []
    for message in messages:
        if isinstance(message, OmniInterleavedMessage):
            normalized.append(message)
            continue
        if not isinstance(message, dict):
            raise TypeError(
                f"omni message must be a dict or OmniInterleavedMessage: {message!r}"
            )
        message_type = message.get("type")
        if message_type == "text":
            content = message.get("text", message.get("content"))
        elif message_type == "image":
            content = message.get("image", message.get("content"))
        else:
            raise ValueError(f"Unsupported omni message type: {message_type!r}")
        if content is None:
            raise ValueError(f"omni {message_type} message is missing content")
        normalized.append(OmniInterleavedMessage(type=message_type, content=content))
    if not normalized:
        raise ValueError("omni interleaved messages must not be empty")
    return normalized
