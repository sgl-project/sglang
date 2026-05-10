# SPDX-License-Identifier: Apache-2.0
"""Middle bridge between generic omni orchestration and SRT session execution."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Protocol

from sglang.srt.omni_session.runtime_protocol import (
    OmniContextBundle,
    OmniContextHandle,
    OmniSessionHandle,
)
from sglang.srt.omni_session.interleaved_protocol import (
    DEFAULT_OMNI_TEXT_MAX_NEW_TOKENS,
    GeneratedSegmentResult,
    GenerationKind,
)
from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniInterleavedMessage,
    OmniSessionRuntime,
    OmniVLMTextGenerationResult,
)

GeneratedSegmentExecutor = Callable[[Any], GeneratedSegmentResult]


class OmniSessionBridge(Protocol):
    generation_kind: GenerationKind

    def prepare_ar_context(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
    ) -> OmniContextBundle: ...

    def prepare_ar_context_from_messages(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
        session_id: str | None = None,
    ) -> OmniContextBundle: ...

    def run_generated_segment(
        self,
        *,
        contexts: OmniContextBundle,
        executor: GeneratedSegmentExecutor,
    ) -> GeneratedSegmentResult: ...

    def commit_generated_segment(
        self, *, contexts: OmniContextBundle, segment: GeneratedSegmentResult
    ) -> None: ...

    def release(self, contexts: OmniContextBundle) -> None: ...

    def continue_ar_decode(
        self, *, contexts: OmniContextBundle
    ) -> OmniDecodeResult: ...

    def generate_vlm_text(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult: ...


class SRTBackedGenerationContextOps:
    """Narrow generation-side view of an SRT-backed omni context.

    generation backends consume this object instead of reaching through the omni bridge or
    runtime. It exposes only model access, logical context positions, and
    temporary query execution.
    """

    def __init__(self, bridge: Any, contexts: OmniContextBundle) -> None:
        self._bridge = bridge
        self._contexts = contexts

    @property
    def generation_kind(self) -> GenerationKind:
        return self._bridge.generation_kind

    @property
    def session_id(self) -> str:
        session = self._contexts.full.session
        if session is None:
            raise ValueError(
                "SRT-backed generation context ops require a session handle"
            )
        return session.session_id

    @property
    def metadata(self) -> dict[str, Any]:
        return self._contexts.full.metadata

    def get_role(self, name: str, default: str) -> str:
        attr_name = name if name.endswith("_role") else f"{name}_role"
        return str(getattr(self._bridge, attr_name, default))

    def get_model(self) -> Any:
        get_srt_model = getattr(self._executor(), "get_srt_model", None)
        if not callable(get_srt_model):
            raise RuntimeError("generation context ops require model access")
        return get_srt_model()

    def get_position_count(self, *, sidecar_role: str | None = None) -> int | None:
        get_position_count = getattr(
            self._executor(),
            "get_latest_session_position_count",
            None,
        )
        if not callable(get_position_count):
            raise RuntimeError(
                "generation context ops require latest context position count"
            )
        return get_position_count(self.session_id, sidecar_role=sidecar_role)

    def build_temporary_forward_batch(
        self,
        *,
        prepared: Any,
        generation_query_embeds: Any,
        timestep: Any,
    ) -> Any:
        build_forward_batch = getattr(
            self._executor(),
            "build_temporary_context_forward_batch_for_session",
            None,
        )
        if not callable(build_forward_batch):
            raise RuntimeError(
                "generation context ops require temporary query forward batches"
            )
        return build_forward_batch(
            prepared=self._to_srt_prepared(prepared),
            generation_query_embeds=generation_query_embeds,
            timestep=timestep,
        )

    def _executor(self) -> Any:
        runtime = getattr(self._bridge, "runtime", None)
        executor = getattr(runtime, "srt_request_executor", None)
        if executor is None:
            raise RuntimeError(
                "SRT-backed generation context ops require a request executor"
            )
        return executor

    def _to_srt_prepared(self, prepared: Any) -> Any:
        if getattr(prepared, "srt_session_id", None) is not None:
            return prepared
        data = dict(getattr(prepared, "__dict__", {}) or {})
        data["srt_session_id"] = data.get("session_id", self.session_id)
        data["srt_sidecar_role"] = data.get("sidecar_role")
        return SimpleNamespace(**data)


class SRTBackedOmniSessionBridge:
    """omni middle bridge that keeps SRT as the session/KV owner.

    It asks the runtime to prefill/decode/commit AR-side chunks, while generation-side
    code receives only `SRTBackedGenerationContextOps`.
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

    def prepare_ar_context(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
    ) -> OmniContextBundle:
        messages = self.runtime.normalize_messages(prompt=prompt, image=image)
        return self.prepare_ar_context_from_messages(
            messages=messages,
            think=think,
            think_max_new_tokens=think_max_new_tokens,
            sampling_params=sampling_params,
        )

    def prepare_ar_context_from_messages(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: Any | None = None,
        session_id: str | None = None,
    ) -> OmniContextBundle:
        del sampling_params
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

    def run_generated_segment(
        self,
        *,
        contexts: OmniContextBundle,
        executor: GeneratedSegmentExecutor,
    ) -> GeneratedSegmentResult:
        if contexts.full.session is None:
            raise ValueError("SRT-backed omni contexts require a session handle")
        segment = executor(SRTBackedGenerationContextOps(self, contexts))
        if segment.type != "image":
            raise ValueError(
                f"omni generated segment expected image output, got {segment.type}"
            )
        return segment

    def commit_generated_segment(
        self, *, contexts: OmniContextBundle, segment: GeneratedSegmentResult
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
        if contexts.full.session is None:
            raise ValueError("SRT-backed omni contexts require a session handle")
        return self.runtime.decode_next_segment(contexts.full.session)

    def _decode_thinking_text(
        self,
        session: OmniSessionHandle,
        *,
        max_new_tokens: int | None,
    ) -> OmniVLMTextGenerationResult:
        decode_vlm_text = getattr(self.runtime.model_runner, "decode_vlm_text", None)
        if not callable(decode_vlm_text):
            raise RuntimeError(
                f"{self.runtime.model_runner.__class__.__name__} does not support "
                "omni think text generation"
            )
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_OMNI_TEXT_MAX_NEW_TOKENS
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError(
                f"omni think text generation requires max_new_tokens > 0, got {max_new_tokens}"
            )
        return decode_vlm_text(
            runtime=self.runtime,
            session=session,
            max_new_tokens=max_new_tokens,
        )

    def generate_vlm_text(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError(
                f"omni VLM text generation requires max_new_tokens > 0, got {max_new_tokens}"
            )
        session = self.runtime.prefill_interleaved(
            normalize_omni_interleaved_messages(messages)
        )
        try:
            decode_vlm_text = getattr(
                self.runtime.model_runner, "decode_vlm_text", None
            )
            if callable(decode_vlm_text):
                return decode_vlm_text(
                    runtime=self.runtime,
                    session=session,
                    max_new_tokens=max_new_tokens,
                )
            segment = self.runtime.decode_next_segment(session)
            if segment.type != "text":
                raise ValueError(
                    "omni VLM text generation expected a text segment, "
                    f"got {segment.type}"
                )
            return OmniVLMTextGenerationResult(
                session=session,
                text=segment.text or "",
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
