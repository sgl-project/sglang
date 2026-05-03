# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol

import torch

from sglang.srt.ug.context import UGContextBundle, UGContextHandle, UGSessionHandle
from sglang.srt.ug.interleaved import DEFAULT_UG_TEXT_MAX_NEW_TOKENS
from sglang.srt.ug.runtime import (
    FakeUGModelRunner,
    UGDecodeResult,
    UGInterleavedMessage,
    UGLatentDecodeRequest,
    UGLatentPrepareRequest,
    UGLatentPrepareResult,
    UGSessionRuntime,
    UGVelocityRequest,
    UGVLMTextGenerationResult,
)


class UGDenoiserBridge(Protocol):
    def build_contexts(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle: ...

    def build_contexts_from_messages(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle: ...

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor: ...

    def release_contexts(self, contexts: UGContextBundle) -> None: ...

    def prepare_latents(
        self,
        *,
        contexts: UGContextBundle,
        sampling_params: Any,
        seed: int | None,
    ) -> UGLatentPrepareResult | None: ...

    def append_generated_image(
        self, *, contexts: UGContextBundle, image: Any | None
    ) -> None: ...

    def decode_latents(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        sampling_params: Any,
    ) -> Any | None: ...

    def decode_next_segment(self, *, contexts: UGContextBundle) -> UGDecodeResult: ...

    def generate_vlm_text(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult: ...


class FakeUGDenoiserBridge:
    def build_contexts(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle:
        del think, think_max_new_tokens
        messages = UGSessionRuntime.normalize_messages(prompt=prompt, image=image)
        return self.build_contexts_from_messages(messages=messages)

    def build_contexts_from_messages(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle:
        del think, think_max_new_tokens
        normalized = normalize_ug_interleaved_messages(messages)
        text_tokens = sum(
            len(str(message.content).split())
            for message in normalized
            if message.type == "text"
        )
        image_tokens = sum(2 for message in normalized if message.type == "image")
        return UGContextBundle(
            full=UGContextHandle("full", text_tokens + image_tokens),
            text_cfg=UGContextHandle("text_cfg", image_tokens),
            image_cfg=UGContextHandle("image_cfg", text_tokens),
        )

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor:
        del latent_position_ids, sampling_params
        scale = 1.0 + contexts.full.token_count * 0.01
        return latent_tokens + scale * timestep.reshape(-1, 1, 1).to(latent_tokens)

    def release_contexts(self, contexts: UGContextBundle) -> None:
        del contexts

    def prepare_latents(
        self,
        *,
        contexts: UGContextBundle,
        sampling_params: Any,
        seed: int | None,
    ) -> UGLatentPrepareResult | None:
        del contexts, sampling_params, seed
        return None

    def append_generated_image(
        self, *, contexts: UGContextBundle, image: Any | None
    ) -> None:
        del contexts, image

    def decode_latents(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        sampling_params: Any,
    ) -> Any | None:
        del contexts, latent_tokens, sampling_params
        return None

    def decode_next_segment(self, *, contexts: UGContextBundle) -> UGDecodeResult:
        del contexts
        return UGDecodeResult(type="done")

    def generate_vlm_text(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        normalized = normalize_ug_interleaved_messages(messages)
        token_ids = tuple(range(1, max(1, int(max_new_tokens)) + 1))
        token_count = sum(
            len(str(message.content).split()) if message.type == "text" else 2
            for message in normalized
        )
        return UGVLMTextGenerationResult(
            session=UGSessionHandle(
                session_id="fake-vlm",
                anchor_request_id="fake-vlm:u0",
                context_length=token_count,
                context_version=0,
            ),
            text="generated_text",
            token_ids=token_ids,
        )


class SRTBackedUGDenoiserBridge:
    """Diffusion-side bridge that delegates UG model work to SRT runtime state."""

    def __init__(
        self,
        runtime: UGSessionRuntime | None = None,
        *,
        max_pre_image_decode_steps: int = 16,
    ) -> None:
        if max_pre_image_decode_steps <= 0:
            raise ValueError(
                "max_pre_image_decode_steps must be positive, got "
                f"{max_pre_image_decode_steps}"
            )
        self.runtime = runtime or UGSessionRuntime(model_runner=FakeUGModelRunner())
        self.max_pre_image_decode_steps = max_pre_image_decode_steps

    def build_contexts(
        self,
        *,
        prompt: str | list[str] | None,
        image: Any | None,
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle:
        messages = self.runtime.normalize_messages(prompt=prompt, image=image)
        return self.build_contexts_from_messages(
            messages=messages,
            think=think,
            think_max_new_tokens=think_max_new_tokens,
        )

    def build_contexts_from_messages(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
    ) -> UGContextBundle:
        messages = normalize_ug_interleaved_messages(messages)
        session = self.runtime.prefill_interleaved(messages)
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
                            "metadata": {"role": "think"},
                        }
                    )
            for _ in range(self.max_pre_image_decode_steps):
                segment = self.runtime.decode_next_segment(session)
                if segment.type == "image_marker":
                    break
                if segment.type == "text":
                    pre_image_segments.append(
                        {"type": "text", "text": segment.text or ""}
                    )
                    continue
                raise ValueError(
                    "UG denoise bridge expected U decode to request an image segment, "
                    f"got {segment.type}"
                )
            else:
                raise ValueError(
                    "UG denoise bridge did not receive an image marker within "
                    f"{self.max_pre_image_decode_steps} U decode steps"
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
        return UGContextBundle(
            full=UGContextHandle(
                session.anchor_request_id,
                session.context_length,
                session=session,
                metadata={"pre_image_segments": pre_image_segments},
            ),
            text_cfg=UGContextHandle(
                f"{session.anchor_request_id}:text_cfg",
                image_tokens,
                session=session,
            ),
            image_cfg=UGContextHandle(
                f"{session.anchor_request_id}:image_cfg",
                text_tokens,
                session=session,
            ),
        )

    def _decode_thinking_text(
        self,
        session: UGSessionHandle,
        *,
        max_new_tokens: int | None,
    ) -> UGVLMTextGenerationResult:
        decode_vlm_text = getattr(self.runtime.model_runner, "decode_vlm_text", None)
        if not callable(decode_vlm_text):
            raise RuntimeError(
                f"{self.runtime.model_runner.__class__.__name__} does not support "
                "UG think text generation"
            )
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_UG_TEXT_MAX_NEW_TOKENS
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError(
                f"UG think text generation requires max_new_tokens > 0, got {max_new_tokens}"
            )
        return decode_vlm_text(
            runtime=self.runtime,
            session=session,
            max_new_tokens=max_new_tokens,
        )

    def generate_vlm_text(
        self,
        *,
        messages: list[UGInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> UGVLMTextGenerationResult:
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError(
                f"UG VLM text generation requires max_new_tokens > 0, got {max_new_tokens}"
            )
        session = self.runtime.prefill_interleaved(
            normalize_ug_interleaved_messages(messages)
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
                    "UG VLM text generation expected a text segment, "
                    f"got {segment.type}"
                )
            return UGVLMTextGenerationResult(
                session=session,
                text=segment.text or "",
            )
        except Exception:
            self.runtime.close_session(session)
            raise

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        response = self.runtime.predict_velocity(
            UGVelocityRequest(
                session=contexts.full.session,
                latent_tokens=latent_tokens,
                timestep=timestep,
                latent_position_ids=latent_position_ids,
                sampling_params=sampling_params,
            )
        )
        contexts.full.session = response.session
        contexts.text_cfg.session = response.session
        contexts.image_cfg.session = response.session
        return response.velocity

    def prepare_latents(
        self,
        *,
        contexts: UGContextBundle,
        sampling_params: Any,
        seed: int | None,
    ) -> UGLatentPrepareResult | None:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        return self.runtime.prepare_latents(
            UGLatentPrepareRequest(
                session=contexts.full.session,
                sampling_params=sampling_params,
                seed=seed,
            )
        )

    def release_contexts(self, contexts: UGContextBundle) -> None:
        if contexts.full.session is not None:
            self.runtime.close_session(contexts.full.session)

    def append_generated_image(
        self, *, contexts: UGContextBundle, image: Any | None
    ) -> None:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        session = self.runtime.append_generated_image(
            contexts.full.session, image=image
        )
        contexts.full.request_id = session.anchor_request_id
        contexts.full.token_count = session.context_length
        contexts.full.session = session
        contexts.text_cfg.session = session
        contexts.image_cfg.session = session

    def decode_latents(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        sampling_params: Any,
    ) -> Any | None:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        return self.runtime.decode_latents_to_image(
            UGLatentDecodeRequest(
                session=contexts.full.session,
                latent_tokens=latent_tokens,
                sampling_params=sampling_params,
            )
        )

    def decode_next_segment(self, *, contexts: UGContextBundle) -> UGDecodeResult:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        return self.runtime.decode_next_segment(contexts.full.session)


def normalize_ug_interleaved_messages(
    messages: list[UGInterleavedMessage | dict[str, Any]],
) -> list[UGInterleavedMessage]:
    normalized: list[UGInterleavedMessage] = []
    for message in messages:
        if isinstance(message, UGInterleavedMessage):
            normalized.append(message)
            continue
        if not isinstance(message, dict):
            raise TypeError(
                f"UG message must be a dict or UGInterleavedMessage: {message!r}"
            )
        message_type = message.get("type")
        if message_type == "text":
            content = message.get("text", message.get("content"))
        elif message_type == "image":
            content = message.get("image", message.get("content"))
        else:
            raise ValueError(f"Unsupported UG message type: {message_type!r}")
        if content is None:
            raise ValueError(f"UG {message_type} message is missing content")
        normalized.append(UGInterleavedMessage(type=message_type, content=content))
    if not normalized:
        raise ValueError("UG interleaved messages must not be empty")
    return normalized
