# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

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

    def prepare_srt_u_message_inputs(
        self,
        *,
        session: Any,
        message: Any,
        state: Any,
    ) -> list[Any] | None:
        del session, message, state
        raise _not_wired()

    def observe_srt_u_forward(
        self,
        *,
        session: Any,
        request: Any,
        messages: list[Any],
    ) -> None:
        del session, request, messages
        raise _not_wired()

    def prefill_interleaved(
        self,
        *,
        session: Any,
        messages: list[Any],
    ) -> Any:
        del session, messages
        raise _not_wired()

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
    ) -> Any:
        del session, image
        raise _not_wired()

    def close_session(self, *, session_id: str) -> None:
        del session_id


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
