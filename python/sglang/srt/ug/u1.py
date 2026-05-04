# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from sglang.srt.ug.interleaved import UGGKind


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


def _not_wired() -> NotImplementedError:
    return NotImplementedError(
        "SenseNova U1 UG backend is not wired yet. This shell only declares "
        "the pixel_flow capability; U path, G pixel-flow mechanics, and true "
        "weights are covered by later roadmap items."
    )
