# SPDX-License-Identifier: Apache-2.0
"""Thin omni orchestration loop for interleaved text and media generation.

The coordinator owns only control flow: decode AR text until a modality
boundary, call the matching generation backend, then ask the AR backend to
commit generated media when the model/session requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sglang.omni.protocol import (
    ARBackend,
    GenerationBackend,
    OmniOutputSegment,
    OmniRequest,
    OmniResponse,
)


@dataclass(slots=True)
class OmniCoordinator:
    """Coordinate AR and generation backends without owning model internals."""

    ar_backend: ARBackend
    generation_backend: GenerationBackend
    request_adapter: Callable[[OmniRequest], OmniRequest] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def generate(self, request: OmniRequest) -> OmniResponse:
        response, _ = self.generate_with_context(request)
        return response

    def generate_with_context(
        self,
        request: OmniRequest,
        *,
        context: Any | None = None,
        release_context: bool = True,
        stop_after_generation_limit: bool = False,
    ) -> tuple[OmniResponse, Any]:
        if self.request_adapter is not None:
            request = self.request_adapter(request)

        if request.max_images < 0:
            raise ValueError(
                f"max_images must be non-negative, got {request.max_images}"
            )
        if request.max_text_segments < 0:
            raise ValueError(
                f"max_text_segments must be non-negative, got {request.max_text_segments}"
            )

        if context is None:
            context = self.ar_backend.prepare_context(request)
        else:
            context = self.ar_backend.append_input_segments(context, request)

        segments: list[OmniOutputSegment] = []
        num_text_segments = 0
        num_generated_segments = 0

        try:
            # orchestrator owns modality handoff; backends own token and pixel internals
            while True:
                boundary = self.ar_backend.decode_until_boundary(
                    context,
                    request=request,
                )

                if boundary.type == "done":
                    break

                if boundary.type == "text":
                    if num_text_segments >= request.max_text_segments:
                        break
                    segments.append(OmniOutputSegment.from_boundary(boundary))
                    num_text_segments += 1
                    continue

                if boundary.type not in {"image", "audio", "video"}:
                    raise ValueError(f"Unsupported omni boundary: {boundary.type!r}")

                if (
                    boundary.type == "image"
                    and num_generated_segments >= request.max_images
                ):
                    break

                generated = self.generation_backend.generate_segment(
                    request,
                    self.ar_backend.get_context_ops(context),
                )
                if generated.type != boundary.type:
                    raise ValueError(
                        "Generation backend returned mismatched segment type: "
                        f"expected {boundary.type!r}, got {generated.type!r}"
                    )
                segments.append(OmniOutputSegment.from_generated(generated))
                num_generated_segments += 1
                context = self.ar_backend.append_generated_segment(
                    context,
                    generated,
                    request=request,
                )
                if (
                    stop_after_generation_limit
                    and boundary.type == "image"
                    and num_generated_segments >= request.max_images
                ):
                    break
        finally:
            if release_context:
                self.ar_backend.release(context)

        stats = {
            "num_segments": len(segments),
            "num_text_segments": num_text_segments,
            "num_generated_segments": num_generated_segments,
        }
        response = OmniResponse(
            segments=tuple(segments),
            context=context.full,
            stats=stats,
            metadata=dict(self.metadata),
        )
        return response, context
