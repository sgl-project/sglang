# SPDX-License-Identifier: Apache-2.0
"""Thin omni orchestration loop for interleaved text and media generation.

The coordinator owns only control flow: decode AR text until a modality
boundary, call the matching generation backend, then ask the AR backend to
commit generated media when the model/session requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import BoundedSemaphore
from typing import TYPE_CHECKING, Any, Callable

from sglang.omni.core.interleaved import (
    STREAMED_TEXT_METADATA_KEY,
    TEXT_ROLE_METADATA_KEY,
    TEXT_ROLE_THINK,
)
from sglang.omni.core.protocol import (
    ARBackend,
    ContextOps,
    GeneratedSegment,
    MultimodalGenerationBackend,
    OmniContextBundle,
    OmniOutputSegment,
    OmniRequest,
    OmniResponse,
)

if TYPE_CHECKING:
    from sglang.omni.entrypoints.streaming import OmniStreamSink


@dataclass(slots=True)
class OmniCoordinator:
    """Top-level structs that resides in OmniSchedulerState and coordinate AR and multimodal-generation backends without owning model internals."""

    ar_backend: ARBackend
    mm_generation_backend: MultimodalGenerationBackend
    request_adapter: Callable[[OmniRequest], OmniRequest] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    max_concurrent_generations: int | None = None
    _generation_slots: BoundedSemaphore | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        if self.max_concurrent_generations is None:
            return
        max_concurrent_generations = int(self.max_concurrent_generations)
        if max_concurrent_generations <= 0:
            raise ValueError(
                "max_concurrent_generations must be positive, got "
                f"{max_concurrent_generations}"
            )
        self._generation_slots = BoundedSemaphore(max_concurrent_generations)

    def generate(self, request: OmniRequest) -> OmniResponse:
        response, _ = self.generate_with_context(request)
        return response

    def generate_with_context(
        self,
        request: OmniRequest,
        *,
        context: OmniContextBundle | None = None,
        release_context: bool = True,
        stop_after_generation_limit: bool = False,
        stream_sink: OmniStreamSink | None = None,
    ) -> tuple[OmniResponse, OmniContextBundle]:
        """Main entrypoint, generate a response for an omni request"""
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
            # prepare for the decode loop, usually includes prefill and decode to the first segment
            context: OmniContextBundle = self.ar_backend.begin_request_context(
                request,
                stream_sink=stream_sink,
            )
        else:
            context: OmniContextBundle = self.ar_backend.append_input_segments(
                context,
                request,
                stream_sink=stream_sink,
            )

        segments: list[OmniOutputSegment] = []
        num_text_segments = 0
        num_image_segments = 0
        num_audio_segments = 0
        num_video_segments = 0

        try:
            # the main loop: coordinator owns modality handoff; backends own token and pixel internals
            # each round generates a text segment and an optional image segment
            while True:
                # 1. decode until a boundary is met
                boundary = self.ar_backend.decode_until_boundary(
                    context,
                    request=request,
                    stream_sink=stream_sink,
                )

                if boundary.type == "done":
                    # eos
                    break

                if boundary.type == "text":
                    is_think_text = (
                        boundary.metadata.get(TEXT_ROLE_METADATA_KEY)
                        == TEXT_ROLE_THINK
                    )
                    if is_think_text:
                        # 1. think text may have already streamed; it is not user-visible answer text
                        if (
                            stream_sink is not None
                            and boundary.metadata.get(STREAMED_TEXT_METADATA_KEY)
                        ):
                            stream_sink.finish_text()
                        continue
                    if num_text_segments >= request.max_text_segments:
                        break
                    if stream_sink is not None:
                        if boundary.metadata.get(STREAMED_TEXT_METADATA_KEY):
                            stream_sink.finish_text()
                        else:
                            stream_sink.text_segment(boundary.text or "")
                    # text boundary: append and continue text gen
                    output_segment = OmniOutputSegment.from_boundary(boundary)
                    output_segment.metadata.pop(STREAMED_TEXT_METADATA_KEY, None)
                    segments.append(output_segment)
                    num_text_segments += 1
                    if (
                        stop_after_generation_limit
                        and num_image_segments >= request.max_images
                    ):
                        # 1. stop before probing another image marker
                        break
                    continue

                if boundary.type not in {"image", "audio", "video"}:
                    raise ValueError(f"Unsupported omni boundary: {boundary.type!r}")

                if (
                    boundary.type == "image"
                    and num_image_segments >= request.max_images
                ):
                    break

                stream_segment_id = None
                if stream_sink is not None and boundary.type == "image":
                    stream_segment_id = stream_sink.begin_image()
                # 2. media boundary: switch to multimodal gen and generate one segment
                generated_segment = self._generate_segment(
                    request,
                    self.ar_backend.get_context_ops(context),
                )
                if generated_segment.type != boundary.type:
                    raise ValueError(
                        "Generation backend returned mismatched segment type: "
                        f"expected {boundary.type!r}, got {generated_segment.type!r}"
                    )

                # 3. count the generated segment by output modality
                segments.append(OmniOutputSegment.from_generated(generated_segment))
                if generated_segment.type == "image":
                    num_image_segments += 1
                elif generated_segment.type == "audio":
                    num_audio_segments += 1
                elif generated_segment.type == "video":
                    num_video_segments += 1
                if stream_sink is not None and stream_segment_id is not None:
                    stream_sink.image(
                        segment_id=stream_segment_id,
                        image=generated_segment.image,
                        metadata=generated_segment.metadata,
                    )

                # 4. update the context with ar_backend, get prepared for the next round
                context = self.ar_backend.append_generated_segment(
                    context,
                    generated_segment,
                    request=request,
                )
                if (
                    num_image_segments >= request.max_images
                    and request.max_text_segments == 0
                ):
                    # 1. do not run an extra AR probe when the caller disabled post-image text
                    break

        finally:
            if release_context:
                self.ar_backend.release(context)

        stats = {
            "num_segments": len(segments),
            "num_text_segments": num_text_segments,
            "num_image_segments": num_image_segments,
            "num_audio_segments": num_audio_segments,
            "num_video_segments": num_video_segments,
        }
        response = OmniResponse(
            segments=tuple(segments),
            context=context.full,
            stats=stats,
            metadata=dict(self.metadata),
        )
        return response, context

    def _generate_segment(
        self, request: OmniRequest, context_ops: ContextOps
    ) -> GeneratedSegment:
        if self._generation_slots is None:
            return self.mm_generation_backend.generate_segment(request, context_ops)

        # media generation is the OOM-sensitive part; keep AR/session work outside this gate
        self._generation_slots.acquire()
        try:
            return self.mm_generation_backend.generate_segment(request, context_ops)
        finally:
            self._generation_slots.release()
