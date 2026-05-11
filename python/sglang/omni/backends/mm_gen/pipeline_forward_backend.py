# SPDX-License-Identifier: Apache-2.0
"""Generation backend for in-process multimodal_gen pipeline calls.

This backend adapts the generic omni generation protocol to a multimodal_gen
pipeline call. It passes live SRT context capabilities through ``Req.extra``.
That keeps U1's current same-process KV access explicit while leaving room for
an executor-backed backend once the diffusion runtime owns the schedule.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.omni.protocol import (
    ContextOps,
    GeneratedSegment,
    MultimodalGenerationBackend,
    OmniRequest,
)


@dataclass(slots=True)
class DirectPipelineForwardBackend(MultimodalGenerationBackend):
    """A backend that calls ``ComposedPipelineBase.forward`` directly from omni orchestration."""

    pipeline: ComposedPipelineBase
    server_args: Any

    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment:
        batch = build_pipeline_req(request, context_ops)
        batch: Req = self.pipeline.forward(batch, self.server_args)
        segment = batch.generated_segment
        if segment is None:
            raise ValueError("Direct pipeline forward did not set generated_segment")

        return coerce_generated_segment(segment)


def build_pipeline_req(
    request: OmniRequest,
    context_ops: ContextOps,
) -> Req:
    """Build the multimodal_gen request envelope shared by omni mm backends."""

    extra = {
        "omni_messages": [message.to_dict() for message in request.messages],
        "omni_context_metadata": context_ops.metadata,
    }
    return Req(
        sampling_params=_copy_sampling_params(request.sampling_params),
        prompt=_prompt_from_request(request),
        suppress_logs=True,
        extra=extra,
        omni_context_ops=context_ops,
    )


def coerce_generated_segment(segment: Any) -> GeneratedSegment:
    if isinstance(segment, GeneratedSegment):
        return segment
    segment_type = getattr(segment, "type", None)
    if segment_type is None and getattr(segment, "image", None) is not None:
        segment_type = "image"
    if segment_type is None:
        raise ValueError("Generation backend returned a segment without type")
    commit_payload = getattr(segment, "commit_payload", None)
    if commit_payload is None:
        commit_payload = getattr(segment, "commit_image", None)
    return GeneratedSegment(
        type=segment_type,
        text=getattr(segment, "text", None),
        image=getattr(segment, "image", None),
        audio=getattr(segment, "audio", None),
        video=getattr(segment, "video", None),
        commit_payload=commit_payload,
        metadata=dict(getattr(segment, "metadata", {}) or {}),
    )


def _copy_sampling_params(sampling_params: Any) -> Any:
    if sampling_params is None:
        return None
    return copy.copy(sampling_params)


def _prompt_from_request(request: OmniRequest) -> str:
    return "\n".join(
        message.text or "" for message in request.messages if message.type == "text"
    )
