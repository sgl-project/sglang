# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.omni.protocol import ContextOps, GeneratedSegment, OmniRequest

# standalone diffusion serving backends can implement the same generation protocol


@dataclass(slots=True)
class ColocatedPipelineBackend:
    pipeline: Any
    server_args: Any
    context_ops_extra_key: str
    output_extra_key: str

    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment:
        batch = self._build_batch(request, context_ops)
        self.pipeline.forward(batch, self.server_args)
        segment = batch.extra.get(self.output_extra_key)
        if segment is None:
            raise ValueError(
                f"Colocated pipeline did not set extra[{self.output_extra_key!r}]"
            )
        return coerce_generated_segment(segment)

    def _build_batch(self, request: OmniRequest, context_ops: ContextOps) -> Req:
        extra = {
            "omni_messages": [message.to_dict() for message in request.messages],
            "omni_context_metadata": context_ops.metadata,
            # context_ops is an in-process srt runner capability, not serialized KV
            self.context_ops_extra_key: context_ops,
        }
        return Req(
            sampling_params=_copy_sampling_params(request.sampling_params),
            prompt=_prompt_from_request(request),
            suppress_logs=True,
            extra=extra,
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
        message.text or ""
        for message in request.messages
        if message.type == "text"
    )
