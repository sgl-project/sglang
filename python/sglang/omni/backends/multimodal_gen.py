# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.omni.protocol import ContextOps, GeneratedSegment, OmniRequest


@dataclass(slots=True)
class MultimodalGenBackend:
    executor: Callable[..., Any]
    server_args: Any | None = None
    context_ops_extra_key: str | None = None

    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment:
        batch = self._build_batch(request, context_ops)
        segment = self.executor(
            context_ops=context_ops,
            batch=batch,
            server_args=self.server_args,
        )
        return _coerce_generated_segment(segment)

    @staticmethod
    def _base_extra(request: OmniRequest, context_ops: ContextOps) -> dict[str, Any]:
        return {
            "omni_messages": [message.to_dict() for message in request.messages],
            "omni_context_metadata": context_ops.metadata,
        }

    def _build_batch(self, request: OmniRequest, context_ops: ContextOps) -> Any:
        extra = self._base_extra(request, context_ops)
        if self.context_ops_extra_key is not None:
            extra[self.context_ops_extra_key] = context_ops
        return Req(sampling_params=request.sampling_params, extra=extra)


def _coerce_generated_segment(segment: Any) -> GeneratedSegment:
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
