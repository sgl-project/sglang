from __future__ import annotations

import json
from typing import Any, Dict, Union

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.observability.trace import SpanAttributes


def make_span_attrs(
    *,
    state: ReqState,
    recv_obj: Union[
        BatchStrOutput,
        BatchEmbeddingOutput,
        BatchTokenIDOutput,
    ],
    i: int,
    served_model_name: str,
) -> Dict[str, Any]:
    """Convert attributes to span attributes."""
    span_attrs = {}

    # Token usage attributes
    if not isinstance(recv_obj, BatchEmbeddingOutput):
        span_attrs[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = (
            recv_obj.completion_tokens[i]
        )
    span_attrs[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = recv_obj.prompt_tokens[i]
    span_attrs[SpanAttributes.GEN_AI_USAGE_CACHED_TOKENS] = recv_obj.cached_tokens[i]

    # Request identifiers
    span_attrs[SpanAttributes.GEN_AI_REQUEST_ID] = (
        str(state.obj.rid) if state.obj.rid else None
    )

    # Sampling parameters
    sampling_params = state.obj.sampling_params or {}

    if max_new_tokens := sampling_params.get("max_new_tokens"):
        span_attrs[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = max_new_tokens

    if top_p := sampling_params.get("top_p"):
        span_attrs[SpanAttributes.GEN_AI_REQUEST_TOP_P] = top_p

    if temperature := sampling_params.get("temperature"):
        span_attrs[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = temperature

    if top_k := sampling_params.get("top_k"):
        span_attrs[SpanAttributes.GEN_AI_REQUEST_TOP_K] = top_k

    if n := sampling_params.get("n"):
        span_attrs[SpanAttributes.GEN_AI_REQUEST_N] = n

    # Response attributes
    span_attrs[SpanAttributes.GEN_AI_RESPONSE_MODEL] = served_model_name

    finish_reason = (
        recv_obj.finished_reasons[i].get("type")
        if recv_obj.finished_reasons[i]
        else None
    )
    if finish_reason:
        span_attrs[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS] = json.dumps(
            [finish_reason]
        )

    # Latency attributes
    span_attrs.update(state.time_stats.convert_to_gen_ai_span_attrs())

    return span_attrs
