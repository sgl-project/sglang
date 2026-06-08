from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Dict, List, Optional, Union

import fastapi

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.observability.trace import extract_trace_headers


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Union[GenerateReqInput, EmbeddingReqInput]

    # For performance metrics
    time_stats: APIServerReqTimeStats
    last_completion_tokens: int = 1
    ttft_observed: bool = False

    # For streaming output
    last_output_offset: int = 0

    # Accumulate text lazily so incremental streaming can emit the incoming
    # delta directly without rebuilding the full output prefix.
    text: str = ""
    text_chunks: List[str] = dataclasses.field(default_factory=list)

    def append_text(self, chunk: str):
        if chunk:
            self.text_chunks.append(chunk)

    def get_text(self) -> str:
        if self.text_chunks:
            self.text += "".join(self.text_chunks)
            self.text_chunks.clear()
        return self.text

    def get_crash_dump_output(self) -> Dict[Any, Any]:
        out = {}
        if self.text or self.text_chunks:
            out["text"] = self.get_text()
        if self.output_ids:
            out["output_ids"] = self.output_ids.copy()
        return out

    # For incremental state update.
    # TODO(lianmin): do not initialize some lists if not needed.
    output_ids: List[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)

    # For detokenized logprobs
    input_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)
    customized_info_accumulated: Dict[str, List[Any]] = dataclasses.field(
        default_factory=dict
    )

    # For return_prompt_token_ids: stores prompt token IDs captured after tokenization
    prompt_token_ids: Optional[List[int]] = None


def init_req(
    rid_to_state: Dict[str, ReqState],
    *,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
    enable_trace: bool,
    disagg_mode: DisaggregationMode,
) -> None:
    created_time = obj.received_time

    external_trace_header = None
    if enable_trace:
        if obj.external_trace_header:
            # When the request comes from the rust grpc server or Engine there isn't a
            # real request object but we still need to propagate the trace context from
            # the trace context that is explicitly passed in
            external_trace_header = obj.external_trace_header
        elif request:
            external_trace_header = extract_trace_headers(request.headers)
            obj.external_trace_header = external_trace_header

    # Normalize single/batch into a uniform list of (rid, sub_obj, bootstrap_room)
    if not hasattr(obj, "is_single") or obj.is_single:
        items = [(obj.rid, obj, getattr(obj, "bootstrap_room", None))]
    else:
        items = [
            (
                obj.rid[i],
                obj[i],
                (
                    obj.bootstrap_room[i]
                    if hasattr(obj, "bootstrap_room") and obj.bootstrap_room
                    else None
                ),
            )
            for i in range(len(obj.rid))
        ]

    for rid, sub_obj, bootstrap_room in items:
        if rid in rid_to_state:
            raise ValueError(f"Duplicate request ID detected: {rid}")
        time_stats = APIServerReqTimeStats(disagg_mode=disagg_mode)
        state = ReqState([], False, asyncio.Event(), sub_obj, time_stats)
        rid_to_state[rid] = state
        if enable_trace:
            time_stats.init_trace_ctx(rid, bootstrap_room, external_trace_header)
        time_stats.set_created_time(created_time)
