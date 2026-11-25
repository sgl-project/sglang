"""Drive a TokenizerManager stream that ends in an abort, for tests.

The chunks come from the manager's own output path: each delta goes through
`_handle_batch_output`, the abort through `_handle_abort_req`, and the client
reads them through `_wait_one_response`, so queued chunks are coalesced as they
are when the client falls behind.
"""

from __future__ import annotations

import asyncio
from array import array
from typing import Any, Dict, List, Optional, Sequence

import msgspec

from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchStrOutput,
    GenerateReqInput,
    RequestOutputStats,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

# Each output token's logprob, so a test can count what reaches the client.
_LOGPROB_STEP = -0.25


def _request_stats() -> RequestOutputStats:
    return RequestOutputStats(
        prompt_tokens=2,
        cached_tokens=0,
        cached_tokens_details=None,
        retraction_count=0,
        image_tokens=0,
        audio_tokens=0,
        video_tokens=0,
        time_stats=None,
    )


def _batch_str_output(
    rid: str, delta: str, token_id: int, completion_tokens: int
) -> BatchStrOutput:
    """One row carrying one decoded token; every other field is empty."""
    values: Dict[str, Any] = {
        "rids": [rid],
        "finished_reasons": [None],
        "output_strs": [delta],
        "output_ids_flat": array("q", [token_id]),
        "output_ids_offsets": [0, 1],
        "request_stats": [_request_stats()],
        "completion_tokens": [completion_tokens],
        "output_token_logprobs_val": [[_LOGPROB_STEP * token_id]],
        "output_token_logprobs_idx": [[token_id]],
    }
    for field in msgspec.structs.fields(BatchStrOutput):
        if field.name in values:
            continue
        if field.default is msgspec.NODEFAULT and (
            field.default_factory is msgspec.NODEFAULT
        ):
            # Per-request lists: one empty or unset entry for the single row.
            values[field.name] = [None] if "hidden" in field.name else [[]]
    return BatchStrOutput(**values)


def make_stream_manager(*, incremental: bool) -> TokenizerManager:
    """A TokenizerManager with only the state its output path reads."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.incremental_streaming_output = incremental
    tm.rid_to_state = {}
    tm.enable_metrics = False
    tm.enable_lora = False
    tm.skip_tokenizer_init = True
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    tm.canary_info_hook = None
    tm.request_logger = _NullLogger()
    tm.request_metrics_exporter_manager = _NullExporter()
    return tm


class _NullLogger:
    def log_finished_request(self, *args, **kwargs) -> None:
        pass


class _NullExporter:
    def exporter_enabled(self) -> bool:
        return False


def make_stream_state(
    tm: TokenizerManager, rid: str, *, return_logprob: bool = False
) -> ReqState:
    obj = GenerateReqInput(text="Hi", stream=True, return_logprob=return_logprob)
    obj.rid = rid
    obj.normalize_batch_and_arguments()
    state = ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )
    tm.rid_to_state[rid] = state
    return state


async def stream_until_abort(
    tm: TokenizerManager,
    state: ReqState,
    deltas: Sequence[str],
    *,
    read_after: Sequence[int] = (),
    abort: Optional[AbortReq] = None,
) -> List[Dict[str, Any]]:
    """The chunks a client reads for `deltas`, then an abort.

    The client reads a chunk after each delta whose index is in `read_after`;
    the deltas after the last read are still queued at the abort, and come out
    coalesced with it.
    """
    rid = state.obj.rid
    stream = tm._wait_one_response(state.obj)
    chunks = []
    for i, delta in enumerate(deltas):
        await tm._handle_batch_output(
            _batch_str_output(rid, delta, token_id=100 + i, completion_tokens=i + 1)
        )
        if i in read_after:
            chunks.append(await anext(stream))
    tm._handle_abort_req(abort or AbortReq(rid=rid))
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


def abort_stream_chunks(
    deltas: Sequence[str],
    *,
    incremental: bool,
    rid: str = "abort-stream",
    read_after: Sequence[int] = (),
    return_logprob: bool = False,
) -> List[Dict[str, Any]]:
    """`stream_until_abort` on a fresh manager, run to completion."""
    tm = make_stream_manager(incremental=incremental)
    state = make_stream_state(tm, rid, return_logprob=return_logprob)
    return asyncio.run(stream_until_abort(tm, state, deltas, read_after=read_after))
