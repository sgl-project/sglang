"""How Python's TokenizerManager reports one request's customized_info in each
streaming mode, and what the Rust frontend shows a generate hook for the same
scheduler outputs. Fixtures for the Rust frontend's native `/generate` frames,
and for the hooks model packages install there, are built from these."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

# A unary response, a cumulative stream (the default) and an incremental
# stream (`--incremental-streaming-output`).
MODES = ("non_streaming", "cumulative", "incremental")


@dataclass(frozen=True)
class SchedulerOutput:
    """One scheduler output for one request: its new output ids, the values
    its batch carried for it per customized_info key (`None` for a batch that
    carried none, such as a prefill or a scheduler abort), and its finish
    reason on the terminal output."""

    output_ids: list[int]
    customized_info: Optional[dict[str, list]] = None
    finish_reason: Optional[dict] = None

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.customized_info or ())


def reported_customized_info(
    outputs: list[SchedulerOutput], mode: str
) -> list[dict[str, list]]:
    """The customized_info keys each frame's `meta_info` reports, with their
    values, for a request whose outputs arrive one per frame:

    - `non_streaming`: the one frame, the terminal output's keys with their
      running lists;
    - `cumulative`: every key surfaced so far with its running list, through
      Python's own `_stream_one_response`;
    - `incremental`: each output's own values, the frames the Rust frontend
      sends. Python's slicing can lose values (`python_incremental`).
    """
    if mode == "incremental":
        return [dict(output.customized_info or {}) for output in outputs]
    return asyncio.run(_stream_one_response(outputs, stream=mode == "cumulative"))


def python_incremental(outputs: list[SchedulerOutput]) -> list[dict[str, list]]:
    """Python's incremental frames: the running lists sliced at the request's
    output-token offset (`_slice_streaming_output_meta_info`), which loses
    values once an output lacked a key."""
    from sglang.srt.managers.tokenizer_manager import (
        TokenizerManager,
        _slice_streaming_output_meta_info,
    )

    manager = object.__new__(TokenizerManager)
    state = _request_state(stream=True)
    frames = []
    for output in outputs:
        meta_info: dict[str, Any] = {}
        _update(manager, meta_info, state, output)
        offset = len(state.output_ids)
        state.output_ids.extend(output.output_ids)
        _slice_streaming_output_meta_info(
            meta_info, offset, state.customized_info_accumulated.keys()
        )
        frames.append(copy.deepcopy(meta_info))
    return frames


def hook_inputs(outputs: list[SchedulerOutput], mode: str) -> list[list[list]]:
    """What the Rust frontend shows a generate hook on each frame
    (`GenerateFrameInfo::customized_info`): `[key, new values]` pairs in the
    order the keys first arrived. A cumulative frame shows every key so far,
    with the values since the previous frame; an incremental frame, its
    output's keys with their values; a unary response's one frame, the
    terminal output's keys with all their values."""
    pending: dict[str, list] = {}
    frames = []
    for output in outputs:
        for key, values in (output.customized_info or {}).items():
            pending.setdefault(key, []).extend(values)
        if mode == "non_streaming" and output.finish_reason is None:
            continue
        shown = [key for key in pending if mode == "cumulative" or key in output.keys]
        frames.append([[key, pending[key]] for key in shown])
        for key in shown:
            pending[key] = []
    return frames


def _request_state(stream: bool):
    """A TokenizerManager `ReqState` for one request, as it tracks
    customized_info."""
    from sglang.srt.managers.tokenizer_manager import ReqState
    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=SimpleNamespace(rid="rid", stream=stream, background=False),
        time_stats=APIServerReqTimeStats(),
    )


def _update(manager, meta_info: dict, state, output: SchedulerOutput) -> None:
    """`TokenizerManager.update_request_meta_info` for one output, as the
    request's one row of its batch."""
    if output.customized_info is None:
        return
    manager.update_request_meta_info(
        meta_info,
        state,
        {key: [values] for key, values in output.customized_info.items()},
        0,
        output.finish_reason,
    )


async def _stream_one_response(
    outputs: list[SchedulerOutput], stream: bool
) -> list[dict[str, list]]:
    """Feed the outputs through `TokenizerManager._stream_one_response` one at
    a time, as `_handle_batch_output` queues them (a unary request queues only
    its terminal output), and keep each yielded frame's customized keys."""
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    manager = object.__new__(TokenizerManager)
    manager.incremental_streaming_output = False
    manager.request_logger = Mock()
    manager.request_metrics_exporter_manager = Mock()
    manager.request_metrics_exporter_manager.exporter_enabled.return_value = False
    state = _request_state(stream)
    keys = {key for output in outputs for key in output.keys}
    responses = manager._stream_one_response(obj=state.obj, state=state)
    frames = []
    output_ids: list[int] = []
    for output in outputs:
        meta_info: dict[str, Any] = {"finish_reason": output.finish_reason}
        _update(manager, meta_info, state, output)
        output_ids.extend(output.output_ids)
        finished = output.finish_reason is not None
        if not (stream or finished):
            continue
        state.out_list.append(
            {"text": "", "output_ids": list(output_ids), "meta_info": meta_info}
        )
        state.finished = finished
        state.event.set()
        frame = await anext(responses)
        frames.append(
            {
                key: copy.deepcopy(values)
                for key, values in frame["meta_info"].items()
                if key in keys
            }
        )
    await responses.aclose()
    return frames
