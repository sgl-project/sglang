import asyncio
from types import SimpleNamespace

from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager


def _make_manager(incremental: bool) -> TokenizerManager:
    tm = object.__new__(TokenizerManager)
    tm.rid_to_state = {}
    tm.incremental_streaming_output = incremental
    tm.config_value = lambda name: 0
    return tm


def _make_state() -> ReqState:
    return ReqState(
        obj=SimpleNamespace(stream=True, return_logprob=False),
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        time_stats=SimpleNamespace(
            set_finished_time=lambda: None,
            get_e2e_latency=lambda: 0.0,
        ),
    )


def _run_abort(tm: TokenizerManager, state: ReqState, rid: str) -> dict:
    tm.rid_to_state[rid] = state
    recv_obj = SimpleNamespace(
        rid=rid,
        abort_message=None,
        finished_reason=None,
        weight_versions=None,
    )
    tm._handle_abort_req(recv_obj)
    assert rid not in tm.rid_to_state
    return state.out_list[-1]


def test_abort_chunk_is_incremental_delta():
    tm = _make_manager(incremental=True)
    state = _make_state()
    state.output_ids = [1, 2, 3, 4, 5, 6, 7]
    state.last_output_offset = 5
    state.last_streamed_text_len = 10
    state.append_text("hello world")
    out = _run_abort(tm, state, "r1")

    assert out["text"] == "d"
    assert out["output_ids"] == [6, 7]
    assert out["meta_info"]["completion_tokens"] == 7


def test_abort_before_first_chunk_sends_full_text_as_first_delta():
    tm = _make_manager(incremental=True)
    state = _make_state()
    state.output_ids = [1, 2, 3]
    state.last_output_offset = 0
    state.last_streamed_text_len = 0
    state.append_text("hi")
    out = _run_abort(tm, state, "r2")

    assert out["text"] == "hi"
    assert out["output_ids"] == [1, 2, 3]


def test_abort_non_incremental_keeps_full_text():
    tm = _make_manager(incremental=False)
    state = _make_state()
    state.output_ids = [1, 2, 3]
    state.append_text("hello world")
    out = _run_abort(tm, state, "r3")

    assert out["text"] == "hello world"
    assert out["output_ids"] == [1, 2, 3]
