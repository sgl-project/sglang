"""Timing snapshots survive native token commits, IPC metadata and coalescing."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from sglang_simulator.simulation.coordinator import (
    Cancel,
    Coordinator,
    ProtocolError,
    Ready,
    Submit,
)
from sglang_simulator.simulation.sglang.timing_metadata import (
    METADATA_KEY,
    C_TimingChatHook,
    C_TimingOutputStreamerHook,
    C_TimingTokenizerHook,
    attach_timing_output,
)
from sglang_simulator.simulation.sglang.token_events import TokenEventRecorder


class Request:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = [10] * 7
        self.cached_tokens = 3
        self.done = False

    def finished(self):
        return self.done


def setup_requests(*rids, arrival_ns=0):
    clock = Coordinator()
    clock.now_ns = arrival_ns
    clock.accept_ready(Ready("test", 0, [Submit(rid) for rid in rids]))
    for rid in rids:
        clock.mark_admitted(rid)
    return clock, [Request(rid) for rid in rids]


def commit(clock, reqs, at_ns, commits, previous=None, token_events=None):
    clock.advance(at_ns)
    result = SimpleNamespace(auxiliary_host_output=previous)
    attach_timing_output(result, clock, token_events)
    result.auxiliary_host_output.consume(SimpleNamespace(reqs=reqs), commits)


def tokens(start, *ids):
    return SimpleNamespace(output_index=start, token_ids=ids)


def streamer():
    class Streamer:
        has_additional_customized_info = False

        def build_additional_customized_info(self, reqs):
            return {"unrelated": [[req.rid] for req in reqs]}

    C_TimingOutputStreamerHook.hook(Streamer)
    return Streamer()


def tokenizer():
    class Tokenizer:
        def update_request_meta_info(
            self, meta_info, state, customized_info, index, finish_reason
        ):
            for key, values in customized_info.items():
                accumulated = state.customized_info_accumulated.setdefault(key, [])
                accumulated.extend(values[index])
                meta_info[key] = accumulated

    C_TimingTokenizerHook.hook(Tokenizer)
    return Tokenizer()


def snapshot(req):
    return streamer().build_additional_customized_info([req])[METADATA_KEY][0]


def test_committed_bounds_preserve_exact_arrival_and_prior_outputs():
    arrival_ns = 2**53 + 1
    clock, (req,) = setup_requests("request", arrival_ns=arrival_ns)
    previous = Mock()
    commit(clock, [req], arrival_ns + 10, [tokens(0, 71)], previous)
    first = snapshot(req)
    previous.consume.assert_called_once()
    commit(clock, [req], arrival_ns + 20, [tokens(1, 72, 73)])
    req.done = True
    commit(clock, [req], arrival_ns + 30, [None])
    final = snapshot(req)
    assert final == {
        "version": 1,
        "rid": "request",
        "arrival_ns": arrival_ns,
        "first_token_ns": arrival_ns + 10,
        "last_token_ns": arrival_ns + 20,
        "commit_ns": arrival_ns + 30,
        "terminal_ns": arrival_ns + 30,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cached_tokens": 3,
    }
    assert first["completion_tokens"] == 1
    assert first["terminal_ns"] is None


def test_zero_commit_and_multiple_requests_keep_aligned_snapshots():
    clock, (left, right) = setup_requests("left", "right")
    commit(clock, [left, right], 10, [None, tokens(0, 75)])
    left.done = right.done = True
    commit(clock, [left, right], 20, [None, tokens(1, 76)])
    outputs = streamer().build_additional_customized_info([right, left])
    right_meta, left_meta = outputs[METADATA_KEY]
    assert left_meta["first_token_ns"] is left_meta["last_token_ns"] is None
    assert left_meta["completion_tokens"] == 0
    assert left_meta["terminal_ns"] == 20
    assert right_meta["first_token_ns"] == 10
    assert right_meta["last_token_ns"] == 20
    assert right_meta["completion_tokens"] == 2
    assert outputs["unrelated"] == [["right"], ["left"]]


def test_noncontiguous_native_commits_fail_explicitly():
    clock, (req,) = setup_requests("request")
    with pytest.raises(ProtocolError, match="non-contiguous"):
        commit(clock, [req], 10, [tokens(1, 70)])


def test_token_ledger_uses_visible_commits_and_keeps_partial_cancelled_output(tmp_path):
    path = tmp_path / "tokens.jsonl"
    recorder = TokenEventRecorder(path)
    clock, (completed, cancelled) = setup_requests("completed", "cancelled")
    commit(
        clock,
        [completed, cancelled],
        10,
        [tokens(0, 71), tokens(0, 72, 73)],
        token_events=recorder,
    )
    completed.done = cancelled.done = True
    commit(
        clock,
        [completed, cancelled],
        20,
        [tokens(1, 74), None],
        token_events=recorder,
    )
    recorder.finish("test", 20)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[1]["commits"] == [
        {"rid": "completed", "output_index": 0, "token_count": 1},
        {"rid": "cancelled", "output_index": 0, "token_count": 2},
    ]
    assert events[2]["commits"] == [
        {"rid": "completed", "output_index": 1, "token_count": 1}
    ]
    assert events[-1]["total_output_tokens"] == 4
    assert snapshot(cancelled)["completion_tokens"] == 2


def test_native_cancel_cleanup_append_does_not_count_as_output(tmp_path):
    root = Path(__file__).resolve().parents[3]
    functions = {}
    for filename, names in (
        ("schedule_batch.py", {"update_finish_state"}),
        (
            "scheduler_components/batch_result_processor.py",
            {"_visible_output_len", "_build_auxiliary_commits"},
        ),
    ):
        path = root / "python/sglang/srt/managers" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        assert len(nodes) == len(names)
        for node in nodes:
            node.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        namespace = {"CommittedTokens": SimpleNamespace}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102 - trusted native functions
        functions.update({name: namespace[name] for name in names})

    processor = SimpleNamespace(_visible_output_len=functions["_visible_output_len"])
    build = functions["_build_auxiliary_commits"]
    clock, (req,) = setup_requests("cancelled")
    req.output_ids = [71]
    req.finished_len = None
    req.finished_reason = None
    req.to_finish = None
    req.finished = lambda: req.finished_reason is not None
    batch = SimpleNamespace(reqs=[req])
    path = tmp_path / "cancelled-tokens.jsonl"
    recorder = TokenEventRecorder(path)
    commit(clock, [req], 10, build(processor, batch, [0]), token_events=recorder)

    clock.advance(12)
    clock.grant()
    clock.accept_ready(Ready("test", 1, [Cancel(req.rid)]))
    clock.grant()
    clock.accept_ready(Ready("test", 2, [], finished=True))
    req.to_finish = SimpleNamespace(reason="aborted")
    start = processor._visible_output_len(req)
    # The native decode path extends output_ids before applying to_finish.
    req.output_ids.extend([72])
    functions["update_finish_state"](req, 1)
    native_commits = build(processor, batch, [start])
    assert native_commits[0].token_ids == (72,)
    assert req.finished() and req.finished_len is None
    commit(clock, [req], 20, native_commits, token_events=recorder)
    recorder.finish("test", 20)

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [
        event["at_ns"] for event in events if event["event"] == "output_commit"
    ] == [10]
    assert events[-1]["total_output_tokens"] == 1
    assert snapshot(req)["completion_tokens"] == 1
    assert snapshot(req)["last_token_ns"] == 10


def _native_tokenizer_functions():
    # Exercise the upstream metadata slicing/coalescing code without importing
    # the GPU runtime. These two pure functions are the compatibility boundary.
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/managers/tokenizer_manager.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_slice_streaming_output_meta_info", "_coalesce_streaming_chunks"}
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = {
        "_INCREMENTAL_STREAMING_META_INFO_KEYS": set(),
        "logger": Mock(),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102 - trusted repository functions
    return namespace


def test_snapshots_survive_native_incremental_slicing_and_coalescing():
    functions = _native_tokenizer_functions()
    clock, (req,) = setup_requests("request")
    manager = tokenizer()
    state = SimpleNamespace(customized_info_accumulated={})
    outputs = []
    for index, at_ns in enumerate((10, 20, 30)):
        req.done = index == 2
        commit(clock, [req], at_ns, [tokens(index, 71)])
        meta = {}
        info = streamer().build_additional_customized_info([req])
        manager.update_request_meta_info(meta, state, info, 0, None)
        functions["_slice_streaming_output_meta_info"](
            meta, index, state.customized_info_accumulated.keys()
        )
        outputs.append({"text": "x", "output_ids": [71], "meta_info": meta})
    assert METADATA_KEY not in state.customized_info_accumulated
    combined = functions["_coalesce_streaming_chunks"](
        None, outputs, req.rid, state.customized_info_accumulated.keys()
    )
    assert combined["text"] == "xxx"
    assert combined["output_ids"] == [71, 71, 71]
    assert combined["meta_info"][METADATA_KEY] == snapshot(req)
    assert combined["meta_info"][METADATA_KEY]["first_token_ns"] == 10
    assert combined["meta_info"][METADATA_KEY]["last_token_ns"] == 30


@pytest.mark.parametrize("has_snapshot", [False, True])
def test_tokenizer_preserves_other_metadata_without_accumulating_snapshot(has_snapshot):
    manager = tokenizer()
    state = SimpleNamespace(customized_info_accumulated={})
    meta = {}
    info = {"other": [[1], [2]]}
    if has_snapshot:
        info[METADATA_KEY] = [None, {"rid": "right"}]
    manager.update_request_meta_info(meta, state, info, 1, None)
    assert meta["other"] == [2]
    assert state.customized_info_accumulated == {"other": [2]}
    assert (meta.get(METADATA_KEY) == {"rid": "right"}) is has_snapshot


def test_sse_snapshot_precedes_all_chunks_from_one_native_output():
    class Chat:
        async def _generate_stream_content(self, content, index):
            yield f"data: content-{index}\n\n"
            yield "data: tool-call\n\n"

    C_TimingChatHook.hook(Chat)
    clock, (req,) = setup_requests("request")
    req.done = True
    commit(clock, [req], 10, [tokens(0, 71)])
    expected = snapshot(req)

    async def collect():
        return [
            item
            async for item in Chat()._generate_stream_content(
                content={"meta_info": {METADATA_KEY: expected}}, index=0
            )
        ]

    chunks = asyncio.run(collect())
    assert len(chunks) == 3
    assert chunks[0].startswith(f"event: {METADATA_KEY}\ndata: ")
    assert msgspec.json.decode(chunks[0].split("data: ", 1)[1]) == expected
    assert chunks[1:] == ["data: content-0\n\n", "data: tool-call\n\n"]


def test_missing_metadata_cannot_silently_report_wall_time():
    class Chat:
        async def _generate_stream_content(self, content):
            yield "unexpected"

    C_TimingChatHook.hook(Chat)

    async def collect():
        return [
            item async for item in Chat()._generate_stream_content({"meta_info": {}})
        ]

    with pytest.raises(ProtocolError, match="lacks timing metadata"):
        asyncio.run(collect())


def test_abort_before_any_token_commit_keeps_native_response():
    class Chat:
        async def _generate_stream_content(self, content):
            yield "native abort"

    C_TimingChatHook.hook(Chat)

    async def collect():
        content = {"meta_info": {"finish_reason": {"type": "abort"}}}
        return [item async for item in Chat()._generate_stream_content(content)]

    assert asyncio.run(collect()) == ["native abort"]
