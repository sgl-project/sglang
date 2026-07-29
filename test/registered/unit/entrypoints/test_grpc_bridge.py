import asyncio
from enum import Enum
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Status(Enum):
    Ready = 1
    Pending = 2
    Closed = 3


class _Callback:
    def __init__(self, statuses=None):
        self.calls = []
        self.statuses = iter(statuses or [])

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return next(self.statuses, _Status.Ready)


class _BackpressuredCallback(_Callback):
    def __init__(self):
        super().__init__()
        self.on_ready = None

    def set_on_ready(self, on_ready):
        self.on_ready = on_ready

    def clear_on_ready(self):
        self.on_ready = None

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        if not kwargs.get("finished"):
            asyncio.get_running_loop().call_soon(self.on_ready)
            return _Status.Pending
        return _Status.Ready


class _TokenizerManager:
    def __init__(self, outputs, incremental=False):
        self.outputs = outputs
        self.server_args = SimpleNamespace(incremental_streaming_output=incremental)
        self.aborted = []
        self.yield_scheduler_errors = None

    async def generate_request(self, _obj, request=None, yield_scheduler_errors=False):
        del request
        self.yield_scheduler_errors = yield_scheduler_errors
        for output in self.outputs:
            if isinstance(output, Exception):
                raise output
            yield output

    def abort_request(self, **kwargs):
        self.aborted.append(kwargs)


def _runtime(outputs, incremental=False):
    runtime = RuntimeHandle.__new__(RuntimeHandle)
    runtime.tokenizer_manager = _TokenizerManager(outputs, incremental)
    runtime._event_loop = asyncio.get_running_loop()
    return runtime


@pytest.mark.asyncio
async def test_streaming_generation_waits_for_every_choice_terminal():
    runtime = _runtime(
        [
            {
                "index": 0,
                "output_ids": [1],
                "meta_info": {"finish_reason": {"type": "stop"}},
            },
            {
                "index": 1,
                "output_ids": [2],
                "meta_info": {"finish_reason": {"type": "length"}},
            },
        ],
        incremental=True,
    )
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, True, None)

    assert [call[0]["index"] for call in callback.calls] == [0, 1]
    assert [call[1]["finished"] for call in callback.calls] == [False, True]


@pytest.mark.asyncio
async def test_legacy_incremental_stream_restores_cumulative_choices():
    runtime = _runtime(
        [
            {"index": 0, "output_ids": [10], "meta_info": {}},
            {"index": 1, "output_ids": [20], "meta_info": {}},
            {
                "index": 0,
                "output_ids": [11],
                "meta_info": {"finish_reason": {"type": "stop"}},
            },
            {
                "index": 1,
                "output_ids": [21],
                "meta_info": {"finish_reason": {"type": "length"}},
            },
        ],
        incremental=True,
    )
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, True, None)

    assert [call[0]["output_ids"] for call in callback.calls] == [
        [10],
        [20],
        [10, 11],
        [20, 21],
    ]


@pytest.mark.asyncio
async def test_typed_streaming_normalizes_cumulative_choices_before_callback():
    def _meta(tokens, *, finished=False):
        return {
            "input_token_logprobs": [[-0.1, 100, "prompt"]],
            "input_top_logprobs": [[[-0.1, 100, "prompt"]]],
            "output_token_logprobs": [
                [-0.1 - index, token, str(token)] for index, token in enumerate(tokens)
            ],
            "output_top_logprobs": [
                [[-0.1 - rank, token + rank, str(token + rank)] for rank in range(2)]
                for token in tokens
            ],
            "finish_reason": {"type": "stop"} if finished else None,
        }

    runtime = _runtime(
        [
            {"index": 0, "output_ids": [10], "meta_info": _meta([10])},
            {"index": 1, "output_ids": [20], "meta_info": _meta([20])},
            {
                "index": 0,
                "output_ids": [10, 11],
                "meta_info": _meta([10, 11], finished=True),
            },
            {
                "index": 1,
                "output_ids": [20, 21],
                "meta_info": _meta([20, 21], finished=True),
            },
        ]
    )
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, True, None, typed_generation=True)

    assert [call[0]["output_ids"] for call in callback.calls] == [
        [10],
        [20],
        [11],
        [21],
    ]
    assert [call[0]["legacy_output_ids"] for call in callback.calls] == [
        [10],
        [20],
        [10, 11],
        [20, 21],
    ]
    assert [
        len(call[0]["meta_info"]["output_token_logprobs"]) for call in callback.calls
    ] == [1, 1, 1, 1]
    assert [
        len(call[0]["meta_info"]["output_top_logprobs"]) for call in callback.calls
    ] == [1, 1, 1, 1]
    assert [
        "input_token_logprobs" in call[0]["meta_info"] for call in callback.calls
    ] == [True, True, False, False]
    assert runtime.tokenizer_manager.yield_scheduler_errors is True


@pytest.mark.asyncio
async def test_typed_incremental_stream_preserves_output_deltas():
    runtime = _runtime(
        [
            {
                "output_ids": [10],
                "meta_info": {
                    "input_token_logprobs": [[-0.1, 1, "prompt"]],
                    "output_token_logprobs": [[-0.2, 10, "10"]],
                },
            },
            {
                "output_ids": [11],
                "meta_info": {
                    "input_token_logprobs": [[-0.1, 1, "prompt"]],
                    "output_token_logprobs": [[-0.3, 11, "11"]],
                    "finish_reason": {"type": "stop"},
                },
            },
        ],
        incremental=True,
    )
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 1}, rid="request")

    await runtime._run_generate(obj, callback, True, None, typed_generation=True)

    assert [call[0]["output_ids"] for call in callback.calls] == [[10], [11]]
    assert [call[0]["legacy_output_ids"] for call in callback.calls] == [
        [10],
        [10, 11],
    ]
    assert [
        call[0]["meta_info"]["output_token_logprobs"] for call in callback.calls
    ] == [[[-0.2, 10, "10"]], [[-0.3, 11, "11"]]]
    assert "input_token_logprobs" in callback.calls[0][0]["meta_info"]
    assert "input_token_logprobs" not in callback.calls[1][0]["meta_info"]


@pytest.mark.asyncio
async def test_typed_non_streaming_generation_honors_backpressure():
    runtime = _runtime(
        [
            [
                {"output_ids": [1], "meta_info": {"finish_reason": {"type": "stop"}}},
                {"output_ids": [2], "meta_info": {"finish_reason": {"type": "stop"}}},
            ]
        ]
    )
    callback = _BackpressuredCallback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, False, None, typed_generation=True)

    assert [call[0]["index"] for call in callback.calls] == [0, 1]
    assert [call[1]["finished"] for call in callback.calls] == [False, True]


@pytest.mark.asyncio
async def test_typed_non_streaming_generation_stops_on_closed_callback():
    runtime = _runtime(
        [
            [
                {"output_ids": [1], "meta_info": {"finish_reason": {"type": "stop"}}},
                {"output_ids": [2], "meta_info": {"finish_reason": {"type": "stop"}}},
                {"output_ids": [3], "meta_info": {"finish_reason": {"type": "stop"}}},
            ]
        ]
    )
    callback = _Callback([_Status.Ready, _Status.Closed])
    obj = SimpleNamespace(sampling_params={"n": 3}, rid="request")

    await runtime._run_generate(obj, callback, False, None, typed_generation=True)

    assert [call[0]["index"] for call in callback.calls] == [0, 1]
    assert runtime.tokenizer_manager.aborted == [{"rid": "request"}]


@pytest.mark.asyncio
async def test_typed_non_streaming_error_terminates_every_choice():
    runtime = _runtime([ValueError("invalid sampling parameters")])
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, False, None, typed_generation=True)

    assert [call[0]["index"] for call in callback.calls] == [0, 1]
    assert [call[1]["finished"] for call in callback.calls] == [False, True]
    assert all(
        call[0]["meta_info"]["finish_reason"]["type"] == "error"
        for call in callback.calls
    )
    assert all(
        call[0]["meta_info"]["finish_reason"]["status_code"] == 400
        for call in callback.calls
    )


@pytest.mark.asyncio
async def test_typed_cancellation_terminates_every_unfinished_choice():
    started = asyncio.Event()

    class _BlockingTokenizerManager(_TokenizerManager):
        async def generate_request(
            self, _obj, request=None, yield_scheduler_errors=False
        ):
            del request
            self.yield_scheduler_errors = yield_scheduler_errors
            started.set()
            await asyncio.Event().wait()
            yield

    runtime = _runtime([])
    runtime.tokenizer_manager = _BlockingTokenizerManager([])
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")
    task = asyncio.create_task(
        runtime._run_generate(obj, callback, True, None, typed_generation=True)
    )
    await started.wait()

    task.cancel()
    await task

    assert [call[0]["index"] for call in callback.calls] == [0, 1]
    assert [call[1]["finished"] for call in callback.calls] == [False, True]
    assert all(
        call[0]["meta_info"]["finish_reason"]["status_code"] == 499
        for call in callback.calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (
            [
                {
                    "index": 2,
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }
            ],
            "choice index 2 is outside 0..2",
        ),
        (
            [
                {
                    "index": 0,
                    "meta_info": {"finish_reason": {"type": "stop"}},
                },
                {
                    "index": 0,
                    "meta_info": {"finish_reason": {"type": "stop"}},
                },
            ],
            "duplicate terminal for choice 0",
        ),
        ([{"index": 0, "meta_info": {}}], "without terminal choices: [0, 1]"),
    ],
)
async def test_streaming_generation_rejects_invalid_choice_lifecycles(outputs, message):
    runtime = _runtime(outputs)
    runtime._abort_request_id = lambda _rid: None
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, True, None)

    assert callback.calls[-1][1]["error"]
    assert message in callback.calls[-1][1]["error"]


@pytest.mark.asyncio
async def test_non_streaming_generation_rejects_missing_choice():
    runtime = _runtime(
        [[{"output_ids": [1], "meta_info": {"finish_reason": {"type": "stop"}}}]]
    )
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, False, None)

    assert callback.calls[-1][1]["error"]
    assert "returned 1 choices; expected 2" in callback.calls[-1][1]["error"]
