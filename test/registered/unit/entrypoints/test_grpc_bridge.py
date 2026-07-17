from enum import Enum
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle


class _Status(Enum):
    Ready = 1
    Pending = 2
    Closed = 3


class _Callback:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return _Status.Ready


class _TokenizerManager:
    def __init__(self, outputs, incremental=False):
        self.outputs = outputs
        self.server_args = SimpleNamespace(incremental_streaming_output=incremental)

    async def generate_request(self, _obj, request=None):
        del request
        for output in self.outputs:
            yield output


def _runtime(outputs, incremental=False):
    runtime = RuntimeHandle.__new__(RuntimeHandle)
    runtime.tokenizer_manager = _TokenizerManager(outputs, incremental)
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
    assert all(call[1]["incremental"] for call in callback.calls)


@pytest.mark.asyncio
async def test_non_streaming_generation_assigns_choice_indices():
    runtime = _runtime(
        [
            [
                {"output_ids": [1], "meta_info": {"finish_reason": {"type": "stop"}}},
                {"output_ids": [2], "meta_info": {"finish_reason": {"type": "stop"}}},
            ]
        ]
    )
    callback = _Callback()
    obj = SimpleNamespace(sampling_params={"n": 2}, rid="request")

    await runtime._run_generate(obj, callback, False, None)

    assert [call[0]["index"] for call in callback.calls] == [0, 1]
    assert [call[1]["finished"] for call in callback.calls] == [False, True]
    assert not any(call[1]["incremental"] for call in callback.calls)


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
