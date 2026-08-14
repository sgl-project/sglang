import asyncio
from types import SimpleNamespace

from examples.voicechat.online_session import (
    AsyncSGLangVoiceChatSession,
    SGLangVoiceChatSession,
)


class FakeEngine:
    def __init__(self, results):
        self.results = iter(results)
        self.requests = []
        self.closed = []

    def open_session(self, capacity, streaming):
        assert capacity == 8192 and streaming
        return f"session-{id(self)}"

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.results)

    def close_session(self, session_id):
        self.closed.append(session_id)


def test_two_stage_feedback_contract():
    duplex = FakeEngine(
        [
            {"output_ids": [99], "meta_info": {"function_tokens": [6]}},
            {"output_ids": [7], "meta_info": {"function_tokens": [8]}},
        ]
    )
    eartts = FakeEngine(
        [
            {"output_ids": [0]},
            {"output_ids": [0], "meta_info": {"audio_codes": [[1, 2, 3]]}},
            {"output_ids": [0], "meta_info": {"audio_codes": [[4, 5, 6]]}},
        ]
    )
    session = SGLangVoiceChatSession(duplex, eartts)
    session.start([10, 11], SimpleNamespace(shape=(4, 1152)), pad_token_id=12)
    first = session.step([[0.0] * 4480])
    second = session.step([[1.0] * 4480])

    assert (first.text_token, first.function_token, first.audio_codes) == (
        99,
        6,
        [1, 2, 3],
    )
    assert (second.text_token, second.function_token, second.audio_codes) == (
        7,
        8,
        [4, 5, 6],
    )
    assert duplex.requests[0]["input_ids"] == [10, 11, 12]
    assert duplex.requests[0]["custom_inputs"]["is_initial_prefill"] is True
    assert duplex.requests[0]["custom_inputs"]["prompt_length"] == 2
    assert duplex.requests[-1]["input_ids"] == []
    assert duplex.requests[-1]["custom_inputs"]["input_function_ids"] == [6]
    assert eartts.requests[-1]["custom_inputs"]["text_token"] == 7
    assert eartts.requests[-1]["input_ids"] == []
    assert eartts.requests[-1]["custom_inputs"]["previous_audio_codes"] == [1, 2, 3]


class AsyncFakeEngine(FakeEngine):
    async def async_open_session(self, capacity, streaming):
        return self.open_session(capacity, streaming)

    async def async_generate(self, **kwargs):
        return self.generate(**kwargs)

    async def async_close_session(self, session_id):
        self.close_session(session_id)


def test_async_two_stage_feedback_contract():
    async def run():
        duplex = AsyncFakeEngine(
            [
                {"output_ids": [99], "meta_info": {"function_tokens": [6]}},
                {"output_ids": [7], "meta_info": {"function_tokens": [8]}},
            ]
        )
        eartts = AsyncFakeEngine(
            [
                {"output_ids": [0]},
                {
                    "output_ids": [0],
                    "meta_info": {"audio_codes": [[1, 2, 3]]},
                },
                {"output_ids": [0], "meta_info": {"audio_codes": [[4, 5, 6]]}},
            ]
        )
        session = await AsyncSGLangVoiceChatSession.create(duplex, eartts)
        await session.start([10, 11], SimpleNamespace(shape=(4, 1152)), pad_token_id=12)
        first = await session.step([[0.0] * 4480])
        second = await session.step([[1.0] * 4480])
        await session.close()

        assert (first.text_token, first.function_token, first.audio_codes) == (
            99,
            6,
            [1, 2, 3],
        )
        assert (second.text_token, second.function_token, second.audio_codes) == (
            7,
            8,
            [4, 5, 6],
        )
        assert duplex.requests[0]["input_ids"] == [10, 11, 12]
        assert duplex.requests[0]["custom_inputs"]["is_initial_prefill"] is True
        assert duplex.requests[-1]["custom_inputs"]["input_function_ids"] == [6]
        assert eartts.requests[-1]["custom_inputs"]["text_token"] == 7
        assert len(duplex.closed) == len(eartts.closed) == 1

    asyncio.run(run())
