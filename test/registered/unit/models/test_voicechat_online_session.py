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
            {"output_ids": [99], "meta_info": {"asr_tokens": [6]}},
            {"output_ids": [7], "meta_info": {"asr_tokens": [8]}},
        ]
    )
    eartts = FakeEngine(
        [
            {"output_ids": [0]},
            {"output_ids": [0], "meta_info": {"audio_codes": [[1, 2, 3]]}},
        ]
    )
    session = SGLangVoiceChatSession(duplex, eartts)
    session.start([10, 11], SimpleNamespace(shape=(4, 1152)))
    result = session.step([[0.0] * 4480])

    assert (result.text_token, result.asr_token, result.audio_codes) == (
        7,
        8,
        [1, 2, 3],
    )
    assert duplex.requests[-1]["input_ids"] == []
    assert duplex.requests[-1]["custom_inputs"]["input_asr_ids"] == [6]
    assert eartts.requests[-1]["custom_inputs"]["text_token"] == 7
    assert eartts.requests[-1]["input_ids"] == []
    assert eartts.requests[-1]["custom_inputs"]["previous_audio_codes"] is None


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
                {"output_ids": [99], "meta_info": {"asr_tokens": [6]}},
                {"output_ids": [7], "meta_info": {"asr_tokens": [8]}},
            ]
        )
        eartts = AsyncFakeEngine(
            [
                {"output_ids": [0]},
                {
                    "output_ids": [0],
                    "meta_info": {"audio_codes": [[1, 2, 3]]},
                },
            ]
        )
        session = await AsyncSGLangVoiceChatSession.create(duplex, eartts)
        await session.start([10, 11], SimpleNamespace(shape=(4, 1152)))
        result = await session.step([[0.0] * 4480])
        await session.close()

        assert (result.text_token, result.asr_token, result.audio_codes) == (
            7,
            8,
            [1, 2, 3],
        )
        assert duplex.requests[-1]["custom_inputs"]["input_asr_ids"] == [6]
        assert eartts.requests[-1]["custom_inputs"]["text_token"] == 7
        assert len(duplex.closed) == len(eartts.closed) == 1

    asyncio.run(run())
