"""Engine.encode / async_encode request construction, with a stub tokenizer manager."""

import asyncio
import unittest

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_RESULT = {"embedding": [0.1, 0.2], "meta_info": {"prompt_tokens": 3}}


class _RecordingTokenizerManager:
    def __init__(self):
        self.requests = []

    def generate_request(self, obj, request):
        self.requests.append(obj)

        async def _gen():
            yield _RESULT

        return _gen()


def _make_engine():
    engine = object.__new__(Engine)
    engine.tokenizer_manager = _RecordingTokenizerManager()
    engine.loop = asyncio.new_event_loop()
    return engine


class TestEngineEncode(CustomTestCase):
    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.loop.close()

    def _last_request(self) -> EmbeddingReqInput:
        return self.engine.tokenizer_manager.requests[-1]

    def test_encode_with_input_ids(self):
        out = self.engine.encode(input_ids=[1, 2, 3])
        self.assertEqual(out, _RESULT)
        req = self._last_request()
        self.assertEqual(req.input_ids, [1, 2, 3])
        self.assertIsNone(req.text)

    def test_async_encode_with_input_ids(self):
        out = self.engine.loop.run_until_complete(
            self.engine.async_encode(input_ids=[[1, 2], [3]], dimensions=8)
        )
        self.assertEqual(out, _RESULT)
        req = self._last_request()
        self.assertEqual(req.input_ids, [[1, 2], [3]])
        self.assertIsNone(req.text)
        self.assertEqual(req.dimensions, 8)

    def test_positional_prompt_still_supported(self):
        self.engine.encode("hello")
        req = self._last_request()
        self.assertEqual(req.text, "hello")
        self.assertIsNone(req.input_ids)


if __name__ == "__main__":
    unittest.main()
