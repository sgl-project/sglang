import asyncio
import unittest
from types import SimpleNamespace

import orjson

from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


async def _chunks(chunks):
    for text, finished in chunks:
        yield {
            "text": text,
            "meta_info": {"finish_reason": {"type": "length"} if finished else None},
        }


async def _read_stream(response):
    return [orjson.loads(chunk) async for chunk in response.body_iterator]


class TestOllamaStreaming(CustomTestCase):
    def _serving(self, chunks, *, incremental):
        manager = SimpleNamespace(
            served_model_name="test-model",
            incremental_streaming_output=incremental,
            generate_request=lambda *_: _chunks(chunks),
        )
        return OllamaServing(manager)

    def test_chat_stream_emits_deltas_for_both_text_shapes(self):
        for chunks, incremental, expected in (
            (
                [("hello world", True)],
                False,
                ["hello world"],
            ),
            (
                [("hello ", False), ("hello world", True)],
                False,
                ["hello ", "world"],
            ),
            (
                [("hello ", False), ("world", True)],
                True,
                ["hello ", "world"],
            ),
        ):
            with self.subTest(incremental=incremental, chunks=chunks):
                response = asyncio.run(
                    self._serving(
                        chunks, incremental=incremental
                    )._stream_chat_response(None, None, "test-model")
                )
                body = asyncio.run(_read_stream(response))
                self.assertEqual(
                    [item["message"]["content"] for item in body], expected
                )
                self.assertTrue(body[-1]["done"])

    def test_generate_stream_emits_deltas_for_cumulative_text(self):
        serving = self._serving(
            [("hello ", False), ("hello world", True)], incremental=False
        )
        response = asyncio.run(
            serving._stream_generate_response(None, None, "test-model")
        )
        body = asyncio.run(_read_stream(response))

        self.assertEqual([item["response"] for item in body], ["hello ", "world"])
        self.assertTrue(body[-1]["done"])


if __name__ == "__main__":
    unittest.main()
