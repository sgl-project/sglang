"""Unit tests for the offline Engine bridge to the Rust request frontend."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.rust_server.offline_engine import (
    RustOfflineEngineClient,
    generation_request_body,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _request(**overrides):
    return GenerateReqInput(
        **{
            "text": "hello",
            "sampling_params": {"max_new_tokens": 4},
            "rid": "request-1",
            **overrides,
        }
    )


class TestRustOfflineEngineClient(CustomTestCase):
    def setUp(self):
        self.responses = []
        self.requests = []

        class Session:
            def __init__(_self, **_kwargs):
                pass

            async def __aenter__(_self):
                return _self

            async def __aexit__(_self, *_args):
                return None

            def post(_self, url, *, json):
                self.requests.append((url, json))
                response = self.responses.pop(0)

                class ResponseContext:
                    async def __aenter__(_response_context):
                        return response

                    async def __aexit__(_response_context, *_args):
                        return None

                return ResponseContext()

        self.client = RustOfflineEngineClient(
            "127.0.0.1", 30000, session_factory=Session
        )

    def test_generation_body_preserves_supported_request_fields(self):
        body = generation_request_body(
            _request(
                input_ids=[1, 2, 3],
                text=None,
                return_logprob=True,
                logprob_start_len=2,
                priority=7,
                cache_salt="tenant-a",
            )
        )
        self.assertEqual(
            body,
            {
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_new_tokens": 4},
                "stream": False,
                "rid": "request-1",
                "return_logprob": True,
                "logprob_start_len": 2,
                "priority": 7,
                "cache_salt": "tenant-a",
            },
        )

    def test_generation_rejects_unmigrated_payloads_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "image_data"):
            generation_request_body(_request(image_data=object()))

    def test_unary_generation_posts_json_and_returns_native_shape(self):
        self.responses.append(
            _FakeResponse(
                payload={"text": "hello!", "output_ids": [4, 5], "meta_info": {}}
            )
        )
        result = asyncio.run(
            self.client.generate(generation_request_body(_request()), stream=False)
        )
        self.assertEqual(result["output_ids"], [4, 5])
        self.assertEqual(self.requests[0][0], "http://127.0.0.1:30000/generate")
        self.assertEqual(self.requests[0][1]["text"], "hello")
        self.assertFalse(self.requests[0][1]["stream"])

    def test_streaming_yields_each_frame_and_stops_at_done(self):
        self.responses.append(
            _FakeResponse(
                lines=[
                    b'data: {"text": "hello", "output_ids": [4]}\n',
                    b"\n",
                    b'data: {"text": "hello!", "output_ids": [4, 5]}\n',
                    b"\n",
                    b"data: [DONE]\n",
                ]
            )
        )

        async def collect():
            stream = await self.client.generate(
                generation_request_body(_request(stream=True)), stream=True
            )
            return [frame async for frame in stream]

        frames = asyncio.run(collect())
        self.assertEqual([frame["output_ids"] for frame in frames], [[4], [4, 5]])
        self.assertTrue(self.requests[0][1]["stream"])

    def test_wildcard_bind_hosts_map_to_loopback(self):
        self.assertEqual(
            RustOfflineEngineClient("0.0.0.0", 30000).base_url,
            "http://127.0.0.1:30000",
        )
        self.assertEqual(
            RustOfflineEngineClient("::", 30000).base_url,
            "http://[::1]:30000",
        )


class _FakeResponse:
    def __init__(self, *, payload: Any = None, lines: list[bytes] | None = None):
        self.status = 200
        self._payload = payload
        self.content = _AsyncLines(lines or [])

    async def text(self) -> str:
        return json.dumps(self._payload)


class _AsyncLines:
    def __init__(self, lines: list[bytes]):
        self.lines = lines

    def __aiter__(self) -> AsyncIterator[bytes]:
        async def iterate():
            for line in self.lines:
                yield line

        return iterate()


if __name__ == "__main__":
    import unittest

    unittest.main()
