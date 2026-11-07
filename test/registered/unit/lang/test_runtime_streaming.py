"""Runtime streaming must depend on SSE events, not HTTP read boundaries."""

import asyncio
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import aiohttp
from aiohttp.base_protocol import BaseProtocol

from sglang.lang.backend.runtime_endpoint import Runtime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def event(payload):
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


class TestRuntimeStreaming(CustomTestCase):
    def make_stream(self):
        loop = asyncio.get_running_loop()
        protocol = BaseProtocol(loop)
        transport = MagicMock(spec=asyncio.Transport)
        transport.get_extra_info.return_value = None
        protocol.connection_made(transport)
        return aiohttp.StreamReader(protocol, limit=2**16, loop=loop)

    @contextmanager
    def generate(self, stream, token_ids=False, alias=False):
        # Bypass only model-server startup; exercise the real Runtime method
        # and aiohttp reader, with the HTTP request as the mocked boundary.
        runtime = Runtime.__new__(Runtime)
        runtime.pid = None
        runtime.server_args = SimpleNamespace(skip_tokenizer_init=token_ids)
        runtime.generate_url = "http://runtime.test/generate"
        response = MagicMock()
        response.__aenter__.return_value = SimpleNamespace(content=stream)
        session = MagicMock()
        session.__aenter__.return_value = session
        session.post.return_value = response
        prompt = [1, 2] if token_ids else "hello"
        with patch(
            "sglang.lang.backend.runtime_endpoint.aiohttp.ClientSession",
            return_value=session,
        ):
            method = runtime.add_request if alias else runtime.async_generate
            yield method(prompt, {"max_new_tokens": 8}, session_id="session-1")
        self.assertEqual(
            session.post.call_args.kwargs["json"],
            {
                "input_ids" if token_ids else "text": prompt,
                "sampling_params": {"max_new_tokens": 8},
                "stream": True,
                "session_id": "session-1",
            },
        )

    async def collect(self, chunks, **kwargs):
        stream = self.make_stream()
        with self.generate(stream, **kwargs) as generator:

            async def consume():
                return [item async for item in generator]

            consumer = asyncio.create_task(consume())
            try:
                for chunk in chunks:
                    stream.begin_http_chunk_receiving()
                    stream.feed_data(chunk)
                    stream.end_http_chunk_receiving()
                    # Let the consumer exhaust each fragment before supplying
                    # the next one, without relying on wall-clock delays.
                    await asyncio.sleep(0)
                stream.feed_eof()
                return await asyncio.wait_for(consumer, 5)
            finally:
                stream.feed_eof()
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
                await generator.aclose()

    def test_fragmented_events(self):
        """Splitting the prefix, JSON, UTF-8 or event delimiter must not lose text."""
        body = event({"text": "Ol\u00e1\U0001f30d"}) + b"data: [DONE]\n\n"
        for cut in range(1, len(body)):
            with self.subTest(cut=cut):
                self.assertEqual(
                    asyncio.run(self.collect([body[:cut], body[cut:]])),
                    ["Ol\u00e1\U0001f30d"],
                )

    def test_coalesced_events_preserve_deltas_and_done(self):
        """Coalescing must preserve deltas, suppress duplicates and stop at DONE."""
        body = (
            event({"text": "O"})
            + event({"text": "O"})
            + event({"text": "Ol\u00e1"})
            + b"data: [DONE]\n\ndata: {invalid}\n\n"
        )
        self.assertEqual(
            asyncio.run(self.collect([body], alias=True)), ["O", "l\u00e1"]
        )

    def test_large_event(self):
        """Cumulative output can exceed aiohttp's line limit."""
        text = "x" * (1024 * 1024)
        body = event({"text": text}) + b"data: [DONE]\n\n"
        chunks = [body[i : i + 16384] for i in range(0, len(body), 16384)]
        self.assertEqual(asyncio.run(self.collect(chunks)), [text])

    def test_event_framing(self):
        """Comments and CRLF/multiline data must not corrupt an event's JSON."""
        for newline in (b"\n", b"\r\n"):
            with self.subTest(newline=newline):
                body = newline.join(
                    [
                        b": heartbeat",
                        b"",
                        b"event: message",
                        b'data: {"text":',
                        b'data:"hello"}',
                        b"",
                        b"data: [DONE]",
                        b"",
                        b"",
                    ]
                )
                self.assertEqual(
                    asyncio.run(
                        self.collect([body[i : i + 1] for i in range(len(body))])
                    ),
                    ["hello"],
                )

    def test_non_text_payloads(self):
        """Token IDs and error events must remain dictionaries after reassembly."""
        for token_ids, payload in (
            (True, {"output_ids": [3, 4], "meta_info": {"completion_tokens": 2}}),
            (False, {"error": {"message": "bad sampling parameter"}}),
        ):
            with self.subTest(token_ids=token_ids):
                body = event(payload) + b"data: [DONE]\n\n"
                self.assertEqual(
                    asyncio.run(
                        self.collect([body[:9], body[9:]], token_ids=token_ids)
                    ),
                    [payload],
                )

    def test_incremental_delivery(self):
        """An event must be yielded before the next event or EOF arrives."""

        async def check():
            stream = self.make_stream()
            with self.generate(stream) as generator:
                try:
                    stream.feed_data(event({"text": "O"}))
                    self.assertEqual(await asyncio.wait_for(anext(generator), 5), "O")
                    stream.feed_data(event({"text": "Ol\u00e1"}) + b"data: [DONE]\n\n")
                    self.assertEqual(
                        await asyncio.wait_for(anext(generator), 5), "l\u00e1"
                    )
                    with self.assertRaises(StopAsyncIteration):
                        await asyncio.wait_for(anext(generator), 5)
                finally:
                    stream.feed_eof()
                    await generator.aclose()

        asyncio.run(check())

    def test_incomplete_event_at_eof(self):
        """EOF must not turn an unterminated SSE event into a partial response."""
        for suffix in (b'data: {"text": "partial"}', b'data: {"text": "partial"}\n'):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    asyncio.run(self.collect([event({"text": "O"}), suffix])), ["O"]
                )

    def test_invalid_json_is_not_swallowed(self):
        """Framing valid SSE must not silently discard malformed JSON."""
        with self.assertRaises(json.JSONDecodeError):
            asyncio.run(self.collect([b"data: {invalid}\n\n"]))


if __name__ == "__main__":
    unittest.main()
