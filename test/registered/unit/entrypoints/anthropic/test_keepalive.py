import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from starlette.background import BackgroundTask  # noqa: E402
from starlette.responses import StreamingResponse  # noqa: E402

from sglang.srt.entrypoints.anthropic.serving import (  # noqa: E402
    AnthropicServing,
    _sse_keepalive,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSSEKeepalive(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def test_disabled_preserves_frames(self):
        frames = ["event: message_start\ndata: {}\n\n", "data: end\n\n"]
        for interval in (0, -1):
            with self.subTest(interval=interval):

                async def source():
                    for frame in frames:
                        yield frame

                self.assertEqual(
                    [frame async for frame in _sse_keepalive(source(), interval)],
                    frames,
                )

    async def test_repeated_keepalives_do_not_cancel_or_restart_read(self):
        ready = asyncio.Event()
        reads = 0
        closed = False

        async def source():
            nonlocal reads, closed
            try:
                reads += 1
                await ready.wait()
                yield "data: first\n\n"
                yield "data: second\n\n"
            finally:
                closed = True

        stream = _sse_keepalive(source(), 0.001)
        try:
            for _ in range(3):
                self.assertEqual(await anext(stream), ": keepalive\n\n")
            self.assertEqual(reads, 1)
            self.assertFalse(closed)
            ready.set()
            self.assertEqual(
                [frame async for frame in stream],
                ["data: first\n\n", "data: second\n\n"],
            )
            self.assertTrue(closed)
        finally:
            await stream.aclose()

    async def test_close_while_read_is_pending(self):
        closed = asyncio.Event()

        async def source():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        stream = _sse_keepalive(source(), 0.001)
        self.assertEqual(await anext(stream), ": keepalive\n\n")
        await stream.aclose()
        self.assertTrue(closed.is_set())

    async def test_cancellation_reaches_source(self):
        for interval in (0, 60):
            with self.subTest(interval=interval):
                started = asyncio.Event()
                closed = asyncio.Event()

                async def source():
                    try:
                        started.set()
                        await asyncio.Event().wait()
                        yield "unreachable"
                    finally:
                        closed.set()

                stream = _sse_keepalive(source(), interval)
                pending = asyncio.ensure_future(anext(stream))
                await asyncio.wait_for(started.wait(), timeout=1)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertTrue(closed.is_set())
                await stream.aclose()

    async def test_close_after_frame_closes_source(self):
        for interval in (0, 0.001):
            with self.subTest(interval=interval):
                closed = False

                async def source():
                    nonlocal closed
                    try:
                        yield "data: first\n\n"
                        yield "data: second\n\n"
                    finally:
                        closed = True

                stream = _sse_keepalive(source(), interval)
                self.assertEqual(await anext(stream), "data: first\n\n")
                await stream.aclose()
                self.assertTrue(closed)

    async def test_source_error_propagates(self):
        async def source():
            yield "data: first\n\n"
            raise ValueError("synthetic stream failure")

        stream = _sse_keepalive(source(), 0.001)
        self.assertEqual(await anext(stream), "data: first\n\n")
        with self.assertRaisesRegex(ValueError, "synthetic stream failure"):
            await anext(stream)

    async def test_empty_source_finishes(self):
        async def source():
            if False:
                yield "unreachable"

        self.assertEqual([frame async for frame in _sse_keepalive(source(), 1)], [])

    async def test_streaming_handler_wraps_source_and_retains_abort_task(self):
        closed = asyncio.Event()

        async def source(*args):
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        adapted = SimpleNamespace()
        abort_task = Mock()
        openai = Mock()
        openai._validate_request.return_value = None
        openai._convert_to_internal_request.return_value = (adapted, object())
        openai.tokenizer_manager.create_abort_task.return_value = abort_task
        serving = AnthropicServing(openai)
        serving._generate_anthropic_stream = source
        with envs.SGLANG_SSE_KEEPALIVE_S.override(0.001):
            response = await serving._handle_streaming(object(), object(), object())
        self.assertIs(response.background, abort_task)
        openai.tokenizer_manager.create_abort_task.assert_called_once_with(adapted)
        self.assertEqual(await anext(response.body_iterator), ": keepalive\n\n")
        await response.body_iterator.aclose()
        self.assertTrue(closed.is_set())

    async def test_asgi_disconnect_cleans_up_pending_read(self):
        frame_sent = asyncio.Event()
        closed = asyncio.Event()
        aborted = asyncio.Event()

        async def source():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        async def receive():
            await frame_sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                self.assertEqual(message["body"], b": keepalive\n\n")
                frame_sent.set()

        async def abort():
            aborted.set()

        response = StreamingResponse(
            _sse_keepalive(source(), 0.001), background=BackgroundTask(abort)
        )
        await asyncio.wait_for(
            response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send),
            timeout=2,
        )
        self.assertTrue(closed.is_set())
        self.assertTrue(aborted.is_set())


if __name__ == "__main__":
    unittest.main()
