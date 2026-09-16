import asyncio
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.serving_hook import handle_other_validations
from sglang.srt.entrypoints.openai.request_conversion import RequestConversionExecutor
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


async def wait_started(event):
    async def poll():
        while not event.is_set():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 1)


class Handler(OpenAIServingBase):
    def _request_id_prefix(self):
        return "test"

    def _validate_request(self, request):
        request.validated_on = threading.get_ident()
        return getattr(request, "validation_error", None)

    def _convert_to_internal_request(self, request, raw_request=None):
        request.converted_on = threading.get_ident()
        if getattr(request, "conversion_error", False):
            raise ValueError("invalid conversion")
        return SimpleNamespace(), request

    async def _handle_non_streaming_request(self, adapted, processed, raw_request):
        return processed


class TestRequestConversionExecutor(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_retains_worker_slot(self):
        executor = RequestConversionExecutor(1)
        started, release, second_started = (threading.Event() for _ in range(3))

        def first():
            started.set()
            if not release.wait(2):
                raise TimeoutError("test worker did not release")
            raise ValueError("detached worker failed")

        first_task = asyncio.create_task(executor.run(first))
        try:
            await wait_started(started)
            first_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first_task
            second = asyncio.create_task(
                executor.run(lambda: second_started.set() or 2)
            )
            await asyncio.sleep(0)
            self.assertFalse(second_started.is_set())
            release.set()
            self.assertEqual(await asyncio.wait_for(second, 1), 2)
        finally:
            release.set()

    async def test_cancellation_while_queued_does_not_submit(self):
        executor = RequestConversionExecutor(1)
        started, release, queued_started = (threading.Event() for _ in range(3))

        def first():
            started.set()
            release.wait(2)

        first_task = asyncio.create_task(executor.run(first))
        try:
            await wait_started(started)
            queued = asyncio.create_task(executor.run(queued_started.set))
            await asyncio.sleep(0)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            release.set()
            await asyncio.wait_for(first_task, 1)
            self.assertFalse(queued_started.is_set())
            self.assertEqual(await executor.run(lambda: 7), 7)
        finally:
            release.set()

    async def test_worker_exception_releases_slot(self):
        executor = RequestConversionExecutor(1)
        with self.assertRaisesRegex(ValueError, "bad schema"):
            await executor.run(lambda: (_ for _ in ()).throw(ValueError("bad schema")))
        self.assertEqual(await asyncio.wait_for(executor.run(lambda: 1), 1), 1)

    async def test_event_loop_progresses_while_worker_blocks(self):
        executor = RequestConversionExecutor(1)
        started, release = threading.Event(), threading.Event()

        def blocking():
            started.set()
            if not release.wait(2):
                raise TimeoutError("event loop could not release thread")
            return threading.get_ident()

        task = asyncio.create_task(executor.run(blocking))
        try:
            await wait_started(started)
            # This coroutine can run only if conversion did not block the event loop.
            release.set()
            self.assertNotEqual(await task, threading.get_ident())
        finally:
            release.set()

    async def test_handlers_share_limit_and_error_semantics(self):
        manager = SimpleNamespace(
            server_args=None, request_logger=SimpleNamespace(log_requests=False)
        )
        with get_context().override_server_args(request_conversion_concurrency=1):
            first, second = Handler(manager), Handler(manager)
        self.assertIs(
            first.request_conversion_executor, second.request_conversion_executor
        )
        req = SimpleNamespace(stream=False)
        self.assertIs(await first.handle_request(req, None), req)
        self.assertNotEqual(req.validated_on, threading.get_ident())
        self.assertNotEqual(req.converted_on, threading.get_ident())

        invalid = SimpleNamespace(stream=False, validation_error="invalid schema")
        response = await second.handle_request(invalid, None)
        self.assertEqual(json.loads(response.body)["message"], "invalid schema")
        self.assertFalse(hasattr(invalid, "converted_on"))

        invalid = SimpleNamespace(stream=False, conversion_error=True)
        response = await first.handle_request(invalid, None)
        self.assertEqual(json.loads(response.body)["message"], "invalid conversion")
        self.assertEqual(response.status_code, 400)

    async def test_disabled_preserves_inline_behavior(self):
        with get_context().override_server_args(request_conversion_concurrency=0):
            handler = Handler(
                SimpleNamespace(
                    server_args=None, request_logger=SimpleNamespace(log_requests=False)
                )
            )
        req = SimpleNamespace(stream=False)
        await handler.handle_request(req, None)
        self.assertEqual(req.validated_on, threading.get_ident())
        self.assertEqual(req.converted_on, threading.get_ident())

    def test_invalid_concurrency(self):
        for value in (-1, 0):
            with self.assertRaises(ValueError):
                RequestConversionExecutor(value)
        with patch(
            "sglang.srt.arg_groups.serving_hook.resolving_view",
            return_value=SimpleNamespace(request_conversion_concurrency=-1),
        ):
            with self.assertRaisesRegex(ValueError, "must be nonnegative"):
                handle_other_validations(object())
