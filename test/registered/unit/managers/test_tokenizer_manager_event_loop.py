"""Event-loop responsiveness tests for TokenizerManager tokenization."""

import asyncio
import threading
import unittest
from contextvars import ContextVar

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _BlockingFastTokenizer:
    is_fast = True

    def __init__(self, tokenization_started, release_tokenization):
        self.tokenization_started = tokenization_started
        self.release_tokenization = release_tokenization

    def __call__(self, texts, **kwargs):
        self.tokenization_started.set()
        if not self.release_tokenization.wait(timeout=5):
            raise TimeoutError("test did not release tokenization")
        return {"input_ids": [[1] for _ in texts]}


class TestTokenizerManagerEventLoop(CustomTestCase):
    def test_regular_tokenizer_does_not_block_event_loop(self):
        tokenization_started = threading.Event()
        loop_progressed = threading.Event()
        release_tokenization = threading.Event()
        loop_was_starved = threading.Event()
        tokenization_did_not_start = threading.Event()

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = _BlockingFastTokenizer(
            tokenization_started, release_tokenization
        )
        manager.async_dynamic_batch_tokenizer = None
        manager.init_request_preprocessor()

        def release_after_loop_progress():
            if not tokenization_started.wait(timeout=2):
                tokenization_did_not_start.set()
                release_tokenization.set()
                return
            if not loop_progressed.wait(timeout=1):
                loop_was_starved.set()
            release_tokenization.set()

        watchdog = threading.Thread(target=release_after_loop_progress)
        watchdog.start()

        async def tokenize_and_probe_loop():
            tokenize_task = asyncio.create_task(manager._tokenize_texts("hello"))
            await asyncio.sleep(0)
            loop_progressed.set()
            return await tokenize_task

        try:
            input_ids, token_type_ids = asyncio.run(tokenize_and_probe_loop())
        finally:
            release_tokenization.set()
            watchdog.join(timeout=2)
            manager._request_preprocessor_executor.shutdown(wait=True)

        self.assertEqual(input_ids, [1])
        self.assertIsNone(token_type_ids)
        self.assertFalse(tokenization_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            loop_was_starved.is_set(),
            "synchronous tokenization blocked the API event loop",
        )

    def test_request_preprocessing_is_serialized(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.init_request_preprocessor()

        def first_call():
            first_started.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release first preprocessing call")

        def second_call():
            second_started.set()

        async def run_concurrent_calls():
            first_task = asyncio.create_task(
                manager.run_in_request_preprocessor(first_call)
            )
            started = await asyncio.to_thread(first_started.wait, 1)
            self.assertTrue(started)

            second_task = asyncio.create_task(
                manager.run_in_request_preprocessor(second_call)
            )
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())

            release_first.set()
            await asyncio.gather(first_task, second_task)

        try:
            asyncio.run(run_concurrent_calls())
        finally:
            release_first.set()
            manager._request_preprocessor_executor.shutdown(wait=True)

        self.assertTrue(second_started.is_set())

    def test_request_preprocessing_preserves_context(self):
        request_context = ContextVar("request_context")
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.init_request_preprocessor()

        async def read_context_from_preprocessor():
            request_context.set("request-value")
            return await manager.run_in_request_preprocessor(request_context.get)

        try:
            result = asyncio.run(read_context_from_preprocessor())
        finally:
            manager._request_preprocessor_executor.shutdown(wait=True)

        self.assertEqual(result, "request-value")


if __name__ == "__main__":
    unittest.main()
