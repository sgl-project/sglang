"""Event-loop responsiveness tests for TokenizerManager tokenization."""

import asyncio
import threading
import unittest

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

        self.assertEqual(input_ids, [1])
        self.assertIsNone(token_type_ids)
        self.assertFalse(tokenization_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            loop_was_starved.is_set(),
            "synchronous tokenization blocked the API event loop",
        )


if __name__ == "__main__":
    unittest.main()
