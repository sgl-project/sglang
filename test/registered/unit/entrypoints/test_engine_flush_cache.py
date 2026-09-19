import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import FlushCacheReqOutput

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEngineFlushCache(CustomTestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)
        self.flush = AsyncMock(return_value=FlushCacheReqOutput(success=True))
        self.engine = SimpleNamespace(
            loop=self.loop,
            tokenizer_manager=SimpleNamespace(flush_cache=self.flush),
        )

    def test_default_returns_flush_result(self):
        result = Engine.flush_cache(self.engine)

        self.assertIs(result, self.flush.return_value)
        self.assertTrue(result.success)
        self.flush.assert_awaited_once()
        self.assertIsNone(self.flush.await_args.kwargs.get("timeout_s"))

    def test_forwards_explicit_timeout(self):
        for timeout_s in (None, 0.0, 2.5):
            with self.subTest(timeout_s=timeout_s):
                self.flush.reset_mock()

                result = Engine.flush_cache(self.engine, timeout_s=timeout_s)

                self.assertIs(result, self.flush.return_value)
                self.assertTrue(result.success)
                self.flush.assert_awaited_once_with(timeout_s=timeout_s)

    def test_preserves_failed_flush_result(self):
        self.flush.return_value = FlushCacheReqOutput(
            success=False, message="Timed out waiting for idle state."
        )

        result = Engine.flush_cache(self.engine, timeout_s=1.0)

        self.assertIs(result, self.flush.return_value)
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Timed out waiting for idle state.")

    def test_propagates_manager_error(self):
        self.flush.side_effect = RuntimeError("flush communication failed")

        with self.assertRaisesRegex(RuntimeError, "flush communication failed"):
            Engine.flush_cache(self.engine, timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
