import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTokenizerFlushCache(unittest.IsolatedAsyncioTestCase):
    async def test_all_worker_results_and_cache_side_effects(self):
        ok = FlushCacheReqOutput(success=True)
        failed = FlushCacheReqOutput(
            success=False, message="Timed out waiting for idle state."
        )
        for replies, success in (
            ([ok], True),
            ([failed], False),
            ([ok, ok], True),
            ([failed, failed], False),
            ([ok, failed], False),
            ([failed, ok], False),
        ):
            for timeout_s in (None, 5.0):
                with self.subTest(
                    replies=[r.success for r in replies], timeout_s=timeout_s
                ):
                    manager = SimpleNamespace(
                        auto_create_handle_loop=Mock(),
                        flush_cache_communicator=AsyncMock(return_value=replies),
                        mm_processor=Mock(),
                    )
                    result = await TokenizerControlMixin.flush_cache(
                        manager, timeout_s=timeout_s
                    )

                    self.assertIsInstance(result, FlushCacheReqOutput)
                    self.assertEqual(result.success, success)
                    if not success:
                        self.assertIn(failed.message, result.message)
                    manager.auto_create_handle_loop.assert_called_once_with()
                    manager.flush_cache_communicator.assert_awaited_once_with(
                        FlushCacheReqInput(timeout_s=timeout_s)
                    )
                    if success:
                        manager.mm_processor.clear_preprocess_cache.assert_called_once_with()
                    else:
                        manager.mm_processor.clear_preprocess_cache.assert_not_called()

    async def test_text_only_manager(self):
        for success in (True, False):
            with self.subTest(success=success):
                manager = SimpleNamespace(
                    auto_create_handle_loop=Mock(),
                    flush_cache_communicator=AsyncMock(
                        return_value=[FlushCacheReqOutput(success=success)]
                    ),
                    mm_processor=None,
                )
                result = await TokenizerControlMixin.flush_cache(manager)
                self.assertEqual(result.success, success)


if __name__ == "__main__":
    unittest.main()
