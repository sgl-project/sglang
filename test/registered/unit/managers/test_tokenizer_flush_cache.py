import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTokenizerFlushCache(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    async def test_all_worker_results_and_cache_side_effects(self):
        """A busy rank fails the flush without masking the default HTTP error."""
        ok = FlushCacheReqOutput(success=True)
        busy = FlushCacheReqOutput(success=False)
        timed_out = FlushCacheReqOutput(
            success=False, message="Timed out waiting for idle state."
        )
        for replies, timeout_s, success, message in (
            ([ok, ok], None, True, ""),
            ([ok, busy], 0.0, False, ""),
            ([ok, timed_out], 5.0, False, timed_out.message),
            ([timed_out, ok], 5.0, False, timed_out.message),
        ):
            with self.subTest(replies=replies, timeout_s=timeout_s):
                manager = SimpleNamespace(
                    auto_create_handle_loop=Mock(),
                    flush_cache_communicator=AsyncMock(return_value=replies),
                    mm_processor=Mock(),
                )
                result = await TokenizerControlMixin.flush_cache(
                    manager, timeout_s=timeout_s
                )

                self.assertEqual(
                    result, FlushCacheReqOutput(success=success, message=message)
                )
                manager.flush_cache_communicator.assert_awaited_once_with(
                    FlushCacheReqInput(timeout_s=timeout_s)
                )
                if success:
                    manager.mm_processor.clear_preprocess_cache.assert_called_once_with()
                else:
                    manager.mm_processor.clear_preprocess_cache.assert_not_called()

    async def test_text_only_manager(self):
        ok = FlushCacheReqOutput(success=True)
        manager = SimpleNamespace(
            auto_create_handle_loop=Mock(),
            flush_cache_communicator=AsyncMock(return_value=[ok]),
            mm_processor=None,
        )
        self.assertEqual(await TokenizerControlMixin.flush_cache(manager), ok)


if __name__ == "__main__":
    unittest.main()
