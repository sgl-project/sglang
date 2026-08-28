import asyncio
import unittest
from unittest.mock import Mock, patch

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cpu_ci(est_time=1, suite="base-c-test-cpu")


class TestTokenizerEventLoopWatchdog(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_feeds_watchdog(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.event_loop_watchdog = Mock()
        manager.event_loop_watchdog_timeout = 30

        with patch(
            "sglang.srt.managers.tokenizer_manager.asyncio.sleep",
            side_effect=asyncio.CancelledError,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await manager.event_loop_watchdog_loop()

        manager.event_loop_watchdog.feed.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
