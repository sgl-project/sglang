import asyncio
import unittest
from unittest.mock import AsyncMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_manager() -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager._async_dispatch_to_scheduler = AsyncMock()
    return manager


class TestTokenizerManagerPauseValidation(CustomTestCase):
    def test_invalid_mode_rejected_at_tokenizer_manager(self):
        """An invalid mode raises ValueError before any pause state is touched."""
        manager = _make_manager()
        request = PauseGenerationReqInput(mode="bogus")

        with self.assertRaisesRegex(ValueError, "Invalid pause_generation mode"):
            asyncio.run(manager.pause_generation(request))

        self.assertFalse(manager.is_pause)
        manager._async_dispatch_to_scheduler.assert_not_called()

    def test_valid_non_abort_modes_dispatch_to_scheduler(self):
        """in_place and retract pass validation and dispatch to the scheduler."""
        for mode in ("in_place", "retract"):
            manager = _make_manager()
            request = PauseGenerationReqInput(mode=mode)

            asyncio.run(manager.pause_generation(request))

            self.assertTrue(manager.is_pause)
            manager._async_dispatch_to_scheduler.assert_awaited_once_with(request)


if __name__ == "__main__":
    unittest.main()
