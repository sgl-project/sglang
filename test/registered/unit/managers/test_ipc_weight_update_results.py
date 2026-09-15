"""IPC update failures must not depend on scheduler reply order."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestIPCWeightUpdateResults(unittest.TestCase):
    def test_all_replies_control_success_and_side_effects(self):
        for paused in (False, True):
            for statuses in (
                (True,),
                (False,),
                (True, True),
                (True, False),
                (False, True),
            ):
                with self.subTest(paused=paused, statuses=statuses):
                    asyncio.run(self._check_update(paused, statuses))

    async def _check_update(self, paused, statuses):
        replies = [
            SimpleNamespace(success=success, message=f"worker-{i}: {success}")
            for i, success in enumerate(statuses)
        ]
        manager = SimpleNamespace(
            auto_create_handle_loop=Mock(),
            is_pause_cond=asyncio.Condition(),
            is_pause=paused,
            model_update_lock=SimpleNamespace(writer_lock=asyncio.Lock()),
            update_weights_from_ipc_communicator=AsyncMock(return_value=replies),
            mm_processor=SimpleNamespace(clear_preprocess_cache=Mock()),
            _update_weight_version_if_provided=Mock(),
        )
        request = SimpleNamespace(flush_cache=True, weight_version="v2")
        with patch(
            "sglang.srt.managers.tokenizer_control_mixin.get_parallel",
            return_value=SimpleNamespace(
                dp_size=len(statuses), enable_dp_attention=True
            ),
        ):
            success, message = await TokenizerControlMixin.update_weights_from_ipc(
                manager, request
            )

        self.assertEqual(success, all(statuses))
        for reply in replies:
            self.assertIn(reply.message, message)
        manager.update_weights_from_ipc_communicator.assert_awaited_once_with(request)
        if success:
            manager.mm_processor.clear_preprocess_cache.assert_called_once_with()
            manager._update_weight_version_if_provided.assert_called_once_with("v2")
        else:
            manager.mm_processor.clear_preprocess_cache.assert_not_called()
            manager._update_weight_version_if_provided.assert_not_called()


if __name__ == "__main__":
    unittest.main()
