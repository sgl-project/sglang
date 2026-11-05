"""Native Engine control tests use existing drain locks and reject failed flushes."""

import asyncio
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import ContinueGenerationReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertSwapEngine(CustomTestCase):
    def test_async_update_waits_for_existing_generation_reader(self):
        owner = self.fixture()

        async def scenario():
            owner.tokenizer_manager.model_update_lock = RWLock()
            requested = [[5, 6, 7, 0]]
            async with owner.tokenizer_manager.model_update_lock.reader_lock:
                update = asyncio.create_task(
                    Engine.async_hot_swap_expert_set(owner, requested)
                )
                await asyncio.sleep(0)
                owner.collective_rpc.assert_not_called()
                self.assertFalse(update.done())
                requested[0][:] = [0, 1, 2, 3]
            await update
            owner.collective_rpc.assert_called_once()
            self.assertEqual(
                owner.collective_rpc.call_args.kwargs["requested"], [[7, 6, 5, 0]]
            )

        try:
            owner.loop.run_until_complete(scenario())
        finally:
            owner.loop.close()

    def fixture(self):
        loop = asyncio.new_event_loop()
        manager = Mock(spec=TokenizerManager)
        manager.is_pause = False
        manager.is_pause_cond = asyncio.Condition()
        manager.model_update_lock = SimpleNamespace(writer_lock=nullcontext())
        manager.flush_cache = AsyncMock(return_value=SimpleNamespace(success=True))
        owner = SimpleNamespace(
            _attached=False,
            tokenizer_manager=manager,
            loop=loop,
            server_args=SimpleNamespace(
                moe_expert_profile='{"num_experts":8,"keep":[[7,1,5,3]]}'
            ),
            collective_rpc=Mock(),
        )
        owner.async_hot_swap_expert_set = lambda *args, **kwargs: (
            Engine.async_hot_swap_expert_set(owner, *args, **kwargs)
        )
        return owner

    def test_failed_flush_cannot_dispatch_a_weight_mutation(self):
        owner = self.fixture()
        owner.tokenizer_manager.flush_cache.return_value = SimpleNamespace(
            success=False
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "cache flush"):
                Engine.hot_swap_expert_set(owner, [[5, 6, 7, 0]])
            owner.collective_rpc.assert_not_called()
            self.assertFalse(hasattr(owner, "_hot_expert_mapping"))
        finally:
            owner.loop.close()

    def test_unsafe_worker_failure_keeps_frontend_paused(self):
        owner = self.fixture()
        owner.collective_rpc.side_effect = AssertionError(
            "HOT_SWAP_UNSAFE: prepare consensus unavailable"
        )
        try:
            with self.assertRaisesRegex(AssertionError, "HOT_SWAP_UNSAFE"):
                Engine.hot_swap_expert_set(owner, [[5, 6, 7, 0]])
            self.assertTrue(owner.tokenizer_manager.is_pause)
            self.assertTrue(owner.tokenizer_manager.hot_swap_failed)
            with self.assertRaisesRegex(RuntimeError, "restart"):
                owner.loop.run_until_complete(
                    TokenizerManager.continue_generation(
                        owner.tokenizer_manager,
                        ContinueGenerationReqInput(torch_empty_cache=False),
                    )
                )
            self.assertTrue(owner.tokenizer_manager.is_pause)
            self.assertFalse(hasattr(owner, "_hot_expert_mapping"))
        finally:
            owner.loop.close()

    def test_continue_waiter_rechecks_latch_after_the_pause_condition(self):
        loop = asyncio.new_event_loop()
        manager = SimpleNamespace(
            hot_swap_failed=False,
            is_pause=True,
            tokenizer_ipc_name=None,
            is_pause_cond=asyncio.Condition(),
            _async_dispatch_to_scheduler=AsyncMock(),
        )

        async def scenario():
            async with manager.is_pause_cond:
                waiter = asyncio.create_task(
                    TokenizerManager.continue_generation(
                        manager, ContinueGenerationReqInput(torch_empty_cache=False)
                    )
                )
                await asyncio.sleep(0)
                manager.hot_swap_failed = True
            with self.assertRaisesRegex(RuntimeError, "restart"):
                await waiter
            self.assertTrue(manager.is_pause)
            manager._async_dispatch_to_scheduler.assert_not_called()

        try:
            loop.run_until_complete(scenario())
        finally:
            loop.close()

    def test_success_keeps_actual_mapping_for_the_next_expected_version(self):
        owner = self.fixture()
        try:
            result = Engine.hot_swap_expert_set(owner, [[5, 6, 7, 0]])
            self.assertEqual(result["physical_to_logical_map"], [[7, 6, 5, 0]])
            self.assertEqual(
                owner.collective_rpc.call_args.kwargs["expected"], [[7, 1, 5, 3]]
            )
            Engine.hot_swap_expert_set(owner, [[7, 1, 5, 3]])
            self.assertEqual(
                owner.collective_rpc.call_args.kwargs["expected"], [[7, 6, 5, 0]]
            )
        finally:
            owner.loop.close()


if __name__ == "__main__":
    unittest.main()
