import asyncio
import gc
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.encoder import http_server
from sglang.srt.disaggregation.encoder import server as encoder_server
from sglang.srt.disaggregation.encoder.server import EncoderMetaRegistry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEncoderMetadataWait(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=60)
        # Preserve the registry's actual ownership policy in each isolated loop.
        self.conditions = type(encoder_server.rid_to_cond)()
        patches = (
            patch.object(encoder_server, "rid_to_cond", self.conditions),
            patch.object(encoder_server, "cond_dict_lock", asyncio.Lock()),
            patch.object(encoder_server, "rid_lock", asyncio.Lock()),
            patch.object(encoder_server, "meta_registry", self.registry),
            patch.object(http_server, "dp_dispatcher", None),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        if self.registry._sweeper_task is not None:
            self.registry._sweeper_task.cancel()
            await asyncio.gather(self.registry._sweeper_task, return_exceptions=True)

    async def test_http_timeouts_do_not_retain_unpublished_conditions(self):
        self.registry.wait_timeout = 0.001
        responses = await asyncio.gather(
            *(
                http_server.handle_scheduler_receive_meta_data(
                    {"req_id": f"unpublished-{i}", "part_idx": 0}
                )
                for i in range(16)
            )
        )
        self.assertTrue(all(response.status_code == 504 for response in responses))
        self.assertEqual(self.registry._pending_at, {})
        self.assertEqual(self.registry._rid_to_meta, {})
        gc.collect()
        self.assertEqual(len(self.conditions), 0)

    async def test_cancelled_waiters_release_unpublished_conditions(self):
        self.registry.wait_timeout = 60
        tasks = [
            asyncio.create_task(self.registry.wait(f"cancelled-{i}")) for i in range(8)
        ]
        try:
            # Each waiter reaches wait_for after creating its condition.
            await asyncio.sleep(0)
            self.assertEqual(len(self.conditions), len(tasks))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # Finished tasks and cancellation tracebacks may still own their frames.
        del tasks, task
        await asyncio.sleep(0)
        gc.collect()
        self.assertEqual(len(self.conditions), 0)

    async def test_cancelling_one_waiter_preserves_another_waiter(self):
        req_id = "shared"
        first = asyncio.create_task(self.registry.wait(req_id))
        second = asyncio.create_task(self.registry.wait(req_id))
        try:
            # Let both callers acquire the rendezvous without keeping another
            # strong condition reference in this test.
            await asyncio.sleep(0)
            self.assertIn(req_id, self.conditions)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            first = None
            gc.collect()
            self.assertIn(req_id, self.conditions)
            await self.registry.publish(req_id, 32, 4, 2)
            self.assertEqual(
                await asyncio.wait_for(second, 1),
                {"embedding_size": 32, "embedding_len": 4, "embedding_dim": 2},
            )
        finally:
            tasks = [task for task in (first, second) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_publish_before_wait_preserves_metadata(self):
        await self.registry.publish("published", 32, 4, 2)
        gc.collect()
        self.assertEqual(len(self.conditions), 0)
        self.assertEqual(
            await self.registry.wait("published"),
            {"embedding_size": 32, "embedding_len": 4, "embedding_dim": 2},
        )
        self.assertIn("published", self.registry._pending_at)
        self.assertIn("published", self.registry._rid_to_meta)

    async def test_condition_stays_shared_while_a_caller_owns_it(self):
        first = await encoder_server._get_receive_condition("held")
        gc.collect()
        second = await encoder_server._get_receive_condition("held")
        self.assertIs(first, second)
        del first, second
        gc.collect()
        self.assertNotIn("held", self.conditions)


if __name__ == "__main__":
    unittest.main()
