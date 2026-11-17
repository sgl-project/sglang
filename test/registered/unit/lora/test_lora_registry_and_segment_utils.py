"""CPU unit tests for LoRARegistry lifecycle invariants and LoRA segment chunking."""

import asyncio
import unittest

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.lora.utils import (
    build_lm_head_pass_segments,
    merge_and_chunk_segments,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLoRARegistryAndSegmentUtils(CustomTestCase):
    def test_merge_and_chunk_segments_rejects_non_positive_chunk_size(self):
        for invalid_chunk_size in (0, -1, -8):
            with self.assertRaises(ValueError):
                merge_and_chunk_segments(
                    weight_indices=[0, 1],
                    pruned_lens=[4, 2],
                    chunk_size=invalid_chunk_size,
                )

    def test_build_lm_head_pass_segments_rejects_non_positive_chunk_size(self):
        for invalid_chunk_size in (0, -1, -4):
            with self.assertRaises(ValueError):
                build_lm_head_pass_segments(
                    weight_indices=[0, 1],
                    pruned_lens=[4, 2],
                    logprobs_chunk_size=invalid_chunk_size,
                )

    def test_wait_for_unload_asserts_adapter_id_is_unregistered(self):
        async def _run():
            ref = LoRARef(
                lora_id="adapter-uuid-123",
                lora_name="my-adapter",
                lora_path="/tmp/adapter",
            )
            registry = LoRARegistry([ref])
            with self.assertRaises(AssertionError):
                await registry.wait_for_unload("adapter-uuid-123")

            unregistered_id = await registry.unregister("my-adapter")
            self.assertEqual(unregistered_id, "adapter-uuid-123")
            await registry.wait_for_unload(unregistered_id)

        asyncio.run(_run())

    def test_acquire_increments_counter_under_writer_lock_before_unregister(self):
        async def _run():
            ref = LoRARef(
                lora_id="adapter-uuid-456",
                lora_name="race-adapter",
                lora_path="/tmp/adapter",
            )
            registry = LoRARegistry([ref])

            acquire_task = asyncio.create_task(registry.acquire("race-adapter"))
            await asyncio.sleep(0)

            unregistered_id = await registry.unregister("race-adapter")
            unload_task = asyncio.create_task(registry.wait_for_unload(unregistered_id))

            acquired_id = await acquire_task
            self.assertEqual(acquired_id, "adapter-uuid-456")
            self.assertFalse(unload_task.done())

            await registry.release(acquired_id)
            await unload_task
            self.assertTrue(unload_task.done())

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
