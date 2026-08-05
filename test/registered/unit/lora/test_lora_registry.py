"""Unit tests for srt/lora/lora_registry.py - no server, no model loading."""

import asyncio
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry  

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestLoRARef(CustomTestCase):
    def test_ref_generates_unique_id_and_string_omits_none_fields(self):
        """LoRARef creates stable IDs and stringifies only populated fields."""
        ref = LoRARef(lora_name="adapter-a", lora_path="/tmp/adapter-a")

        self.assertIsNotNone(ref.lora_id)
        self.assertIn("lora_id=", str(ref))
        self.assertIn("lora_name=adapter-a", str(ref))
        self.assertIn("lora_path=/tmp/adapter-a", str(ref))
        self.assertNotIn("pinned=", str(ref))

    def test_ref_rejects_none_lora_id(self):
        """LoRARef rejects None IDs so registry counter keys are always valid."""
        with self.assertRaisesRegex(ValueError, "lora_id cannot be None"):
            LoRARef(lora_id=None, lora_name="adapter-a")


class TestLoRARegistry(CustomTestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_initializes_with_refs_and_reports_all_adapters(self):
        """The registry stores initial LoRA refs by name in insertion order."""
        ref_a = LoRARef(lora_id="id-a", lora_name="adapter-a")
        ref_b = LoRARef(lora_id="id-b", lora_name="adapter-b", pinned=True)

        registry = LoRARegistry([ref_a, ref_b])

        self.assertEqual(registry.num_registered_loras, 2)
        self.assertEqual(
            registry.get_all_adapters(),
            {"adapter-a": ref_a, "adapter-b": ref_b},
        )

    def test_register_rejects_duplicate_lora_name(self):
        """Registering a duplicate adapter name raises instead of overwriting it."""
        registry = LoRARegistry([LoRARef(lora_id="id-a", lora_name="adapter-a")])

        with self.assertRaisesRegex(ValueError, "already exists"):
            self._run(registry.register(LoRARef(lora_id="id-b", lora_name="adapter-a")))

    def test_unregister_removes_adapter_and_returns_lora_id(self):
        """Unregister removes the name entry and returns the removed adapter ID."""
        registry = LoRARegistry([LoRARef(lora_id="id-a", lora_name="adapter-a")])

        removed_id = self._run(registry.unregister("adapter-a"))

        self.assertEqual(removed_id, "id-a")
        self.assertEqual(registry.num_registered_loras, 0)
        self.assertEqual(registry.get_all_adapters(), {})

    def test_unregister_unknown_name_raises(self):
        """Unregistering an unloaded adapter reports the missing name."""
        registry = LoRARegistry()

        with self.assertRaisesRegex(ValueError, "does not exist"):
            self._run(registry.unregister("missing-adapter"))

    def test_acquire_supports_single_name_and_updates_lru_order(self):
        """Acquire returns the adapter ID and marks the adapter as recently used."""
        registry = LoRARegistry(
            [
                LoRARef(lora_id="id-a", lora_name="adapter-a"),
                LoRARef(lora_id="id-b", lora_name="adapter-b"),
            ]
        )

        acquired_id = self._run(registry.acquire("adapter-a"))

        self.assertEqual(acquired_id, "id-a")
        self.assertEqual(self._run(registry.lru_lora_name()), "adapter-b")
        self._run(registry.release(acquired_id))

    def test_acquire_list_preserves_none_and_rejects_missing_names(self):
        """Batch acquire preserves base-model slots and validates every LoRA name."""
        registry = LoRARegistry(
            [
                LoRARef(lora_id="id-a", lora_name="adapter-a"),
                LoRARef(lora_id="id-b", lora_name="adapter-b"),
            ]
        )

        acquired_ids = self._run(registry.acquire(["adapter-a", None, "adapter-b"]))

        self.assertEqual(acquired_ids, ["id-a", None, "id-b"])
        self._run(registry.release(acquired_ids))
        with self.assertRaisesRegex(ValueError, "not loaded"):
            self._run(registry.acquire(["adapter-a", "missing-adapter"]))

    def test_get_unregistered_loras_returns_missing_names_and_refreshes_lru(self):
        """Missing-name lookup reports unloaded adapters and refreshes loaded ones."""
        registry = LoRARegistry(
            [
                LoRARef(lora_id="id-a", lora_name="adapter-a"),
                LoRARef(lora_id="id-b", lora_name="adapter-b"),
            ]
        )

        missing = self._run(
            registry.get_unregistered_loras({"adapter-a", "missing-adapter"})
        )

        self.assertEqual(missing, ["missing-adapter"])
        self.assertEqual(self._run(registry.lru_lora_name()), "adapter-b")

    def test_lru_lora_name_can_exclude_pinned_adapters(self):
        """LRU lookup can skip pinned adapters when selecting unload candidates."""
        registry = LoRARegistry(
            [
                LoRARef(lora_id="id-a", lora_name="adapter-a", pinned=True),
                LoRARef(lora_id="id-b", lora_name="adapter-b", pinned=False),
            ]
        )

        self.assertEqual(self._run(registry.lru_lora_name()), "adapter-a")
        self.assertEqual(
            self._run(registry.lru_lora_name(exclude_pinned=True)), "adapter-b"
        )

    def test_wait_for_unload_blocks_until_acquired_adapter_is_released(self):
        """wait_for_unload keeps counters alive until all in-flight users release."""

        async def scenario():
            registry = LoRARegistry([LoRARef(lora_id="id-a", lora_name="adapter-a")])
            lora_id = await registry.acquire("adapter-a")
            removed_id = await registry.unregister("adapter-a")
            wait_task = asyncio.create_task(registry.wait_for_unload(removed_id))

            await asyncio.sleep(0)
            self.assertFalse(wait_task.done())
            await registry.release(lora_id)
            await asyncio.wait_for(wait_task, timeout=1)
            self.assertNotIn(removed_id, registry._counters)

        self._run(scenario())


if __name__ == "__main__":
    unittest.main()
