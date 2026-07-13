"""Unit tests for the LoRA upsert (in-place refresh) path.

With upsert=True, loading an adapter that is already registered refreshes
its weights in place, reusing the existing lora_id and memory-pool slot
instead of failing with a duplicate error. Covers:

  * LoRARegistry.get_lora_id / register_or_reuse / refresh
  * LoRAManager.load_lora_adapter_from_tensors upsert semantics
  * failed-upsert rollback (no half-updated live adapter)
  * num_pinned_loras consistency across pinned flips
  * LoRAManager.validate_new_adapter duplicate-name / starvation checks
  * TokenizerControlMixin from_distributed AND from_tensors id reuse
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromDistributedReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CONFIG_DICT = {"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}


class TestLoRARegistryGetLoraId(CustomTestCase):
    def test_returns_id_for_registered_adapter(self):
        registry = LoRARegistry()
        ref = LoRARef(lora_name="a", lora_path="/x")
        asyncio.run(registry.register(ref))

        self.assertEqual(asyncio.run(registry.get_lora_id("a")), ref.lora_id)

    def test_returns_none_for_unregistered_adapter(self):
        registry = LoRARegistry()
        self.assertIsNone(asyncio.run(registry.get_lora_id("missing")))


def _make_manager() -> LoRAManager:
    """Create a LoRAManager via __new__ with only the fields the load path reads."""
    manager = LoRAManager.__new__(LoRAManager)
    manager.configs = {}
    manager.loras = {}
    manager.lora_refs = {}
    manager.num_pinned_loras = 0
    manager.max_loras_per_batch = 4
    manager.base_hf_config = MagicMock(vocab_size=32000)
    manager.lora_modules = []
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    manager.device = torch.device("cpu")
    manager.memory_pool = MagicMock()
    manager.memory_pool.can_support.return_value = True
    manager.memory_pool.uid_to_buffer_id = {}
    # Weight loading needs a real base model / backend; just build a stub.
    manager._create_lora_adapter_from_tensors = Mock(
        side_effect=lambda ref, config, tensors: MagicMock()
    )
    return manager


class TestLoRAManagerUpsert(CustomTestCase):
    def test_fresh_load_registers_adapter(self):
        manager = _make_manager()
        ref = LoRARef(lora_name="a", lora_path="__tensor__", pinned=True)

        result = manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)

        self.assertTrue(result.success)
        self.assertIn(ref.lora_id, manager.loras)
        self.assertIs(manager.lora_refs[ref.lora_id], ref)
        self.assertEqual(manager.num_pinned_loras, 1)

    def test_duplicate_load_without_upsert_asserts(self):
        manager = _make_manager()
        ref = LoRARef(lora_name="a", lora_path="__tensor__")
        manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)

        with self.assertRaises(AssertionError):
            manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)

    def test_upsert_refreshes_loaded_adapter_in_place(self):
        manager = _make_manager()
        ref = LoRARef(lora_name="a", lora_path="__tensor__", pinned=True)
        manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)
        # Simulate the adapter occupying a memory-pool buffer slot.
        manager.memory_pool.uid_to_buffer_id = {ref.lora_id: 3}

        new_config = dict(CONFIG_DICT, lora_alpha=32)
        result = manager.load_lora_adapter_from_tensors(
            ref, {}, new_config, upsert=True
        )

        self.assertTrue(result.success)
        self.assertEqual(manager.configs[ref.lora_id].lora_alpha, 32)
        # Weights are re-copied into the existing buffer slot.
        manager.memory_pool.load_lora_weight_to_buffer.assert_called_once()
        call = manager.memory_pool.load_lora_weight_to_buffer.call_args
        self.assertEqual(call.args[0], ref.lora_id)
        self.assertEqual(call.args[1], 3)
        # The pinned slot is not double-counted.
        self.assertEqual(manager.num_pinned_loras, 1)

    def test_upsert_skips_pool_copy_when_not_resident(self):
        manager = _make_manager()
        ref = LoRARef(lora_name="a", lora_path="__tensor__")
        manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)

        result = manager.load_lora_adapter_from_tensors(
            ref, {}, CONFIG_DICT, upsert=True
        )

        self.assertTrue(result.success)
        manager.memory_pool.load_lora_weight_to_buffer.assert_not_called()

    def test_upsert_falls_back_to_register_when_not_loaded(self):
        manager = _make_manager()
        ref = LoRARef(lora_name="a", lora_path="__tensor__", pinned=True)

        result = manager.load_lora_adapter_from_tensors(
            ref, {}, CONFIG_DICT, upsert=True
        )

        self.assertTrue(result.success)
        self.assertIn(ref.lora_id, manager.loras)
        self.assertEqual(manager.num_pinned_loras, 1)


class TestLoRARegistryRegisterOrReuse(CustomTestCase):
    def test_upsert_reuses_id_of_registered_adapter(self):
        registry = LoRARegistry()
        existing = LoRARef(lora_name="a", lora_path="/x", pinned=False)
        asyncio.run(registry.register(existing))

        candidate = LoRARef(lora_name="a", lora_path="__tensor__", pinned=True)
        resolved, reused = asyncio.run(registry.register_or_reuse(candidate, True))

        self.assertTrue(reused)
        self.assertEqual(resolved.lora_id, existing.lora_id)
        self.assertEqual(resolved.lora_path, "__tensor__")
        self.assertTrue(resolved.pinned)

    def test_upsert_without_registered_adapter_keeps_fresh_id(self):
        registry = LoRARegistry()
        candidate = LoRARef(lora_name="a", lora_path="__tensor__")

        resolved, reused = asyncio.run(registry.register_or_reuse(candidate, True))

        self.assertFalse(reused)
        self.assertIs(resolved, candidate)

    def test_non_upsert_never_reuses(self):
        registry = LoRARegistry()
        asyncio.run(registry.register(LoRARef(lora_name="a", lora_path="/x")))

        candidate = LoRARef(lora_name="a", lora_path="/x")
        resolved, reused = asyncio.run(registry.register_or_reuse(candidate, False))

        self.assertFalse(reused)
        self.assertIs(resolved, candidate)

    def test_refresh_replaces_ref_in_place(self):
        registry = LoRARegistry()
        existing = LoRARef(lora_name="a", lora_path="/x", pinned=False)
        asyncio.run(registry.register(existing))

        refreshed = LoRARef(
            lora_id=existing.lora_id,
            lora_name="a",
            lora_path="__tensor__",
            pinned=True,
        )
        asyncio.run(registry.refresh(refreshed))

        self.assertEqual(registry.get_all_adapters()["a"].pinned, True)
        self.assertEqual(asyncio.run(registry.get_lora_id("a")), existing.lora_id)

    def test_refresh_rejects_id_mismatch(self):
        registry = LoRARegistry()
        asyncio.run(registry.register(LoRARef(lora_name="a", lora_path="/x")))

        with self.assertRaises(AssertionError):
            asyncio.run(
                registry.refresh(LoRARef(lora_name="a", lora_path="__tensor__"))
            )


class TestUpsertRollback(CustomTestCase):
    """A failed load/upsert must not leave a live adapter half-updated."""

    def test_failed_fresh_load_leaves_no_state(self):
        manager = _make_manager()
        manager._create_lora_adapter_from_tensors = Mock(
            side_effect=ValueError("bad tensors")
        )
        ref = LoRARef(lora_name="a", lora_path="__tensor__")

        result = manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)

        self.assertFalse(result.success)
        self.assertIn("bad tensors", result.error_message)
        self.assertEqual(manager.configs, {})
        self.assertEqual(manager.loras, {})
        self.assertEqual(manager.lora_refs, {})

    def test_failed_upsert_staging_keeps_old_adapter_serving(self):
        manager = _make_manager()
        ref = LoRARef(lora_name="a", lora_path="__tensor__")
        manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)
        old_config = manager.configs[ref.lora_id]
        old_lora = manager.loras[ref.lora_id]

        manager._create_lora_adapter_from_tensors = Mock(
            side_effect=ValueError("rank mismatch")
        )
        result = manager.load_lora_adapter_from_tensors(
            ref, {}, CONFIG_DICT, upsert=True
        )

        self.assertFalse(result.success)
        self.assertIs(manager.configs[ref.lora_id], old_config)
        self.assertIs(manager.loras[ref.lora_id], old_lora)
        manager.memory_pool.load_lora_weight_to_buffer.assert_not_called()

    def test_failed_buffer_rewrite_restores_old_weights(self):
        manager = _make_manager()
        ref = LoRARef(lora_name="a", lora_path="__tensor__")
        manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)
        old_config = manager.configs[ref.lora_id]
        old_lora = manager.loras[ref.lora_id]
        manager.memory_pool.uid_to_buffer_id = {ref.lora_id: 3}
        manager.memory_pool.load_lora_weight_to_buffer.side_effect = [
            RuntimeError("copy failed at layer k"),
            None,  # the restore pass
        ]

        result = manager.load_lora_adapter_from_tensors(
            ref, {}, CONFIG_DICT, upsert=True
        )

        self.assertFalse(result.success)
        self.assertIn("copy failed", result.error_message)
        # CPU-side state rolled back...
        self.assertIs(manager.configs[ref.lora_id], old_config)
        self.assertIs(manager.loras[ref.lora_id], old_lora)
        # ...and the served buffer was rewritten back from the old adapter.
        calls = manager.memory_pool.load_lora_weight_to_buffer.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[1].args[2], old_lora)


class TestUpsertPinnedAccounting(CustomTestCase):
    def test_pinned_flip_updates_counter_both_ways(self):
        manager = _make_manager()
        unpinned = LoRARef(lora_name="a", lora_path="__tensor__", pinned=False)
        manager.load_lora_adapter_from_tensors(unpinned, {}, CONFIG_DICT)
        self.assertEqual(manager.num_pinned_loras, 0)

        pinned = LoRARef(
            lora_id=unpinned.lora_id,
            lora_name="a",
            lora_path="__tensor__",
            pinned=True,
        )
        manager.load_lora_adapter_from_tensors(pinned, {}, CONFIG_DICT, upsert=True)
        self.assertEqual(manager.num_pinned_loras, 1)

        manager.load_lora_adapter_from_tensors(unpinned, {}, CONFIG_DICT, upsert=True)
        self.assertEqual(manager.num_pinned_loras, 0)

    def test_unload_after_pinned_flip_keeps_counter_consistent(self):
        manager = _make_manager()
        unpinned = LoRARef(lora_name="a", lora_path="__tensor__", pinned=False)
        manager.load_lora_adapter_from_tensors(unpinned, {}, CONFIG_DICT)
        pinned = LoRARef(
            lora_id=unpinned.lora_id,
            lora_name="a",
            lora_path="__tensor__",
            pinned=True,
        )
        manager.load_lora_adapter_from_tensors(pinned, {}, CONFIG_DICT, upsert=True)

        result = manager.unload_lora_adapter(pinned)

        self.assertTrue(result.success)
        self.assertEqual(manager.num_pinned_loras, 0)

    def test_pinned_refresh_allowed_at_pin_limit(self):
        """Refreshing a pinned adapter adds no pinned slot; rejecting it would
        freeze RL serving on the step-1 weights."""
        manager = _make_manager()
        manager.max_loras_per_batch = 2
        ref = LoRARef(lora_name="a", lora_path="__tensor__", pinned=True)
        manager.load_lora_adapter_from_tensors(ref, {}, CONFIG_DICT)
        self.assertEqual(manager.num_pinned_loras, 1)

        result = manager.load_lora_adapter_from_tensors(
            ref, {}, CONFIG_DICT, upsert=True
        )

        self.assertTrue(result.success)
        self.assertEqual(manager.num_pinned_loras, 1)

    def test_fresh_pinned_load_still_rejected_at_pin_limit(self):
        manager = _make_manager()
        manager.max_loras_per_batch = 2
        manager.load_lora_adapter_from_tensors(
            LoRARef(lora_name="a", lora_path="__tensor__", pinned=True),
            {},
            CONFIG_DICT,
        )

        result = manager.load_lora_adapter_from_tensors(
            LoRARef(lora_name="b", lora_path="__tensor__", pinned=True),
            {},
            CONFIG_DICT,
        )

        self.assertFalse(result.success)
        self.assertIn("not allowed to pin all slots", result.error_message)


class TestValidateNewAdapterDuplicates(CustomTestCase):
    def test_duplicate_name_rejected_without_update(self):
        manager = _make_manager()
        existing = LoRARef(lora_name="a", lora_path="/x")
        manager.lora_refs[existing.lora_id] = existing
        config = MagicMock(lora_added_tokens_size=0, use_dora=False)

        with self.assertRaisesRegex(ValueError, "already loaded"):
            manager.validate_new_adapter(config, LoRARef(lora_name="a", lora_path="/y"))

    def test_duplicate_name_allowed_for_update(self):
        manager = _make_manager()
        existing = LoRARef(lora_name="a", lora_path="/x")
        manager.lora_refs[existing.lora_id] = existing
        config = MagicMock(lora_added_tokens_size=0, use_dora=False)

        manager.validate_new_adapter(config, existing, is_update=True)


def _make_tokenizer_manager(tokenizer_worker_num: int = 1) -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.enable_lora = True
    tm.server_args.dp_size = 1
    tm.server_args.max_loaded_loras = None
    tm.server_args.tokenizer_worker_num = tokenizer_worker_num
    tm.auto_create_handle_loop = Mock()
    tm.lora_update_lock = asyncio.Lock()
    tm.lora_registry = LoRARegistry()
    tm.lora_ref_cache = {}
    tm.update_lora_adapter_communicator = AsyncMock(
        return_value=[MagicMock(success=True)]
    )
    return tm


def _make_distributed_req(upsert: bool) -> LoadLoRAAdapterFromDistributedReqInput:
    return LoadLoRAAdapterFromDistributedReqInput(
        lora_name="a",
        config_dict=CONFIG_DICT,
        names=[],
        dtypes=[],
        shapes=[],
        upsert=upsert,
    )


def _make_tensors_req(upsert: bool, pinned: bool = False) -> LoadLoRAAdapterFromTensorsReqInput:
    return LoadLoRAAdapterFromTensorsReqInput(
        lora_name="a",
        config_dict=CONFIG_DICT,
        serialized_named_tensors=[],
        pinned=pinned,
        upsert=upsert,
    )


class TestLoadFromDistributedUpsert(CustomTestCase):
    def test_upsert_reuses_existing_lora_id(self):
        tm = _make_tokenizer_manager()
        existing = LoRARef(lora_name="a", lora_path="__distributed__")
        asyncio.run(tm.lora_registry.register(existing))

        obj = _make_distributed_req(upsert=True)
        result = asyncio.run(tm.load_lora_adapter_from_distributed(obj))

        self.assertTrue(result.success)
        self.assertEqual(obj.lora_id, existing.lora_id)
        tm.update_lora_adapter_communicator.assert_awaited_once_with(obj)
        # Not re-registered: still exactly one adapter with the original id.
        self.assertEqual(tm.lora_registry.num_registered_loras, 1)
        self.assertEqual(
            asyncio.run(tm.lora_registry.get_lora_id("a")), existing.lora_id
        )
        self.assertEqual(tm.lora_ref_cache["a"].lora_id, existing.lora_id)

    def test_upsert_registers_when_missing(self):
        tm = _make_tokenizer_manager()

        obj = _make_distributed_req(upsert=True)
        result = asyncio.run(tm.load_lora_adapter_from_distributed(obj))

        self.assertTrue(result.success)
        self.assertIsNotNone(obj.lora_id)
        self.assertEqual(asyncio.run(tm.lora_registry.get_lora_id("a")), obj.lora_id)

    def test_non_upsert_duplicate_fails(self):
        tm = _make_tokenizer_manager()
        asyncio.run(
            tm.lora_registry.register(
                LoRARef(lora_name="a", lora_path="__distributed__")
            )
        )

        obj = _make_distributed_req(upsert=False)
        result = asyncio.run(tm.load_lora_adapter_from_distributed(obj))

        self.assertFalse(result.success)
        self.assertIn("already exists", result.error_message)

    def test_upsert_refreshes_registered_ref(self):
        # The registry ref (not just lora_ref_cache) must adopt the new
        # metadata: LRU eviction reads ``pinned`` from the registry.
        tm = _make_tokenizer_manager()
        existing = LoRARef(lora_name="a", lora_path="__distributed__", pinned=False)
        asyncio.run(tm.lora_registry.register(existing))

        obj = _make_distributed_req(upsert=True)
        obj.pinned = True
        result = asyncio.run(tm.load_lora_adapter_from_distributed(obj))

        self.assertTrue(result.success)
        registered = tm.lora_registry.get_all_adapters()["a"]
        self.assertEqual(registered.lora_id, existing.lora_id)
        self.assertTrue(registered.pinned)

    def test_failed_backend_load_keeps_registry_untouched(self):
        tm = _make_tokenizer_manager()
        existing = LoRARef(lora_name="a", lora_path="__distributed__", pinned=False)
        asyncio.run(tm.lora_registry.register(existing))
        tm.update_lora_adapter_communicator = AsyncMock(
            return_value=[MagicMock(success=False, error_message="boom")]
        )

        obj = _make_distributed_req(upsert=True)
        obj.pinned = True
        result = asyncio.run(tm.load_lora_adapter_from_distributed(obj))

        self.assertFalse(result.success)
        self.assertIs(tm.lora_registry.get_all_adapters()["a"], existing)
        self.assertNotIn("a", tm.lora_ref_cache)


class TestLoadFromTensorsUpsert(CustomTestCase):
    """The from_tensors route must resolve upsert identically to
    from_distributed — a fresh uuid per request would never match
    ``lora_ref.lora_id in self.loras`` on the backend."""

    def test_upsert_reuses_existing_lora_id(self):
        tm = _make_tokenizer_manager()
        existing = LoRARef(lora_name="a", lora_path="__tensor__")
        asyncio.run(tm.lora_registry.register(existing))

        obj = _make_tensors_req(upsert=True)
        result = asyncio.run(tm.load_lora_adapter_from_tensors(obj))

        self.assertTrue(result.success)
        self.assertEqual(obj.lora_id, existing.lora_id)
        tm.update_lora_adapter_communicator.assert_awaited_once_with(obj)
        self.assertEqual(tm.lora_registry.num_registered_loras, 1)
        self.assertEqual(tm.lora_ref_cache["a"].lora_id, existing.lora_id)

    def test_upsert_registers_when_missing(self):
        tm = _make_tokenizer_manager()

        obj = _make_tensors_req(upsert=True)
        result = asyncio.run(tm.load_lora_adapter_from_tensors(obj))

        self.assertTrue(result.success)
        self.assertIsNotNone(obj.lora_id)
        self.assertEqual(asyncio.run(tm.lora_registry.get_lora_id("a")), obj.lora_id)

    def test_non_upsert_duplicate_fails(self):
        tm = _make_tokenizer_manager()
        asyncio.run(
            tm.lora_registry.register(LoRARef(lora_name="a", lora_path="__tensor__"))
        )

        obj = _make_tensors_req(upsert=False)
        result = asyncio.run(tm.load_lora_adapter_from_tensors(obj))

        self.assertFalse(result.success)
        self.assertIn("already exists", result.error_message)


class TestUpsertMultiTokenizerWorkerGuard(CustomTestCase):
    """Upsert resolves names against a per-process registry; with >1 tokenizer
    workers that resolution is nondeterministic, so it must fail loudly."""

    def test_tensors_upsert_rejected_with_multiple_workers(self):
        tm = _make_tokenizer_manager(tokenizer_worker_num=2)

        result = asyncio.run(tm.load_lora_adapter_from_tensors(_make_tensors_req(True)))

        self.assertFalse(result.success)
        self.assertIn("tokenizer_worker_num", result.error_message)
        tm.update_lora_adapter_communicator.assert_not_awaited()

    def test_distributed_upsert_rejected_with_multiple_workers(self):
        tm = _make_tokenizer_manager(tokenizer_worker_num=2)

        result = asyncio.run(
            tm.load_lora_adapter_from_distributed(_make_distributed_req(True))
        )

        self.assertFalse(result.success)
        self.assertIn("tokenizer_worker_num", result.error_message)

    def test_non_upsert_load_unaffected_by_multiple_workers(self):
        tm = _make_tokenizer_manager(tokenizer_worker_num=2)

        result = asyncio.run(
            tm.load_lora_adapter_from_tensors(_make_tensors_req(False))
        )

        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main(verbosity=2)
