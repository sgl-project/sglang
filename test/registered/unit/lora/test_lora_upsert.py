"""Unit tests for the LoRA upsert (in-place refresh) path.

With upsert=True, loading an adapter that is already registered refreshes
its weights in place, reusing the existing lora_id and memory-pool slot
instead of failing with a duplicate error. Covers:

  * LoRARegistry.get_lora_id lookups
  * LoRAManager.load_lora_adapter_from_tensors upsert semantics
  * LoRAManager.validate_new_adapter duplicate-name check skip
  * TokenizerControlMixin.load_lora_adapter_from_distributed id reuse
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import LoadLoRAAdapterFromDistributedReqInput
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
    manager.memory_pool = MagicMock()
    manager.memory_pool.can_support.return_value = True
    manager.memory_pool.uid_to_buffer_id = {}
    # Weight loading needs a real base model / backend; just record the adapter.
    manager.load_lora_weights_from_tensors = Mock(
        side_effect=lambda ref, tensors: manager.loras.__setitem__(
            ref.lora_id, MagicMock()
        )
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


def _make_tokenizer_manager() -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.enable_lora = True
    tm.server_args.dp_size = 1
    tm.server_args.max_loaded_loras = None
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
