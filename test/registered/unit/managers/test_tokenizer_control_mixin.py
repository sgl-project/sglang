"""
Unit tests for dynamic LoRA guards in srt/managers/tokenizer_control_mixin.py.

With ``--tokenizer-worker-num > 1``, each tokenizer worker process holds its own
``lora_registry`` / ``lora_ref_cache`` and dynamic LoRA updates are applied only
by the worker serving the HTTP request, with no cross-worker synchronization
(https://github.com/sgl-project/sglang/issues/31084). The dynamic LoRA endpoints
must therefore reject such requests with a clear error instead of silently
creating per-worker-divergent state.

Covers:
  - load_lora_adapter / load_lora_adapter_from_tensors / unload_lora_adapter
    fail fast with tokenizer_worker_num > 1: structured failure output, no
    backend communicator call, no registry / ref-cache mutation.
  - The tokenizer_worker_num == 1 path is unchanged: backend communicator is
    called and registry / ref-cache are updated on success.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARegistry
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager(TokenizerControlMixin):
    """Minimal stand-in exposing only the state the LoRA handlers touch."""

    def __init__(self, tokenizer_worker_num: int):
        self.server_args = SimpleNamespace(
            enable_lora=True,
            dp_size=1,
            tokenizer_worker_num=tokenizer_worker_num,
            max_loaded_loras=None,
        )
        self.auto_create_handle_loop = MagicMock()
        self.lora_update_lock = asyncio.Lock()
        self.lora_registry = LoRARegistry()
        self.lora_ref_cache = {}
        self.update_lora_adapter_communicator = AsyncMock(
            return_value=[LoRAUpdateOutput(success=True, loaded_adapters={})]
        )


def _load_req() -> LoadLoRAAdapterReqInput:
    return LoadLoRAAdapterReqInput(lora_name="adapter_a", lora_path="/tmp/adapter_a")


def _load_from_tensors_req() -> LoadLoRAAdapterFromTensorsReqInput:
    return LoadLoRAAdapterFromTensorsReqInput(
        lora_name="adapter_a",
        config_dict={"r": 8},
        serialized_tensors="",
    )


def _unload_req() -> UnloadLoRAAdapterReqInput:
    return UnloadLoRAAdapterReqInput(lora_name="adapter_a")


class TestDynamicLoRAMultiTokenizerWorkerGuard(CustomTestCase):
    """Dynamic LoRA ops must be rejected when tokenizer_worker_num > 1."""

    def _assert_rejected(self, manager: _FakeTokenizerManager, result):
        self.assertFalse(result.success)
        self.assertIn("--tokenizer-worker-num", result.error_message)
        self.assertIn("31084", result.error_message)
        # The backend must never be reached, and this worker's local LoRA
        # state must not diverge from its siblings'.
        manager.update_lora_adapter_communicator.assert_not_awaited()
        self.assertEqual(manager.lora_registry.num_registered_loras, 0)
        self.assertEqual(manager.lora_ref_cache, {})

    def test_load_lora_adapter_rejected(self):
        manager = _FakeTokenizerManager(tokenizer_worker_num=2)
        result = asyncio.run(manager.load_lora_adapter(_load_req()))
        self._assert_rejected(manager, result)

    def test_load_lora_adapter_from_tensors_rejected(self):
        manager = _FakeTokenizerManager(tokenizer_worker_num=2)
        result = asyncio.run(
            manager.load_lora_adapter_from_tensors(_load_from_tensors_req())
        )
        self._assert_rejected(manager, result)

    def test_unload_lora_adapter_rejected(self):
        manager = _FakeTokenizerManager(tokenizer_worker_num=2)
        result = asyncio.run(manager.unload_lora_adapter(_unload_req()))
        self._assert_rejected(manager, result)

    def test_all_workers_stay_consistent(self):
        """Simulate N per-worker registries: a rejected dynamic load must
        leave every worker's registry identical (all empty)."""
        workers = [_FakeTokenizerManager(tokenizer_worker_num=3) for _ in range(3)]
        serving_worker = workers[0]
        result = asyncio.run(serving_worker.load_lora_adapter(_load_req()))
        self.assertFalse(result.success)
        registered = {w.lora_registry.num_registered_loras for w in workers}
        self.assertEqual(registered, {0})


class TestDynamicLoRASingleTokenizerWorkerUnchanged(CustomTestCase):
    """The tokenizer_worker_num == 1 path must behave exactly as before."""

    def test_load_lora_adapter_succeeds(self):
        manager = _FakeTokenizerManager(tokenizer_worker_num=1)
        result = asyncio.run(manager.load_lora_adapter(_load_req()))
        self.assertTrue(result.success)
        manager.update_lora_adapter_communicator.assert_awaited_once()
        self.assertEqual(manager.lora_registry.num_registered_loras, 1)
        self.assertIn("adapter_a", manager.lora_ref_cache)

    def test_load_lora_adapter_from_tensors_succeeds(self):
        manager = _FakeTokenizerManager(tokenizer_worker_num=1)
        result = asyncio.run(
            manager.load_lora_adapter_from_tensors(_load_from_tensors_req())
        )
        self.assertTrue(result.success)
        manager.update_lora_adapter_communicator.assert_awaited_once()
        self.assertEqual(manager.lora_registry.num_registered_loras, 1)
        self.assertIn("adapter_a", manager.lora_ref_cache)

    def test_load_then_unload_succeeds(self):
        manager = _FakeTokenizerManager(tokenizer_worker_num=1)
        load_result = asyncio.run(manager.load_lora_adapter(_load_req()))
        self.assertTrue(load_result.success)
        unload_result = asyncio.run(manager.unload_lora_adapter(_unload_req()))
        self.assertTrue(unload_result.success)
        self.assertEqual(manager.lora_registry.num_registered_loras, 0)


if __name__ == "__main__":
    unittest.main()
