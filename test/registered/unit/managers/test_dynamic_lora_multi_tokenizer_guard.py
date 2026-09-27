"""Dynamic LoRA must fail fast when tokenizer_worker_num > 1 (issue #31084).

Each tokenizer worker keeps its own LoRA registry. Dynamic load/unload only
mutates the worker that served the HTTP request, so multi-worker mode would
leave registries divergent. This pins the guard that rejects those endpoints
before any backend call or local registry mutation.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from sglang.srt.lora.lora_registry import LoRARegistry
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager() -> TokenizerControlMixin:
    manager = TokenizerControlMixin()
    manager.auto_create_handle_loop = Mock()
    manager.lora_update_lock = asyncio.Lock()
    manager.lora_registry = LoRARegistry()
    manager.pending_lora_unloads = {}
    manager.lora_ref_cache = {}
    manager.update_lora_adapter_communicator = AsyncMock(
        return_value=[LoRAUpdateOutput(success=True, loaded_adapters={})]
    )
    return manager


class TestDynamicLoRAMultiTokenizerGuard(CustomTestCase):
    def _assert_rejected(self, manager: TokenizerControlMixin, result) -> None:
        self.assertFalse(result.success)
        self.assertIn("--tokenizer-worker-num", result.error_message)
        self.assertIn("31084", result.error_message)
        manager.update_lora_adapter_communicator.assert_not_awaited()
        self.assertEqual(manager.lora_registry.num_registered_loras, 0)
        self.assertEqual(manager.lora_ref_cache, {})

    def test_load_rejected_with_multiple_tokenizer_workers(self):
        manager = _manager()
        with get_context().override_server_args(
            enable_lora=True,
            dp_size=1,
            tokenizer_worker_num=2,
        ):
            result = asyncio.run(
                manager.load_lora_adapter(
                    LoadLoRAAdapterReqInput(
                        lora_name="adapter_a", lora_path="/tmp/adapter_a"
                    )
                )
            )
        self._assert_rejected(manager, result)

    def test_load_from_tensors_rejected_with_multiple_tokenizer_workers(self):
        manager = _manager()
        with get_context().override_server_args(
            enable_lora=True,
            dp_size=1,
            tokenizer_worker_num=2,
        ):
            result = asyncio.run(
                manager.load_lora_adapter_from_tensors(
                    LoadLoRAAdapterFromTensorsReqInput(
                        lora_name="adapter_a",
                        config_dict={"r": 8},
                        serialized_named_tensors=[b"tp0"],
                    )
                )
            )
        self._assert_rejected(manager, result)

    def test_unload_rejected_with_multiple_tokenizer_workers(self):
        manager = _manager()
        with get_context().override_server_args(
            enable_lora=True,
            dp_size=1,
            tokenizer_worker_num=2,
        ):
            result = asyncio.run(
                manager.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(lora_name="adapter_a")
                )
            )
        self._assert_rejected(manager, result)

    def test_guard_still_fires_when_dp_attention_allows_dp_size_above_one(self):
        """dp attention relaxed the neighboring dp_size check; tokenizer
        registries remain per-process, so the multi-worker guard must still
        reject."""
        manager = _manager()
        with get_context().override_server_args(
            enable_lora=True,
            dp_size=2,
            enable_dp_attention=True,
            tokenizer_worker_num=2,
        ):
            result = asyncio.run(
                manager.load_lora_adapter(
                    LoadLoRAAdapterReqInput(
                        lora_name="adapter_a", lora_path="/tmp/adapter_a"
                    )
                )
            )
        self._assert_rejected(manager, result)

    def test_single_tokenizer_worker_load_still_succeeds(self):
        """If the guard accidentally always-rejects, the single-worker path
        must still reach the backend and register the adapter."""
        manager = _manager()
        with get_context().override_server_args(
            enable_lora=True,
            dp_size=1,
            tokenizer_worker_num=1,
        ):
            result = asyncio.run(
                manager.load_lora_adapter(
                    LoadLoRAAdapterReqInput(
                        lora_name="adapter_a", lora_path="/tmp/adapter_a"
                    )
                )
            )
        self.assertTrue(result.success)
        manager.update_lora_adapter_communicator.assert_awaited_once()
        self.assertEqual(manager.lora_registry.num_registered_loras, 1)
        self.assertIn("adapter_a", manager.lora_ref_cache)


if __name__ == "__main__":
    unittest.main()
