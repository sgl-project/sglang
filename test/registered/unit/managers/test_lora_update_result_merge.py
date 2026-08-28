# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for merging per-rank LoRA update replies from the control fan-out."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARegistry
from sglang.srt.managers import tokenizer_control_mixin as control_module
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    TokenizerControlMixin,
)
from sglang.srt.server_args import LoRARef
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _ok(adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(success=True, loaded_adapters=adapters or {})


def _err(message, adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(
        success=False,
        error_message=message,
        loaded_adapters=adapters or {},
    )


class TestLoRAUpdateRecovery(CustomTestCase):
    def test_partial_local_load_state_can_be_unloaded(self):
        memory_pool = SimpleNamespace(remove_lora=Mock(return_value=2))
        notify_slots_updated = Mock()
        manager = SimpleNamespace(
            lora_refs={},
            configs={},
            loras={},
            memory_pool=memory_pool,
            pending_lora_load_events={},
            num_pinned_loras=0,
            _notify_lora_slots_updated=notify_slots_updated,
            create_lora_update_result=lambda **kwargs: LoRAUpdateOutput(**kwargs),
        )

        result = LoRAManager._unload_lora_adapter(
            manager,
            LoRARef(lora_id="id", lora_name="adapter", lora_path="/adapter"),
        )

        self.assertTrue(result.success)
        memory_pool.remove_lora.assert_called_once_with("id")
        notify_slots_updated.assert_called_once_with({2})


class _FakeTokenizerControl(TokenizerControlMixin):
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent_lora_ids = []
        self.lora_update_lock = asyncio.Lock()
        self.lora_registry = LoRARegistry()
        self.lora_ref_cache = {}
        self.pending_lora_unloads = {}
        self.server_args = SimpleNamespace(enable_lora=True, max_loaded_loras=None)

    def auto_create_handle_loop(self):
        pass

    async def update_lora_adapter_communicator(self, obj):
        self.sent_lora_ids.append(obj.lora_id)
        return self.replies.pop(0)


class TestLoRAUpdateStateMachine(CustomTestCase):
    def _runtime_context(self):
        return (
            patch.object(
                control_module,
                "get_lora",
                return_value=SimpleNamespace(enable_lora=True),
                create=True,
            ),
            patch.object(
                control_module,
                "get_parallel",
                return_value=SimpleNamespace(dp_size=2, enable_dp_attention=True),
            ),
        )

    def test_partial_updates_converge_on_retry(self):
        async def run():
            manager = _FakeTokenizerControl(
                [
                    [_err("invalid adapter"), _err("invalid adapter")],
                    [_ok({"adapter": "/adapter"}), _err("rank 1 load failed")],
                    [_ok(), _ok()],
                    [_ok({"adapter": "/adapter"}), _ok({"adapter": "/adapter"})],
                    [_ok(), _err("rank 1 unload failed")],
                    [_ok(), _ok()],
                ]
            )

            rejected = await manager.load_lora_adapter(
                LoadLoRAAdapterReqInput(lora_name="adapter", lora_path="/invalid")
            )
            self.assertFalse(rejected.success)
            self.assertEqual(rejected.error_message, "invalid adapter")
            rejected_lora_id = manager.sent_lora_ids[0]
            self.assertNotIn("adapter", manager.pending_lora_unloads)

            first = await manager.load_lora_adapter(
                LoadLoRAAdapterReqInput(lora_name="adapter", lora_path="/adapter")
            )
            self.assertFalse(first.success)
            self.assertEqual(first.error_message, "rank 1 load failed")
            failed_lora_id = manager.sent_lora_ids[1]
            self.assertNotEqual(failed_lora_id, rejected_lora_id)
            self.assertIn("adapter", manager.pending_lora_unloads)
            self.assertEqual(manager.lora_registry.get_all_adapters(), {})

            cleanup = await manager.unload_lora_adapter(
                UnloadLoRAAdapterReqInput(lora_name="adapter")
            )
            self.assertTrue(cleanup.success)
            self.assertEqual(manager.sent_lora_ids[2], failed_lora_id)
            self.assertNotIn("adapter", manager.pending_lora_unloads)

            retry = await manager.load_lora_adapter(
                LoadLoRAAdapterReqInput(lora_name="adapter", lora_path="/adapter")
            )
            self.assertTrue(retry.success)
            self.assertNotEqual(manager.sent_lora_ids[3], failed_lora_id)
            self.assertEqual(
                manager.lora_registry.get_all_adapters()["adapter"].lora_id,
                manager.sent_lora_ids[3],
            )

            unload = await manager.unload_lora_adapter(
                UnloadLoRAAdapterReqInput(lora_name="adapter")
            )
            self.assertFalse(unload.success)
            self.assertEqual(unload.error_message, "rank 1 unload failed")
            loaded_lora_id = manager.sent_lora_ids[3]
            self.assertEqual(manager.sent_lora_ids[4], loaded_lora_id)
            self.assertIn("adapter", manager.pending_lora_unloads)
            self.assertEqual(manager.lora_registry.get_all_adapters(), {})

            unload_retry = await manager.unload_lora_adapter(
                UnloadLoRAAdapterReqInput(lora_name="adapter")
            )
            self.assertTrue(unload_retry.success)
            self.assertEqual(manager.sent_lora_ids[5], loaded_lora_id)
            self.assertNotIn("adapter", manager.pending_lora_unloads)

        lora_context, parallel_context = self._runtime_context()
        with lora_context, parallel_context:
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
