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

"""``--max-loaded-loras`` eviction in the tokenizer manager. Unloading the
wrong adapter, a pinned one, or none at all leaves the registry and the ranks
disagreeing about which adapters are resident, with no error."""

import asyncio
import unittest
from unittest.mock import Mock

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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAX_LOADED_LORAS = 2


class _Ranks:
    """Stands in for the scheduler fan-out: holds the adapter table the ranks
    would hold and answers every load or unload with it, as they do."""

    def __init__(self):
        self.adapters = {}
        self.unloaded = []

    async def __call__(self, obj):
        if isinstance(obj, UnloadLoRAAdapterReqInput):
            del self.adapters[obj.lora_name]
            self.unloaded.append(obj.lora_name)
        else:
            self.adapters[obj.lora_name] = getattr(obj, "lora_path", "__tensor__")
        return [LoRAUpdateOutput(success=True, loaded_adapters=dict(self.adapters))]


def _manager(ranks):
    manager = TokenizerControlMixin()
    manager.auto_create_handle_loop = Mock()
    manager.lora_update_lock = asyncio.Lock()
    manager.lora_registry = LoRARegistry([])
    manager.pending_lora_unloads = {}
    manager.lora_ref_cache = {}
    manager.update_lora_adapter_communicator = ranks
    return manager


async def _load(manager, name, from_tensors, pinned=False):
    if from_tensors:
        request = LoadLoRAAdapterFromTensorsReqInput(
            lora_name=name,
            config_dict={},
            serialized_named_tensors=[b""],
            pinned=pinned,
        )
        return await manager.load_lora_adapter_from_tensors(request)
    request = LoadLoRAAdapterReqInput(
        lora_name=name, lora_path=f"/{name}", pinned=pinned
    )
    return await manager.load_lora_adapter(request)


class TestLoRALRUUnload(CustomTestCase):
    def _run(self, scenario):
        with get_context().override_server_args(
            enable_lora=True, dp_size=1, max_loaded_loras=MAX_LOADED_LORAS
        ):
            for from_tensors in (False, True):
                with self.subTest(from_tensors=from_tensors):
                    asyncio.run(scenario(from_tensors))

    def test_least_recently_used_adapter_is_unloaded(self):
        async def scenario(from_tensors):
            ranks = _Ranks()
            manager = _manager(ranks)
            for name in ("a", "b"):
                self.assertTrue((await _load(manager, name, from_tensors)).success)
            # At the limit, not over it: nothing is unloaded yet.
            self.assertEqual(ranks.unloaded, [])
            # A request on "a" makes "b" the least recently used adapter.
            await manager.lora_registry.release(
                await manager.lora_registry.acquire("a")
            )

            result = await _load(manager, "c", from_tensors)

            self.assertTrue(result.success, result.error_message)
            self.assertEqual(ranks.unloaded, ["b"])
            self.assertEqual(set(manager.lora_registry.get_all_adapters()), {"a", "c"})
            self.assertEqual(set(result.loaded_adapters), {"a", "c"})
            self.assertEqual(manager.pending_lora_unloads, {})

        self._run(scenario)

    def test_pinned_adapter_is_skipped(self):
        async def scenario(from_tensors):
            ranks = _Ranks()
            manager = _manager(ranks)
            self.assertTrue(
                (await _load(manager, "a", from_tensors, pinned=True)).success
            )
            self.assertTrue((await _load(manager, "b", from_tensors)).success)

            result = await _load(manager, "c", from_tensors)

            self.assertTrue(result.success, result.error_message)
            self.assertEqual(ranks.unloaded, ["b"])
            self.assertEqual(set(manager.lora_registry.get_all_adapters()), {"a", "c"})

        self._run(scenario)


if __name__ == "__main__":
    unittest.main()
