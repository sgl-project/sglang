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

"""LoRA unloads in the tokenizer manager. Evicting the wrong adapter, a pinned
one, or none at all leaves the registry and the ranks disagreeing about which
adapters are resident, with no error; an unload that reaches the ranks while a
request still holds the adapter frees weights that request is reading."""

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
# Upper bound for steps that finish at once when correct; a broken counter or
# notification fails here instead of hanging the CPU job.
STEP_TIMEOUT_S = 5


class _Ranks:
    """Stands in for the scheduler fan-out: holds the adapter table the ranks
    would hold and answers every load or unload with it, as they do."""

    def __init__(self):
        self.adapters = {}
        self.unloaded = []
        self.unloaded_ids = []

    async def __call__(self, obj):
        if isinstance(obj, UnloadLoRAAdapterReqInput):
            del self.adapters[obj.lora_name]
            self.unloaded.append(obj.lora_name)
            self.unloaded_ids.append(obj.lora_id)
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

    def test_unload_waits_for_active_requests(self):
        """The unload parks until the last request holding the adapter releases
        it, then sends the ranks the ID those requests acquired."""

        async def scenario(from_tensors):
            ranks = _Ranks()
            manager = _manager(ranks)
            registry = manager.lora_registry
            self.assertTrue((await _load(manager, "a", from_tensors)).success)
            # Two requests hold "a".
            lora_id = await registry.acquire("a")
            self.assertEqual(await registry.acquire("a"), lora_id)

            waiting = asyncio.Event()
            wait_for_unload = registry.wait_for_unload

            async def spy(waited_id):
                waiting.set()
                await wait_for_unload(waited_id)

            registry.wait_for_unload = spy
            unload = asyncio.create_task(
                manager.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="a"))
            )
            waited = asyncio.create_task(waiting.wait())
            try:
                # Resume once the unload starts waiting or, if it never waits, ends.
                await asyncio.wait(
                    {unload, waited},
                    timeout=STEP_TIMEOUT_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                self.assertTrue(waiting.is_set(), "the unload never waited")
                # The fan-out stub never yields, so an unload that did not block on
                # the requests would have finished before this coroutine resumed.
                self.assertFalse(unload.done())

                await registry.release(lora_id)
                # Event-loop turns, not wall time: the woken unload re-checks the
                # count and must park again behind the second request.
                for _ in range(5):
                    await asyncio.sleep(0)
                self.assertFalse(unload.done(), "unloaded while a request held it")
                self.assertEqual(ranks.unloaded_ids, [])

                await registry.release(lora_id)
                result = await asyncio.wait_for(unload, STEP_TIMEOUT_S)
            finally:
                for task in (unload, waited):
                    task.cancel()
                await asyncio.gather(unload, waited, return_exceptions=True)

            self.assertTrue(result.success, result.error_message)
            self.assertEqual(ranks.unloaded_ids, [lora_id])
            self.assertEqual(manager.pending_lora_unloads, {})

        self._run(scenario)


if __name__ == "__main__":
    unittest.main()
