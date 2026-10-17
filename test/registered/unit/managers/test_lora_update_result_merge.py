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
from unittest.mock import AsyncMock, Mock

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    TokenizerControlMixin,
    _merge_lora_update_results,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _ok(adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(success=True, loaded_adapters=adapters or {})


def _err(message, adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(
        success=False, error_message=message, loaded_adapters=adapters or {}
    )


class TestMergeLoRAUpdateResults(CustomTestCase):
    def test_all_success_returns_first_rank_result(self):
        """On success the merge must hand back a rank's own reply: callers
        mutate result.loaded_adapters in place during LRU eviction, which a
        synthesized empty result would silently break."""
        results = [_ok({"a": "path"}), _ok({"a": "path"})]
        merged = _merge_lora_update_results(results)
        self.assertIs(merged, results[0])
        self.assertTrue(merged.success)

    def test_any_rank_failure_wins(self):
        """Regression guard for the pre-merge behavior of returning
        results[0]: a failure on a non-zero rank was reported as success,
        letting the tokenizer-side registry drift from that rank's actual
        adapter state."""
        merged = _merge_lora_update_results(
            [_ok({"a": "path"}), _err("out of memory", {"stale": "path"})]
        )
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "out of memory")
        self.assertEqual(merged.loaded_adapters, {"stale": "path"})

    def test_duplicate_error_messages_deduplicated(self):
        """All ranks usually fail identically (e.g. "already loaded"); the
        joined message must not repeat per rank, but distinct causes must all
        be kept."""
        merged = _merge_lora_update_results(
            [_err("already loaded"), _err("already loaded"), _err("bad rank")]
        )
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "already loaded | bad rank")

    def test_failure_without_message(self):
        """A rank replying success=False with error_message=None must not
        crash the join."""
        merged = _merge_lora_update_results([_err(None), _ok()])
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "")

    def test_failed_load_does_not_redirect_existing_adapter_unload(self):
        """Rejected duplicates must keep the original adapter's unload ID;
        fresh partial loads must remain cleanable via their new ID."""

        async def scenario(from_tensors, duplicate):
            original = LoRARef(lora_name="a", lora_path="path", pinned=False)
            manager = TokenizerControlMixin()
            manager.auto_create_handle_loop = Mock()
            manager.lora_update_lock = asyncio.Lock()
            manager.lora_registry = LoRARegistry([original] if duplicate else [])
            manager.pending_lora_unloads = {}
            manager.lora_ref_cache = {}
            replies = (
                [_err("already loaded", {"a": "path"})]
                if duplicate
                else [_ok({"a": "path"}), _err("rank failed")]
            )
            manager.update_lora_adapter_communicator = AsyncMock(
                side_effect=[replies, [_ok()]]
            )
            if from_tensors:
                request = LoadLoRAAdapterFromTensorsReqInput(
                    lora_name="a", config_dict={}, serialized_named_tensors=[b""]
                )
                result = await manager.load_lora_adapter_from_tensors(request)
            else:
                request = LoadLoRAAdapterReqInput(lora_name="a", lora_path="path")
                result = await manager.load_lora_adapter(request)

            self.assertFalse(result.success)
            self.assertEqual(result.loaded_adapters, {"a": "path"} if duplicate else {})
            self.assertEqual(
                manager.pending_lora_unloads,
                {} if duplicate else {"a": request.lora_id},
            )
            self.assertEqual(
                manager.lora_registry.get_all_adapters(),
                {"a": original} if duplicate else {},
            )
            unload = UnloadLoRAAdapterReqInput(lora_name="a")
            result = await manager.unload_lora_adapter(unload)
            self.assertTrue(result.success)
            self.assertEqual(
                unload.lora_id, original.lora_id if duplicate else request.lora_id
            )
            manager.update_lora_adapter_communicator.assert_awaited_with(unload)
            self.assertEqual(manager.update_lora_adapter_communicator.await_count, 2)
            self.assertEqual(manager.lora_registry.get_all_adapters(), {})
            self.assertEqual(manager.pending_lora_unloads, {})

        with get_context().override_server_args(enable_lora=True, dp_size=1):
            for from_tensors in (False, True):
                for duplicate in (True, False):
                    with self.subTest(from_tensors=from_tensors, duplicate=duplicate):
                        asyncio.run(scenario(from_tensors, duplicate))


class TestLoadLoRACapacity(CustomTestCase):
    def make_manager(self, adapters):
        manager = TokenizerControlMixin()
        manager.auto_create_handle_loop = Mock()
        manager.lora_update_lock = asyncio.Lock()
        manager.lora_registry = LoRARegistry(adapters)
        manager.pending_lora_unloads = {}
        manager.lora_ref_cache = {ref.lora_name: ref for ref in adapters}
        backend = {ref.lora_name: ref.lora_path for ref in adapters}

        async def update(request):
            # Yield so concurrent loads exercise the real update lock.
            await asyncio.sleep(0)
            if isinstance(request, UnloadLoRAAdapterReqInput):
                del backend[request.lora_name]
            elif request.lora_name in backend:
                return [_err("already loaded", dict(backend))]
            else:
                backend[request.lora_name] = getattr(request, "lora_path", "__tensor__")
            return [_ok(dict(backend))]

        manager.update_lora_adapter_communicator = AsyncMock(side_effect=update)
        return manager, backend

    async def load(self, manager, from_tensors, name="new", pinned=True):
        if from_tensors:
            request = LoadLoRAAdapterFromTensorsReqInput(
                lora_name=name,
                config_dict={},
                serialized_named_tensors=[b""],
                pinned=pinned,
            )
            result = await manager.load_lora_adapter_from_tensors(request)
        else:
            request = LoadLoRAAdapterReqInput(
                lora_name=name, lora_path=f"{name}-path", pinned=pinned
            )
            result = await manager.load_lora_adapter(request)
        return request, result

    def test_pinned_capacity_rejects_before_loading(self):
        async def scenario(from_tensors, pinned):
            adapters = [
                LoRARef(lora_name=name, lora_path=name, pinned=True)
                for name in ("a", "b")
            ]
            manager, backend = self.make_manager(adapters)
            request, result = await self.load(manager, from_tensors, pinned=pinned)
            self.assertFalse(result.success)
            manager.update_lora_adapter_communicator.assert_not_awaited()
            self.assertIn("max_loaded_loras", result.error_message)
            self.assertIsNone(request.lora_id)
            self.assertEqual(
                manager.lora_registry.get_all_adapters(), manager.lora_ref_cache
            )
            self.assertEqual(set(backend), {"a", "b"})
            self.assertEqual(manager.lora_registry.num_registered_loras, 2)

        with get_context().override_server_args(
            enable_lora=True, dp_size=1, max_loaded_loras=2
        ):
            for from_tensors in (False, True):
                for pinned in (False, True):
                    with self.subTest(from_tensors=from_tensors, pinned=pinned):
                        asyncio.run(scenario(from_tensors, pinned))

    def test_evictable_adapter_still_makes_room(self):
        async def scenario(from_tensors, pinned):
            adapters = [
                LoRARef(lora_name="keep", lora_path="keep-path", pinned=True),
                LoRARef(lora_name="evict", lora_path="evict-path", pinned=False),
            ]
            manager, backend = self.make_manager(adapters)
            _, result = await self.load(manager, from_tensors, pinned=pinned)
            self.assertTrue(result.success)
            self.assertEqual(set(backend), {"keep", "new"})
            self.assertEqual(
                set(manager.lora_registry.get_all_adapters()), {"keep", "new"}
            )
            self.assertEqual(set(result.loaded_adapters), {"keep", "new"})
            self.assertEqual(manager.update_lora_adapter_communicator.await_count, 2)
            self.assertNotIn(adapters[1].lora_id, manager.lora_registry._counters)

        with get_context().override_server_args(
            enable_lora=True, dp_size=1, max_loaded_loras=2
        ):
            for from_tensors in (False, True):
                for pinned in (False, True):
                    with self.subTest(from_tensors=from_tensors, pinned=pinned):
                        asyncio.run(scenario(from_tensors, pinned))

    def test_pinned_adapters_allow_spare_or_unlimited_capacity(self):
        async def scenario(from_tensors):
            manager, backend = self.make_manager(
                [LoRARef(lora_name="old", lora_path="old-path", pinned=True)]
            )
            _, result = await self.load(manager, from_tensors)
            self.assertTrue(result.success)
            self.assertEqual(set(backend), {"old", "new"})
            self.assertEqual(
                set(manager.lora_registry.get_all_adapters()), {"old", "new"}
            )
            self.assertEqual(manager.update_lora_adapter_communicator.await_count, 1)

        for limit in (2, None):
            with get_context().override_server_args(
                enable_lora=True, dp_size=1, max_loaded_loras=limit
            ):
                for from_tensors in (False, True):
                    with self.subTest(limit=limit, from_tensors=from_tensors):
                        asyncio.run(scenario(from_tensors))

    def test_duplicate_at_pinned_capacity_keeps_backend_error(self):
        async def scenario(from_tensors):
            existing = LoRARef(lora_name="old", lora_path="old-path", pinned=True)
            manager, backend = self.make_manager([existing])
            _, result = await self.load(manager, from_tensors, name="old")
            self.assertFalse(result.success)
            self.assertEqual(result.error_message, "already loaded")
            self.assertEqual(manager.update_lora_adapter_communicator.await_count, 1)
            self.assertEqual(
                manager.lora_registry.get_all_adapters(), {"old": existing}
            )
            self.assertEqual(backend, {"old": "old-path"})
            self.assertEqual(manager.pending_lora_unloads, {})

        with get_context().override_server_args(
            enable_lora=True, dp_size=1, max_loaded_loras=1
        ):
            for from_tensors in (False, True):
                with self.subTest(from_tensors=from_tensors):
                    asyncio.run(scenario(from_tensors))

    def test_concurrent_pinned_loads_cannot_exceed_capacity(self):
        async def scenario():
            manager, backend = self.make_manager([])
            responses = await asyncio.gather(
                self.load(manager, False, name="disk"),
                self.load(manager, True, name="tensor"),
            )
            self.assertEqual(sum(result.success for _, result in responses), 1)
            self.assertEqual(manager.update_lora_adapter_communicator.await_count, 1)
            self.assertEqual(len(backend), 1)
            self.assertEqual(
                set(manager.lora_registry.get_all_adapters()), set(backend)
            )
            self.assertEqual(set(manager.lora_ref_cache), set(backend))

        with get_context().override_server_args(
            enable_lora=True, dp_size=1, max_loaded_loras=1
        ):
            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
