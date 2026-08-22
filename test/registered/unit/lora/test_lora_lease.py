"""Unit tests for the exactly-once LoRA lease lifecycle.

A leaked lease leaves the adapter's in-flight counter above zero forever, so
unload's wait_for_zero never returns; a double release drives the counter
negative, which hangs it just the same (it waits for exactly zero). Covers:

  * TokenizerManager._finalize_lora_lease is the single, idempotent release
    owner for every terminal path
  * pre-acquire failures (lora_id never set) release nothing
  * LoRARegistry.acquire: lookup + counter increment are one atomic admission
    step under the writer lock, so acquire racing an unload fails cleanly
    instead of touching a deleted counter
  * LoRARegistry.lru_lora_name never picks non-reloadable (wire-loaded) refs
  * explicit unload drops the lora_ref_cache entry (DELETE) while the locked
    helper alone (the max_loaded_loras EVICT path) keeps it
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tm(enable_lora: bool = True) -> TokenizerManager:
    """TokenizerManager with only the fields the lease path reads."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.enable_lora = enable_lora
    tm.lora_registry = MagicMock()
    tm.lora_registry.release = AsyncMock()
    return tm


def _make_state(lora_path="adapter-a", lora_id="id-1") -> ReqState:
    obj = SimpleNamespace(lora_path=lora_path, lora_id=lora_id, rid="rid-1")
    return ReqState([], False, asyncio.Event(), obj, Mock())


class TestFinalizeLoraLease(CustomTestCase):
    def _finalize(self, tm, state, times=1):
        async def run():
            for _ in range(times):
                tm._finalize_lora_lease(state)
            # Let the create_task'd release coroutines run.
            await asyncio.sleep(0)

        asyncio.run(run())

    def test_release_exactly_once_on_double_finalize(self):
        tm = _make_tm()
        state = _make_state()
        self._finalize(tm, state, times=3)
        tm.lora_registry.release.assert_awaited_once_with("id-1")
        self.assertTrue(state.lora_lease_released)

    def test_pre_acquire_failure_releases_nothing(self):
        # _init_req_state runs before the LoRA acquire; a failure in between
        # leaves lora_id unset and there is no lease to release.
        tm = _make_tm()
        state = _make_state(lora_id=None)
        self._finalize(tm, state)
        tm.lora_registry.release.assert_not_awaited()
        self.assertFalse(state.lora_lease_released)

    def test_no_lora_path_is_noop(self):
        tm = _make_tm()
        state = _make_state(lora_path=None)
        self._finalize(tm, state)
        tm.lora_registry.release.assert_not_awaited()

    def test_lora_disabled_is_noop(self):
        tm = _make_tm(enable_lora=False)
        state = _make_state()
        self._finalize(tm, state)
        tm.lora_registry.release.assert_not_awaited()

    def test_none_state_is_noop(self):
        tm = _make_tm()

        async def run():
            tm._finalize_lora_lease(None)

        asyncio.run(run())
        tm.lora_registry.release.assert_not_awaited()


class TestAcquireAtomicity(CustomTestCase):
    def test_acquire_after_unregister_raises_value_error(self):
        async def run():
            registry = LoRARegistry()
            ref = LoRARef(lora_name="a", lora_path="/x")
            await registry.register(ref)
            await registry.unregister("a")
            with self.assertRaises(ValueError):
                await registry.acquire("a")

        asyncio.run(run())

    def test_unload_waits_for_inflight_lease_then_completes(self):
        async def run():
            registry = LoRARegistry()
            ref = LoRARef(lora_name="a", lora_path="/x")
            await registry.register(ref)
            lora_id = await registry.acquire("a")
            await registry.unregister("a")

            wait_task = asyncio.create_task(registry.wait_for_unload(lora_id))
            await asyncio.sleep(0.01)
            self.assertFalse(wait_task.done(), "unload must wait for the lease")

            await registry.release(lora_id)
            await asyncio.wait_for(wait_task, timeout=2)

        asyncio.run(run())

    def test_batch_acquire_release_roundtrip(self):
        async def run():
            registry = LoRARegistry()
            for name in ("a", "b"):
                await registry.register(LoRARef(lora_name=name, lora_path=f"/{name}"))
            lora_ids = await registry.acquire(["a", "b", None])
            self.assertEqual(len(lora_ids), 3)
            self.assertIsNone(lora_ids[2])
            await registry.release([i for i in lora_ids if i is not None])
            for name in ("a", "b"):
                await registry.unregister(name)

        asyncio.run(run())


class TestLruSkipsNonReloadable(CustomTestCase):
    def test_lru_prefers_reloadable_even_if_older_entry_is_wire_loaded(self):
        async def run():
            registry = LoRARegistry()
            # Registration order = LRU order; the wire-loaded ref is oldest.
            await registry.register(
                LoRARef(lora_name="wire", lora_path="__distributed__", reloadable=False)
            )
            await registry.register(LoRARef(lora_name="disk", lora_path="/d"))
            self.assertEqual(await registry.lru_lora_name(), "disk")

        asyncio.run(run())

    def test_all_non_reloadable_yields_no_victim(self):
        async def run():
            registry = LoRARegistry()
            await registry.register(
                LoRARef(lora_name="w1", lora_path="__tensor__", reloadable=False)
            )
            await registry.register(
                LoRARef(lora_name="w2", lora_path="__distributed__", reloadable=False)
            )
            self.assertIsNone(await registry.lru_lora_name())
            self.assertIsNone(await registry.lru_lora_name(exclude_pinned=True))

        asyncio.run(run())

    def test_exclude_pinned_still_respected(self):
        async def run():
            registry = LoRARegistry()
            await registry.register(
                LoRARef(lora_name="pinned-disk", lora_path="/p", pinned=True)
            )
            await registry.register(LoRARef(lora_name="plain-disk", lora_path="/q"))
            self.assertEqual(
                await registry.lru_lora_name(exclude_pinned=True), "plain-disk"
            )
            self.assertEqual(await registry.lru_lora_name(), "pinned-disk")

        asyncio.run(run())


class TestUnloadRefCacheCleanup(CustomTestCase):
    def _make_unload_tm(self, success: bool) -> TokenizerManager:
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.auto_create_handle_loop = Mock()
        tm.server_args = MagicMock()
        tm.server_args.enable_lora = True
        tm.server_args.dp_size = 1
        tm.lora_update_lock = asyncio.Lock()
        tm._unload_lora_adapter_locked = AsyncMock(
            return_value=SimpleNamespace(success=success)
        )
        tm.lora_ref_cache = {"a": LoRARef(lora_name="a", lora_path="/x")}
        return tm

    def test_explicit_unload_drops_ref_cache_entry(self):
        tm = self._make_unload_tm(success=True)
        asyncio.run(tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="a")))
        self.assertNotIn("a", tm.lora_ref_cache)

    def test_failed_unload_keeps_ref_cache_entry(self):
        tm = self._make_unload_tm(success=False)
        asyncio.run(tm.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name="a")))
        self.assertIn("a", tm.lora_ref_cache)

    def test_locked_helper_alone_keeps_ref_cache_entry(self):
        # The max_loaded_loras LRU loop calls _unload_lora_adapter_locked
        # directly — an EVICT — and a disk-backed adapter must keep its
        # reload-catalog entry or it can never be implicitly reloaded.
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.lora_update_lock = asyncio.Lock()
        tm.lora_registry = MagicMock()
        tm.lora_registry.unregister = AsyncMock(return_value="id-a")
        tm.lora_registry.wait_for_unload = AsyncMock()
        tm.update_lora_adapter_communicator = AsyncMock(
            return_value=[SimpleNamespace(success=True)]
        )
        tm.lora_ref_cache = {"a": LoRARef(lora_name="a", lora_path="/x")}

        async def run():
            async with tm.lora_update_lock:
                return await tm._unload_lora_adapter_locked(
                    UnloadLoRAAdapterReqInput(lora_name="a")
                )

        result = asyncio.run(run())
        self.assertTrue(result.success)
        self.assertIn("a", tm.lora_ref_cache)


if __name__ == "__main__":
    unittest.main()
