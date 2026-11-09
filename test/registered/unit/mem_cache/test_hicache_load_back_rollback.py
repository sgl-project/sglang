# SPDX-License-Identifier: Apache-2.0
"""Unsubmitted load-back failures must return reservations and admission pins.

These CPU fault-injection tests use real slot allocators and Mamba preparation
hooks. Transfer submission and TreeCore callbacks are dependency boundaries;
no CUDA, host registration, or compiled TreeCore fixture is needed.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.group import PoolEntry
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import index_slots

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestDeviceReservationRollback(CustomTestCase):
    def _controller(self, entries, *, full=None):
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(entry_map=entries)
        controller.mem_pool_device_allocator = full
        controller.device = "cpu"
        controller.load_queue = []
        return controller

    def _entry(self, name, allocator, **callbacks):
        if allocator is None:
            allocator = MambaSlotAllocator(0, "cpu")
        return PoolEntry(
            name=name,
            host_pool=None,
            device_pool=allocator,
            layer_mapper=lambda layer: layer,
            **callbacks,
        )

    def test_side_exception_returns_only_controller_owned_anchor(self):
        """A side-pool error must not strand freshly reserved FULL rows."""
        for caller_owned in (False, True):
            with self.subTest(caller_owned=caller_owned):
                full = MambaSlotAllocator(8, "cpu")
                failure = RuntimeError("side allocation failed")
                entry = self._entry(
                    PoolName.MAMBA,
                    None,
                    device_alloc_fn=mock.Mock(side_effect=failure),
                )
                controller = self._controller({PoolName.MAMBA: entry}, full=full)
                owned = full.alloc(2) if caller_owned else None
                with self.assertRaises(RuntimeError) as caught:
                    controller.load(
                        torch.tensor([2, 3]),
                        device_indices=owned,
                        extra_pools=[
                            PoolTransfer(
                                name=PoolName.MAMBA, host_indices=torch.tensor([1])
                            )
                        ],
                    )
                self.assertIs(caught.exception, failure)
                self.assertEqual(full.available_size(), 6 if caller_owned else 8)
                self.assertEqual(controller.load_queue, [])

    def test_later_callback_error_returns_earlier_side_slots(self):
        """Both allocation and eviction errors must undo earlier side pools."""
        for fail_at in ("allocate", "evict"):
            with self.subTest(fail_at=fail_at):
                slots = MambaSlotAllocator(2, "cpu")
                failure = RuntimeError(f"injected {fail_at}")
                second = self._entry(
                    PoolName.SWA,
                    None,
                    device_alloc_fn=mock.Mock(
                        side_effect=failure if fail_at == "allocate" else None,
                        return_value=None,
                    ),
                    device_evict_fn=mock.Mock(side_effect=failure),
                )
                controller = self._controller(
                    {
                        PoolName.MAMBA: self._entry(PoolName.MAMBA, slots),
                        PoolName.SWA: second,
                    }
                )
                transfers = [
                    PoolTransfer(name=name, host_indices=torch.tensor([1]))
                    for name in (PoolName.MAMBA, PoolName.SWA)
                ]
                with self.assertRaises(RuntimeError) as caught:
                    controller._resolve_device_transfers(transfers)
                self.assertIs(caught.exception, failure)
                self.assertEqual(slots.available_size(), 2)
                self.assertTrue(all(t.device_indices is None for t in transfers))

    def test_rejected_resolution_restores_derived_aliases(self):
        """A rejected batch must not retain aliases to a returned source slot."""
        for keep_destination in (False, True):
            with self.subTest(keep_destination=keep_destination):
                slots = MambaSlotAllocator(2, "cpu")
                controller = self._controller(
                    {PoolName.MAMBA: self._entry(PoolName.MAMBA, slots)}
                )
                source = PoolTransfer(
                    name=PoolName.MAMBA, host_indices=torch.tensor([1])
                )
                original_host = torch.tensor([7])
                original_device = torch.tensor([8]) if keep_destination else None
                derived = PoolTransfer(
                    name=PoolName.INDEXER,
                    host_indices=original_host,
                    device_indices=original_device,
                    indices_from_pool=PoolName.MAMBA,
                )
                missing = PoolTransfer(
                    name=PoolName.SWA, indices_from_pool=PoolName.DRAFT
                )
                self.assertIsNone(
                    controller._resolve_device_transfers([source, derived, missing])
                )
                self.assertEqual(slots.available_size(), 2)
                self.assertIsNone(source.device_indices)
                self.assertIs(derived.host_indices, original_host)
                self.assertIs(derived.device_indices, original_device)

    def test_successful_derived_pool_preserves_split_source_spans(self):
        """The rollback fix must retain main's multi-transfer source merging."""
        slots = MambaSlotAllocator(2, "cpu")
        controller = self._controller(
            {PoolName.MAMBA: self._entry(PoolName.MAMBA, slots)}
        )
        sources = [
            PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([i]))
            for i in (3, 4)
        ]
        derived = PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.MAMBA)
        self.assertIsNotNone(controller._resolve_device_transfers([*sources, derived]))
        self.assertEqual(derived.host_indices.tolist(), [3, 4])
        self.assertEqual(derived.device_indices.tolist(), [1, 2])


class TestLoadBackAdmissionRollback(CustomTestCase):
    def _admission(self, *, host_only=False):
        cache = object.__new__(UnifiedRadixCache)
        index_slots(cache, (ComponentType.FULL, ComponentType.MAMBA))
        cache.token_to_kv_pool_allocator = SimpleNamespace()
        cache.load_back_threshold = 1
        cache.ongoing_load_back = {}
        cache._mamba_host_only_checkpoint = lambda: host_only
        cache._build_sidecar_transfers = lambda *args: []
        cache._component_available_size = lambda component: 8
        cache._apply_cache_actions = lambda actions: None
        cache.cache_controller = SimpleNamespace(
            load=lambda **kwargs: torch.tensor([3, 4])
        )
        host = torch.tensor([1, 2])
        cache.tree_core = SimpleNamespace(
            component_has_host_value_only=lambda *args: True,
            build_load_back_spec=lambda *args, **kwargs: (
                PoolTransfer(name=PoolName.KV, host_indices=host),
                {},
            ),
            commit_load_back=lambda *args: [],
        )
        locks = {"host": 0, "device": 0}

        def acquire(kind):
            locks[kind] += 1
            return SimpleNamespace(delta=0, to_dec_params=lambda: kind)

        def release(kind):
            self.assertGreater(locks[kind], 0, "double release")
            locks[kind] -= 1

        cache.inc_host_lock_ref = lambda node: acquire("host")
        cache.inc_lock_ref = lambda node: acquire("device")
        cache.dec_host_lock_ref = lambda node, receipt: release("host")
        cache.dec_lock_ref = lambda node, receipt: release("device")
        slots = MambaSlotAllocator(2, "cpu")
        storage = SimpleNamespace(
            tree_sub_pool=lambda component: component,
            take_state_slots=lambda pool, count: slots.alloc(count),
            free_state_slots=lambda pool, indices: slots.free(indices),
        )
        cache.kv_storage = lambda: storage
        component = object.__new__(MambaComponent)
        component.cache = cache
        component.tree_core = cache.tree_core
        cache._components_tuple = (component,)
        req = SimpleNamespace(kv=ReqKvInfo(mamba_pool_idx=None))
        return cache, req, slots, locks

    def _assert_failed_admission_released(self, req, slots, locks):
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(slots.available_size(), 2)
        self.assertEqual(locks, {"host": 0, "device": 0})

    def test_device_pin_exception_releases_host_pin(self):
        cache, req, slots, locks = self._admission()
        with (
            mock.patch.object(cache, "inc_lock_ref", side_effect=RuntimeError("pin")),
            self.assertRaisesRegex(RuntimeError, "pin"),
        ):
            cache.load_back(7, req=req)
        self._assert_failed_admission_released(req, slots, locks)

    def test_first_prepare_error_releases_admission_pins(self):
        cache, req, slots, locks = self._admission()
        with (
            mock.patch.object(
                cache._components_tuple[0],
                "prepare_load_back",
                side_effect=RuntimeError("prepare"),
            ),
            self.assertRaisesRegex(RuntimeError, "prepare"),
        ):
            cache.load_back(7, req=req)
        self._assert_failed_admission_released(req, slots, locks)

    def test_later_prepare_error_returns_real_mamba_preparation(self):
        cache, req, slots, locks = self._admission()
        cache._components_tuple += (
            SimpleNamespace(
                component_type=ComponentType.SWA,
                prepare_load_back=mock.Mock(side_effect=RuntimeError("later prepare")),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "later prepare"):
            cache.load_back(7, req=req)
        self._assert_failed_admission_released(req, slots, locks)

    def test_exhausted_request_pool_releases_admission_pins(self):
        """The real Mamba preparation assertion must not strand cache locks."""
        cache, req, slots, locks = self._admission()
        owned = slots.alloc(2)
        cache.evict_for_alloc = lambda params: None
        with self.assertRaisesRegex(AssertionError, "Cannot alloc mamba for load_back"):
            cache.load_back(7, req=req)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(slots.available_size(), 0)
        self.assertEqual(locks, {"host": 0, "device": 0})
        slots.free(owned)

    def test_spec_error_returns_preparation_and_admission_pins(self):
        cache, req, slots, locks = self._admission()
        with (
            mock.patch.object(
                cache.tree_core,
                "build_load_back_spec",
                side_effect=RuntimeError("spec"),
            ),
            self.assertRaisesRegex(RuntimeError, "spec"),
        ):
            cache.load_back(7, req=req)
        self._assert_failed_admission_released(req, slots, locks)

    def test_controller_error_returns_preparation_and_admission_pins(self):
        cache, req, slots, locks = self._admission()
        with (
            mock.patch.object(
                cache.cache_controller, "load", side_effect=RuntimeError("controller")
            ),
            self.assertRaisesRegex(RuntimeError, "controller"),
        ):
            cache.load_back(7, req=req)
        self._assert_failed_admission_released(req, slots, locks)

    def test_commit_callback_keeps_the_device_pin(self):
        """A commit callback must not see queued H2D rows become evictable."""
        cache, req, slots, locks = self._admission()
        pins_at_commit = []
        cache._apply_cache_actions = lambda actions: pins_at_commit.append(
            locks["device"]
        )
        self.assertTrue(cache.load_back(7, req=req))
        self.assertEqual(pins_at_commit, [1])
        self.assertEqual(locks, {"host": 1, "device": 1})

    def test_rejection_preserves_a_preexisting_request_slot(self):
        cache, req, slots, locks = self._admission()
        req.kv.mamba_pool_idx = slots.alloc(1)[0]
        original = req.kv.mamba_pool_idx
        self.assertFalse(cache.load_back(7, req=req, mem_quota=0))
        self.assertIs(req.kv.mamba_pool_idx, original)
        self.assertEqual(slots.available_size(), 1)
        self.assertEqual(locks, {"host": 0, "device": 0})

    def test_rejection_returns_a_new_request_slot(self):
        cache, req, slots, locks = self._admission()
        self.assertFalse(cache.load_back(7, req=req, mem_quota=0))
        self._assert_failed_admission_released(req, slots, locks)

    def test_success_transfers_pins_to_each_async_receipt(self):
        for host_only in (False, True):
            with self.subTest(host_only=host_only):
                cache, req, slots, locks = self._admission(host_only=host_only)
                count = 2 if host_only else 1
                for _ in range(count):
                    self.assertTrue(cache.load_back(7, req=req))
                receipts = cache.ongoing_load_back[7]
                if not host_only:
                    receipts = [receipts]
                self.assertEqual(len(receipts), count)
                self.assertEqual(locks, {"host": count, "device": count})
                for receipt in receipts:
                    cache.dec_lock_ref(7, receipt.lock_params)
                    cache.dec_host_lock_ref(7, receipt.host_lock_params)
                self.assertEqual(locks, {"host": 0, "device": 0})


if __name__ == "__main__":
    unittest.main()
