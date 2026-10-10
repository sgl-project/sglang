"""Finite-I/O regressions with real Unified cache, host pool and file storage.

Run on CPU with SGLANG_USE_CPU_ENGINE=1 and Python TreeCore. CPU runs use
unpinned host tensors; no GPU transfer or model inference is claimed.
"""

import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path
from unittest import mock

import test_unified_radix_cache_unittest as fixtures
import torch

from sglang.srt.managers.cache_controller import PrefetchAck
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, MatchPrefixParams
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.runtime_context import get_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestFiniteIO(fixtures.CustomTestCase):
    cfg = fixtures.CacheConfig(
        page_size=4,
        num_layers=2,
        head_num=1,
        head_dim=16,
        kv_size=64,
        max_context_len=64,
    )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache, self.allocator, _ = fixtures.build_fixture(self.cfg)
        original = MHATokenToKVPoolHost.__init__

        def cpu_host(pool, *args, **kwargs):
            if not torch.cuda.is_available():
                kwargs["pin_memory"] = False
            original(pool, *args, **kwargs)

        # Production reserves 10 GiB; this fixture allocates less than 64 KiB.
        # Ignore that deployment admission floor on constrained CPU runners.
        with (
            mock.patch.object(MHATokenToKVPoolHost, "__init__", cpu_host),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
                side_effect=lambda requested: requested,
            ),
        ):
            fixtures.UnifiedRadixCacheSuite._init_hicache(
                self,
                self.cache,
                storage_backend="file",
                storage_dir=self.directory.name,
                prefetch_threshold=4,
            )
        # Match the running Unified cache flag: upstream catches some batch
        # exceptions only when this flag is enabled.
        unified_memory = get_memory().override(enable_unified_memory=True)
        unified_memory.__enter__()
        self.addCleanup(unified_memory.__exit__, None, None, None)
        self.cc = self.cache.cache_controller
        self.pool = self.cc.mem_pool_host
        self.initial_slots = self.pool.available_size()
        self.cache.enable_storage_metrics = True
        self.cache.storage_metrics_collector = mock.Mock()
        self.tokens = array("q", range(1, 13))
        self.key = RadixKey(self.tokens)
        self.hashes = get_storage_hash_str(self.key, page_size=4)
        self.backend = self.cc.storage_backend
        host = self.pool.anchor_entry.host_pool
        page = host.get_dummy_flat_data_page()
        page.fill_(7)
        for key in self.hashes:
            self.assertTrue(self.backend.set(key, page))
        self.handle_counter = 0

    def submit(self, handle=None):
        self.handle_counter += 1
        handle = handle or CacheRequestHandle(f"finite-{self.handle_counter}", 0)
        self.cache.prefetch_from_storage(
            handle, self.cache.root_node_handle(), self.tokens, None, None
        )
        return handle

    def pump_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.cache.drain_storage_control_queues()
            if predicate():
                return
            time.sleep(0.005)
        self.fail("prefetch lifecycle did not settle")

    def settle(self, handle):
        self.pump_until(lambda: handle not in self.cache.ongoing_prefetch)
        self.cache.drain_storage_control_queues()

    def conservation(self, handle, resident=0):
        if resident:
            self.cache.finish_storage_prefetch_admission(handle, resident, reason=None)
        self.cache.pop_prefetch_loaded_span(handle)
        self.cc.mem_pool_host.anchor_entry.host_pool._merge_release_slots()
        self.assertEqual(self.pool.available_size(), self.initial_slots - resident)
        self.assertEqual(
            int(self.pool.anchor_entry.host_pool.slot_used.sum()), resident
        )
        self.assertEqual(self.cache.ongoing_prefetch, {})
        self.assertEqual(self.cc.prefetch_tokens_occupied, 0)
        self.assertFalse(self.cache._storage_prefetch_hit_remaining_by_reqid)
        self.assertFalse(self.cache.prefetch_loaded_tokens_by_reqid)
        self.assertFalse(self.cache.prefetch_loaded_storage_start_by_reqid)
        self.cache.sanity_check()

    def next_prefetch(self):
        handle = self.submit()
        self.settle(handle)
        match = self.cache.match_prefix(MatchPrefixParams(key=self.key))
        self.assertEqual(match.host_hit_length, len(self.tokens))
        self.assertTrue(self.cc.prefetch_io_aux_thread.is_alive())
        node = self.cache.tree_core.node_by_id(match.last_host_node)
        slots = node.component_data[fixtures.ComponentType.FULL].host_value
        page = self.pool.anchor_entry.host_pool.get_data_page(int(slots[0]), flat=True)
        torch.testing.assert_close(page, torch.full_like(page, 7))
        self.conservation(handle, resident=len(self.tokens))

    def test_finite_read_exception_worker_survives(self):
        with mock.patch.object(
            self.backend, "batch_get", side_effect=RuntimeError("read failed")
        ):
            handle = self.submit()
            self.settle(handle)
        self.conservation(handle)
        self.next_prefetch()

    def test_finite_short_file_worker_survives(self):
        path = Path(self.directory.name) / (
            self.backend._get_suffixed_key(self.hashes[0]) + ".bin"
        )
        original = path.read_bytes()
        path.write_bytes(b"x")
        handle = self.submit()
        self.settle(handle)
        self.conservation(handle)
        path.write_bytes(original)
        self.next_prefetch()

    def test_finite_query_exception_worker_survives(self):
        with mock.patch.object(
            self.backend, "batch_exists", side_effect=OSError("query failed")
        ):
            handle = self.submit()
            self.settle(handle)
        self.conservation(handle)
        self.assertTrue(self.cc.prefetch_thread.is_alive())
        self.next_prefetch()

    def test_finite_file_disappears_after_query(self):
        original = self.backend.batch_get

        def disappeared(*args, **kwargs):
            for p in Path(self.directory.name).glob("*.bin"):
                p.unlink()
            return original(*args, **kwargs)

        with mock.patch.object(self.backend, "batch_get", side_effect=disappeared):
            handle = self.submit()
            self.settle(handle)
        self.conservation(handle)

    def test_finite_cancel_running_late_success_and_duplicate_terminal(self):
        entered, resume = threading.Event(), threading.Event()
        original = self.backend.batch_get

        def blocked(*args, **kwargs):
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("test gate not released")
            return original(*args, **kwargs)

        try:
            with mock.patch.object(self.backend, "batch_get", side_effect=blocked):
                handle = self.submit()
                self.pump_until(entered.is_set)
                operation = self.cache.ongoing_prefetch[handle].operation
                self.cache.release_aborted_request(handle)
                self.cache.drain_storage_control_queues()
                self.assertLess(self.pool.available_size(), self.initial_slots)
                self.assertEqual(
                    self.cache.match_prefix(
                        MatchPrefixParams(key=self.key)
                    ).host_hit_length,
                    0,
                )
                resume.set()
                self.pump_until(
                    lambda: self.pool.available_size() == self.initial_slots
                )
            self.conservation(handle)
            terminal = PrefetchAck(operation.request_id, operation, completed_req=True)
            self.cc.ack_prefetch_queue.put(terminal)
            self.cache.drain_storage_control_queues()
            self.conservation(handle)
            self.assertEqual(
                self.cache.match_prefix(
                    MatchPrefixParams(key=self.key)
                ).host_hit_length,
                0,
            )
            self.next_prefetch()
        finally:
            resume.set()

    def test_finite_cancel_before_allocation(self):
        entered, resume = threading.Event(), threading.Event()
        original = self.backend.batch_exists

        def blocked(*args, **kwargs):
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("test gate not released")
            return original(*args, **kwargs)

        try:
            with mock.patch.object(self.backend, "batch_exists", side_effect=blocked):
                handle = self.submit()
                self.pump_until(entered.is_set)
                operation = self.cache.ongoing_prefetch[handle].operation
                self.assertIsNone(operation.host_indices)
                self.cache.release_aborted_request(handle)
                self.conservation(handle)
                resume.set()
                self.pump_until(lambda: operation.storage_hit_count > 0)
                self.conservation(handle)
            self.next_prefetch()
        finally:
            resume.set()

    def test_finite_cancel_allocated_before_read(self):
        entered, resume = threading.Event(), threading.Event()
        original = self.cc._page_transfer

        def blocked(operation):
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("test gate not released")
            return original(operation)

        try:
            with (
                mock.patch.object(self.cc, "_page_transfer", side_effect=blocked),
                mock.patch.object(
                    self.backend, "batch_get", wraps=self.backend.batch_get
                ) as read,
            ):
                handle = self.submit()
                self.pump_until(entered.is_set)
                operation = self.cache.ongoing_prefetch[handle].operation
                self.assertIsNotNone(operation.host_indices)
                self.cache.release_aborted_request(handle)
                self.cache.drain_storage_control_queues()
                self.assertLess(self.pool.available_size(), self.initial_slots)
                resume.set()
                self.pump_until(
                    lambda: self.pool.available_size() == self.initial_slots
                )
                read.assert_not_called()
                self.conservation(handle)
            self.next_prefetch()
        finally:
            resume.set()

    def test_finite_partial_progress_then_failure(self):
        original = self.backend.batch_get
        calls = 0

        def partial(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second batch failed")
            return original(*args, **kwargs)

        with (
            mock.patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1),
            mock.patch.object(self.backend, "batch_get", side_effect=partial),
        ):
            handle = self.submit()
            operation = self.cache.ongoing_prefetch[handle].operation
            self.settle(handle)
        self.assertEqual(operation.completed_tokens, 4)
        self.conservation(handle)
        self.assertEqual(operation.terminal_outcome, "FAILURE")
        self.assertTrue(operation.terminal_ack_consumed)
        self.next_prefetch()

    def test_finite_failure_releases_nonroot_anchor_lock(self):
        full_tokens, full_key = self.tokens, self.key
        self.tokens = self.tokens[:4]
        handle = self.submit()
        self.settle(handle)
        self.conservation(handle, resident=4)
        self.tokens, self.key = full_tokens, full_key
        anchor = self.cache.match_prefix(MatchPrefixParams(key=self.key)).last_host_node
        node = self.cache.tree_core.node_by_id(anchor)
        full = node.component_data[fixtures.ComponentType.FULL]
        baseline = full.host_lock_ref
        with mock.patch.object(
            self.backend, "batch_get", side_effect=RuntimeError("anchor read failed")
        ):
            handle = CacheRequestHandle("anchor-failure", 0)
            self.cache.prefetch_from_storage(
                handle,
                anchor,
                self.tokens[4:],
                self.cache.get_last_hash_value(anchor),
                None,
                matched_prefix_tokens=list(self.tokens[:4]),
            )
            self.settle(handle)
        self.assertEqual(full.host_lock_ref, baseline)
        self.conservation(handle, resident=4)

    def test_finite_late_completion_keeps_replacement_operation(self):
        entered, resume = threading.Event(), threading.Event()
        original = self.backend.batch_get
        calls = 0

        def blocked(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                if not resume.wait(10):
                    raise RuntimeError("test gate not released")
            return original(*args, **kwargs)

        try:
            with mock.patch.object(self.backend, "batch_get", side_effect=blocked):
                handle = self.submit()
                self.pump_until(entered.is_set)
                old = self.cache.ongoing_prefetch[handle].operation
                self.cache.release_aborted_request(handle)
                self.submit(handle)
                replacement = self.cache.ongoing_prefetch[handle].operation
                self.assertIsNot(old, replacement)
                resume.set()
                self.settle(handle)
            self.assertEqual(
                self.cache.match_prefix(
                    MatchPrefixParams(key=self.key)
                ).host_hit_length,
                len(self.tokens),
            )
            self.conservation(handle, resident=len(self.tokens))
        finally:
            resume.set()


if __name__ == "__main__":
    unittest.main()
