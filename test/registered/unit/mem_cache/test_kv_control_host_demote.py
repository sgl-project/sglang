# SPDX-License-Identifier: Apache-2.0
"""Host DEMOTE uses actual file I/O, source receipts and CPU pool reclamation."""

import threading
import unittest
from unittest.mock import patch

import test_kv_control_cache as cache_fixtures
import torch
from test_kv_control_transfers import (
    FULL,
    backup_prefix,
    current,
    hicache,
    persist_prefix,
    public_transfer,
    seed,
    start_inference_backup,
    start_inference_load,
    wait,
    wait_for,
)

from sglang.srt.managers.kv_control.contracts import Status
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def ranges(outcomes):
    return [(o.coverage.start_token, o.coverage.end_token) for o in outcomes]


class TestKVControlHostDemote(cache_fixtures.KVCacheTestCase):
    def test_legacy_file_positive_metadata_does_not_advertise_host_reclamation(self):
        from sglang.srt.mem_cache.hicache_storage import MetadataCache

        with hicache(self, storage=True) as owner:
            _, cache, service, _ = owner
            backend = cache.cache_controller.storage_backend
            self.assertIsNone(backend._unified_suffixes)
            with patch.object(backend, "metadata_cache", MetadataCache(60)):
                caps = service.capabilities()["public_protocol"]
                self.assertNotIn(("host", "storage"), caps["paths"]["sglang.demote"])
                result = public_transfer(service, "demote", "host", "storage", "cold")
                self.assertEqual(result.status, Status.UNSUPPORTED)

    def test_reclaims_full_host_scope_and_recovers_file_bytes_in_both_write_policies(
        self,
    ):
        for policy in ("write_back", "write_through"):
            with (
                self.subTest(policy=policy),
                hicache(self, storage=True, write_policy=policy) as owner,
            ):
                _, cache, service, dma = owner
                node = seed(cache)
                backup_prefix(cache, dma, node, evict=True)
                pool = cache.host_pool_group.anchor_entry.host_pool
                initial = pool.available_size()
                self.assertIn(
                    ("host", "storage"),
                    service.capabilities()["public_protocol"]["paths"]["sglang.demote"],
                )
                result = public_transfer(service, "demote", "host", "storage", "cold")
                done = wait(service, result)
                self.assertEqual(done.status, Status.COMPLETED, done)
                self.assertEqual(ranges(done.public_result.preserved_scope), [(0, 4)])
                self.assertEqual(ranges(done.public_result.reclaimed_scope), [(0, 4)])
                self.assertEqual(pool.available_size(), initial + 4)
                self.assertFalse(cache.tree_core.root_node.children)
                self.assertNotIn(node, cache.tree_core.evictable_host_leaves)
                cache.sanity_check()
                # Recovery must publish the exact persisted K/V, then survive
                # H2D back to a request-usable device allocation.
                loaded = public_transfer(service, "prefetch", "storage", "host", "read")
                self.assertEqual(wait(service, loaded).status, Status.COMPLETED)
                rows = service.control.executor._walk(
                    (1, 2, 3, 4),
                    service.control.executor._compatibility(cache_fixtures.target()),
                )
                start_inference_load(cache, rows[-1][0])
                dma.finish()
                cache.loading_check()
                indices = torch.cat([n.component_data[FULL].value for n, *_ in rows])
                device = cache.token_to_kv_pool_allocator.get_kvcache()
                self.assertTrue(torch.all(device.k_buffer[0][indices] == 7))
                self.assertTrue(torch.all(device.v_buffer[0][indices] == 8))
                self.assertEqual(current(service, result), done)
                cache.sanity_check()

    def test_shared_ancestor_is_preserved_and_only_selected_suffix_reclaimed(self):
        with hicache(self, storage=True, write_policy="write_through") as owner:
            _, cache, service, dma = owner
            ancestor = seed(cache, (1, 2))
            leaf = seed(cache)
            sibling = seed(cache, (1, 2, 5, 6), value=11)
            backup_prefix(cache, dma, leaf, evict=True)
            backup_prefix(cache, dma, sibling, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            with patch.object(cache.cache_controller, "write_storage") as write:
                blocked = public_transfer(
                    service, "demote", "host", "storage", "ancestor", tokens=(1, 2)
                )
                self.assertEqual(blocked.status, Status.DEFERRED)
                self.assertEqual(blocked.update.code, "structural_dependency")
                write.assert_not_called()
            result = public_transfer(service, "demote", "host", "storage", "cold")
            done = wait(service, result)
            self.assertEqual(done.status, Status.PARTIAL)
            self.assertEqual(
                ranges(done.public_result.preserved_scope), [(0, 2), (2, 4)]
            )
            self.assertEqual(ranges(done.public_result.reclaimed_scope), [(2, 4)])
            self.assertEqual(
                [
                    (o.coverage.end_token, o.code)
                    for o in done.public_result.blocked_scope
                ],
                [(2, "structural_dependency")],
            )
            self.assertEqual(pool.available_size(), initial + 2)
            self.assertIsNotNone(ancestor.component_data[FULL].host_value)
            self.assertIsNotNone(sibling.component_data[FULL].host_value)
            self.assertIsNone(leaf.component_data[FULL].host_value)
            cache.sanity_check()

    def test_gpu_resident_host_copy_is_not_queued_for_storage(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            with patch.object(cache.cache_controller, "write_storage") as write:
                done = public_transfer(service, "demote", "host", "storage", "cold")
                write.assert_not_called()
            self.assertEqual(done.status, Status.DEFERRED)
            self.assertTrue(done.update.quiescent)
            self.assertFalse(done.public_result.preserved_scope)
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(
                {o.code for o in done.public_result.blocked_scope},
                {"source_gpu_resident"},
            )
            self.assertEqual(pool.available_size(), initial)
            self.assertIsNotNone(node.component_data[FULL].value)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_unpublished_backup_does_not_enqueue_a_storage_write(self):
        with hicache(self, storage=True, write_policy="write_through") as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            start_inference_backup(cache, node)
            self.assertTrue(dma.pending)
            with patch.object(cache.cache_controller, "write_storage") as write:
                blocked = public_transfer(service, "demote", "host", "storage", "cold")
                self.assertEqual(blocked.status, Status.DEFERRED)
                self.assertEqual(blocked.update.code, "inflight_backup")
                write.assert_not_called()
            dma.finish()
            cache.writing_check()
            cache.sanity_check()

    def test_pin_and_computation_refs_skip_storage_and_allow_retry_after_release(self):
        from test_kv_control_protection import run_pin

        for protection in ("pin", "compute"):
            with (
                self.subTest(protection=protection),
                hicache(self, storage=True, write_policy="write_through") as owner,
            ):
                _, cache, service, dma = owner
                node = seed(cache)
                persist_prefix(cache, service, dma, node)
                pool = cache.host_pool_group.anchor_entry.host_pool
                initial = pool.available_size()
                if protection == "pin":
                    self.assertEqual(run_pin(service)["update"]["status"], "completed")
                    lock = None
                else:
                    lock = cache.lock(node.id)
                try:
                    with patch.object(cache.cache_controller, "write_storage") as write:
                        done = public_transfer(
                            service, "demote", "host", "storage", "cold"
                        )
                        write.assert_not_called()
                    self.assertEqual(done.status, Status.DEFERRED)
                    self.assertEqual(
                        {o.code for o in done.public_result.blocked_scope},
                        {"active_reference"},
                    )
                    self.assertFalse(done.public_result.preserved_scope)
                    self.assertFalse(done.public_result.reclaimed_scope)
                    self.assertEqual(pool.available_size(), initial)
                    self.assertGreater(
                        node.component_data[FULL].host_lock_ref
                        if protection == "pin"
                        else node.component_data[FULL].lock_ref,
                        0,
                    )
                finally:
                    if lock is not None:
                        cache.unlock(lock)
                    else:
                        run_pin(service, mode="release", revision=2, message="release")
                # Existing bytes must be confirmed without rewriting them.
                with patch(
                    "sglang.srt.mem_cache.hicache_storage.os.replace",
                    side_effect=AssertionError("redundant storage write"),
                ):
                    retried = public_transfer(
                        service, "demote", "host", "storage", "retry"
                    )
                    self.assertEqual(wait(service, retried).status, Status.COMPLETED)
                self.assertEqual(pool.available_size(), initial + 4)
                cache.sanity_check()

    def test_radix_split_during_write_retains_ack_geometry_and_reclaims_each_slot_once(
        self,
    ):
        from array import array

        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            entered, release = threading.Event(), threading.Event()
            cc = cache.cache_controller
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = public_transfer(
                        service, "demote", "host", "storage", "cold"
                    )
                    self.assertTrue(entered.wait(5))
                    cache.match_prefix(
                        MatchPrefixParams(key=RadixKey(array("q", [1, 2])))
                    )
                    self.assertEqual(len(cache.tree_core.root_node.children), 1)
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            self.assertEqual(done.status, Status.COMPLETED)
            self.assertEqual(ranges(done.public_result.reclaimed_scope), [(0, 4)])
            self.assertEqual(pool.available_size(), initial + 4)
            self.assertFalse(cache.tree_core.root_node.children)
            # Exactly replaying an operation uses its immutable receipt.
            replayed = public_transfer(service, "demote", "host", "storage", "cold")
            self.assertEqual(replayed, done)
            self.assertEqual(pool.available_size(), initial + 4)
            cache.sanity_check()

    def test_foreign_h2d_keeps_host_source_until_its_own_fence(self):
        with hicache(self, storage=True, write_policy="write_through") as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            entered, release = threading.Event(), threading.Event()
            cc = cache.cache_controller
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = public_transfer(
                        service, "demote", "host", "storage", "cold"
                    )
                    self.assertTrue(entered.wait(5))
                    start_inference_load(cache, node)
                    self.assertTrue(dma.pending)
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            self.assertEqual(done.status, Status.PARTIAL)
            self.assertEqual(
                {o.code for o in done.public_result.blocked_scope}, {"inflight_load"}
            )
            self.assertEqual(pool.available_size(), initial)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            dma.finish()
            cache.loading_check()
            self.assertTrue(
                torch.all(
                    cache.token_to_kv_pool_allocator.get_kvcache().k_buffer[0][
                        node.component_data[FULL].value
                    ]
                    == 7
                )
            )
            cache.sanity_check()

    def test_partial_write_keeps_all_source_slots_and_reports_actual_saved_prefix(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            cc = cache.cache_controller
            original, calls = cc.page_set_func, 0

            def partial(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs) if calls == 1 else False

            with (
                patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2),
                patch.object(cc, "page_set_func", partial),
            ):
                done = wait(
                    service,
                    public_transfer(service, "demote", "host", "storage", "partial"),
                )
            self.assertEqual(done.status, Status.FAILED)
            self.assertEqual(done.update.code, "incomplete_storage_write")
            self.assertEqual(ranges(done.public_result.preserved_scope), [(0, 2)])
            self.assertEqual(ranges(done.public_result.missing_scope), [(2, 4)])
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(pool.available_size(), initial)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_missing_prior_backup_is_rewritten_before_host_reclamation(self):
        from pathlib import Path

        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            cache.write_backup_storage(node.id)

            def saved():
                cache.drain_storage_control_queues()
                return not cache.ongoing_backup

            wait_for(service, saved)
            cc = cache.cache_controller
            hashes = cache.tree_core.get_hash_values(node.id)
            files = [
                Path(cc.storage_backend.file_path, key + ".bin")
                for h in hashes
                for key in cc.storage_backend._get_component_keys(h)
            ]
            for path in files:
                path.unlink()
            # A previous native receipt or node backup flag cannot substitute
            # for the current verified worker result.
            done = wait(
                service, public_transfer(service, "demote", "host", "storage", "cold")
            )
            self.assertEqual(done.status, Status.COMPLETED)
            self.assertTrue(all(path.is_file() for path in files))
            self.assertFalse(cache.tree_core.root_node.children)
            cache.sanity_check()

    def test_inside_node_endpoint_is_rejected_without_reclaiming_extra_tokens(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            result = public_transfer(
                service, "demote", "host", "storage", "short", tokens=(1, 2)
            )
            self.assertEqual(result.status, Status.UNSUPPORTED)
            self.assertFalse(result.public_result.reclaimed_scope)
            self.assertEqual(pool.available_size(), initial)
            self.assertEqual(len(node.component_data[FULL].host_value), 4)
            cache.sanity_check()

    def test_cancel_waits_for_storage_quiescence_and_never_reclaims_source(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            entered, release = threading.Event(), threading.Event()
            cc = cache.cache_controller
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                success = original(*args, **kwargs)
                entered.set()
                assert release.wait(5)
                return success

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = public_transfer(
                        service, "demote", "host", "storage", "cold"
                    )
                    self.assertTrue(entered.wait(5))
                    cancelled = service.control.cancel(result.ref)
                    self.assertFalse(cancelled.update.quiescent)
                    self.assertGreater(node.component_data[FULL].host_lock_ref, 0)
                    self.assertEqual(pool.available_size(), initial)
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            self.assertEqual(done.status, Status.CANCELLED)
            self.assertTrue(done.public_result.preserved_scope)
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(pool.available_size(), initial)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(node.component_data[FULL].lock_ref, 0)
            cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
