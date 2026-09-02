"""Contracts for multi-pool HiCache prefetch reads."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class KVDerivedPrefetchTest(unittest.TestCase):
    def _controller(self, results):
        controller = object.__new__(HiCacheController)
        controller.page_get_func = Mock(return_value=2)
        controller.storage_backend = Mock()
        controller.storage_backend.batch_get_v2.return_value = results
        return controller

    def test_forwards_router_hint_extra_info(self):
        controller = self._controller({PoolName.INDEXER: [True, True]})
        extra_info = HiCacheStorageExtraInfo(
            extra_info={"kv_hints": {"protocol_version": "0.1"}}
        )
        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
        )

        hit_pages = controller._page_transfer_kv_batch(
            Mock(), ["h0", "h1"], Mock(), extra_info, [transfer]
        )

        self.assertEqual(hit_pages, 2)
        call = controller.storage_backend.batch_get_v2.call_args
        self.assertIs(call.args[1], extra_info)

    def test_missing_or_malformed_pool_result_clamps_the_batch_to_zero(self):
        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
        )
        for results in (
            {},
            {PoolName.INDEXER: [True]},
            {PoolName.INDEXER: [1, True]},
        ):
            with self.subTest(results=results):
                controller = self._controller(results)
                hit_pages = controller._page_transfer_kv_batch(
                    Mock(),
                    ["h0", "h1"],
                    Mock(),
                    HiCacheStorageExtraInfo(),
                    [transfer],
                )
                self.assertEqual(hit_pages, 0)


class NonKVDerivedPrefetchTest(unittest.TestCase):
    def test_forwards_router_hint_extra_info(self):
        controller = object.__new__(HybridCacheController)
        controller.storage_backend = Mock()
        controller.storage_backend.batch_get_v2.return_value = {
            PoolName.MAMBA: [True, True]
        }
        controller.prefetch_sync_queue = Queue()
        transfer = PoolTransfer(name=PoolName.MAMBA, keys=["h0", "h1"])
        router_hint = {"protocol_version": "0.1", "actions": []}
        operation = SimpleNamespace(
            request_id="request-0",
            hash_value=["h0", "h1"],
            pool_transfers=[transfer],
            router_hint=router_hint,
            is_terminated=lambda: False,
        )

        controller._page_transfer_sidecar(operation, kv_completed_pages=2)

        call = controller.storage_backend.batch_get_v2.call_args
        self.assertEqual(call.args[1].extra_info, {"kv_hints": router_hint})
        ack = controller.prefetch_sync_queue.get_nowait()
        self.assertEqual(ack.pool_hits, {PoolName.MAMBA.value: 2})

    def test_missing_or_malformed_pool_result_is_reported_as_zero(self):
        transfer = PoolTransfer(name=PoolName.MAMBA, keys=["h0", "h1"])
        operation = SimpleNamespace(
            request_id="request-0",
            hash_value=["h0", "h1"],
            pool_transfers=[transfer],
            router_hint=None,
            is_terminated=lambda: False,
        )
        for results in (
            {},
            {PoolName.MAMBA: [True]},
            {PoolName.MAMBA: [1, True]},
        ):
            with self.subTest(results=results):
                controller = object.__new__(HybridCacheController)
                controller.storage_backend = Mock()
                controller.storage_backend.batch_get_v2.return_value = results
                controller.prefetch_sync_queue = Queue()

                controller._page_transfer_sidecar(operation, kv_completed_pages=2)

                ack = controller.prefetch_sync_queue.get_nowait()
                self.assertEqual(ack.pool_hits, {PoolName.MAMBA.value: 0})


if __name__ == "__main__":
    unittest.main()
