"""Unit tests for request_id propagation into HiCacheStorageExtraInfo on the
read/query paths of HiCacheController and HybridCacheController, introduced by
augusto.yjh since dae126d5.

The controllers carry `operation.request_id` inside
`HiCacheStorageExtraInfo.extra_info` on every *read/query* storage call
(`_storage_hit_query` -> batch_exists(_v2); `_page_transfer`/_page_transfer_kv_batch
-> batch_get_v1; KV-derived / non-KV sidecar -> batch_get_v2) while the *write*
path (_page_backup -> page_set_func / batch_set_*) keeps `extra_info=None`.

The controllers are built with `__new__` + the minimal attributes the target
methods actually touch, so the tests stay CPU-only and do not materialise host
pools or buffers. Operations are lightweight `SimpleNamespace` stand-ins.

Usage:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_controller_request_id_unit.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

from sglang.srt.managers import cache_controller as cc_module
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchAck
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hicache_ctrl(page_size=2):
    c = HiCacheController.__new__(HiCacheController)
    c.page_size = page_size
    c.storage_backend = MagicMock()
    return c


def _hybrid_ctrl(page_size=2):
    c = HybridCacheController.__new__(HybridCacheController)
    c.page_size = page_size
    c.storage_backend = MagicMock()
    return c


class TestHiCacheControllerRequestId(unittest.TestCase):
    def test_storage_hit_query_injects_request_id_into_batch_exists(self):
        ctrl = _hicache_ctrl()
        ctrl.get_hash_str = MagicMock(return_value=["h0", "h1", "h2", "h3"])
        ctrl.storage_backend.batch_exists.return_value = 3  # 3 of 4 pages hit

        op = SimpleNamespace(
            last_hash=None,
            token_ids=[1, 2, 3],
            prefix_keys=None,
            request_id="r-1",
        )
        hash_value, count = ctrl._storage_hit_query(op)

        extra = ctrl.storage_backend.batch_exists.call_args[0][1]
        self.assertIsInstance(extra, HiCacheStorageExtraInfo)
        self.assertIsNone(extra.prefix_keys)
        self.assertEqual(extra.extra_info, {"request_id": "r-1"})
        self.assertEqual(hash_value, ["h0", "h1", "h2"])
        self.assertEqual(count, 3 * ctrl.page_size)

    @mock.patch.object(cc_module, "STORAGE_BATCH_SIZE", 2)
    def test_storage_hit_query_prefix_keys_grow_per_batch(self):
        ctrl = _hicache_ctrl()
        ctrl.get_hash_str = MagicMock(return_value=["h0", "h1", "h2", "h3"])

        # `_storage_hit_query` mutates the shared ``prefix_keys`` list in place
        # after each batch returns, so snapshot each call's arguments as the
        # real backend would see them (copy the list at call time).
        seen = []

        def capture(batch_hashes, extra_info):
            seen.append(
                (
                    list(batch_hashes),
                    list(extra_info.prefix_keys),
                    extra_info.extra_info,
                )
            )
            return 2 if len(seen) == 1 else 1  # first batch full hit, second partial

        ctrl.storage_backend.batch_exists.side_effect = capture

        op = SimpleNamespace(
            last_hash=None,
            token_ids=[1, 2, 3],
            prefix_keys=["p0"],
            request_id="r-1",
        )
        ctrl._storage_hit_query(op)

        self.assertEqual(len(seen), 2)
        # First batch carries the original prefix; every batch carries request_id.
        self.assertEqual(seen[0], (["h0", "h1"], ["p0"], {"request_id": "r-1"}))
        # Second batch: prefix grew by the first batch's confirmed hit hashes.
        self.assertEqual(
            seen[1], (["h2", "h3"], ["p0", "h0", "h1"], {"request_id": "r-1"})
        )

    def test_page_transfer_forwards_request_id_to_batch_get_v1_and_sidecar(self):
        ctrl = _hicache_ctrl(page_size=2)
        ctrl.page_get_func = MagicMock(return_value=3)  # full KV hit for 3 pages
        ctrl.storage_backend.batch_get_v2.return_value = {"indexer": [True, True, True]}
        ctrl.prefetch_sync_queue = MagicMock()

        op = SimpleNamespace(
            request_id="r-1",
            hash_value=["h0", "h1", "h2"],
            host_indices=list(range(6)),  # 3 pages * page_size 2
            prefix_keys=None,
            is_terminated=lambda: False,
            pool_transfers=[
                PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
            ],
        )
        completed = ctrl._page_transfer(op)

        self.assertEqual(completed, 3)
        # _page_transfer_kv_batch routes KV reads through page_get_func with
        # the request_id-bearing extra_info.
        pg_extra = ctrl.page_get_func.call_args[0][3]
        self.assertEqual(pg_extra.extra_info, {"request_id": "r-1"})
        self.assertIsNone(pg_extra.prefix_keys)
        # KV-derived sidecar batch_get_v2 receives the same extra_info.
        gv_extra = ctrl.storage_backend.batch_get_v2.call_args[0][1]
        self.assertEqual(gv_extra.extra_info, {"request_id": "r-1"})
        # The PrefetchAck carries the request id.
        ack = ctrl.prefetch_sync_queue.put.call_args[0][0]
        self.assertIsInstance(ack, PrefetchAck)
        self.assertEqual(ack.rid, "r-1")

    def test_page_backup_write_path_carries_no_request_id(self):
        ctrl = _hicache_ctrl(page_size=2)
        captured = {}

        def fake_set(batch_hashes, batch_host_indices, extra_info):
            captured["extra"] = extra_info
            return True

        ctrl.page_set_func = fake_set
        op = SimpleNamespace(
            prefix_keys=None,
            hash_value=["h0", "h1", "h2"],
            host_indices=list(range(6)),
            completed_tokens=0,
        )
        ctrl._page_backup(op)

        self.assertIsNone(captured["extra"].prefix_keys)
        # Write path never injects request_id -> extra_info is the default None.
        self.assertIsNone(captured["extra"].extra_info)


class TestHybridCacheControllerRequestId(unittest.TestCase):
    def test_storage_hit_query_no_pools_uses_batch_exists_with_request_id(self):
        ctrl = _hybrid_ctrl()
        ctrl.get_hash_str = MagicMock(return_value=["h0", "h1", "h2"])
        ctrl.storage_backend.batch_exists.return_value = 2

        op = SimpleNamespace(
            token_ids=[1, 2, 3],
            last_hash=None,
            prefix_keys=["p0"],
            request_id="r-1",
            pool_transfers=None,
            pool_storage_result=MagicMock(),
        )
        hash_value, count = ctrl._storage_hit_query(op)

        extra = ctrl.storage_backend.batch_exists.call_args[0][1]
        self.assertEqual(extra.prefix_keys, ["p0"])
        self.assertEqual(extra.extra_info, {"request_id": "r-1"})
        op.pool_storage_result.update_kv_hit_pages.assert_called_once_with(2)
        self.assertEqual(hash_value, ["h0", "h1"])
        self.assertEqual(count, 2 * ctrl.page_size)

    def test_storage_hit_query_with_pools_uses_batch_exists_v2_with_request_id(self):
        ctrl = _hybrid_ctrl()
        ctrl.get_hash_str = MagicMock(return_value=["h0", "h1", "h2"])
        ctrl.storage_backend.batch_exists_v2.return_value = PoolTransferResult(
            kv_hit_pages=2, extra_pool_hit_pages={}
        )

        transfers = [PoolTransfer(name=PoolName.SWA, indices_from_pool=None)]
        op = SimpleNamespace(
            token_ids=[1, 2, 3],
            last_hash=None,
            prefix_keys=["p0"],
            request_id="r-1",
            pool_transfers=transfers,
            pool_storage_result=MagicMock(),
        )
        ctrl._storage_hit_query(op)

        args = ctrl.storage_backend.batch_exists_v2.call_args[0]
        self.assertEqual(args[0], ["h0", "h1", "h2"])
        self.assertEqual(args[1], transfers)
        self.assertEqual(args[2].prefix_keys, ["p0"])
        self.assertEqual(args[2].extra_info, {"request_id": "r-1"})

    def test_page_transfer_sidecar_injects_request_id_into_batch_get_v2(self):
        ctrl = _hybrid_ctrl()
        ctrl._sync_trailing_keys = MagicMock()
        ctrl._resolve_sidecar_nonkv_derived_pool_transfers = MagicMock()
        ctrl.storage_backend.batch_get_v2.return_value = {"swa": [True, True]}
        ctrl.prefetch_sync_queue = MagicMock()

        op = SimpleNamespace(
            pool_transfers=[PoolTransfer(name=PoolName.SWA, indices_from_pool=None)],
            is_terminated=lambda: False,
            hash_value=["h0", "h1"],
            request_id="r-1",
        )
        ctrl._page_transfer_sidecar(op, kv_completed_pages=2)

        # batch_get_v2 carries a request_id-only extra_info (no prefix_keys).
        gv_extra = ctrl.storage_backend.batch_get_v2.call_args[0][1]
        self.assertEqual(gv_extra.extra_info, {"request_id": "r-1"})
        self.assertIsNone(gv_extra.prefix_keys)
        # The non-KV sidecar plumbing was driven.
        ctrl._sync_trailing_keys.assert_called_once()
        ctrl._resolve_sidecar_nonkv_derived_pool_transfers.assert_called_once_with(op)
        # PrefetchAck on the sync queue carries the request id + counted pool hits.
        ack = ctrl.prefetch_sync_queue.put.call_args[0][0]
        self.assertIsInstance(ack, PrefetchAck)
        self.assertEqual(ack.rid, "r-1")
        self.assertEqual(ack.pool_hits, {"swa": 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
