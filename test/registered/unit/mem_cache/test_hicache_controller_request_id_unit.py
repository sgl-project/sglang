"""Unit tests for hybrid-controller request context propagation.

The trace context logic is intentionally scoped to ``HybridCacheController``;
the generic ``HiCacheController`` remains unchanged. These tests verify that
all storage RPCs issued by the hybrid controller carry request/caller metadata
in ``HiCacheStorageExtraInfo.extra_info``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller as hybrid_cc_module
from sglang.srt.managers.cache_controller import PrefetchAck
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


def _hybrid_ctrl(page_size=2):
    ctrl = HybridCacheController.__new__(HybridCacheController)
    ctrl.page_size = page_size
    ctrl.storage_backend = MagicMock()
    return ctrl


class TestHybridCacheControllerRequestId(unittest.TestCase):
    def test_thread_trace_info_registers_storage_config_ranks(self):
        """Hybrid storage threads read rank placement from storage_config."""
        ctrl = _hybrid_ctrl()
        ctrl.storage_config = SimpleNamespace(tp_rank=2, dp_rank=1, pp_rank=3)
        with mock.patch.object(hybrid_cc_module, "trace_set_thread_info") as set_info:
            ctrl._register_thread_trace_info("Prefetch")

        set_info.assert_called_once_with("Prefetch", 2, 1, 3)

    def test_thread_trace_info_defaults_missing_config_ranks_to_none(self):
        ctrl = _hybrid_ctrl()
        ctrl.storage_config = SimpleNamespace()
        with mock.patch.object(hybrid_cc_module, "trace_set_thread_info") as set_info:
            ctrl._register_thread_trace_info("Backup")

        set_info.assert_called_once_with("Backup", None, None, None)

    def test_storage_hit_query_no_pools_uses_batch_exists_with_request_id(self):
        ctrl = _hybrid_ctrl()
        ctrl.storage_backend.batch_exists.return_value = 2

        op = SimpleNamespace(
            token_ids=[1, 2, 3],
            last_hash=None,
            prefix_keys=["p0"],
            request_id="r-1",
            pool_transfers=None,
            pool_storage_result=MagicMock(),
            assume_stored=False,
        )
        with (
            mock.patch.object(
                hybrid_cc_module, "get_thread_caller_info", return_value=None
            ),
            mock.patch.object(
                hybrid_cc_module,
                "get_storage_hash_str",
                return_value=["h0", "h1", "h2"],
            ),
        ):
            hash_value, count = ctrl._storage_hit_query(op)
            extra = ctrl.storage_backend.batch_exists.call_args[0][1]

        self.assertEqual(extra.prefix_keys, ["p0"])
        self.assertEqual(extra.extra_info, {"request_id": "r-1"})
        op.pool_storage_result.update_kv_hit_pages.assert_called_once_with(2)
        self.assertEqual(hash_value, ["h0", "h1"])
        self.assertEqual(count, 2 * ctrl.page_size)

    def test_storage_hit_query_with_pools_uses_batch_exists_v2_with_request_id(self):
        ctrl = _hybrid_ctrl()
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
            assume_stored=False,
        )
        with (
            mock.patch.object(
                hybrid_cc_module, "get_thread_caller_info", return_value=None
            ),
            mock.patch.object(
                hybrid_cc_module,
                "get_storage_hash_str",
                return_value=["h0", "h1", "h2"],
            ),
        ):
            ctrl._storage_hit_query(op)

        args = ctrl.storage_backend.batch_exists_v2.call_args[0]
        self.assertEqual(args[0], ["h0", "h1", "h2"])
        self.assertEqual(args[1], transfers)
        self.assertEqual(args[2].prefix_keys, ["p0"])
        self.assertEqual(args[2].extra_info, {"request_id": "r-1"})

    def test_storage_hit_query_injects_caller_attribution(self):
        ctrl = _hybrid_ctrl()
        ctrl.storage_backend.batch_exists.return_value = 2
        op = SimpleNamespace(
            token_ids=[1, 2],
            last_hash=None,
            prefix_keys=None,
            request_id="r-1",
            pool_transfers=None,
            pool_storage_result=MagicMock(),
            assume_stored=False,
        )
        with (
            mock.patch.object(
                hybrid_cc_module,
                "get_thread_caller_info",
                return_value=("sglang-tp1", "Prefetch"),
            ),
            mock.patch.object(
                hybrid_cc_module, "get_storage_hash_str", return_value=["h0", "h1"]
            ),
        ):
            ctrl._storage_hit_query(op)

        extra = ctrl.storage_backend.batch_exists.call_args[0][1]
        self.assertEqual(
            extra.extra_info,
            {
                "caller_id": "sglang-tp1",
                "caller_role": "Prefetch",
                "request_id": "r-1",
            },
        )

    def test_storage_hit_query_forwards_exported_trace_fields(self):
        ctrl = _hybrid_ctrl()
        ctrl.storage_backend.batch_exists.return_value = 2
        op = SimpleNamespace(
            token_ids=[1, 2],
            last_hash=None,
            prefix_keys=None,
            request_id="r-1",
            pool_transfers=None,
            pool_storage_result=MagicMock(),
            assume_stored=False,
            trace_ctx=object(),
            trace_id="0" * 31 + "1",
            span_id="0" * 15 + "2",
        )
        with (
            mock.patch.object(
                hybrid_cc_module,
                "get_thread_caller_info",
                return_value=("sglang-tp1", "Prefetch"),
            ),
            mock.patch.object(
                hybrid_cc_module, "get_storage_hash_str", return_value=["h0", "h1"]
            ),
        ):
            ctrl._storage_hit_query(op)

        extra = ctrl.storage_backend.batch_exists.call_args[0][1]
        self.assertEqual(
            extra.extra_info,
            {
                "caller_id": "sglang-tp1",
                "caller_role": "Prefetch",
                "request_id": "r-1",
                "trace_id": "0" * 31 + "1",
                "span_id": "0" * 15 + "2",
            },
        )

    def test_page_transfer_kv_batch_forwards_request_id_to_kv_and_sidecar(self):
        ctrl = _hybrid_ctrl(page_size=2)
        ctrl.page_get_func = MagicMock(return_value=3)
        ctrl.storage_backend.batch_get_v2.return_value = {"indexer": [True, True, True]}

        transfers = [PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)]
        op = SimpleNamespace(
            request_id="r-1",
            trace_ctx=object(),
        )
        ctrl._page_transfer_kv_batch(
            op,
            batch_hashes=["h0", "h1", "h2"],
            batch_host_indices=list(range(6)),
            extra_info=HiCacheStorageExtraInfo(prefix_keys=["p0"]),
            kv_derived_transfers=transfers,
        )

        page_get_extra = ctrl.page_get_func.call_args[0][3]
        self.assertEqual(page_get_extra.prefix_keys, ["p0"])
        self.assertEqual(page_get_extra.extra_info, {"request_id": "r-1"})
        sidecar_extra = ctrl.storage_backend.batch_get_v2.call_args.kwargs["extra_info"]
        self.assertEqual(sidecar_extra.extra_info, {"request_id": "r-1"})

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
            prefix_keys=None,
            sidecar_hash_values=None,
            sidecar_hit_pages=0,
        )
        with mock.patch.object(
            hybrid_cc_module, "get_thread_caller_info", return_value=None
        ):
            ctrl._page_transfer_sidecar(op, kv_completed_pages=2)

        sidecar_extra = ctrl.storage_backend.batch_get_v2.call_args.kwargs["extra_info"]
        self.assertEqual(sidecar_extra.extra_info, {"request_id": "r-1"})
        self.assertIsNone(sidecar_extra.prefix_keys)
        ctrl._sync_trailing_keys.assert_called_once()
        ctrl._resolve_sidecar_nonkv_derived_pool_transfers.assert_called_once_with(op)
        ack = ctrl.prefetch_sync_queue.put.call_args[0][0]
        self.assertIsInstance(ack, PrefetchAck)
        self.assertEqual(ack.rid, "r-1")
        self.assertEqual(ack.pool_hits, {"swa": 2})

    def test_page_backup_carries_caller_attribution_not_request_id(self):
        ctrl = _hybrid_ctrl(page_size=2)
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
            id=7,
            trace_ctx=object(),
        )
        with mock.patch.object(
            hybrid_cc_module,
            "get_thread_caller_info",
            return_value=("sglang-tp0", "Backup"),
        ):
            ctrl._page_backup_kv_with_trace(op)

        self.assertEqual(
            captured["extra"].extra_info,
            {"caller_id": "sglang-tp0", "caller_role": "Backup"},
        )
        self.assertNotIn("request_id", captured["extra"].extra_info or {})
        self.assertEqual(op.completed_tokens, 6)

    def test_page_backup_forwards_trace_fields_when_exported(self):
        ctrl = _hybrid_ctrl(page_size=2)
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
            trace_ctx=object(),
            trace_id="0" * 31 + "1",
            span_id="0" * 15 + "2",
        )
        with mock.patch.object(
            hybrid_cc_module,
            "get_thread_caller_info",
            return_value=("sglang-tp0", "Backup"),
        ):
            ctrl._page_backup_kv_with_trace(op)

        self.assertEqual(
            captured["extra"].extra_info,
            {
                "caller_id": "sglang-tp0",
                "caller_role": "Backup",
                "trace_id": "0" * 31 + "1",
                "span_id": "0" * 15 + "2",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
