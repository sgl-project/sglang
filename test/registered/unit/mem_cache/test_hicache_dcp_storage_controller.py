"""CPU controller/file tests for DCP backup ownership and acknowledgment lifetime.

Run: python test/registered/unit/mem_cache/test_hicache_dcp_storage_controller.py -v
"""

import json
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _AckQueue(Queue):
    def __init__(self):
        super().__init__()
        self.ready = threading.Event()
        self.operations = []

    def put(self, operation, *args, **kwargs):
        super().put(operation, *args, **kwargs)
        self.operations.append(operation)
        self.ready.set()


def _parallel(rank, dcp_size):
    return SimpleNamespace(
        tp_rank=rank,
        tp_size=4,
        pp_rank=0,
        pp_size=1,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=rank % dcp_size,
    )


def _controller(rank, dcp_size=2, is_mla=True):
    # GPU allocation and process-group creation are outside this component test.
    device = SimpleNamespace(
        size=256,
        host_capacity_tokens=None,
        store_dtype=torch.bfloat16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        start_layer=0,
        end_layer=1,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )
    host = MLATokenToKVPoolHost(
        device,
        host_to_device_ratio=2,
        host_size=0,
        page_size=64 * dcp_size,
        layout="page_first",
        pin_memory=False,
        device="cpu",
        dcp_size=dcp_size,
        dcp_rank=rank % dcp_size,
    )
    cc = HiCacheController.__new__(HiCacheController)
    cc.mem_pool_device = (
        MLATokenToKVPool.__new__(MLATokenToKVPool) if is_mla else object()
    )
    cc.mem_pool_host = cc.storage_host_pool = host
    cc.page_size = host.logical_page_size
    cc.attn_cp_group = None
    cc.enable_storage_metrics = False
    cc.enable_storage = False
    cc.storage_stop_event = threading.Event()
    cc.backup_queue = Queue()
    cc.ack_backup_queue = _AckQueue()
    cc.prefetch_hit_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.host_mem_release_queue = Queue()
    with (
        mock.patch(
            "sglang.srt.managers.cache_controller.get_parallel",
            return_value=_parallel(rank, dcp_size),
        ),
        mock.patch(
            "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
            return_value=False,
        ),
    ):
        cc.storage_config = cc._generate_storage_config(
            "controller-test",
            {
                "max_size": "3072",
                "min_free_space": "0",
                "eviction_ratio": 1.0,
                "enable_metadata_cache": False,
            },
        )
    cc.storage_backend = HiCacheFile(cc.storage_config)
    cc.page_set_func = cc._generic_page_set
    return cc


class TestDcpStorageController(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(self.directory)
        )

    def start_worker(self, cc):
        worker = threading.Thread(target=cc.backup_thread_func, daemon=True)
        worker.start()

        def stop():
            cc.storage_stop_event.set()
            cc.backup_queue.put(None)
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(), "backup worker did not stop")

        self.stack.callback(stop)

    @staticmethod
    def drain_backup(cache):
        cache._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=None,
            n_release=0,
            extra_release_counts=None,
            log_metrics=False,
        )

    def test_one_writer_per_shard_and_acknowledgment_lifetime(self):
        controllers = [_controller(rank) for rank in range(4)]
        entered = threading.Event()
        release = threading.Event()
        delayed_backend = controllers[1].storage_backend
        original_set = delayed_backend.batch_set

        def delayed_set(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=15):
                raise TimeoutError("test did not release the delayed writer")
            return original_set(*args, **kwargs)

        caches = []
        indices_by_rank = []
        for rank, cc in enumerate(controllers):
            pool = cc.mem_pool_host
            self.assertEqual(cc.storage_config.logical_page_size, 128)
            self.assertEqual(cc.storage_config.kv_cache_dtype, torch.bfloat16)
            self.assertEqual(cc.storage_config.host_layout, "page_first")
            pool.kv_buffer.fill_(rank % 2 + 1)
            indices = pool.alloc(128)
            indices_by_rank.append(indices)
            cc.storage_backend.batch_set = mock.Mock(
                wraps=delayed_set if rank == 1 else cc.storage_backend.batch_set
            )
            operation_id = cc.write_storage(
                indices, list(range(128)), hash_value=["page-a"]
            )
            cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
            cache.cache_controller = cc
            cache.host_memory_mode = "cache"
            cache.ongoing_backup = {operation_id: (rank, indices)}
            # Model reclaiming a page after its tree lock is released. Queue
            # draining is real; the tree's lock callback reclaims a real host allocation.
            cache.dec_host_lock_ref = mock.Mock(
                side_effect=lambda node, slots, pool=pool: pool.free(slots)
            )
            caches.append(cache)
            self.start_worker(cc)
        # Cleanup must release a blocked writer before joining its thread.
        self.stack.callback(release.set)

        self.assertTrue(entered.wait(timeout=10))
        for rank in (0, 2, 3):
            self.assertTrue(controllers[rank].ack_backup_queue.ready.wait(timeout=10))
        self.assertEqual(
            [len(cc.ack_backup_queue.operations) for cc in controllers], [1, 0, 1, 1]
        )
        self.drain_backup(caches[1])
        caches[1].dec_host_lock_ref.assert_not_called()
        self.assertTrue(
            controllers[1].mem_pool_host.slot_used[indices_by_rank[1]].all()
        )

        release.set()
        self.assertTrue(controllers[1].ack_backup_queue.ready.wait(timeout=10))
        self.assertEqual(len(list(Path(self.directory).glob("*.bin"))), 2)
        self.assertEqual(
            [cc.storage_backend.batch_set.call_count for cc in controllers],
            [1, 1, 0, 0],
        )
        self.assertEqual(
            [cc.ack_backup_queue.operations[0].completed_tokens for cc in controllers],
            [128, 128, 0, 0],
        )

        report = []
        for rank, (cc, cache) in enumerate(zip(controllers, caches)):
            self.drain_backup(cache)
            cache.dec_host_lock_ref.assert_called_once()
            self.assertFalse(cache.ongoing_backup)
            self.assertTrue(cc.ack_backup_queue.empty())
            pool = cc.mem_pool_host
            self.assertEqual(pool.available_size(), pool.logical_size)

            # Every rank reads its shard into a new allocation, including replicas.
            target = pool.alloc(128)
            operation = PrefetchOperation(f"read-{rank}", list(range(128)))
            pool.set_from_flat_data_page(
                int(target[0]), torch.full_like(pool.get_dummy_flat_data_page(), -1)
            )
            self.assertEqual(cc._generic_page_get(operation, ["page-a"], target), 1)
            torch.testing.assert_close(
                pool.get_data_page(int(target[0])),
                torch.full_like(pool.get_dummy_flat_data_page(), rank % 2 + 1),
            )
            pool.free(target)
            self.assertEqual(pool.available_size(), pool.logical_size)
            self.assertEqual(int(pool.slot_used.sum()), 0)
            evictor = cc.storage_backend._evictor
            self.assertEqual(evictor.is_storage_owner, rank < 2)
            self.assertEqual(evictor._total_bytes, 3072 if rank < 2 else 0)
            report.append(
                {
                    "rank": rank,
                    "writer": cc.storage_config.is_storage_writer,
                    "writes": cc.storage_backend.batch_set.call_count,
                    "acks": len(cc.ack_backup_queue.operations),
                    "remaining_slots": int(pool.slot_used.sum()),
                }
            )
        print("DCP_BACKUP_REPORT=" + json.dumps(report))

    def test_eviction_is_owned_by_both_shard_writers(self):
        controllers = [_controller(rank) for rank in range(4)]
        payload = torch.ones(1536, dtype=torch.bfloat16)
        for cc in controllers[:2]:
            self.assertTrue(cc.storage_backend.set("page-a", payload))
        for cc in controllers[2:]:
            self.assertFalse(cc.storage_backend.set("page-b", payload))
            self.assertEqual(cc.storage_backend._evictor._total_bytes, 0)
        self.assertTrue(controllers[0].storage_backend.set("page-b", payload))
        self.assertFalse(controllers[0].storage_backend.exists("page-a"))
        self.assertTrue(controllers[1].storage_backend.exists("page-a"))
        self.assertTrue(controllers[1].storage_backend.set("page-b", payload))
        self.assertFalse(controllers[1].storage_backend.exists("page-a"))
        self.assertEqual(len(list(Path(self.directory).glob("*.bin"))), 2)
        self.assertEqual(
            [cc.storage_backend._evictor._total_bytes for cc in controllers],
            [3072, 3072, 0, 0],
        )

    def test_dcp_one_preserves_controller_writer_rules(self):
        for rank in range(4):
            for is_mla in (True, False):
                with self.subTest(rank=rank, is_mla=is_mla):
                    cc = _controller(rank, dcp_size=1, is_mla=is_mla)
                    self.assertEqual(cc.backup_skip, is_mla and rank != 0)
                    self.assertEqual(
                        cc.storage_backend._evictor.is_storage_owner, not cc.backup_skip
                    )

    def test_runtime_attach_rejects_dcp_before_side_effects(self):
        cc = HiCacheController.__new__(HiCacheController)
        cc.enable_storage = False
        cc._stop_storage_threads = mock.Mock()
        cc._start_storage_threads = mock.Mock()
        cc._generate_storage_config = mock.Mock()
        for dcp_size in (2, 4):
            with (
                self.subTest(dcp_size=dcp_size),
                mock.patch(
                    "sglang.srt.managers.cache_controller.get_parallel",
                    return_value=_parallel(0, dcp_size),
                ),
                self.assertRaisesRegex(NotImplementedError, "runtime attachment"),
            ):
                cc.attach_storage_backend("file")
        cc._stop_storage_threads.assert_not_called()
        cc._start_storage_threads.assert_not_called()
        cc._generate_storage_config.assert_not_called()
        self.assertFalse(cc.enable_storage)


if __name__ == "__main__":
    unittest.main()
