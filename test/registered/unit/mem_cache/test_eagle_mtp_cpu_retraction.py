import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import discard_kv_cache_backup
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def make_pool(base, *, dsa=False):
    # Real MLA/DSA CPU-copy methods over small CPU tensors. No GPU allocation
    # or kernels; only device synchronization is patched by the test.
    pool = object.__new__(DSATokenToKVPool if dsa else MLATokenToKVPool)
    pool.layer_num = 2
    pool.page_size = 1
    pool.cpu_offloading_chunk_size = 7
    pool.kv_buffer = [
        (torch.arange(96 * 8, dtype=torch.float32) + base + layer * 10000).view(
            96, 1, 8
        )
        for layer in range(pool.layer_num)
    ]
    if dsa:
        pool.kpool_use_compress = False
        pool.index_key_cache = object.__new__(IndexKeyCache)
        pool.index_key_cache.pool = pool
        # Includes packed index-key and scale bytes, not just the MLA KV.
        pool.index_key_cache.buffer = [
            (torch.arange(192 * 132).view(192, 132) + base + layer).to(torch.uint8)
            for layer in range(pool.layer_num)
        ]
    return pool


def make_req(indices):
    req = object.__new__(Req)
    req.kv = ReqKvInfo(req_pool_idx=0)
    req.origin_input_ids = [1] * (len(indices) + 1)
    req.output_ids = []
    req_pool = SimpleNamespace(req_to_token=indices.unsqueeze(0).clone())
    return req, req_pool


class TestEagleMtpCpuRetraction(unittest.TestCase):
    def setUp(self):
        cm = get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        sync = patch("torch.cuda.synchronize")
        sync.start()
        self.addCleanup(sync.stop)

    def test_poison_and_relocate_restores_every_draft_and_dsa_indexer(self):
        for dcp_size, rank, draft_count in ((1, 0, 1), (1, 0, 3), (2, 1, 2)):
            with self.subTest(dcp_size=dcp_size, draft_count=draft_count):
                with get_parallel().override(
                    attn_dcp_size=dcp_size, attn_dcp_rank=rank
                ):
                    target = make_pool(100)
                    drafts = tuple(
                        make_pool(1000 * (i + 1), dsa=True) for i in range(draft_count)
                    )
                    allocator = TokenToKVPoolAllocator(
                        192, torch.float32, "cpu", target, need_sort=False
                    )
                    allocator.register_cpu_retraction_draft_pools(drafts)
                    # DCP copy APIs consume page-aligned logical runs.
                    old = torch.arange(2, 21)
                    new = old + 64
                    req, req_pool = make_req(old)
                    owned = old.remainder(dcp_size) == rank
                    source_rows = old[owned] // dcp_size
                    dest_rows = new[owned] // dcp_size
                    expected = [
                        [buf[source_rows].clone() for buf in pool.kv_buffer]
                        for pool in (target, *drafts)
                    ]
                    expected_index = [
                        [buf[old].clone() for buf in pool.index_key_cache.buffer]
                        for pool in drafts
                    ]
                    req.offload_kv_cache(req_pool, allocator)
                    self.assertEqual(
                        len(req.kv.retraction_backup.draft_cpu_tensors), draft_count
                    )
                    for pool in (target, *drafts):
                        for buf in pool.kv_buffer:
                            buf.fill_(-100)
                    for pool in drafts:
                        for buf in pool.index_key_cache.buffer:
                            buf.fill_(0)
                    req_pool.req_to_token = new.unsqueeze(0)
                    req.load_kv_cache(req_pool, allocator)
                    self.assertIsNone(req.kv.retraction_backup)
                    for pool, saved in zip((target, *drafts), expected, strict=True):
                        for buf, value in zip(pool.kv_buffer, saved, strict=True):
                            torch.testing.assert_close(buf[dest_rows], value)
                            # Restoring must not mutate someone else's rows.
                            untouched = torch.ones(buf.shape[0], dtype=torch.bool)
                            untouched[dest_rows] = False
                            self.assertTrue(torch.all(buf[untouched] == -100))
                    for pool, saved in zip(drafts, expected_index, strict=True):
                        for buf, value in zip(
                            pool.index_key_cache.buffer, saved, strict=True
                        ):
                            torch.testing.assert_close(buf[new], value)

    def test_registration_deduplicates_shared_and_target_pools(self):
        target, draft = make_pool(100), make_pool(200)
        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator._kvcache = target
        allocator.register_cpu_retraction_draft_pools((target, draft, draft))
        self.assertEqual(allocator.cpu_retraction_draft_pools, (draft,))

    def test_unregistered_draft_negative_control_loses_history(self):
        target, draft = make_pool(100), make_pool(200, dsa=True)
        allocator = TokenToKVPoolAllocator(
            96, torch.float32, "cpu", target, need_sort=False
        )
        old = torch.arange(2, 10)
        new = old + 64
        req, req_pool = make_req(old)
        expected = draft.kv_buffer[0][old].clone()
        req.offload_kv_cache(req_pool, allocator)
        self.assertEqual(req.kv.retraction_backup.draft_cpu_tensors, ())
        # The baseline only copies target state. Reused draft slots retain
        # unrelated history after target-only restore at new locations.
        draft.kv_buffer[0].fill_(-100)
        req_pool.req_to_token = new.unsqueeze(0)
        req.load_kv_cache(req_pool, allocator)
        self.assertFalse(torch.equal(draft.kv_buffer[0][new], expected))

    def test_no_draft_keeps_existing_cpu_backup(self):
        allocator = TokenToKVPoolAllocator(
            96, torch.float32, "cpu", make_pool(100), need_sort=False
        )
        req, req_pool = make_req(torch.arange(1, 9))
        req.offload_kv_cache(req_pool, allocator)
        self.assertEqual(req.kv.retraction_backup.draft_cpu_tensors, ())
        req.load_kv_cache(req_pool, allocator)
        self.assertIsNone(req.kv.retraction_backup)

    def test_discard_releases_request_owned_draft_snapshots(self):
        target, draft = make_pool(100), make_pool(200, dsa=True)
        allocator = TokenToKVPoolAllocator(
            96, torch.float32, "cpu", target, need_sort=False
        )
        allocator.register_cpu_retraction_draft_pools((draft,))
        req, req_pool = make_req(torch.arange(1, 9))
        req.offload_kv_cache(req_pool, allocator)
        saved = weakref.ref(req.kv.retraction_backup.draft_cpu_tensors[0]["kv"][0][0])
        discard_kv_cache_backup(req, None, "cpu_tensor")
        gc.collect()
        self.assertIsNone(req.kv.retraction_backup)
        self.assertIsNone(saved())
        # Idempotent discard: abort/teardown can share this cleanup path.
        discard_kv_cache_backup(req, None, "cpu_tensor")

    def test_failed_restore_retains_backup_for_cleanup(self):
        target, draft = make_pool(100), make_pool(200)
        allocator = TokenToKVPoolAllocator(
            96, torch.float32, "cpu", target, need_sort=False
        )
        allocator.register_cpu_retraction_draft_pools((draft,))
        req, req_pool = make_req(torch.arange(1, 9))
        req.offload_kv_cache(req_pool, allocator)
        with patch.object(draft, "load_cpu_copy", side_effect=RuntimeError("restore")):
            with self.assertRaisesRegex(RuntimeError, "restore"):
                req.load_kv_cache(req_pool, allocator)
        self.assertIsNotNone(req.kv.retraction_backup)
        discard_kv_cache_backup(req, None, "cpu_tensor")
        self.assertIsNone(req.kv.retraction_backup)

    def test_class_default_does_not_require_base_constructor(self):
        self.assertEqual(BaseTokenToKVPoolAllocator.cpu_retraction_draft_pools, ())


class TestEagleCpuRetractionRegistration(unittest.TestCase):
    def test_registration_after_allocation_is_independent_of_hicache(self):
        for algorithm, unified, count, registered in (
            (SpeculativeAlgorithm.EAGLE, False, 1, True),
            (SpeculativeAlgorithm.EAGLE, False, 3, True),
            (SpeculativeAlgorithm.EAGLE3, False, 1, True),
            (SpeculativeAlgorithm.EAGLE, True, 1, False),
            (SpeculativeAlgorithm.DFLASH, False, 1, False),
        ):
            with self.subTest(algorithm=algorithm, unified=unified, count=count):
                pools = tuple(object() for _ in range(count))
                worker = BaseSpecWorker()
                worker._target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(spec_algorithm=algorithm)
                )
                worker._draft_worker = SimpleNamespace(
                    alloc_memory_pool=Mock(),
                    draft_runners=[
                        SimpleNamespace(token_to_kv_pool=pool) for pool in pools
                    ],
                )
                allocator = Mock()
                with patch(
                    "sglang.srt.speculative.base_spec_worker.get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=False,
                        enable_unified_memory=unified,
                    ),
                ):
                    worker.alloc_memory_pool(
                        req_to_token_pool=object(),
                        token_to_kv_pool_allocator=allocator,
                    )
                worker.draft_worker.alloc_memory_pool.assert_called_once()
                if registered:
                    allocator.register_cpu_retraction_draft_pools.assert_called_once_with(
                        pools
                    )
                else:
                    allocator.register_cpu_retraction_draft_pools.assert_not_called()


if __name__ == "__main__":
    unittest.main()
