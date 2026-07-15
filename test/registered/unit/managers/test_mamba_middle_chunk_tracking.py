import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import alloc_req_slots
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeMambaAllocator:
    def __init__(self, slots):
        self.free_slots = list(slots)
        self.alloc_calls = []

    def alloc(self, n):
        self.alloc_calls.append(n)
        if n > len(self.free_slots):
            return None
        slots = self.free_slots[:n]
        self.free_slots = self.free_slots[n:]
        return torch.tensor(slots, dtype=torch.int64)

    def schedulable_available_size(self):
        return len(self.free_slots)


class _FakeMambaTreeCache:
    def __init__(self):
        self.evicted_mamba = []

    def supports_mamba(self):
        return True

    def evict(self, params):
        self.evicted_mamba.append(params.mamba_num)


class TestMambaMiddleChunkTracking(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(
            SimpleNamespace(
                mamba_cache_chunk_size=4,
                enable_mamba_extra_buffer_lazy=lambda: False,
            )
        )

    def _make_batch(self):
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.req_to_token_pool = SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx
        )
        return batch

    def _make_req(self, *, inflight_middle_chunks: int):
        return SimpleNamespace(
            req_pool_idx=None,
            kv_committed_len=0,
            prefix_indices=torch.arange(8, dtype=torch.int64),
            extend_range=Range(8, 16),
            inflight_middle_chunks=inflight_middle_chunks,
            mamba_pool_idx=None,
            mamba_needs_clear=False,
            mamba_ping_pong_track_buffer=torch.tensor([11, 22], dtype=torch.int64),
            mamba_next_track_idx=0,
            mamba_branching_seqlen=None,
            mamba_last_track_seqlen=None,
        )

    def test_middle_chunk_skips_mamba_tracking(self):
        batch = self._make_batch()
        req = self._make_req(inflight_middle_chunks=1)
        req.mamba_ping_pong_track_buffer = None

        track_entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)

        self.assertFalse(track_entry.track_mask)
        self.assertEqual(track_entry.track_index, 0)
        self.assertEqual(track_entry.track_seqlen, -1)
        self.assertEqual(req.mamba_next_track_idx, 0)
        self.assertIsNone(req.mamba_last_track_seqlen)

    def test_final_chunk_still_tracks_mamba_state(self):
        batch = self._make_batch()
        req = self._make_req(inflight_middle_chunks=0)

        track_entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)

        self.assertTrue(track_entry.track_mask)
        self.assertEqual(track_entry.track_index, 11)
        self.assertEqual(track_entry.track_seqlen, 16)
        self.assertEqual(req.mamba_next_track_idx, 1)
        self.assertEqual(req.mamba_last_track_seqlen, 16)

    def _make_req_to_token_pool(self, slots):
        pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        pool.free_slots = [1]
        pool.mamba_allocator = _FakeMambaAllocator(slots)
        pool.mamba_pool = SimpleNamespace(size=16, replayssm_write_pos=None)
        pool.enable_mamba_extra_buffer = True
        pool.enable_mamba_extra_buffer_lazy = False
        pool.mamba_ping_pong_track_buffer_size = 2
        pool.req_index_to_mamba_index_mapping = torch.zeros(2, dtype=torch.int32)
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            (2, 2), dtype=torch.int64
        )
        return pool

    def test_middle_chunk_does_not_allocate_ping_pong_slots(self):
        pool = self._make_req_to_token_pool([101, 102, 103])
        req = self._make_req(inflight_middle_chunks=1)
        req.mamba_ping_pong_track_buffer = None

        req_pool_indices = pool.alloc([req])

        self.assertEqual(req_pool_indices, [1])
        self.assertEqual(req.mamba_pool_idx.item(), 101)
        self.assertEqual(pool.mamba_allocator.alloc_calls, [1])
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertTrue(
            torch.equal(
                pool.req_index_to_mamba_ping_pong_track_buffer_mapping[1],
                torch.tensor([-1, -1], dtype=torch.int64),
            )
        )

    def test_final_chunk_allocates_ping_pong_slots_after_middle_skip(self):
        pool = self._make_req_to_token_pool([101, 102, 103])
        req = self._make_req(inflight_middle_chunks=1)
        req.mamba_ping_pong_track_buffer = None
        pool.alloc([req])

        req.inflight_middle_chunks = 0
        req.kv_committed_len = 8
        req_pool_indices = pool.alloc([req])

        self.assertEqual(req_pool_indices, [1])
        self.assertEqual(pool.mamba_allocator.alloc_calls, [1, 2])
        self.assertTrue(
            torch.equal(req.mamba_ping_pong_track_buffer, torch.tensor([102, 103]))
        )
        self.assertEqual(req.mamba_next_track_idx, 0)

    def test_middle_chunk_headroom_excludes_ping_pong_slots(self):
        pool = self._make_req_to_token_pool([101])
        req = self._make_req(inflight_middle_chunks=1)
        req.mamba_ping_pong_track_buffer = None
        tree_cache = _FakeMambaTreeCache()

        alloc_req_slots(pool, [req], tree_cache)

        self.assertEqual(tree_cache.evicted_mamba, [])

    def test_final_chunk_headroom_keeps_ping_pong_slots(self):
        # For a final (non-middle) chunk with prefix cache + extra buffer,
        # factor = MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3 (1 pool slot + 2
        # ping-pong slots).  With 5 mamba slots available and 3 needed,
        # no eviction is required.
        pool = self._make_req_to_token_pool([101, 102, 103, 104, 105])
        req = self._make_req(inflight_middle_chunks=0)
        req.mamba_ping_pong_track_buffer = None
        tree_cache = _FakeMambaTreeCache()

        alloc_req_slots(pool, [req], tree_cache)

        # mamba_state_needed=3, available=5 → no eviction needed
        self.assertEqual(tree_cache.evicted_mamba, [])


if __name__ == "__main__":
    unittest.main()
