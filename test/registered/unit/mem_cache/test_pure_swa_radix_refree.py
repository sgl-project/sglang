"""Regression test: PureSWARadixCache must not re-cache window-evicted KV.

Bug: on a non-prefill-aware all-SWA model, ``req.swa_evict_floor`` stays 0, so
``PureSWARadixCache.cache_finished_req`` took its ``else`` branch and set
``insert_end = keys_len``, inserting the ENTIRE prefix [0, keys_len). But during
decode ``free_swa_out_of_window_slots`` already freed the out-of-window range
[cache_protected_len, swa_evicted_seqlen). Re-inserting those freed slots made
``match_prefix`` hand them out as a "cache hit" (stale KV -- the single-pool
``PureSWATokenToKVPoolAllocator`` recycles ``free``/``free_swa`` from the same
pool) and made ``evict()`` free them a second time (double-free).

Fix: mirror decode's protection boundary -- when a freed hole exists
(swa_evicted_seqlen > cache_protected_len) only [0, cache_protected_len) is a
live contiguous prefix, so cache no further. The still-in-window tail is freed
back to the allocator as before.

Uses real tree/allocator/pool with a mock Req, mirroring
test_swa_eviction_boundary.py.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _build_pure_swa_tree(sliding_window_size, kv_size_swa=512):
    """All-SWA model: every layer is sliding-window; page_size == 1."""
    head_num, head_dim, num_layers = 8, 128, 8
    dtype = torch.bfloat16
    device = get_device()
    pool = ReqToTokenPool(
        size=8, max_context_len=2048, device=device, enable_memory_saver=False
    )
    kv_pool = SWAKVPool(
        size=0,
        size_swa=kv_size_swa,
        page_size=1,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=list(range(num_layers)),
        full_attention_layer_ids=[],
        device=device,
    )
    allocator = PureSWATokenToKVPoolAllocator(
        kv_size_swa,
        page_size=1,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = PureSWARadixCache(
        params=CacheInitParams(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            is_eagle=False,
            sliding_window_size=sliding_window_size,
        ),
    )
    return tree, allocator, pool


def _make_req(req_pool_idx, token_ids, tree):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        origin_input_ids=token_ids,
        output_ids=[],
        cache_protected_len=0,
        swa_evict_floor=0,  # non-prefill-aware all-SWA model
        kv=SimpleNamespace(swa_evicted_seqlen=0),
        extra_key=None,
        last_node=tree.root_node,
    )


class TestPureSWARadixRefree(unittest.TestCase):
    def _decode_evict(self, tree, allocator, pool, req, seq_len, window):
        """Drive the real decode-time SWA eviction (frees out-of-window slots)."""
        free_swa_out_of_window_slots(
            req,
            seq_len - 1,
            sliding_window_size=window,
            page_size=1,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
        )

    def test_window_evicted_prefix_not_refree_or_realiased(self):
        window = 4
        tree, allocator, pool = _build_pure_swa_tree(window)
        cap = allocator.swa_available_size()  # free capacity (robust to reserved slots)

        seq_len = 32  # >> window, so decode evicts an out-of-window prefix
        kv = allocator.alloc(seq_len)
        self.assertIsNotNone(kv)
        pool.write((0, slice(0, seq_len)), kv)

        req = _make_req(0, list(range(seq_len)), tree)
        self._decode_evict(tree, allocator, pool, req, seq_len, window)
        evicted = req.kv.swa_evicted_seqlen
        self.assertGreater(evicted, 0, "decode should have window-evicted a prefix")

        tree.cache_finished_req(req, is_insert=True, kv_len_to_handle=seq_len)

        # The freed out-of-window prefix must NOT be re-cached: with a hole, no
        # contiguous prefix survives (cache_protected_len == 0), so nothing new
        # is evictable, and no slot is both allocator-free and tree-referenced.
        self.assertEqual(
            tree.evictable_size(),
            0,
            "window-evicted prefix must not be re-inserted into the tree",
        )
        # Conservation: a slot double-counted (free-list AND tree) would push the
        # sum above the pool capacity -- exactly the aliasing/double-free bug.
        self.assertEqual(
            allocator.swa_available_size() + tree.evictable_size(),
            cap,
            "every slot must be either free or tree-held, never both",
        )

    def test_deterministic_mode_no_double_free(self):
        # is_insert=False (deterministic mode) took the same buggy free_end path.
        window = 4
        tree, allocator, pool = _build_pure_swa_tree(window)
        cap = allocator.swa_available_size()

        seq_len = 32
        kv = allocator.alloc(seq_len)
        pool.write((0, slice(0, seq_len)), kv)
        req = _make_req(0, list(range(seq_len)), tree)
        self._decode_evict(tree, allocator, pool, req, seq_len, window)
        self.assertGreater(req.kv.swa_evicted_seqlen, 0)

        tree.cache_finished_req(req, is_insert=False, kv_len_to_handle=seq_len)

        # Nothing cached; all slots freed exactly once (a double-free would make
        # available exceed capacity).
        self.assertEqual(tree.evictable_size(), 0)
        self.assertEqual(allocator.swa_available_size(), cap)

    def test_short_request_still_fully_cached(self):
        # Positive control (no caching regression): a request that never leaves
        # the window (swa_evicted_seqlen stays 0) must still cache its whole prefix.
        window = 8
        tree, allocator, pool = _build_pure_swa_tree(window)
        cap = allocator.swa_available_size()

        seq_len = 5  # < window -> decode evicts nothing
        kv = allocator.alloc(seq_len)
        pool.write((0, slice(0, seq_len)), kv)
        req = _make_req(0, list(range(seq_len)), tree)
        self._decode_evict(tree, allocator, pool, req, seq_len, window)
        self.assertEqual(req.kv.swa_evicted_seqlen, 0)

        tree.cache_finished_req(req, is_insert=True, kv_len_to_handle=seq_len)

        self.assertEqual(
            tree.evictable_size(), seq_len, "full prefix must remain cached"
        )
        self.assertEqual(allocator.swa_available_size() + tree.evictable_size(), cap)


if __name__ == "__main__":
    unittest.main()
