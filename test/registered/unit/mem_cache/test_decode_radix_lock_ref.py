"""
Unit tests for lock_ref correctness in decode disagg radix cache scenarios.

Verifies that inc_lock_ref / dec_lock_ref are balanced across the four
transfer scenarios identified in PR #19746:

1. Incremental transfer & success (prefix match > 0)
   inc_lock_ref(pop_preallocated) -> dec+inc(cache_unfinished_req) -> dec(cache_finished_req)

2. Full transfer & success (prefix match == 0, full KV transferred)
   inc_lock_ref(get_new_prebuilt_batch) -> dec+inc(cache_unfinished_req) -> dec(cache_finished_req)

3. Incremental transfer & failure (prefix match > 0, transfer fails)
   inc_lock_ref(pop_preallocated) -> dec(cache_finished_req via release_kv_cache is_insert=False)

4. Full transfer & failure (prefix match == 0, transfer fails)
   no inc_lock_ref -> dec(root_node) is no-op since root lock_ref starts at 1

Additionally tests mamba-aware decode radix cache scenarios:

5. skip_mamba_match: prefix hit through split nodes where mamba_value=None
6. Mamba budget gate in pop_preallocated
7. Lock_ref balance with MambaRadixCache (incremental / full transfer)

Usage:
    python -m pytest test/registered/unit/mem_cache/test_decode_radix_lock_ref.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.decode_hicache_mixin import DecodePrefixMatch
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.utils.common import Range


def _make_cache_with_pools(page_size=1):
    """Create a RadixCache with mock pools sufficient for cache_unfinished/finished_req."""
    mock_allocator = MagicMock()
    mock_allocator.device = torch.device("cpu")

    # req_to_token pool: stores kv indices per request slot
    max_seq_len = 64
    max_batch = 4
    req_to_token = torch.zeros(max_batch, max_seq_len, dtype=torch.int64)

    mock_pool = MagicMock()
    mock_pool.req_to_token = req_to_token
    mock_pool.write = lambda idx_tuple, values: req_to_token.__setitem__(
        idx_tuple, values
    )

    cache = RadixCache.create_simulated(
        mock_allocator=mock_allocator, page_size=page_size
    )
    cache.req_to_token_pool = mock_pool
    return cache, req_to_token


class MockReq:
    """Minimal mock Req with fields needed by cache_unfinished/finished_req."""

    def __init__(self, fill_ids, req_pool_idx=0, cache_protected_len=0, last_node=None):
        self.full_untruncated_fill_ids = array("q", fill_ids)
        self.extend_range = Range(0, len(self.full_untruncated_fill_ids))
        self.origin_input_ids = array(
            "q", fill_ids[:-1] if len(fill_ids) > 1 else fill_ids
        )
        self.output_ids = array("q", [fill_ids[-1]] if len(fill_ids) > 1 else [])
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = cache_protected_len
        self.last_node = last_node
        self.extra_key = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0
        self.kv_committed_len = len(fill_ids)
        self.kv_allocated_len = len(fill_ids)
        self.kv_committed_freed = False

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]

    def pop_committed_kv_cache(self):
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self):
        return (self.kv_committed_len, self.kv_allocated_len)


def _make_req(fill_ids, req_pool_idx=0, cache_protected_len=0, last_node=None):
    return MockReq(fill_ids, req_pool_idx, cache_protected_len, last_node)


class TestDecodeLockRefScenarios(unittest.TestCase):
    """Test lock_ref balance across decode transfer scenarios."""

    def _populate_prefix(self, cache, prefix_ids, prefix_values):
        """Insert a prefix into the tree so future requests can match it."""
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", prefix_ids)),
                value=torch.tensor(prefix_values, dtype=torch.int64),
            )
        )

    def test_incremental_transfer_success(self):
        """Scenario 1: prefix match > 0, transfer succeeds.

        Flow: inc_lock_ref(pop_preallocated)
              -> dec_lock_ref + inc_lock_ref(cache_unfinished_req)
              -> dec_lock_ref(cache_finished_req)
        """
        cache, req_to_token = _make_cache_with_pools()

        # Pre-populate a prefix [1,2,3] in the tree
        prefix = [1, 2, 3]
        prefix_vals = [10, 20, 30]
        self._populate_prefix(cache, prefix, prefix_vals)

        # Match prefix (simulates _match_prefix_and_lock in pop_preallocated)
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prefix))))
        matched_node = result.last_device_node
        prefix_len = len(result.device_indices)
        self.assertEqual(prefix_len, 3)

        # Step 1: inc_lock_ref (pop_preallocated locks the matched node)
        cache.inc_lock_ref(matched_node)
        self.assertGreater(matched_node.lock_ref, 0)

        # Simulate _pre_alloc: write prefix + new tokens to req_to_token
        full_ids = [1, 2, 3, 4, 5]  # prefix + 2 new tokens
        full_vals = [10, 20, 30, 40, 50]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=prefix_len,
            last_node=matched_node,
        )

        # Step 2: cache_unfinished_req (dec old lock, inc new lock)
        cache.cache_unfinished_req(req)

        # Step 3: cache_finished_req with is_insert=True (dec lock)
        cache.cache_finished_req(req)

        # Verify: all non-root nodes should have lock_ref == 0
        # (root always has lock_ref == 1)
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), 0)
        # The evictable size should equal total inserted tokens
        self.assertEqual(cache.evictable_size(), len(full_ids))

    def test_full_transfer_success(self):
        """Scenario 2: no prefix match, full KV transferred, succeeds.

        Flow: inc_lock_ref(root, via init_next_round_input/get_new_prebuilt_batch)
              -> dec_lock_ref + inc_lock_ref(cache_unfinished_req)
              -> dec_lock_ref(cache_finished_req)
        """
        cache, req_to_token = _make_cache_with_pools()

        # No prefix in tree -- match returns root
        full_ids = [10, 20, 30]
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", full_ids)))
        )
        matched_node = result.last_device_node
        self.assertEqual(len(result.device_indices), 0)  # no match
        # matched_node is root

        root_lock_before = cache.root_node.lock_ref
        # Step 1: inc_lock_ref on root (simulates get_new_prebuilt_batch)
        # Note: inc/dec_lock_ref skip the root node (while node != root_node),
        # so this is a no-op. Root always keeps lock_ref=1.
        cache.inc_lock_ref(matched_node)
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)  # no-op on root

        # Write full KV to pool
        full_vals = [100, 200, 300]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=matched_node,
        )

        # Step 2: cache_unfinished_req (dec root=no-op, inc new leaf)
        cache.cache_unfinished_req(req)

        # Step 3: cache_finished_req (dec leaf)
        cache.cache_finished_req(req)

        # Root lock unchanged, all nodes unlocked
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), len(full_ids))

    def test_incremental_transfer_failure(self):
        """Scenario 3: prefix match > 0, transfer fails.

        Flow: inc_lock_ref(pop_preallocated)
              -> dec_lock_ref(cache_finished_req via release_kv_cache is_insert=False)
        """
        cache, req_to_token = _make_cache_with_pools()

        # Pre-populate prefix
        prefix = [1, 2, 3]
        prefix_vals = [10, 20, 30]
        self._populate_prefix(cache, prefix, prefix_vals)

        # Match and lock
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prefix))))
        matched_node = result.last_device_node
        prefix_len = len(result.device_indices)

        cache.inc_lock_ref(matched_node)
        # Prefix tokens should now be protected (locked)
        self.assertGreater(cache.protected_size(), 0)

        # Simulate _pre_alloc with additional tokens
        full_ids = [1, 2, 3, 4, 5]
        full_vals = [10, 20, 30, 40, 50]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=prefix_len,
            last_node=matched_node,
        )

        # Transfer fails -> cache_finished_req with is_insert=False
        # This frees delta tokens and dec_lock_ref on last_node
        cache.cache_finished_req(req, is_insert=False)

        # The prefix node should be unlocked (back to evictable)
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), 0)
        # Prefix tokens should still be in tree and evictable
        self.assertEqual(cache.evictable_size(), len(prefix))

    def test_full_transfer_failure(self):
        """Scenario 4: no prefix match, transfer fails.

        Flow: _match_prefix_and_lock sets last_node=root and calls
              inc_lock_ref(root) which is a no-op. On failure,
              cache_finished_req calls dec_lock_ref(root) which is also
              a no-op. Net: balanced.
        """
        cache, req_to_token = _make_cache_with_pools()

        root_lock_before = cache.root_node.lock_ref

        # No prefix in tree -- match returns root (simulates _match_prefix_and_lock)
        full_ids = [10, 20, 30]
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", full_ids)))
        )
        matched_node = result.last_device_node
        self.assertIs(matched_node, cache.root_node)

        # inc_lock_ref(root) is a no-op
        cache.inc_lock_ref(matched_node)
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)

        full_vals = [100, 200, 300]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        # last_node = root (as set by _match_prefix_and_lock)
        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=matched_node,
        )

        # Transfer fails -> cache_finished_req with is_insert=False
        # dec_lock_ref(root) is a no-op
        cache.cache_finished_req(req, is_insert=False)

        # Root lock unchanged, nothing protected or evictable
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)

    def test_pop_preallocated_rechecks_budget_after_lock(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)

        req = MagicMock()
        req.rid = "req-1"
        req.origin_input_ids = list(range(8))
        req.output_ids = [99]
        req.last_node = object()
        req.finished_reason = None
        req.cache_protected_len = 0
        req.sampling_params.max_new_tokens = 16

        decode_req = MagicMock()
        decode_req.req = req
        decode_req.waiting_for_input = True

        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._match_prefix_and_lock = MagicMock(
            return_value=DecodePrefixMatch(
                prefix_indices=torch.arange(4, dtype=torch.int64),
                l2_host_hit_length=0,
                l3_storage_hit_length=0,
                last_device_node=req.last_node,
            )
        )
        queue._pre_alloc = MagicMock(
            side_effect=AssertionError("_pre_alloc should not run")
        )
        queue.transfer_queue = MagicMock(queue=[], enable_staging=False)
        queue.tree_cache = MagicMock()
        queue.tree_cache.dec_lock_ref = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 4

        running_batch = MagicMock()
        running_batch.reqs = []
        server_args = MagicMock()
        server_args.disaggregation_decode_enable_radix_cache = True
        scheduler = MagicMock()
        scheduler.running_batch = running_batch
        scheduler.server_args = server_args
        scheduler.enable_hisparse = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Initial budget says the request fits; post-lock budget says it does not.
        queue._allocatable_token_budgets = MagicMock(side_effect=[8, 3])

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        queue._pre_alloc.assert_not_called()
        queue.tree_cache.dec_lock_ref.assert_called_once_with(req.last_node)
        self.assertEqual(queue._allocatable_token_budgets.call_count, 2)

    def test_repeated_incremental_no_leak(self):
        """Multiple incremental transfers shouldn't leak lock_refs."""
        cache, req_to_token = _make_cache_with_pools()

        prefix = [1, 2, 3]
        prefix_vals = [10, 20, 30]
        self._populate_prefix(cache, prefix, prefix_vals)

        for iteration in range(5):
            result = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", prefix)))
            )
            matched_node = result.last_device_node
            prefix_len = len(result.device_indices)

            cache.inc_lock_ref(matched_node)

            suffix_token = 40 + iteration
            full_ids = prefix + [suffix_token]
            full_vals = prefix_vals + [100 + iteration]
            req_to_token[0, : len(full_vals)] = torch.tensor(
                full_vals, dtype=torch.int64
            )

            req = _make_req(
                fill_ids=full_ids,
                req_pool_idx=0,
                cache_protected_len=prefix_len,
                last_node=matched_node,
            )

            cache.cache_unfinished_req(req)
            cache.cache_finished_req(req)

        # After all iterations, root lock should be 1, no protected nodes
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), 0)


class TestDecodeMambaLockRefScenarios(unittest.TestCase):
    """Test lock_ref balance and skip_mamba_match with MambaRadixCache.

    Covers PD decode scenarios where the mamba state is transferred from
    prefill (RDMA), so the cached mamba_value in the tree is irrelevant
    for computation and matching should be KV-only (skip_mamba_match=True).
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        server_args = ServerArgs(model_path="dummy", page_size=1)
        from sglang.srt.layers.attention.fla.chunk_delta_h import (
            CHUNK_SIZE as FLA_CHUNK_SIZE,
        )

        server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        set_global_server_args_for_scheduler(server_args)

    def _make_mamba_cache(self):
        from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            HybridReqToTokenPool,
        )
        from sglang.srt.utils import get_device

        device = get_device()
        num_layers = 48
        global_interval = 4
        full_attention_layer_ids = list(
            range(global_interval - 1, num_layers, global_interval)
        )
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]

        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=10,
            mamba_size=20,
            mamba_spec_state_size=10,
            max_context_len=128,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=128,
            dtype=torch.bfloat16,
            page_size=1,
            head_num=2,
            head_dim=256,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=128,
            dtype=torch.bfloat16,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
        )
        tree = MambaRadixCache(params=params)
        return tree, allocator, req_to_token_pool

    def _make_mamba_req(self, req_to_token_pool, fill_ids, req_pool_idx=0):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=array("q", fill_ids),
            sampling_params=sampling_params,
        )
        req_to_token_pool.alloc([req])
        req.fill_ids = array("q", fill_ids)
        req.output_ids = array("q", [])
        req.cache_protected_len = 0
        req.extra_key = None
        req.priority = 0
        req.kv_committed_len = len(fill_ids)
        req.kv_allocated_len = len(fill_ids)
        req.kv_committed_freed = False
        req.mamba_last_track_seqlen = None
        return req

    def _insert_with_mamba(self, tree, allocator, token_ids, mamba_slot):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        kv_indices = allocator.alloc(len(token_ids))
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", token_ids)),
                value=kv_indices,
                mamba_value=mamba_slot,
            )
        )
        return kv_indices

    def test_skip_mamba_match_through_split_node(self):
        """After a split, the parent node has mamba_value=None.

        With skip_mamba_match=False, matching stops at the split point.
        With skip_mamba_match=True (PD decode), matching continues through.
        """
        tree, allocator, req_to_token_pool = self._make_mamba_cache()
        mamba_pool = req_to_token_pool.mamba_pool

        mamba_slot1 = mamba_pool.alloc(1)
        self.assertIsNotNone(mamba_slot1)
        self._insert_with_mamba(tree, allocator, [1, 2, 3], mamba_slot1)

        mamba_slot2 = mamba_pool.alloc(1)
        self.assertIsNotNone(mamba_slot2)
        self._insert_with_mamba(tree, allocator, [1, 2, 3, 4, 5], mamba_slot2)

        # Without skip_mamba_match: should match [1,2,3] since that node has mamba_value
        result_normal = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4, 5])),
                skip_mamba_match=False,
            )
        )

        # With skip_mamba_match: should match all 5 tokens
        result_skip = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4, 5])),
                skip_mamba_match=True,
            )
        )

        self.assertGreaterEqual(
            len(result_skip.device_indices), len(result_normal.device_indices)
        )
        self.assertEqual(len(result_skip.device_indices), 5)

    def test_skip_mamba_match_split_by_divergent_suffix(self):
        """Two sequences sharing a prefix cause a split; skip_mamba_match
        lets the shared prefix match even though the split node has no mamba.
        """
        tree, allocator, req_to_token_pool = self._make_mamba_cache()
        mamba_pool = req_to_token_pool.mamba_pool

        mamba_slot1 = mamba_pool.alloc(1)
        self._insert_with_mamba(tree, allocator, [1, 2, 3, 4], mamba_slot1)

        mamba_slot2 = mamba_pool.alloc(1)
        self._insert_with_mamba(tree, allocator, [1, 2, 3, 5], mamba_slot2)

        # Tree after two inserts with divergent suffix at position 3:
        #   root -> [1,2,3](mamba=None, split node) -> [4](mamba=slot1)
        #                                            -> [5](mamba=slot2)

        query = [1, 2, 3, 6]

        result_normal = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", query)), skip_mamba_match=False)
        )
        result_skip = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", query)), skip_mamba_match=True)
        )

        # Without skip: mamba_value=None on [1,2,3] node means 0 match
        self.assertEqual(len(result_normal.device_indices), 0)
        # With skip: KV-only, matches 3 tokens
        self.assertEqual(len(result_skip.device_indices), 3)

    def test_mamba_lock_ref_incremental_transfer_success(self):
        """Mamba cache: prefix match > 0, transfer succeeds.

        Same flow as KV-only but using MambaRadixCache. Verifies that
        full_lock_ref and mamba_lock_ref are balanced.
        """
        tree, allocator, req_to_token_pool = self._make_mamba_cache()
        mamba_pool = req_to_token_pool.mamba_pool

        mamba_slot = mamba_pool.alloc(1)
        prefix = [1, 2, 3]
        prefix_kv = self._insert_with_mamba(tree, allocator, prefix, mamba_slot)

        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", prefix)),
                skip_mamba_match=True,
            )
        )
        matched_node = result.last_device_node
        self.assertEqual(len(result.device_indices), 3)

        tree.inc_lock_ref(matched_node)
        self.assertGreater(tree.full_protected_size(), 0)

        req = self._make_mamba_req(req_to_token_pool, [1, 2, 3, 4, 5])
        new_kv = allocator.alloc(2)
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, 3)),
            prefix_kv,
        )
        req_to_token_pool.write(
            (req.req_pool_idx, slice(3, 5)),
            new_kv,
        )
        req.cache_protected_len = 3
        req.last_node = matched_node

        tree.cache_unfinished_req(req)
        tree.cache_finished_req(req)

        self.assertEqual(tree.full_protected_size(), 0)

    def test_mamba_lock_ref_full_transfer_success(self):
        """Mamba cache: no prefix match, full KV transferred, succeeds."""
        tree, allocator, req_to_token_pool = self._make_mamba_cache()

        full_ids = [10, 20, 30]
        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", full_ids)),
                skip_mamba_match=True,
            )
        )
        self.assertEqual(len(result.device_indices), 0)

        matched_node = result.last_device_node
        tree.inc_lock_ref(matched_node)

        req = self._make_mamba_req(req_to_token_pool, full_ids)
        kv_indices = allocator.alloc(len(full_ids))
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, len(full_ids))),
            kv_indices,
        )
        req.last_node = matched_node
        req.cache_protected_len = 0

        tree.cache_unfinished_req(req)
        tree.cache_finished_req(req)

        self.assertEqual(tree.full_protected_size(), 0)
        self.assertGreater(tree.full_evictable_size(), 0)

    def test_mamba_lock_ref_incremental_transfer_failure(self):
        """Mamba cache: prefix match > 0, transfer fails.

        After failure, dec_lock_ref should restore all counters.
        """
        tree, allocator, req_to_token_pool = self._make_mamba_cache()
        mamba_pool = req_to_token_pool.mamba_pool

        mamba_slot = mamba_pool.alloc(1)
        prefix = [1, 2, 3]
        self._insert_with_mamba(tree, allocator, prefix, mamba_slot)

        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", prefix)),
                skip_mamba_match=True,
            )
        )
        matched_node = result.last_device_node
        tree.inc_lock_ref(matched_node)
        self.assertGreater(tree.full_protected_size(), 0)

        req = self._make_mamba_req(req_to_token_pool, [1, 2, 3, 4, 5])
        kv_indices = allocator.alloc(5)
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, 5)),
            kv_indices,
        )
        req.cache_protected_len = 3
        req.last_node = matched_node

        tree.cache_finished_req(req, is_insert=False)

        self.assertEqual(tree.full_protected_size(), 0)
        self.assertGreater(tree.full_evictable_size(), 0)

    def test_pop_preallocated_mamba_budget_gate(self):
        """When mamba pool is exhausted, pop_preallocated should not admit
        the request and should release any KV lock already taken.
        """
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)

        req = MagicMock()
        req.rid = "req-mamba-1"
        req.origin_input_ids = list(range(8))
        req.output_ids = [99]
        req.last_node = object()
        req.finished_reason = None
        req.cache_protected_len = 0
        req.sampling_params.max_new_tokens = 16
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None

        decode_req = MagicMock()
        decode_req.req = req
        decode_req.waiting_for_input = True

        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._match_prefix_and_lock = MagicMock(
            return_value=DecodePrefixMatch(
                prefix_indices=torch.arange(4, dtype=torch.int64),
                l2_host_hit_length=0,
                l3_storage_hit_length=0,
                last_device_node=req.last_node,
            )
        )
        queue._pre_alloc = MagicMock(
            side_effect=AssertionError("_pre_alloc should not run")
        )
        queue.transfer_queue = MagicMock(queue=[], enable_staging=False)
        queue.tree_cache = MagicMock()
        queue.tree_cache.dec_lock_ref = MagicMock()
        queue.tree_cache.supports_mamba = MagicMock(return_value=True)
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_token_pool.enable_mamba_extra_buffer = False
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 1

        running_batch = MagicMock()
        running_batch.reqs = []
        server_args = MagicMock()
        server_args.disaggregation_decode_enable_radix_cache = True
        scheduler = MagicMock()
        scheduler.running_batch = running_batch
        scheduler.server_args = server_args
        scheduler.enable_hisparse = False
        scheduler.enable_priority_scheduling = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # KV budget is fine
        queue._allocatable_token_budgets = MagicMock(return_value=100)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._required_alloc_tokens = MagicMock(return_value=4)
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)

        # Mamba budget is exhausted
        queue._required_alloc_mamba_states = MagicMock(return_value=1)
        queue._allocatable_mamba_budgets = MagicMock(return_value=0)

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        queue._pre_alloc.assert_not_called()
        queue.tree_cache.dec_lock_ref.assert_called_once_with(req.last_node)

    def test_pop_preallocated_mamba_budget_sufficient(self):
        """When mamba budget is sufficient, the request should proceed to _pre_alloc."""
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)

        req = MagicMock()
        req.rid = "req-mamba-2"
        req.origin_input_ids = list(range(8))
        req.output_ids = [99]
        req.last_node = object()
        req.finished_reason = None
        req.cache_protected_len = 0
        req.sampling_params.max_new_tokens = 16
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None

        decode_req = MagicMock()
        decode_req.req = req
        decode_req.waiting_for_input = True
        decode_req.prefix_match = None

        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()

        prefix_indices = torch.arange(4, dtype=torch.int64)
        prefix_match = DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node=req.last_node,
        )
        queue._match_prefix_and_lock = MagicMock(return_value=prefix_match)
        queue._pre_alloc = MagicMock(return_value=torch.arange(8, dtype=torch.int64))
        queue._start_hicache_prefetch = MagicMock()
        queue.transfer_queue = MagicMock(queue=[], enable_staging=False)
        queue.tree_cache = MagicMock()
        queue.tree_cache.dec_lock_ref = MagicMock()
        queue.tree_cache.supports_mamba = MagicMock(return_value=True)
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_token_pool.enable_mamba_extra_buffer = False
        queue.req_to_token_pool.req_to_token = torch.zeros(4, 64, dtype=torch.int64)
        queue.req_to_token_pool.req_index_to_mamba_index_mapping = torch.zeros(
            4, dtype=torch.int64
        )
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator.alloc.return_value = 0
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool.page_size = 1
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 1

        running_batch = MagicMock()
        running_batch.reqs = []
        server_args = MagicMock()
        server_args.disaggregation_decode_enable_radix_cache = True
        scheduler = MagicMock()
        scheduler.running_batch = running_batch
        scheduler.server_args = server_args
        scheduler.enable_hisparse = False
        scheduler.enable_priority_scheduling = False
        scheduler.enable_decode_hicache = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        queue._allocatable_token_budgets = MagicMock(return_value=100)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._required_alloc_tokens = MagicMock(return_value=4)
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)

        # Mamba budget is sufficient
        queue._required_alloc_mamba_states = MagicMock(return_value=1)
        queue._allocatable_mamba_budgets = MagicMock(return_value=5)

        # Need to mock kv_manager for the state_types iteration

        queue.kv_manager = MagicMock()
        queue.kv_manager.kv_args.state_types = []

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(len(preallocated), 1)
        queue._pre_alloc.assert_called_once()

    def test_tombstone_restoration_lock_ref(self):
        """Restoring a tombstone via cache_unfinished_req must not crash.

        When skip_mamba_match=True matches to a tombstone (mamba_value=None,
        created by _split_node), inc_lock_ref skips mamba_lock_ref. The
        subsequent cache_unfinished_req inserts a mamba_value on that node
        (_insert_helper tombstone restoration). Without the direct_locks fix
        in _insert_helper, dec_lock_ref would assert mamba_lock_ref > 0.
        """
        tree, allocator, req_to_token_pool = self._make_mamba_cache()
        mamba_pool = req_to_token_pool.mamba_pool

        # Insert [1,2,3,4,5] → single leaf node with mamba_value
        mamba_slot = mamba_pool.alloc(1)
        self._insert_with_mamba(tree, allocator, [1, 2, 3, 4, 5], mamba_slot)

        # Match [1,2,3] with skip_mamba_match=True → _split_node creates:
        #   [1,2,3] (tombstone, mamba_value=None) → [4,5] (has mamba_value)
        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3])),
                skip_mamba_match=True,
            )
        )
        tombstone = result.last_device_node
        self.assertEqual(len(result.device_indices), 3)
        self.assertIsNone(tombstone.mamba_value)

        # Simulate _match_prefix_and_lock: lock the tombstone
        tree.inc_lock_ref(tombstone)
        self.assertEqual(tombstone.mamba_lock_ref, 0)

        # Create request with fill_ids=[1,2,3], write KV indices
        req = self._make_mamba_req(req_to_token_pool, [1, 2, 3])
        kv_indices = allocator.alloc(3)
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, 3)),
            kv_indices,
        )
        req.last_node = tombstone
        req.cache_protected_len = 0

        # cache_unfinished_req restores tombstone's mamba_value, then
        # dec_lock_ref + inc_lock_ref — must not crash
        tree.cache_unfinished_req(req)

        # Finish the request
        tree.cache_finished_req(req)

        self.assertEqual(tree.full_protected_size(), 0)
        self.assertEqual(tree.mamba_protected_size(), 0)

    def test_tombstone_restoration_multiple_locks(self):
        """Multiple requests locked at a tombstone via skip_mamba_match.

        direct_locks > 1: _insert_helper must set mamba_lock_ref for all
        requests, not just the one whose cache_unfinished_req triggers the
        tombstone restoration.
        """
        tree, allocator, req_to_token_pool = self._make_mamba_cache()
        mamba_pool = req_to_token_pool.mamba_pool

        # Insert [1,2,3,4,5] → single leaf
        mamba_slot = mamba_pool.alloc(1)
        self._insert_with_mamba(tree, allocator, [1, 2, 3, 4, 5], mamba_slot)

        # Match [1,2,3] → split, creating tombstone
        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3])),
                skip_mamba_match=True,
            )
        )
        tombstone = result.last_device_node
        self.assertIsNone(tombstone.mamba_value)

        # Two requests lock the same tombstone
        tree.inc_lock_ref(tombstone)
        tree.inc_lock_ref(tombstone)
        self.assertEqual(tombstone.mamba_lock_ref, 0)

        # Request A restores the tombstone
        req_a = self._make_mamba_req(req_to_token_pool, [1, 2, 3], req_pool_idx=0)
        kv_a = allocator.alloc(3)
        req_to_token_pool.write((req_a.req_pool_idx, slice(0, 3)), kv_a)
        req_a.last_node = tombstone
        req_a.cache_protected_len = 0

        tree.cache_unfinished_req(req_a)
        # After restoration, mamba_lock_ref should account for both locks
        # (req A's dec+inc leaves net 1, plus req B's original lock = 1)

        # Request B: tombstone already restored, insert finds mamba exists
        req_b = self._make_mamba_req(req_to_token_pool, [1, 2, 3], req_pool_idx=1)
        kv_b = allocator.alloc(3)
        req_to_token_pool.write((req_b.req_pool_idx, slice(0, 3)), kv_b)
        req_b.last_node = tombstone
        req_b.cache_protected_len = 0

        # Must not crash — req B's dec_lock_ref needs mamba_lock_ref > 0
        tree.cache_unfinished_req(req_b)

        # Finish both
        tree.cache_finished_req(req_a)
        tree.cache_finished_req(req_b)

        self.assertEqual(tree.full_protected_size(), 0)
        self.assertEqual(tree.mamba_protected_size(), 0)


if __name__ == "__main__":
    unittest.main()
