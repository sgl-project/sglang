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

Usage:
    python -m pytest test/registered/unit/mem_cache/test_decode_radix_lock_ref.py -v
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-test-1-gpu-small")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


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
        self.fill_ids = list(fill_ids)
        self.origin_input_ids = (
            list(fill_ids[:-1]) if len(fill_ids) > 1 else list(fill_ids)
        )
        self.output_ids = [fill_ids[-1]] if len(fill_ids) > 1 else []
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = cache_protected_len
        self.last_node = last_node
        self.extra_key = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0
        self.kv_committed_len = len(fill_ids)
        self.kv_allocated_len = len(fill_ids)
        self.kv_committed_freed = False

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
                key=RadixKey(prefix_ids),
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
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(prefix)))
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
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids)))
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
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(prefix)))
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
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids)))
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
            return_value=(torch.arange(4, dtype=torch.int64), 4)
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
        scheduler.stream_output = MagicMock()
        queue.scheduler = scheduler

        # Initial budget says the request fits; post-lock budget says it does not.
        queue._allocatable_tokens = MagicMock(side_effect=[8, 3])

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        queue._pre_alloc.assert_not_called()
        queue.tree_cache.dec_lock_ref.assert_called_once_with(req.last_node)
        self.assertEqual(queue._allocatable_tokens.call_count, 2)

    def test_repeated_incremental_no_leak(self):
        """Multiple incremental transfers shouldn't leak lock_refs."""
        cache, req_to_token = _make_cache_with_pools()

        prefix = [1, 2, 3]
        prefix_vals = [10, 20, 30]
        self._populate_prefix(cache, prefix, prefix_vals)

        for iteration in range(5):
            result = cache.match_prefix(MatchPrefixParams(key=RadixKey(prefix)))
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


if __name__ == "__main__":
    unittest.main()
