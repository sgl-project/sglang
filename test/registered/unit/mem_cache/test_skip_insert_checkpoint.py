"""A request kept out of the tree still checkpoints its progress: after a chunk,
its next extend resumes at the chunk end on the slots it wrote, and the tree is untouched."""

import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SkipInsertReq:
    """The fields checkpoint_kv_cache reads, for a request barred from the tree."""

    rid = "skip-insert"
    skip_radix_cache_insert = True

    def __init__(self, fill_ids, *, extend_range, cache_protected_len, last_node):
        self.full_untruncated_fill_ids = array("q", fill_ids)
        self.extend_range = extend_range
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.last_node = last_node
        self.kv = ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=extend_range.end,
            kv_allocated_len=extend_range.end,
            cache_protected_len=cache_protected_len,
        )

    def finished(self):
        return False


class TestSkipInsertCheckpoint(CustomTestCase):
    def test_chunk_past_a_tree_prefix_resumes_without_touching_the_tree(self):
        allocator = MagicMock()
        allocator.device = torch.device("cpu")
        allocator.page_size = 1
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        req_to_token = torch.zeros(2, 16, dtype=torch.int64)
        cache.req_to_token_pool = MagicMock(req_to_token=req_to_token)

        # Tree prefix [1, 2, 3] on slots 10..30, matched and locked by the request.
        prefix = array("q", [1, 2, 3])
        cache.insert(
            InsertParams(
                key=RadixKey(prefix),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(prefix)))
        node = match.last_device_node
        cache.inc_lock_ref(node)
        tree_size, lock_ref = cache.total_size(), node.lock_ref

        # The request has computed a chunk [3, 6) of an 8-token prompt.
        req_to_token[0, :6] = torch.tensor([10, 20, 30, 40, 50, 60])
        req = _SkipInsertReq(
            [1, 2, 3, 4, 5, 6, 7, 8],
            extend_range=Range(3, 6),
            cache_protected_len=3,
            last_node=node,
        )

        checkpoint_kv_cache(req, cache)

        self.assertEqual(req.prefix_indices.tolist(), [10, 20, 30, 40, 50, 60])
        self.assertEqual(cache.total_size(), tree_size)
        self.assertEqual(req.kv.cache_protected_len, 3)
        self.assertIs(req.last_node, node)
        self.assertEqual(node.lock_ref, lock_ref)


if __name__ == "__main__":
    unittest.main()
