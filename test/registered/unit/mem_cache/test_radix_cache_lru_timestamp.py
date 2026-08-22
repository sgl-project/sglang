"""Unit tests for radix cache split access timestamps.

Structural split coverage lives in
test_radix_cache_unit.py::TestRadixCache::test_advanced_prefix_match_with_node_splits.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.test_utils import CustomTestCase


class TestRadixCacheSplitTimestamp(CustomTestCase):
    def test_match_split_keeps_suffix_timestamp_for_lru(self):
        for page_size in [1, 2]:
            with self.subTest(page_size=page_size):
                mock_allocator = unittest.mock.Mock()
                mock_allocator.device = torch.device("cpu")
                cache = RadixCache.create_simulated(
                    mock_allocator=mock_allocator, page_size=page_size
                )
                cache.insert(
                    InsertParams(
                        key=RadixKey([1, 2, 3, 4]),
                        value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
                    )
                )
                first_edge = next(iter(cache.root_node.children.values()))
                first_edge.last_access_time = 10.0

                cache.insert(
                    InsertParams(
                        key=RadixKey([7, 8]),
                        value=torch.tensor([70, 80], dtype=torch.int64),
                    )
                )
                newer_leaf = next(
                    child
                    for child in cache.root_node.children.values()
                    if child.key.token_ids == [7, 8]
                )
                newer_leaf.last_access_time = 20.0

                with unittest.mock.patch(
                    "sglang.srt.mem_cache.radix_cache.time.monotonic",
                    return_value=30.0,
                ):
                    match = cache.match_prefix(
                        MatchPrefixParams(key=RadixKey([1, 2, 9, 9]))
                    )

                prefix = next(
                    child
                    for child in cache.root_node.children.values()
                    if child.key.token_ids == [1, 2]
                )
                suffix = next(iter(prefix.children.values()))
                self.assertEqual(suffix.key.token_ids, [3, 4])
                torch.testing.assert_close(
                    match.device_indices, torch.tensor([10, 20], dtype=torch.int64)
                )
                self.assertEqual(prefix.last_access_time, 30.0)
                self.assertEqual(suffix.last_access_time, 10.0)
                self.assertEqual(newer_leaf.last_access_time, 20.0)

                cache.evict(EvictParams(num_tokens=2))
                freed_values = [
                    call.args[0].tolist() for call in mock_allocator.free.call_args_list
                ]
                self.assertEqual(freed_values, [[30, 40]])

    def test_match_complete_child_refreshes_timestamp(self):
        cache = RadixCache.create_simulated()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2]),
                value=torch.tensor([10, 20], dtype=torch.int64),
            )
        )
        child = cache.root_node.children[1]
        child.last_access_time = 10.0

        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.time.monotonic", return_value=30.0
        ):
            match = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2])))

        torch.testing.assert_close(
            match.device_indices, torch.tensor([10, 20], dtype=torch.int64)
        )
        self.assertEqual(child.last_access_time, 30.0)

    def test_insert_split_keeps_suffix_timestamp(self):
        for page_size in [1, 2]:
            with self.subTest(page_size=page_size):
                cache = RadixCache.create_simulated(page_size=page_size)
                cache.insert(
                    InsertParams(
                        key=RadixKey([1, 2, 3, 4]),
                        value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
                    )
                )
                first_edge = next(iter(cache.root_node.children.values()))
                first_edge.last_access_time = 10.0

                with unittest.mock.patch(
                    "sglang.srt.mem_cache.radix_cache.time.monotonic",
                    return_value=30.0,
                ):
                    result = cache.insert(
                        InsertParams(
                            key=RadixKey([1, 2, 9, 9]),
                            value=torch.tensor([10, 20, 90, 90], dtype=torch.int64),
                        )
                    )

                prefix = next(
                    child
                    for child in cache.root_node.children.values()
                    if child.key.token_ids == [1, 2]
                )
                suffix = next(iter(prefix.children.values()))
                self.assertEqual(result.prefix_len, 2)
                self.assertEqual(suffix.key.token_ids, [3, 4])
                self.assertEqual(prefix.last_access_time, 30.0)
                self.assertEqual(suffix.last_access_time, 10.0)

    def test_insert_complete_child_refreshes_timestamp(self):
        cache = RadixCache.create_simulated()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2]),
                value=torch.tensor([10, 20], dtype=torch.int64),
            )
        )
        child = cache.root_node.children[1]
        child.last_access_time = 10.0

        with unittest.mock.patch(
            "sglang.srt.mem_cache.radix_cache.time.monotonic", return_value=30.0
        ):
            result = cache.insert(
                InsertParams(
                    key=RadixKey([1, 2, 3]),
                    value=torch.tensor([10, 20, 30], dtype=torch.int64),
                )
            )

        self.assertEqual(result.prefix_len, 2)
        self.assertEqual(child.last_access_time, 30.0)


if __name__ == "__main__":
    unittest.main()
