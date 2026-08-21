"""Unit tests for the FULL/device-only C++ unified TreeCore backend."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.cpp_unified_tree_core import (
    CppUnifiedTreeCore,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _params(**kwargs) -> CacheInitParams:
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=2,
        tree_components=(ComponentType.FULL,),
        **kwargs,
    )


def _key(*tokens: int) -> RadixKey:
    return RadixKey(array("q", tokens))


class CppUnifiedTreeCoreTest(CustomTestCase):
    def setUp(self):
        component = SimpleNamespace(tree_core=None)
        self.core = CppUnifiedTreeCore(_params(), {ComponentType.FULL: component})

    def _insert(self, tokens, values, prev_prefix_len=0):
        step = self.core.begin_insert(
            InsertParams(
                key=_key(*tokens),
                value=torch.tensor(values, dtype=torch.int64),
                prev_prefix_len=prev_prefix_len,
            )
        )
        self.assertIsNotNone(step.result)
        self.core.end_insert()
        return step

    def test_insert_match_split_and_flatten(self):
        step = self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        self.assertEqual(step.result.prefix_len, 0)

        result = self.core.match_prefix(MatchPrefixParams(key=_key(1, 2, 9, 9)))
        self.assertEqual(result.device_indices.tolist(), [10, 11])
        self.assertEqual(result.full_kv_hit_length, 2)
        self.assertNotEqual(result.last_device_node, 0)
        self.assertCountEqual(self.core.all_values_flatten().tolist(), [10, 11, 12, 13])
        self.assertEqual(self.core.total_size(), (4, 0))

    def test_duplicate_insert_returns_unowned_slots_as_action(self):
        self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        step = self._insert([1, 2, 3, 4], [20, 21, 22, 23], prev_prefix_len=1)
        self.assertEqual(step.result.prefix_len, 4)
        self.assertEqual(len(step.actions), 1)
        self.assertIsInstance(step.actions[0], FreeDeviceKV)
        self.assertEqual(step.actions[0].indices[0].tolist(), [21, 22, 23])

        result = self.core.match_prefix(MatchPrefixParams(key=_key(1, 2, 3, 4)))
        self.assertEqual(result.device_indices.tolist(), [10, 11, 12, 13])

    def test_lock_accounting_and_eager_eviction_bridge(self):
        self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        result = self.core.match_prefix(MatchPrefixParams(key=_key(1, 2)))
        node_id = result.last_device_node

        self.core.inc_lock_ref(node_id)
        self.assertEqual(self.core.protected_size(), 2)
        self.assertEqual(self.core.evictable_size(), 2)
        self.core.dec_lock_ref(node_id)
        self.assertEqual(self.core.protected_size(), 0)

        self.core.evict_device_start(ComponentType.FULL, 3)
        evicted = self.core.evict_device_next_node(
            ComponentType.FULL, {ComponentType.FULL: 0}
        )
        try:
            self.assertIsNone(evicted.node_id)
            self.assertGreaterEqual(evicted.tracker[ComponentType.FULL], 3)
            self.assertEqual(
                sum(len(v) for v in evicted.device_frees[ComponentType.FULL]),
                evicted.tracker[ComponentType.FULL],
            )
        finally:
            # Results assert that allocator-owned values are consumed.
            evicted.device_frees.clear()
            evicted.host_frees.clear()
        self.core.evict_device_end(ComponentType.FULL)
        self.assertEqual(self.core.total_size(), (0, 0))

    def test_reset_clears_tree_and_restores_cpp_root_handle(self):
        self._insert([1, 2], [10, 11])
        self.core.reset()
        self.assertEqual(self.core.root_node_handle(), 0)
        self.assertEqual(self.core.total_size(), (0, 0))
        self.assertEqual(
            self.core.match_prefix(MatchPrefixParams(key=_key(1, 2))).last_device_node,
            0,
        )

    def test_unsupported_namespace_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "extra_key"):
            self.core.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [1, 2]), "tenant"))
            )

    def test_hicache_configuration_fails_closed(self):
        with self.assertRaisesRegex(NotImplementedError, "HiCache"):
            CppUnifiedTreeCore(
                _params(enable_hicache=True),
                {ComponentType.FULL: SimpleNamespace(tree_core=None)},
            )

    def test_rejected_insert_does_not_poison_the_next_insert(self):
        with self.assertRaisesRegex(ValueError, "extra_key"):
            self.core.begin_insert(
                InsertParams(
                    key=RadixKey(array("q", [1, 2]), "tenant"),
                    value=torch.tensor([10, 11], dtype=torch.int64),
                )
            )
        self.assertFalse(self.core.has_ongoing_insert())

        step = self._insert([3, 4], [12, 13])
        self.assertEqual(step.result.prefix_len, 0)
        self.assertEqual(
            self.core.match_prefix(
                MatchPrefixParams(key=_key(3, 4))
            ).device_indices.tolist(),
            [12, 13],
        )


if __name__ == "__main__":
    unittest.main()
