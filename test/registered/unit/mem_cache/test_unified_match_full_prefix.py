"""UnifiedRadixCache.match_full_prefix: FULL KV lookup independent of component state."""

import unittest
from array import array

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import (
    BASE_COMPONENT_TYPE,
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 2


class _TombstonedSwaComponent(TreeComponent):
    """Every node's SWA state is tombstoned: the all-component match sees nothing.

    On the P/D decode tier this is the shared-prefix shape: the SWA tail is
    transferred fresh per request and belongs to each request's own end.
    """

    component_type = ComponentType.SWA

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: False

    def build_hicache_transfers(self, node, phase, **kwargs):
        # Like the real SWA component: a tombstoned node has nothing to build.
        raise AssertionError("tombstoned SWA state has no transfer")

    def redistribute_on_node_split(self, new_parent, child):
        return None

    def evict_component(
        self, node, device_frees, host_frees, target: EvictLayer = EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0

    def acquire_component_lock(self, node, result):
        return result

    def release_component_lock(self, node, params):
        return None

    def _evict_device_start(self, request_cnt) -> None:
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        pass

    def _dec_session_coverage(self, session_id, leaf) -> None:
        pass

    def _advance_session_coverage(self, session_id, leaf, old_ancestor) -> None:
        pass

    def _recede_session_coverage(self, session_id, leaf, fallback) -> None:
        pass


def _cache() -> UnifiedRadixCache:
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=PAGE_SIZE,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            component_registry_override={ComponentType.SWA: _TombstonedSwaComponent},
        )
    )


def _key(tokens) -> RadixKey:
    return RadixKey(array("q", tokens))


def _add_node(tree_core, parent, tokens, *, device: bool, host: bool):
    node = tree_core._new_node()
    node.parent = parent
    node.key = _key(tokens)
    node.hash_value = []
    cd = node.component_data[BASE_COMPONENT_TYPE]
    base = 100 * node.id
    if device:
        cd.value = torch.arange(base, base + len(tokens))
    if host:
        cd.host_value = torch.arange(base, base + len(tokens))
    parent.children[node.key.child_key(PAGE_SIZE)] = node
    return node


class TestMatchFullPrefix(CustomTestCase):
    def setUp(self):
        self.cache = _cache()
        self.tree_core = self.cache.tree_core
        root = self.tree_core.root_node
        self.a = _add_node(self.tree_core, root, [1, 2, 3, 4], device=True, host=False)
        self.b = _add_node(
            self.tree_core, self.a, [5, 6, 7, 8], device=True, host=False
        )

    def test_finds_full_kv_behind_tombstoned_component_state(self):
        key = _key([1, 2, 3, 4, 5, 6, 7, 8])

        result = self.cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(result.host_hit_length, 0)

        matched_len, node_id = self.cache.match_full_prefix(key)

        self.assertEqual(matched_len, 8)
        self.assertEqual(node_id, self.b.id)
        self.assertEqual(
            self.tree_core.collect_full_device_indices(
                node_id, self.tree_core.root_node.id
            ).tolist(),
            [
                *self.a.component_data[BASE_COMPONENT_TYPE].value.tolist(),
                *self.b.component_data[BASE_COMPONENT_TYPE].value.tolist(),
            ],
        )

    def test_host_only_full_kv_matches_and_dead_node_ends_it(self):
        c = _add_node(self.tree_core, self.b, [9, 10], device=False, host=True)
        _add_node(self.tree_core, c, [11, 12], device=False, host=False)

        matched_len, node_id = self.cache.match_full_prefix(
            _key([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        )

        self.assertEqual(matched_len, 10)
        self.assertEqual(node_id, c.id)
        self.assertTrue(self.tree_core.is_full_device_evicted(node_id))

    def test_splits_the_deepest_node_at_the_key_end(self):
        matched_len, node_id = self.cache.match_full_prefix(_key([1, 2, 3, 4, 5, 6]))

        self.assertEqual(matched_len, 6)
        split = self.tree_core.node_by_id(node_id)
        self.assertEqual(split.key.token_ids.tolist(), [5, 6])
        self.assertIs(split.parent, self.a)
        self.assertIs(self.b.parent, split)
        self.assertEqual(self.b.key.token_ids.tolist(), [7, 8])
        self.assertEqual(
            split.component_data[BASE_COMPONENT_TYPE].value.tolist(),
            [self.b.id * 100, self.b.id * 100 + 1],
        )

    def test_diverging_key_stops_at_the_last_full_node(self):
        matched_len, node_id = self.cache.match_full_prefix(_key([1, 2, 3, 4, 50, 60]))

        self.assertEqual(matched_len, 4)
        self.assertEqual(node_id, self.a.id)

    def test_kv_only_load_back_spec_builds_no_component_transfers(self):
        # The KV-only restore of an evicted FULL node behind tombstoned SWA
        # state must not ask the SWA component for a transfer (it asserts).
        c = _add_node(self.tree_core, self.b, [9, 10], device=False, host=True)

        with self.assertRaises(AssertionError):
            self.tree_core.build_load_back_spec(c.id)

        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(c.id, kv_only=True)

        self.assertEqual(comp_xfers, {})
        self.assertEqual(
            kv_xfer.host_indices.tolist(),
            c.component_data[BASE_COMPONENT_TYPE].host_value.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
