"""Unit tests for the per-path SWA window cap."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest
from array import array
from collections import defaultdict

import torch

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList, UnifiedTreeNode
from sglang.srt.server_args import ServerArgs
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase


class _FakeTreeCore:
    tree_components = (ComponentType.FULL, ComponentType.SWA)
    page_size = 1

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_device_leaves = set()
        self.component_evictable_size_ = {ComponentType.SWA: 0}
        self.component_protected_size_ = {ComponentType.SWA: 0}
        self.lru_lists = {
            ComponentType.SWA: UnifiedLRUList(ComponentType.SWA, self.tree_components)
        }
        self.host_lru_lists = {
            ComponentType.SWA: UnifiedLRUList(
                ComponentType.SWA, self.tree_components, use_host_ptr=True
            )
        }
        self.evicted = []
        self._is_tracking_unbacked_tokens = False
        self._tracked_unbacked_tokens = 0

    def _evict_component_and_detach_lru(self, node, component, *args, **kwargs):
        self.evicted.append(node)
        return UnifiedTreeCore._evict_component_and_detach_lru(
            self, node, component, *args, **kwargs
        )

    def _cascade_evict(
        self, node, component, tracker, device_frees, host_frees, cache_actions
    ):
        pass


class _FakeUnifiedCache:
    tree_components = _FakeTreeCore.tree_components


def _build_chain(cap, length=4):
    core = _FakeTreeCore()
    component = object.__new__(SWAComponent)
    component.cache = _FakeUnifiedCache()
    component.tree_core = core
    component.swa_max_states_per_path = cap
    component.sliding_window_size = 1

    nodes = []
    parent = core.root_node
    for index in range(length):
        node = UnifiedTreeNode(core.tree_components)
        node.parent = parent
        node.key = RadixKey(array("q", [index]))
        node.component_data[ComponentType.FULL].value = torch.tensor([100 + index])
        node.component_data[ComponentType.SWA].value = torch.tensor([index])
        parent.children[index] = node
        core.component_evictable_size_[ComponentType.SWA] += 1
        core.lru_lists[ComponentType.SWA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, core


def _trim(component, tail):
    device_frees = defaultdict(list)
    cache_actions = []
    component._evict_excess_path_states(
        tail, device_frees, defaultdict(list), cache_actions
    )
    assert cache_actions == []
    return device_frees


class TestSWAPathWindowCap(unittest.TestCase):
    def test_server_arg_rejects_zero_and_values_below_negative_one(self):
        for value in (0, -2):
            args = ServerArgs(model_path="dummy", swa_max_states_per_path=value)
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "must be -1 \\(unlimited\\) or a positive integer"
                ),
            ):
                handle_cache_compatibility(args)

    def test_removes_shallow_windows_but_keeps_full_kv(self):
        component, nodes, core = _build_chain(cap=1)

        device_frees = _trim(component, nodes[-1])

        self.assertEqual(core.evicted, [nodes[0], nodes[1]])
        swa = [node.component_data[ComponentType.SWA].value for node in nodes]
        self.assertEqual([value is None for value in swa], [True, True, False, False])
        # SWA slots are freed through their FULL indices, never the SWA ones.
        self.assertEqual(
            [value.item() for value in device_frees[ComponentType.SWA]], [100, 101]
        )
        self.assertTrue(
            all(
                node.component_data[ComponentType.FULL].value is not None
                for node in nodes
            )
        )
        self.assertEqual(core.component_evictable_size_[ComponentType.SWA], 2)


class _SWAPathCapFixtureBase(UnifiedRadixCacheFixtures):
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.SWA), sliding_window_size=4
    )
    tree_core_backend: str

    def _build(self, cap):
        return build_fixture(
            self.cfg,
            tree_core_backend=self.tree_core_backend,
            swa_max_states_per_path=cap,
        )

    def _insert(self, cache, allocator, tokens, swa_evicted_seqlen):
        value = self._alloc(allocator, len(tokens))
        # A request frees its out-of-window SWA slots before it inserts.
        allocator.free_swa(value[:swa_evicted_seqlen])
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value[: len(tokens)],
                component_evicted_seqlens={ComponentType.SWA: swa_evicted_seqlen},
            )
        )

    def _match_len(self, cache, tokens):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        return match.device_prefix_len, match.full_kv_hit_length

    def test_trims_shallow_window_but_keeps_full_kv_and_tail_match(self):
        cache, allocator, _ = self._build(cap=1)
        tokens = list(range(16))

        self._insert(cache, allocator, tokens[:8], swa_evicted_seqlen=4)
        self._insert(cache, allocator, tokens, swa_evicted_seqlen=12)

        # The first chunk's window is a tombstone; its full KV stays.
        self.assertEqual(self._match_len(cache, tokens[:8]), (0, 8))
        self.assertEqual(self._match_len(cache, tokens), (16, 16))
        cache.sanity_check()

    def test_keeps_parent_window_completing_a_short_tail(self):
        """A tail shorter than the window keeps the parent window it needs,
        even when no request locks that parent."""
        cache, allocator, _ = self._build(cap=1)
        tokens = list(range(14))

        self._insert(cache, allocator, tokens[:12], swa_evicted_seqlen=8)
        self._insert(cache, allocator, tokens, swa_evicted_seqlen=10)

        self.assertEqual(self._match_len(cache, tokens), (14, 14))
        cache.sanity_check()

    def test_keeps_window_for_a_repeat_of_the_cached_input(self):
        """A repeated input matches up to one page short of the cached tail;
        the window ending there must survive the cap."""
        cache, allocator, _ = self._build(cap=1)
        tokens = list(range(10))

        self._insert(cache, allocator, tokens, swa_evicted_seqlen=4)

        self.assertEqual(self._match_len(cache, tokens[:-1]), (9, 9))
        cache.sanity_check()

    def test_write_through_backup_chain_survives_path_cap(self):
        """Windows trimmed by the cap must not break a pending write-through
        backup of their ancestors."""
        cache, allocator, _ = self._build(cap=1)
        self._init_hicache(cache)
        cache.write_through_threshold = 2
        tokens = list(range(24))

        for end in (8, 16, 24, 24):
            self._insert(cache, allocator, tokens[:end], swa_evicted_seqlen=0)
        cache.writing_check(write_back=True)

        self.assertEqual(self._match_len(cache, tokens[:-1]), (23, 23))
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "SWA pool fixtures need CUDA")
class TestSWAPathCapPythonTreeCore(_SWAPathCapFixtureBase, CustomTestCase):
    tree_core_backend = "python"


@unittest.skipUnless(torch.cuda.is_available(), "SWA pool fixtures need CUDA")
class TestSWAPathCapRustTreeCore(_SWAPathCapFixtureBase, CustomTestCase):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
