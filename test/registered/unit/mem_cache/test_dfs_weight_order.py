import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import slot_of, tree_slots_of

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_tree(backend):
    if backend == "classic":
        return RadixCache.create_simulated()
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_slots=tree_slots_of((ComponentType.FULL,)),
    )
    cache = SimpleNamespace(enable_session_radix_cache=False)
    component = FullComponent(cache, params, slot_of(params, ComponentType.FULL))
    return UnifiedTreeCore(params, {ComponentType.FULL: component})


def insert(tree, tokens):
    params = InsertParams(
        key=RadixKey(array("q", tokens)),
        value=torch.arange(len(tokens), dtype=torch.int64),
    )
    if isinstance(tree, RadixCache):
        return tree.insert(params).last_device_node
    # CPU index tensors only; no physical KV pool or storage actions are needed.
    step = tree.begin_insert(params)
    while step.result is None:
        step = tree.resume_insert()
    tree.end_insert()
    return step.result.last_device_node


def match(tree, tokens):
    return tree.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    ).last_device_node


def resolve(tree, handle):
    return (
        tree.resolve_node_handle(handle)
        if isinstance(tree, RadixCache)
        else tree.node_by_id(handle)
    )


class CountingChildren(dict):
    """Count real child visits without substituting the tree or ordering code."""

    visits = 0

    def values(self):
        for child in super().values():
            self.visits += 1
            yield child


class TestDFSWeightOrder(unittest.TestCase):
    def test_heaviest_subtree_then_descendants_then_node(self):
        for backend in ("classic", "python"):
            with self.subTest(backend=backend):
                tree = make_tree(backend)
                a1 = insert(tree, [1, 10])
                a2 = insert(tree, [1, 11])
                b = insert(tree, [2, 20])
                a = match(tree, [1])
                root = tree.root_node_handle()
                self.assertEqual(
                    tree.dfs_weight_order([b, a2, a1, a1, a, root]),
                    [2, 3, 1, 4, 0, 5],
                )
                # Equal subtree weights follow child insertion order, not the
                # order in which requests happen to reach the waiting queue.
                self.assertEqual(tree.dfs_weight_order([b, a2]), [1, 0])
                self.assertEqual(tree.dfs_weight_order([a2, a1]), [1, 0])

    def test_sparse_queue_does_not_visit_unrelated_descendants(self):
        for backend in ("classic", "python"):
            with self.subTest(backend=backend):
                tree = make_tree(backend)
                active = insert(tree, [1, 10])
                for index in range(32):
                    insert(tree, [2, 20, index])
                inactive = resolve(tree, match(tree, [2, 20]))
                inactive.children = CountingChildren(inactive.children)
                self.assertEqual(tree.dfs_weight_order([active, active]), [0, 1])
                self.assertEqual(inactive.children.visits, 0)

    def test_empty_queue_does_not_visit_cached_branches(self):
        for backend in ("classic", "python"):
            with self.subTest(backend=backend):
                tree = make_tree(backend)
                insert(tree, [1])
                root = resolve(tree, tree.root_node_handle())
                root.children = CountingChildren(root.children)
                self.assertEqual(tree.dfs_weight_order([]), [])
                self.assertEqual(root.children.visits, 0)

    def test_scheduler_consumer_preserves_request_objects(self):
        tree = make_tree("classic")
        a = insert(tree, [1])
        b = insert(tree, [2])
        requests = [SimpleNamespace(last_node=node) for node in (b, a, a)]
        expected = [requests[1], requests[2], requests[0]]
        SchedulePolicy._sort_by_dfs_weight(requests, tree)
        self.assertEqual([id(req) for req in requests], [id(req) for req in expected])

    def test_evicted_classic_handle_does_not_add_weight_to_its_parent(self):
        tree = make_tree("classic")
        a = insert(tree, [1, 10])
        evicted = insert(tree, [1, 11])
        b = insert(tree, [2, 20])
        tree._delete_leaf(evicted)
        self.assertEqual(
            tree.dfs_weight_order([a, b, b, evicted, evicted, evicted]), [1, 2, 0]
        )

    def test_root_requests_follow_matching_descendants(self):
        for backend in ("classic", "python"):
            with self.subTest(backend=backend):
                tree = make_tree(backend)
                leaf = insert(tree, [1, 2, 3])
                root = tree.root_node_handle()
                self.assertEqual(tree.dfs_weight_order([root, root]), [0, 1])
                self.assertEqual(tree.dfs_weight_order([root, leaf, root]), [1, 0, 2])

    def test_calc_priority_matches_prefixes_before_ordering(self):
        tree = make_tree("classic")
        for tokens in ([1, 10], [1, 11], [2, 20]):
            insert(tree, tokens)
        requests = [
            Req(
                rid=str(index),
                origin_input_text="",
                origin_input_ids=array("q", tokens),
                sampling_params=SamplingParams(max_new_tokens=1),
            )
            for index, tokens in enumerate(
                ([2, 20, 99], [1, 11, 99], [1, 10, 99], [1, 10, 99])
            )
        ]
        policy = SchedulePolicy(
            policy="dfs-weight",
            tree_cache=tree,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        result = policy.calc_priority(requests)
        self.assertEqual([request.rid for request in result], ["2", "3", "1", "0"])
        self.assertTrue(all(request.prefix_len == 2 for request in result))


if __name__ == "__main__":
    unittest.main()
