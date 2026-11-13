"""CPU-only unit tests for --radix-eviction-policy gds: GreedyDual-Size
eviction of a hybrid model's cached states, each with the KV only it serves."""

import random
import unittest
from array import array
from contextlib import nullcontext
from unittest.mock import patch

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    resolve_tree_core_backend,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.utils import get_eviction_strategy
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


class _Tree:
    """A FULL + MAMBA tree under gds over CPU tensors, page size 1. With the
    default ``prefill_cost`` [1, 0], a unit's gain is the tokens since the
    state above it; a state slot counts as ``state_tokens`` KV tokens unless
    that is None (then it is read from the pools)."""

    def __init__(self, kv_size=256, state_tokens=10, **config):
        cfg = CacheConfig(
            components=(FULL, MAMBA),
            kv_size=kv_size,
            mamba_cache_size=64,
            max_num_reqs=1024,
            eviction_policy="gds",
            eviction_policy_config=config or None,
        )
        # build_fixture defaults to an accelerator, but these tests use CPU tensors.
        with patch(
            "sglang.test.kits.unified_radix_cache.fixtures.get_device",
            return_value="cpu",
        ):
            self.cache, self.allocator, self.pool = build_fixture(cfg)
        self.core = self.cache.tree_core
        if state_tokens is not None:
            self.core.gds.state_tokens = state_tokens
        self.kv_capacity = self.allocator.available_size()
        self.rid = 0

    def insert(self, tokens, *, state=True):
        value = self.allocator.alloc(len(tokens))
        assert value is not None, "the test's KV pool is exhausted"
        params = InsertParams(key=RadixKey(array("q", tokens)), value=value)
        if state:
            self.rid += 1
            req = Req(
                rid=f"gds-{self.rid}",
                origin_input_text="",
                origin_input_ids=array("q"),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            acquire_rows(self.pool, self.pool.state, [req])
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        else:
            params.allow_missing_mamba = True
        self.cache.insert(params)

    def node(self, tokens):
        node, depth = self.core.root_node, 0
        while depth < len(tokens):
            for child in node.children.values():
                key = list(child.key.raw_token_ids())
                if tokens[depth : depth + len(key)] == key:
                    node, depth = child, depth + len(key)
                    break
            else:
                return None
        return node

    def has_state(self, tokens):
        node = self.node(tokens)
        return node is not None and node.component_data[MAMBA].value is not None

    def evict_states(self, n=1):
        return self.cache.evict(EvictParams(mamba_num=n))

    def check(self, test):
        self.cache.sanity_check()
        test.assertEqual(
            self.allocator.available_size() + len(self.core.all_values_flatten()),
            self.kv_capacity,
        )


A, B = [1, 2, 3, 4], list(range(10, 20))


class TestGreedyDualSizeOrder(CustomTestCase):
    def test_the_lowest_gain_per_size_goes_first(self):
        tree = _Tree()
        tree.insert(A)  # gain 4, size 10 + 4
        tree.insert(B)  # gain 10, size 10 + 10
        tree.evict_states()
        self.assertIsNone(tree.node(A))
        self.assertTrue(tree.has_state(B))
        tree.check(self)

    def test_gain_counts_from_the_nearest_state_above(self):
        tree = _Tree()
        tree.insert(list(range(1, 7)))
        tree.insert(list(range(1, 9)))  # gain 2 over the state at 6, size 12
        tree.insert([20, 21, 22])  # gain 3, size 13
        tree.evict_states()
        self.assertFalse(tree.has_state(list(range(1, 9))))
        self.assertTrue(tree.has_state([20, 21, 22]))

    def test_size_counts_the_kv_only_the_unit_serves(self):
        tree = _Tree()
        tree.insert([1, 2, 3, 4])  # interior: gain 4, size 10
        tree.insert(list(range(1, 13)) + [99])  # leaf
        tree.insert(list(range(30, 36)))  # gain 6, size 10 + 6
        tree.evict_states()
        self.assertFalse(tree.has_state(list(range(30, 36))))
        self.assertTrue(tree.has_state([1, 2, 3, 4]))

    def test_a_unit_whose_exclusive_kv_shrank_is_rescored(self):
        tree = _Tree()
        tree.insert([1, 2, 3, 4, 5, 6])  # gain 6, size 16 until the split
        tree.insert([1, 2, 3, 9, 9])  # splits it: size 13, score 0.46
        tree.insert(list(range(60, 67)))  # gain 7, size 17: score 0.41
        tree.evict_states()
        self.assertFalse(tree.has_state(list(range(60, 67))))
        self.assertTrue(tree.has_state([1, 2, 3, 4, 5, 6]))

    def test_a_unit_whose_sibling_went_is_pushed_again(self):
        tree = _Tree()
        tree.insert([1, 2, 3, 4, 5, 6])
        tree.insert([1, 2, 3, 9])  # gain 4, size 11: score 0.36
        tree.insert(list(range(60, 67)))  # gain 7, size 17: score 0.41
        # A hit ranks the first unit at its split size: 6 / 13 = 0.46.
        tree.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 7])))
        )
        tree.evict_states()
        self.assertIsNone(tree.node([1, 2, 3, 9]))
        # It serves [1, 2, 3] alone again: 6 / 16 = 0.375.
        tree.evict_states()
        self.assertFalse(tree.has_state([1, 2, 3, 4, 5, 6]))
        self.assertTrue(tree.has_state(list(range(60, 67))))

    def test_a_unit_whose_last_branch_went_is_pushed_again(self):
        """A unit whose last branch below went is a leaf: it holds its own KV
        now, so its score drops below a unit it used to outrank."""
        tree = _Tree()
        p = list(range(1, 11))
        tree.insert(p)  # interior: 10 / 10, then a leaf: 10 / 20
        tree.insert(p + [99])  # 1 / 11: goes first
        tree.insert(list(range(50, 65)))  # 15 / 25
        tree.evict_states()
        self.assertIsNone(tree.node(p + [99]))
        tree.evict_states()
        self.assertFalse(tree.has_state(p))
        self.assertTrue(tree.has_state(list(range(50, 65))))
        tree.check(self)

    def test_a_unit_below_a_state_the_path_cap_takes_is_rescored(self):
        """The path cap takes the state above a unit outside the walk: the
        unit gains that state's tokens and holds its KV too."""
        tree = _Tree()
        x = list(range(1, 11))
        tree.insert(x)
        tree.insert(list(range(50, 58)))  # 8 / 18
        tree.cache.components[MAMBA].mamba_max_states_per_path = 1
        tree.insert(x + [11, 12, 13, 14])  # 4 / 14, then 14 / 24 without x's
        self.assertFalse(tree.has_state(x))
        tree.evict_states()
        self.assertTrue(tree.has_state(x + [11, 12, 13, 14]))
        self.assertFalse(tree.has_state(list(range(50, 58))))
        tree.check(self)

    def test_a_unit_below_a_state_the_path_cap_takes_can_drop(self):
        """With a cost per token rising with depth, the shallow tokens the
        unit gains cost less than its gain per size, so its score drops and
        it is pushed again at once."""
        # A token at position p costs 1 + p; a state slot counts as 1 token.
        tree = _Tree(state_tokens=1, prefill_cost=[1, 100000])
        x = list(range(1, 11))
        tree.insert(x)
        tree.insert(list(range(50, 67)))  # 161.5 / 18 = 8.97
        tree.cache.components[MAMBA].mamba_max_states_per_path = 1
        tree.insert(x + [11, 12, 13, 14])  # 52 / 5 = 10.4, then 112 / 15 = 7.47
        self.assertFalse(tree.has_state(x))
        tree.evict_states()
        self.assertFalse(tree.has_state(x + [11, 12, 13, 14]))
        self.assertTrue(tree.has_state(list(range(50, 67))))
        tree.check(self)

    def test_a_state_cached_above_units_rescores_them(self):
        """A state cached where an earlier insert split a key takes gain and
        exclusive KV from the unit below it, which then goes first."""
        tree = _Tree()
        tree.insert([1, 2, 3, 4, 5, 6])  # 6 / 16
        tree.insert([1, 2, 3], state=False)
        tree.insert(list(range(20, 25)))  # 5 / 15
        tree.insert([1, 2, 3])  # 3 / 10; the unit below drops to 3 / 13
        tree.evict_states()
        self.assertFalse(tree.has_state([1, 2, 3, 4, 5, 6]))
        self.assertTrue(tree.has_state([1, 2, 3]))
        tree.check(self)

    def test_a_state_cached_above_a_split_rescores_the_units_below_it(self):
        """The units under a stateless split below a newly cached state lose
        gain too, and are pushed again."""
        tree = _Tree()
        prefix = list(range(1, 21))
        tree.insert(prefix + [21, 4, 5, 6])  # 24 / 13, then 4 / 13
        tree.insert(prefix + [21, 7, 7])  # 23 / 12, then 3 / 12
        tree.insert([30, 31, 32, 33])  # 4 / 14
        tree.insert(prefix, state=False)
        tree.insert(prefix)  # interior: 20 / 10
        tree.evict_states()
        self.assertFalse(tree.has_state(prefix + [21, 7, 7]))
        self.assertTrue(tree.has_state([30, 31, 32, 33]))
        tree.check(self)

    def test_a_state_cached_below_makes_the_unit_above_interior(self):
        tree = _Tree(leaf_last=True)
        tree.insert([1, 2])
        tree.insert([1, 2, 3, 4], state=False)
        tree.insert([7])  # leaf: 1 / 11
        tree.insert([1, 2, 3, 4])  # [1, 2] is interior now: it goes first
        tree.evict_states()
        self.assertFalse(tree.has_state([1, 2]))
        self.assertTrue(tree.has_state([7]))
        tree.check(self)

    def test_the_context_term_values_states_deep_in_a_long_context(self):
        """With e, four tokens after 100 cost more to prefill than eight at
        the root; without it they cost less."""
        for e, deep_kept in ((2e5, True), (0.0, False)):
            with self.subTest(e=e):
                tree = _Tree(prefill_cost=[1.0, e])
                tree.insert(list(range(1, 101)))
                tree.insert(list(range(1, 105)))
                tree.insert(list(range(500, 508)))
                tree.evict_states()
                self.assertEqual(tree.has_state(list(range(1, 105))), deep_kept)
                self.assertEqual(tree.has_state(list(range(500, 508))), not deep_kept)

    def test_interior_states_go_before_leaves_with_leaf_last(self):
        for leaf_last in (False, True):
            with self.subTest(leaf_last=leaf_last):
                tree = _Tree(leaf_last=leaf_last)
                tree.insert([1, 2])  # interior once its child is cached: 2 / 10
                tree.insert([1, 2, 3, 4, 5, 6])  # leaf: 4 / 14
                tree.insert([7])  # leaf: 1 / 11
                tree.evict_states()
                self.assertEqual(tree.has_state([1, 2]), not leaf_last)
                self.assertEqual(tree.has_state([7]), leaf_last)
                tree.check(self)

    def test_a_branch_point_ranks_as_a_leaf(self):
        """With leaf_last an interior state goes before a branch point whose
        score is lower; without it the lowest score goes."""
        for leaf_last in (False, True):
            with self.subTest(leaf_last=leaf_last):
                tree = _Tree(leaf_last=leaf_last)
                tree.insert([1, 2])  # branch point: 2 / 10
                tree.insert([1, 2, 3])  # leaf: 1 / 11
                tree.insert([1, 2, 5])  # leaf: 1 / 11
                tree.insert([7, 8, 9])  # interior: 3 / 10
                tree.insert([7, 8, 9, 10, 11, 12])  # leaf: 3 / 13
                tree.evict_states()
                self.assertEqual(tree.has_state([1, 2, 3]), leaf_last)
                self.assertEqual(tree.has_state([7, 8, 9]), not leaf_last)
                self.assertTrue(tree.has_state([1, 2]))
                tree.check(self)

    def test_the_clock_rises_when_a_unit_goes_and_ages_older_units(self):
        tree = _Tree()
        tree.insert(A)
        tree.insert([7, 8, 9, 10])
        tree.evict_states()
        tree.insert([5, 6, 7, 8])  # same gain and size as the survivor, newer
        tree.evict_states()
        self.assertTrue(tree.has_state([5, 6, 7, 8]))
        self.assertFalse(tree.has_state([7, 8, 9, 10]))

    def test_a_hit_refreshes_the_stamp(self):
        for hit in (None, "match", "insert"):
            with self.subTest(hit=hit):
                tree = _Tree()
                tree.insert([50])  # the lowest score: it goes first
                tree.insert([7, 8, 9, 10])
                tree.evict_states()  # raises the clock
                tree.insert([5, 6, 7, 8])  # same gain and size, newer stamp
                if hit == "match":
                    tree.cache.match_prefix(
                        MatchPrefixParams(key=RadixKey(array("q", [7, 8, 9, 10, 11])))
                    )
                elif hit == "insert":
                    tree.insert([7, 8, 9, 10])  # the same state cached again
                tree.evict_states()
                self.assertEqual(tree.has_state([7, 8, 9, 10]), hit is not None)
                self.assertEqual(tree.has_state([5, 6, 7, 8]), hit is None)

    def test_hits_do_not_pile_up_heap_entries(self):
        """Each hit pushes its unit again; the stale entries are dropped once
        they outnumber the units."""
        tree = _Tree()
        tree.insert(A)
        tree.insert(B)
        for _ in range(500):
            tree.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", A))))
        self.assertLessEqual(len(tree.core.gds._heap), 2 * 2 + 64)
        tree.evict_states()
        self.assertFalse(tree.has_state(A))
        self.assertTrue(tree.has_state(B))

    def test_the_sanity_check_reports_a_unit_its_heap_entry_overrates(self):
        """A score that dropped without a push, or a unit with no live
        entry, would make the unit go later than its score says."""
        for drop_entry in (False, True):
            with self.subTest(drop_entry=drop_entry):
                tree = _Tree()
                tree.insert(A)
                gds, node = tree.core.gds, tree.node(A)
                if drop_entry:
                    gds._version[node] = -1
                else:
                    gds._stamp[node] -= 1
                with self.assertRaisesRegex(AssertionError, "heap entry"):
                    tree.cache.sanity_check()

    def test_kv_that_no_state_serves_goes_first_least_recent_first(self):
        for reused in (False, True):
            with self.subTest(reused=reused):
                tree = _Tree()
                tree.insert(A)
                tree.insert([9, 9, 9], state=False)
                tree.insert([8, 8, 8], state=False)
                if reused:
                    tree.insert([9, 9, 9], state=False)  # cached again: used last
                tree.cache.evict(EvictParams(num_tokens=3))
                self.assertEqual(tree.node([9, 9, 9]) is None, not reused)
                self.assertEqual(tree.node([8, 8, 8]) is None, reused)
                self.assertTrue(tree.has_state(A))
                tree.check(self)

    def test_kv_that_no_state_serves_goes_first_once_a_request_releases_it(self):
        tree = _Tree()
        tree.insert(A)  # 4 / 14
        tree.insert(B)  # 10 / 20
        tree.insert([9, 9, 9], state=False)
        held = tree.node([9, 9, 9]).id
        receipt = tree.core.inc_lock_ref(held)
        tree.cache.evict(EvictParams(num_tokens=3))
        self.assertIsNone(tree.node(A))  # the lowest unit instead
        tree.core.dec_lock_ref(held, receipt.to_dec_params())
        tree.cache.evict(EvictParams(num_tokens=3))
        self.assertIsNone(tree.node([9, 9, 9]))
        self.assertTrue(tree.has_state(B))
        tree.check(self)

    def test_a_locked_unit_is_passed_over_and_kept(self):
        tree = _Tree()
        tree.insert(A)
        tree.insert(B)
        receipt = tree.core.inc_lock_ref(tree.node(A).id)
        tree.evict_states()
        self.assertTrue(tree.has_state(A))
        self.assertFalse(tree.has_state(B))
        tree.core.dec_lock_ref(tree.node(A).id, receipt.to_dec_params())
        tree.evict_states()
        self.assertFalse(tree.has_state(A))
        tree.check(self)

    def test_a_token_request_takes_only_units_that_free_kv(self):
        tree = _Tree()
        tree.insert([1, 2])  # interior: frees no KV, the lowest score
        tree.insert(list(range(1, 11)))
        tree.insert(B)
        result = tree.cache.evict(EvictParams(num_tokens=1))
        self.assertGreaterEqual(result.num_tokens_evicted, 1)
        self.assertTrue(tree.has_state([1, 2]))
        tree.check(self)

    def test_a_state_slot_counts_as_the_kv_tokens_of_its_bytes(self):
        tree = _Tree(state_tokens=None)
        kv_pool = tree.allocator.get_kvcache()
        kv_bytes = kv_pool.get_kv_size_bytes()
        kv_bytes = sum(kv_bytes) if isinstance(kv_bytes, tuple) else kv_bytes
        mamba_pool = tree.pool.mamba_pool
        expected = (mamba_pool.get_kv_size_bytes() / mamba_pool.size) / (
            kv_bytes / kv_pool.size
        )
        self.assertAlmostEqual(tree.core.gds.state_tokens, expected)


class TestGreedyDualSizeRandomWorkload(CustomTestCase):
    def test_the_tree_stays_consistent_under_pressure(self):
        """The sanity check also holds every unit's heap entry to its score.
        With a token at position p costing 1 + p and a state slot counting as
        one token, the path cap lowers the scores of the units below it."""
        for leaf_last in (False, True):
            for e, state_tokens, path_cap in ((0, 8, -1), (1e5, 1, -1), (1e5, 1, 1)):
                for seed in range(3):
                    with self.subTest(
                        leaf_last=leaf_last, e=e, path_cap=path_cap, seed=seed
                    ):
                        self._run(seed, leaf_last, e, state_tokens, path_cap)

    def _run(self, seed, leaf_last, e, state_tokens, path_cap):
        rng = random.Random(seed)
        tree = _Tree(
            kv_size=160,
            state_tokens=state_tokens,
            leaf_last=leaf_last,
            prefill_cost=[1, e],
        )
        tree.cache.components[MAMBA].mamba_max_states_per_path = path_cap
        sessions = [[s * 1000] for s in range(6)]
        locks = []
        for step in range(250):
            op = rng.random()
            if op < 0.65:
                i = rng.randrange(len(sessions))
                if rng.random() < 0.15:
                    other = sessions[rng.randrange(len(sessions))]
                    sessions[i] = other[: rng.randint(1, len(other))]
                sessions[i] = sessions[i] + [
                    rng.randrange(1, 900) for _ in range(rng.randint(1, 6))
                ]
                if len(sessions[i]) > 48:
                    sessions[i] = [i * 1000 + step]
                need = len(sessions[i])
                if tree.allocator.available_size() < need:
                    tree.cache.evict(EvictParams(num_tokens=need))
                if tree.allocator.available_size() < need:
                    continue
                if tree.cache._pool_available(MAMBA) < 1:
                    tree.evict_states()
                tree.insert(sessions[i], state=rng.random() < 0.85)
            elif op < 0.72:
                tokens = sessions[rng.randrange(len(sessions))]
                node_id = tree.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens)))
                ).last_device_node
                if node_id != tree.core.root_node.id:
                    locks.append((node_id, tree.core.inc_lock_ref(node_id)))
            elif op < 0.8 and locks:
                node_id, receipt = locks.pop(rng.randrange(len(locks)))
                tree.core.dec_lock_ref(node_id, receipt.to_dec_params())
            elif op < 0.9:
                tree.evict_states(rng.randint(1, 3))
            else:
                tree.cache.evict(EvictParams(num_tokens=rng.randint(1, 8)))
            tree.check(self)


class TestGreedyDualSizeConfiguration(CustomTestCase):
    def test_parameters_are_checked_at_startup(self):
        self.assertEqual(get_eviction_strategy("gds").prefill_cost, (1.0, 0.0))
        for config in (
            {"prefill_cost": [1]},
            {"prefill_cost": [0, 1]},
            {"prefill_cost": [1, -1]},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                get_eviction_strategy("gds", config)
        for config in ({"leaflast": True}, {"leaf_idle_seconds": 60}):
            with self.subTest(config=config), self.assertRaises(TypeError):
                get_eviction_strategy("gds", config)

    def test_server_args_refuse_what_it_does_not_support(self):
        for flag, kwargs in (
            ("disable-radix-cache", dict(disable_radix_cache=True)),
            ("radix-cache-backend", dict(radix_cache_backend="custom")),
            ("hierarchical", dict(enable_hierarchical_cache=True)),
            ("session", dict(enable_session_radix_cache=True)),
        ):
            with self.subTest(flag=flag):
                args = ServerArgs(
                    model_path="dummy",
                    radix_eviction_policy="gds",
                    radix_eviction_policy_config={"prefill_cost": [1.862e-5, 3.33e-6]},
                    **kwargs,
                )
                with self.assertRaisesRegex(ValueError, flag):
                    handle_cache_compatibility(args)

    def test_the_cache_refuses_what_it_cannot_rank(self):
        mamba_tree = dict(components=(FULL, MAMBA), eviction_policy="gds")
        for message, cfg, context in (
            (
                "ranks state units",
                CacheConfig(**mamba_tree),
                patch.object(UnifiedTreeCore, "supports_gds_eviction", False),
            ),
            (
                "cannot size a state slot",
                CacheConfig(**mamba_tree),
                patch.object(MambaPool, "get_kv_size_bytes", return_value=0),
            ),
            (
                "full attention and Mamba states only",
                CacheConfig(
                    components=(FULL, ComponentType.SWA, MAMBA),
                    sliding_window_size=8,
                    eviction_policy="gds",
                ),
                nullcontext(),
            ),
            (
                "hybrid",
                CacheConfig(components=(FULL,), eviction_policy="gds"),
                nullcontext(),
            ),
        ):
            with (
                self.subTest(message),
                patch(
                    "sglang.test.kits.unified_radix_cache.fixtures.get_device",
                    return_value="cpu",
                ),
                context,
                self.assertRaisesRegex(ValueError, message),
            ):
                build_fixture(cfg)

    def test_gds_runs_on_the_python_tree_core(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            eviction_policy="gds",
        )
        self.assertEqual(resolve_tree_core_backend("rust", params), "python")


if __name__ == "__main__":
    unittest.main()
