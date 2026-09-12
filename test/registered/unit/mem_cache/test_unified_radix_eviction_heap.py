"""Persistent lazy eviction heap for the UnifiedRadixCache Full component.

Covers ``_LazyLeafHeap`` directly and proves that eviction order is identical
to the legacy per-call heap rebuild by replaying the same randomized op stream
on two caches: one running the new code and one whose Full component carries
the legacy ``_evict_device_*`` / ``drive_host_eviction`` bodies (installed
through ``component_registry_override``).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import heapq
import random
import unittest
from array import array
from types import SimpleNamespace
from typing import Optional

import torch
from numpy import float64

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components import (
    tree_component as _tree_component,
)
from sglang.srt.mem_cache.unified_cache.components.full_component import (
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import _LazyLeafHeap
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase

POLICIES = ("lru", "lfu", "fifo", "mru", "filo", "priority", "slru")


# ---------------------------------------------------------------------------
# Legacy oracle: the Full component as it was before the persistent heap.
# ---------------------------------------------------------------------------
class LegacyFullComponent(FullComponent):
    """Verbatim pre-change eviction drivers (rebuild the heap every call)."""

    def _evict_device_start(self, request_cnt: int) -> None:
        self._ensure_eviction_strategy()
        self._evict_device_request_cnt = request_cnt
        self._evict_device_last_node = None
        self._evict_device_heap = [
            (self.session_ref_eviction_strategy(n), n)
            for n in self.tree_core.evictable_device_leaves
        ]
        heapq.heapify(self._evict_device_heap)

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        ct = self.component_type
        lv = self._evict_device_last_node
        if (
            lv is not None
            and lv.parent is not None
            and lv.parent in self.tree_core.evictable_device_leaves
        ):
            heapq.heappush(
                self._evict_device_heap,
                (self.session_ref_eviction_strategy(lv.parent), lv.parent),
            )
        self._evict_device_last_node = None
        while tracker[ct] < self._evict_device_request_cnt and self._evict_device_heap:
            _, x = heapq.heappop(self._evict_device_heap)
            if x not in self.tree_core.evictable_device_leaves:
                continue
            self._evict_device_last_node = x
            return x.id
        return None

    def _evict_device_end(self) -> None:
        self._evict_device_heap = []
        self._evict_device_last_node = None

    def drive_host_eviction(self, num_tokens, tracker, device_frees, host_frees):
        self._ensure_eviction_strategy()
        heap = [
            (self.session_ref_eviction_strategy(n), n)
            for n in self.tree_core.evictable_host_leaves
        ]
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.tree_core.evictable_host_leaves:
                continue
            self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)
            if (
                x.parent is not None
                and x.parent in self.tree_core.evictable_host_leaves
            ):
                heapq.heappush(
                    heap,
                    (self.session_ref_eviction_strategy(x.parent), x.parent),
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_cache(
    *,
    policy: str = "lru",
    kv_size: int = 512,
    enable_session: bool = False,
    legacy: bool = False,
    page_size: int = 1,
) -> UnifiedRadixCache:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=kv_size,
        page_size=page_size,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=kv_size, dtype=dtype, device="cpu", kvcache=kv_pool, need_sort=False
    )
    req_pool = ReqToTokenPool(
        size=8, max_context_len=1024, device="cpu", enable_memory_saver=False
    )
    return UnifiedRadixCache(
        params=CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            eviction_policy=policy,
            enable_session_radix_cache=enable_session,
            tree_components=(ComponentType.FULL,),
            component_registry_override=(
                {ComponentType.FULL: LegacyFullComponent} if legacy else None
            ),
        )
    )


def node_path(core, node) -> tuple:
    """Root-to-node token path: identifies a node across two trees."""
    parts = []
    while node is not None and node is not core.root_node:
        parts.append(tuple(node.key.token_ids))
        node = node.parent
    return tuple(reversed(parts))


def reset_time_counter() -> None:
    _tree_component._LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)


def gen_sequences(rng: random.Random, n: int, page_size: int) -> list[list[int]]:
    """Sequences with tree-like prefix sharing (chains + fan-out)."""
    seqs = [[rng.randint(1, 50) for _ in range(page_size)]]
    while len(seqs) < n:
        parent = rng.choice(seqs)
        tail = [rng.randint(1, 50) for _ in range(page_size * rng.randint(1, 6))]
        seqs.append(parent + tail)
    return seqs


def gen_ops(seed: int, num_ops: int, page_size: int, session: bool) -> list[tuple]:
    rng = random.Random(seed)
    seqs = gen_sequences(rng, 60, page_size)
    ops = []
    for _ in range(num_ops):
        r = rng.random()
        if r < 0.40:
            ops.append(("insert", rng.choice(seqs), rng.randint(0, 3)))
        elif r < 0.60:
            ops.append(("match", rng.choice(seqs)))
        elif r < 0.70:
            ops.append(("lock", rng.choice(seqs)))
        elif r < 0.78:
            ops.append(("unlock", rng.randint(0, 1 << 30)))
        elif session and r < 0.84:
            ops.append(("register", rng.choice(seqs), f"s{rng.randint(0, 3)}"))
        elif session and r < 0.88:
            ops.append(("release", f"s{rng.randint(0, 3)}"))
        else:
            ops.append(("evict", page_size * rng.randint(1, 6)))
    return ops


class Replay:
    """Drive one cache through an op stream, recording every eviction victim."""

    def __init__(self, cache: UnifiedRadixCache, check_every: int = 25):
        self.cache = cache
        self.core = cache.tree_core
        self.check_every = check_every
        self.victims: list[list[tuple]] = []
        self.evicted_counts: list[int] = []
        self._current: Optional[list] = None
        self._locked: list[tuple] = []
        orig = self.core.evict_device_leaf
        core = self.core

        def recording_evict_device_leaf(node_id, is_write_back):
            if self._current is not None:
                self._current.append(node_path(core, core.node_by_id(node_id)))
            return orig(node_id, is_write_back)

        self.core.evict_device_leaf = recording_evict_device_leaf

    def _alloc(self, n: int):
        alloc = self.cache.token_to_kv_pool_allocator
        v = alloc.alloc(n)
        if v is None:
            self.cache.evict(EvictParams(num_tokens=n * 2))
            v = alloc.alloc(n)
        return v

    def run(self, ops: list[tuple]) -> None:
        for i, op in enumerate(ops):
            kind = op[0]
            if kind == "insert":
                seq, prio = op[1], op[2]
                v = self._alloc(len(seq))
                if v is not None:
                    self.cache.insert(
                        InsertParams(
                            key=RadixKey(array("q", seq)),
                            value=v.to(torch.int64),
                            priority=prio,
                        )
                    )
            elif kind == "match":
                self.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", op[1])))
                )
            elif kind == "lock":
                res = self.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", op[1])))
                )
                node_id = res.last_device_node
                if node_id != self.cache.root_node_handle():
                    lr = self.cache.inc_lock_ref(node_id)
                    self._locked.append((node_id, lr))
            elif kind == "unlock":
                if self._locked:
                    node_id, lr = self._locked.pop(op[1] % len(self._locked))
                    self.cache.dec_lock_ref(
                        node_id,
                        DecLockRefParams(swa_uuid_for_lock=lr.swa_uuid_for_lock),
                    )
            elif kind == "register":
                seq, sid = op[1], op[2]
                res = self.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq)))
                )
                if res.last_device_node != self.cache.root_node_handle():
                    self.cache.session_refs.register_session_ref(
                        SimpleNamespace(
                            session_id=sid,
                            session_generation=self.cache.ensure_session_generation(
                                sid
                            ),
                            session=None,
                            last_node=res.last_device_node,
                            origin_input_ids=array("q", seq),
                            output_ids=array("q"),
                            extra_key=None,
                        )
                    )
            elif kind == "release":
                self.cache.release_radix_session(op[1])
            elif kind == "evict":
                self._current = []
                res = self.cache.evict(EvictParams(num_tokens=op[1]))
                self.victims.append(self._current)
                self.evicted_counts.append(res.num_tokens_evicted)
                self._current = None
            if i % self.check_every == 0:
                self.cache.sanity_check()
        # release everything so the final state is comparable
        while self._locked:
            node_id, lr = self._locked.pop()
            self.cache.dec_lock_ref(
                node_id, DecLockRefParams(swa_uuid_for_lock=lr.swa_uuid_for_lock)
            )
        self.cache.sanity_check()

    def leaf_paths(self) -> set:
        return {node_path(self.core, n) for n in self.core.evictable_device_leaves}


def replay_pair(
    policy: str, seed: int, session: bool, page_size: int = 1, kill_switch=False
):
    ops = gen_ops(seed, 320, page_size, session)
    reset_time_counter()
    # Pin the lazy side explicitly: the env var is read at UnifiedTreeCore
    # construction, so an exported kill switch would otherwise flip both sides.
    with envs.SGLANG_UNIFIED_RADIX_LAZY_EVICTION_HEAP.override(True):
        new = Replay(
            make_cache(policy=policy, enable_session=session, page_size=page_size)
        )
    new.run(ops)
    reset_time_counter()
    if kill_switch:
        with envs.SGLANG_UNIFIED_RADIX_LAZY_EVICTION_HEAP.override(False):
            ref = Replay(
                make_cache(policy=policy, enable_session=session, page_size=page_size)
            )
    else:
        ref = Replay(
            make_cache(
                policy=policy, enable_session=session, page_size=page_size, legacy=True
            )
        )
    ref.run(ops)
    return new, ref


# ---------------------------------------------------------------------------
# _LazyLeafHeap unit tests
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("id", "key", "parent")

    def __init__(self, id, key):
        self.id = id
        self.key = key
        self.parent = None

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return f"N{self.id}"


class TestLazyLeafHeapUnit(CustomTestCase):
    def _heap(self, nodes, **kw):
        members = set(nodes)
        heap = _LazyLeafHeap(members, lambda: lambda n: n.key, **kw)
        for n in nodes:
            heap.refresh(n)
        return members, heap

    def _drain(self, heap):
        heap.begin_walk()
        out = []
        while True:
            n = heap.pop_next()
            if n is None:
                break
            out.append(n)
        heap.end_walk()
        return out

    def test_pop_order_and_key_updates(self):
        nodes = [_Node(i, k) for i, k in enumerate([5, 3, 9, 1])]
        members, heap = self._heap(nodes)
        errors = []
        heap.check_invariants(errors.append, "t")
        self.assertEqual(errors, [])
        nodes[2].key = 0  # 9 -> 0: must come first after a touch
        heap.touch(nodes[2])
        self.assertEqual(self._drain(heap), [nodes[2], nodes[3], nodes[1], nodes[0]])
        # a full walk yields every member exactly once, then they are all live again
        errors = []
        heap.check_invariants(errors.append, "t")
        self.assertEqual(errors, [])

    def test_touch_ignores_non_members_and_forget_removes(self):
        nodes = [_Node(i, i) for i in range(4)]
        members, heap = self._heap(nodes)
        outsider = _Node(99, -1)
        heap.touch(outsider)
        heap.refresh(outsider)
        self.assertEqual(len(heap), 4)
        members.discard(nodes[0])
        heap.forget(nodes[0])
        self.assertEqual(self._drain(heap), nodes[1:])

    def test_walk_snapshot_entrants_and_promote(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        heap.begin_walk()
        first = heap.pop_next()
        self.assertIs(first, nodes[0])
        entrant = _Node(10, -5)  # smallest key, enters mid-walk
        members.add(entrant)
        heap.refresh(entrant)
        self.assertIs(heap.pop_next(), nodes[1])  # invisible without promote
        heap.promote(entrant)
        self.assertIs(heap.pop_next(), entrant)  # visible right after promote
        self.assertIs(heap.pop_next(), nodes[2])
        self.assertIsNone(heap.pop_next())
        heap.end_walk()
        # every member (including the yielded ones) is live again
        self.assertEqual(set(heap._live), members)
        errors = []
        heap.check_invariants(errors.append, "t")
        self.assertEqual(errors, [])

    def test_mid_walk_touch_is_deferred_to_end_walk(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        heap.begin_walk()
        nodes[2].key = -100
        heap.touch(nodes[2])  # key frozen for this walk
        self.assertIs(heap.pop_next(), nodes[0])
        heap.end_walk()
        # after the walk the fresh key is honoured; the yielded node is live again
        self.assertEqual(self._drain(heap), [nodes[2], nodes[0], nodes[1]])

    def test_nested_walk_is_rejected(self):
        members, heap = self._heap([_Node(0, 0)])
        heap.begin_walk()
        with self.assertRaises(AssertionError):
            heap.begin_walk()
        heap.end_walk()

    def test_compaction_bound(self):
        nodes = [_Node(i, i) for i in range(8)]
        members, heap = self._heap(nodes)
        for t in range(5000):
            nodes[t % 8].key = 1000 + t
            heap.touch(nodes[t % 8])
        self.assertLessEqual(len(heap._heap), 2 * len(heap._live) + 64)
        errors = []
        heap.check_invariants(errors.append, "t")
        self.assertEqual(errors, [])
        self.assertEqual(self._drain(heap), sorted(nodes, key=lambda n: n.key))

    def test_rebuild_each_walk_matches(self):
        nodes = [_Node(i, k) for i, k in enumerate([4, 2, 8, 6])]
        _, lazy = self._heap(nodes)
        _, eager = self._heap(nodes, rebuild_each_walk=True)
        nodes[0].key = 1  # key change without any hook: only the rebuild sees it
        self.assertEqual(self._drain(eager)[0], nodes[0])
        self.assertEqual(self._drain(lazy)[0], nodes[1])

    def test_check_invariants_reports_divergence(self):
        nodes = [_Node(i, i) for i in range(3)]
        members, heap = self._heap(nodes)
        members.discard(nodes[1])  # membership removed without forget()
        errors = []
        heap.check_invariants(errors.append, "t")
        self.assertTrue(any("live but not member" in e for e in errors))
        members.add(nodes[1])
        nodes[2].key = 42  # key changed without touch()
        errors = []
        heap.check_invariants(errors.append, "t")
        self.assertTrue(any("out of date" in e for e in errors))


# ---------------------------------------------------------------------------
# Order parity against the legacy per-call rebuild
# ---------------------------------------------------------------------------
class TestEvictionOrderParity(CustomTestCase):
    def _assert_parity(self, new: Replay, ref: Replay, label: str):
        self.assertEqual(len(new.victims), len(ref.victims), label)
        for i, (a, b) in enumerate(zip(new.victims, ref.victims)):
            self.assertEqual(a, b, f"{label}: victims differ at evict call {i}")
        self.assertEqual(new.evicted_counts, ref.evicted_counts, label)
        self.assertEqual(new.leaf_paths(), ref.leaf_paths(), label)
        self.assertGreater(sum(len(v) for v in new.victims), 0, label)

    def test_parity_all_policies(self):
        for policy in POLICIES:
            for session in (False, True):
                for seed in range(12):
                    label = f"policy={policy} session={session} seed={seed}"
                    with self.subTest(label):
                        new, ref = replay_pair(policy, seed, session)
                        self._assert_parity(new, ref, label)

    def test_parity_page_size_4(self):
        for policy in ("lru", "lfu", "mru"):
            for seed in range(2):
                label = f"page_size=4 policy={policy} seed={seed}"
                with self.subTest(label):
                    new, ref = replay_pair(policy, seed, False, page_size=4)
                    self._assert_parity(new, ref, label)

    def test_kill_switch_parity(self):
        for policy in ("lru", "slru"):
            label = f"kill-switch policy={policy}"
            with self.subTest(label):
                new, ref = replay_pair(policy, 7, True, kill_switch=True)
                self._assert_parity(new, ref, label)
                self.assertTrue(ref.core.full_device_heap.rebuild_each_walk)
                self.assertFalse(new.core.full_device_heap.rebuild_each_walk)


# ---------------------------------------------------------------------------
# Targeted behaviours on a real cache
# ---------------------------------------------------------------------------
def _insert(cache, tokens):
    v = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    cache.insert(
        InsertParams(key=RadixKey(array("q", tokens)), value=v.to(torch.int64))
    )


def _match_len(cache, tokens) -> int:
    return len(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).device_indices
    )


class TestHeapOnRealCache(CustomTestCase):
    def test_mru_touched_leaf_is_evicted_first(self):
        cache = make_cache(policy="mru")
        _insert(cache, [1, 2, 3])
        _insert(cache, [7, 8, 9])
        _match_len(cache, [7, 8, 9])  # most recently used -> first victim under MRU
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(_match_len(cache, [7, 8, 9]), 0)
        self.assertEqual(_match_len(cache, [1, 2, 3]), 3)
        cache.sanity_check()

    def test_lru_touched_leaf_is_protected(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        _insert(cache, [7, 8, 9])
        _match_len(cache, [1, 2, 3])  # refresh -> [7,8,9] is the LRU victim
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(_match_len(cache, [7, 8, 9]), 0)
        self.assertEqual(_match_len(cache, [1, 2, 3]), 3)
        cache.sanity_check()

    def test_session_release_returns_leaf_to_unreferenced_band(self):
        cache = make_cache(policy="lru", enable_session=True)
        _insert(cache, [1, 2, 3, 4])
        _insert(cache, [7, 8, 9])
        res = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
        )
        cache.session_refs.register_session_ref(
            SimpleNamespace(
                session_id="s1",
                session_generation=cache.ensure_session_generation("s1"),
                session=None,
                last_node=res.last_device_node,
                origin_input_ids=array("q", [1, 2, 3, 4]),
                output_ids=array("q"),
                extra_key=None,
            )
        )
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(_match_len(cache, [7, 8, 9]), 0)  # unreferenced goes first
        self.assertEqual(_match_len(cache, [1, 2, 3, 4]), 4)
        cache.release_radix_session("s1")
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(
            _match_len(cache, [1, 2, 3, 4]), 0
        )  # released leaf is evictable again
        cache.sanity_check()

    def test_parent_promotion_within_one_call(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2])
        _insert(cache, [1, 2, 3, 4])
        # One call for 4 tokens must evict the leaf [3,4] and then its parent [1,2].
        res = cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(res.num_tokens_evicted, 4)
        self.assertEqual(_match_len(cache, [1, 2]), 0)
        cache.sanity_check()

    def test_reset_empties_heaps(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        self.assertEqual(len(cache.tree_core.full_device_heap), 1)
        cache.reset()
        self.assertEqual(len(cache.tree_core.full_device_heap), 0)
        self.assertEqual(len(cache.tree_core.full_host_heap), 0)
        cache.sanity_check()

    def test_exception_mid_walk_leaves_heap_consistent(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        _insert(cache, [7, 8, 9])
        core = cache.tree_core
        orig = core.evict_device_leaf

        def boom(node_id, is_write_back):
            raise RuntimeError("injected")

        core.evict_device_leaf = boom
        with self.assertRaises(RuntimeError):
            cache.evict(EvictParams(num_tokens=3))
        core.evict_device_leaf = orig
        self.assertFalse(core.full_device_heap.walking)
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=6))
        self.assertEqual(_match_len(cache, [1, 2, 3]) + _match_len(cache, [7, 8, 9]), 0)
        cache.sanity_check()

    def test_eviction_walk_keeps_heap_compact(self):
        """Regression: a walk must not leave one stale entry per evicted leaf."""
        cache = make_cache(policy="mru")
        seqs = [[1000 + i * 3 + j for j in range(3)] for i in range(30)]
        for s in seqs:
            _insert(cache, s)
        for _ in range(3):
            for s in seqs:
                _match_len(cache, s)
        cache.evict(EvictParams(num_tokens=45))
        heap = cache.tree_core.full_device_heap
        self.assertLessEqual(len(heap._heap), 2 * len(heap._live) + 64)
        cache.sanity_check()

    def test_many_matches_keep_heap_compact(self):
        cache = make_cache(policy="lru")
        _insert(cache, [1, 2, 3])
        for _ in range(5000):
            _match_len(cache, [1, 2, 3])
        heap = cache.tree_core.full_device_heap
        self.assertLessEqual(len(heap._heap), 2 * len(heap._live) + 64)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
