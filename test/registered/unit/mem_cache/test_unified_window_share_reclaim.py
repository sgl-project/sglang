import itertools
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    PAGE,
    InstantWriteBackController,
    SlotOnlyHostPool,
)
from sglang.test.test_utils import CustomTestCase
from sglang.test.token_demand import token_demand
from sglang.test.tree_slots import tree_slots_of
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_swa_cache,
    build_tri_cache,
    build_unified_bundle,
    build_window_chain,
    largest_take,
    setup_allocator_context,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_BUILD_IS_CUDA = "sglang.srt.mem_cache.kv.build.is_cuda"
_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16)


def _low_mark(share):
    return share - share // 16


def _window_storage(windows, *, cuda, page_size=4):
    plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=_MHA,
            windows=[
                WindowLayers(layer_ids=(i + 1,), geometry=_MHA, window=w)
                for i, w in enumerate(windows)
            ],
        ),
        page_size=page_size,
    )
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=32 * page_size,
            window_tokens=16 * page_size,
            state_slots=0,
            max_running_requests=3,
            context_len=16 * page_size,
            alignment=1,
        ),
    )
    with mock.patch(_BUILD_IS_CUDA, return_value=cuda):
        storage = build_kv_storage(
            plan,
            sized,
            UnifiedBuildContext(
                device="cpu",
                start_layer=0,
                end_layer=len(windows) + 1,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=3,
                model_context_len=16 * page_size,
                lazy_compaction=True,
            ),
        )
    return storage, sized


class TestShareWiring(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def test_a_cuda_window_chain_takes_the_sized_share(self):
        storage, sized = _window_storage((16, 8), cuda=True)
        allocator = storage.allocator
        self.assertEqual(allocator.window_share_rows, sized.window_tokens)
        for name in allocator.window_pool_names():
            self.assertEqual(allocator.window_share_tokens(name), sized.window_tokens)

    def test_other_platforms_and_state_chains_take_none(self):
        storage, _ = _window_storage((16,), cuda=False)
        self.assertIsNone(storage.allocator.window_share_rows)
        self.assertIsNone(storage.allocator.window_share_tokens("window_16"))
        with mock.patch(_BUILD_IS_CUDA, return_value=True):
            tri = build_unified_bundle(
                UnifiedGeometry(
                    chain="tri",
                    full_layer_ids=(0, 1),
                    swa_layer_ids=(2, 3),
                    state_layer_ids=(0, 2),
                )
            )
        self.assertIsNone(tri.token_to_kv_pool_allocator.window_share_rows)

    def test_resize_takes_the_resized_share(self):
        storage, sized = _window_storage((16,), cuda=True)
        allocator = storage.allocator
        allocator.arena.backed_after_capture = True
        allocator.resize(
            SimpleNamespace(
                unified_memory_pool_bytes=allocator.arena.total_bytes,
                unified_full_cap_tokens=None,
                full_max_total_num_tokens=sized.full_tokens,
                swa_max_total_num_tokens=sized.window_tokens - 16,
                max_total_num_tokens=sized.full_tokens,
            )
        )
        self.assertEqual(
            allocator.window_share_tokens("window_16"), sized.window_tokens - 16
        )


class TestWindowShareTrim(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def _reclaim(self, page_size, occupancy, share_rows, windows_first, locked=False):
        cache, allocator = _write_back_cache(
            occupancy, window_host=True, page_size=page_size
        )
        owner = allocator.full_attn_allocator.sub_pool_name
        (window,) = allocator.window_pool_names()
        allocator.window_share_rows = share_rows
        # Found without a match, which would make it the most recent entry.
        oldest = _node_id(cache, list(range(page_size)))
        lock = cache.lock(oldest) if locked else None
        try:
            demand = KVDemand(largest_take(allocator, owner) + 2, {window: 0})
            waits = []
            with _count_waits(cache, waits):
                result = cache.reclaim_jointly(demand, windows_first=windows_first)
            self.assertTrue(allocator.ensure_capacity(demand))
            outcome = (
                result.num_tokens_evicted // page_size,
                cache.window_evictable_size(window) // page_size,
                len(waits),
                _on_device(cache, oldest),
            )
        finally:
            cache.unlock(lock)
        cache.sanity_check()
        return outcome

    def test_a_trim_starts_past_the_high_mark_and_ends_at_the_low_mark(self):
        for windows_first, page_size, occupancy in itertools.product(
            (False, True), (1, 16, 64), (33, 34, 35)
        ):
            with self.subTest(
                windows_first=windows_first, page_size=page_size, occupancy=occupancy
            ):
                trimmed = self._reclaim(
                    page_size, occupancy, 32 * page_size, windows_first
                )
                unshared = self._reclaim(page_size, occupancy, None, windows_first)
                self.assertEqual(unshared, (1, occupancy - 1, 1, False))
                if occupancy <= 34:
                    self.assertEqual(trimmed, unshared)
                else:
                    self.assertEqual(trimmed, (5, 30, 5, False))

    def test_a_locked_prefix_is_kept_out_of_the_trim(self):
        for windows_first, page_size in itertools.product((False, True), (1, 16, 64)):
            with self.subTest(windows_first=windows_first, page_size=page_size):
                share = 32 * page_size
                unlocked = self._reclaim(page_size, 36, share, windows_first)
                self.assertFalse(unlocked[3])
                self.assertEqual(
                    self._reclaim(page_size, 36, share, windows_first, locked=True),
                    (5, 30, 5, True),
                )

    def test_joint_lru_evicts_by_recency_alone(self):
        _, allocator, _ = build_window_chain(
            n_full_slots=48, n_swa_slots=48, full_layer_num=2, swa_layer_num=6
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                disable=False,
                sliding_window_size=4,
                tree_slots=tree_slots_of(
                    (ComponentType.FULL, ComponentType.SWA), window=4
                ),
                tree_core_backend="python",
                eviction_policy="joint_lru",
            )
        )
        self.assertTrue(cache.evicts_by_joint_lru)
        keys, start = [], 1000
        while (ids := allocator.alloc(4)) is not None:
            keys.append(list(range(start, start + 4)))
            cache.insert(InsertParams(key=RadixKey(array("q", keys[-1])), value=ids))
            start += 100
        cache.match_prefix(MatchPrefixParams(RadixKey(array("q", keys[0]))))
        allocator.window_share_rows = 1
        (window,) = allocator.window_pool_names()
        cache.evict_for_alloc(KVDemand(4, {window: 4}))
        cached = [_cached(cache, key) for key in keys]
        self.assertEqual(cached, [i != 1 for i in range(len(keys))])


def _node_id(cache, key):
    stack = list(cache.tree_core.root_node.children.values())
    while stack:
        node = stack.pop()
        if list(node.key) == key:
            return node.id
        stack.extend(node.children.values())
    return None


def _cached(cache, key):
    return _node_id(cache, key) is not None


def _on_device(cache, node_id):
    return all(
        cache.tree_core.get_component_device_value(node_id, ct) is not None
        for ct in (ComponentType.FULL, *cache.window_slots())
    )


class TestSecondaryWindowShare(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def _cache(self, page_size, prefix_tokens=None):
        storage, _ = _window_storage((16, 8, 4), cuda=True, page_size=page_size)
        allocator = storage.allocator
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=storage.requests,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                sliding_window_size=16,
                tree_slots=storage.tree_slots(),
            )
        )
        start = 1000
        tokens = prefix_tokens or page_size
        while (ids := allocator.alloc(tokens)) is not None:
            key = RadixKey(array("q", range(start, start + tokens)))
            cache.insert(InsertParams(key=key, value=ids.clone()))
            start += tokens
        return storage, allocator, cache

    def _reclaim(self, page_size, shares):
        storage, allocator, cache = self._cache(page_size)
        windows = allocator.window_pool_names()
        before = cache.window_evictable_size("window_8") // page_size
        with mock.patch.object(
            allocator, "window_share_tokens", side_effect=shares.get
        ):
            result = cache.evict_for_alloc(
                KVDemand(page_size, {name: 0 for name in windows})
            )
        self.assertTrue(allocator.units_fit({"full": 1}))
        self.assertEqual(storage.accounting_violations(), [])
        return (
            before,
            result.num_tokens_evicted // page_size,
            cache.window_evictable_size("window_8") // page_size,
        )

    def test_only_a_secondary_window_past_its_share_is_trimmed(self):
        for page_size in (1, 16, 64):
            with self.subTest(page_size=page_size):
                roomy = dict.fromkeys(("window_16", "window_8", "window_4"), 1 << 30)
                before, victims, _ = self._reclaim(page_size, roomy)
                self.assertEqual(victims, 1)
                share = {**roomy, "window_8": 16 * page_size}
                before, victims, cached = self._reclaim(page_size, share)
                self.assertGreater(before, 17)
                self.assertEqual((victims, cached), (before - 15, 15))

    def test_a_window_demand_takes_its_own_entries_first(self):
        storage, allocator, cache = self._cache(4, prefix_tokens=16)
        share = allocator.window_share_tokens("window_4")
        self.assertGreater(cache.window_evictable_size("window_4"), share + share // 16)
        self.assertFalse(allocator.units_fit({"window_4": 1}))
        full, own = cache.full_evictable_size(), cache.window_evictable_size("window_4")
        with mock.patch.object(urc, "_is_cuda", True):
            self.assertTrue(cache.reclaim({"window_4": 1}))
        self.assertLess(cache.window_evictable_size("window_4"), own)
        self.assertEqual(cache.full_evictable_size(), full)
        self.assertEqual(storage.accounting_violations(), [])


def _write_back_cache(occupancy, *, window_host, page_size=PAGE, host_pages=64):
    allocator, cache = build_swa_cache(occupancy=occupancy, page_size=page_size)
    host_pool = SlotOnlyHostPool(host_pages * page_size, page_size)
    entries = [
        PoolEntry(
            name=PoolName.KV,
            host_pool=host_pool,
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
            is_primary_index_anchor=True,
        )
    ]
    window_pool = None
    if window_host:
        window_pool = SlotOnlyHostPool(64 * page_size, page_size)
        entries.append(
            PoolEntry(
                name=PoolName.SWA,
                host_pool=window_pool,
                device_pool=None,
                layer_mapper=lambda layer_id: layer_id,
            )
        )
    host_group = HostPoolGroup(entries)
    cache.host_pool_group = host_group
    cache.cache_controller = InstantWriteBackController(host_group, allocator)
    cache.components[ComponentType.FULL]._full_kv_pool_host = host_pool
    if window_host:
        (window,) = allocator.window_pool_names()
        cache.components[cache.window_slot_of(window)]._swa_kv_pool_host = window_pool
        cache.tree_core.has_swa_host_pool = True
    cache.tree_core.set_hicache_enabled()
    cache.is_write_back = True
    cache.write_through_threshold = 1 << 30
    cache.load_back_threshold = 10
    return cache, allocator


def _count_waits(cache, waits):
    writing_check = cache.writing_check

    def count(write_back=False, finish_count=None):
        if write_back and cache.ongoing_write_through:
            waits.append(len(cache.ongoing_write_through))
        return writing_check(write_back=write_back, finish_count=finish_count)

    return mock.patch.object(cache, "writing_check", side_effect=count)


class TestSuccessiveSmallDemandsWithWindowWriteBack(CustomTestCase):
    ROUNDS = 12

    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def _rounds(self, share_rows, windows_first):
        cache, allocator = _write_back_cache(40, window_host=True)
        (window,) = allocator.window_pool_names()
        allocator.window_share_rows = share_rows
        while allocator.alloc(PAGE) is not None:
            pass
        waits, rounds = [], []
        start = 1 << 20
        with _count_waits(cache, waits):
            for _ in range(self.ROUNDS):
                waited, cached = len(waits), cache.window_evictable_size(window)
                cache.reclaim_jointly(
                    KVDemand(PAGE, {window: PAGE}), windows_first=windows_first
                )
                ids = allocator.alloc(PAGE)
                self.assertIsNotNone(ids)
                rounds.append(
                    (len(waits) - waited, cached - cache.window_evictable_size(window))
                )
                key = RadixKey(array("q", range(start, start + PAGE)))
                cache.insert(InsertParams(key=key, value=ids))
                start += 1000
        cache.sanity_check()
        return rounds, cache.window_evictable_size(window)

    def test_without_a_share_every_demand_waits(self):
        for windows_first in (False, True):
            with self.subTest(windows_first=windows_first):
                rounds, _ = self._rounds(None, windows_first)
                self.assertEqual(
                    [waits > 0 for waits, _ in rounds], [True] * self.ROUNDS
                )

    def test_a_trim_waits_in_one_round_and_serves_the_next_demands(self):
        share = 64
        for windows_first in (False, True):
            with self.subTest(windows_first=windows_first):
                rounds, cached = self._rounds(share, windows_first)
                (first_waits, first_evicted), rest = rounds[0], rounds[1:]
                self.assertGreater(first_waits, 0)
                self.assertGreater(first_evicted, PAGE)
                self.assertEqual(rest, [(0, 0)] * (self.ROUNDS - 1))
                self.assertEqual(cached, _low_mark(share) + self.ROUNDS * PAGE)


class TestOwnerReclaimBesideWindows(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def _reclaim(self, sized, *, occupancy, owner_pages, window_pages, capped):
        cache, allocator = _write_back_cache(occupancy, window_host=False)
        while allocator.alloc(PAGE) is not None:
            pass
        owner = allocator.full_attn_allocator
        if capped:
            owner.max_live_tokens = owner.allocated_count()
        (window,) = allocator.window_pool_names()
        demand = KVDemand(owner_pages * PAGE, {window: window_pages * PAGE})
        waits = []
        with mock.patch.object(urc, "_is_cuda", sized), _count_waits(cache, waits):
            result = cache.evict_for_alloc(demand)
        self.assertTrue(allocator.ensure_capacity(demand))
        cache.sanity_check()
        return waits, result.num_tokens_evicted // PAGE

    def test_queued_victims_are_the_per_victim_walks_behind_one_wait(self):
        for capped, window_pages in itertools.product((False, True), (0, 4, 8)):
            with self.subTest(capped=capped, window_pages=window_pages):
                args = dict(
                    occupancy=16,
                    owner_pages=4,
                    window_pages=window_pages,
                    capped=capped,
                )
                unsized, victims = self._reclaim(False, **args)
                self.assertEqual(unsized, [1] * victims)
                self.assertEqual(self._reclaim(True, **args), ([victims], victims))

    def test_a_long_walk_waits_once(self):
        unsized, victims = self._reclaim(
            False, occupancy=64, owner_pages=48, window_pages=48, capped=False
        )
        self.assertGreaterEqual(victims, 40)
        self.assertEqual(unsized, [1] * victims)
        self.assertEqual(
            self._reclaim(
                True, occupancy=64, owner_pages=48, window_pages=48, capped=False
            ),
            ([victims], victims),
        )


class TestStateChainKeepsPerVictimWalk(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def _reclaim(self, cuda):
        _, allocator, cache = build_tri_cache()
        keys = []
        for i in range(6):
            key = RadixKey(array("q", range(i * 1000, i * 1000 + 4)))
            state = allocator.state_slots.alloc(1).reshape(1)
            cache.insert(
                InsertParams(key=key, value=allocator.alloc(4), mamba_value=state)
            )
            keys.append(key)
        while allocator.state_slots.alloc(1) is not None:
            pass
        host_pool = SlotOnlyHostPool(1024, 1)
        host_group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=host_pool,
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                )
            ]
        )
        cache.host_pool_group = host_group
        cache.cache_controller = InstantWriteBackController(host_group, allocator)
        cache.components[ComponentType.FULL]._full_kv_pool_host = host_pool
        cache.tree_core.set_hicache_enabled()
        cache.is_write_back = True
        cache.write_through_threshold = 1 << 30
        while allocator.ensure_capacity(token_demand(allocator, 4, 4)):
            allocator.free_swa(allocator.alloc(4))
        demand = token_demand(
            allocator, allocator.full_available_size() + 12, 0, state_slots=1
        )
        waits = []
        with mock.patch.object(urc, "_is_cuda", cuda), _count_waits(cache, waits):
            result = cache.evict_for_alloc(demand)
        self.assertTrue(allocator.ensure_capacity(demand))
        cache.sanity_check()
        cached = [
            cache.match_prefix(MatchPrefixParams(key)).device_prefix_len for key in keys
        ]
        return waits, result.num_tokens_evicted, result.mamba_num_evicted, cached

    def test_cuda_walks_victims_as_other_platforms_do(self):
        per_victim = self._reclaim(cuda=False)
        waits, _, states, _ = per_victim
        self.assertEqual(waits, [1] * len(waits))
        self.assertGreaterEqual(len(waits), 2)
        self.assertGreater(states, 0)
        self.assertEqual(self._reclaim(cuda=True), per_victim)


class _QueuedOwnerWalkOnTrees:
    tree_core_backend: str
    CHAINS = 6

    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def _reclaim(self, sized, *, host_pages=64, fail_first_write=False, lock=None):
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(
            self.tree_core_backend
        ):
            cache, allocator = _write_back_cache(
                1, window_host=False, host_pages=host_pages
            )
        chains = [
            list(range(5000 + 100 * i, 5000 + 100 * i + 2 * PAGE))
            for i in range(self.CHAINS)
        ]
        for seq in chains:
            parent = allocator.alloc(PAGE)
            cache.insert(
                InsertParams(key=RadixKey(array("q", seq[:PAGE])), value=parent)
            )
            child = torch.cat([parent, allocator.alloc(PAGE)])
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", seq)), value=child, prev_prefix_len=PAGE
                )
            )
        if lock is not None:
            match = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", chains[lock])))
            )
            cache.inc_lock_ref(match.last_device_node)
        while allocator.alloc(PAGE) is not None:
            pass
        (window,) = allocator.window_pool_names()
        demand = KVDemand(10 * PAGE, {window: 10 * PAGE})
        waits = []
        write = cache.cache_controller.write
        writes = []

        def fail_first(*args, **kwargs):
            writes.append(args)
            if fail_first_write and len(writes) == 1:
                return None
            return write(*args, **kwargs)

        with (
            mock.patch.object(urc, "_is_cuda", sized),
            mock.patch.object(cache.cache_controller, "write", side_effect=fail_first),
            _count_waits(cache, waits),
        ):
            evicted = cache.evict_for_alloc(demand).num_tokens_evicted // PAGE
        self.assertTrue(allocator.ensure_capacity(demand))
        cache.sanity_check()
        cached = [
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", seq)))
            ).device_prefix_len
            for seq in chains
        ]
        return evicted, cached, waits

    def _check(self, expected_waits, **kwargs):
        evicted, cached, unsized_waits = self._reclaim(False, **kwargs)
        self.assertEqual(unsized_waits, [1] * len(unsized_waits))
        self.assertEqual(
            self._reclaim(True, **kwargs), (evicted, cached, expected_waits)
        )
        return evicted, cached, unsized_waits

    def test_chain_parents_follow_their_queued_children(self):
        evicted, cached, _ = self._check([10])
        self.assertEqual((evicted, cached), (10, [0, 0, 0, 0, PAGE, 2 * PAGE]))

    def test_a_failed_backup_drops_and_a_locked_chain_stays(self):
        evicted, cached, _ = self._check([9], fail_first_write=True, lock=1)
        self.assertEqual((evicted, cached), (10, [0, 2 * PAGE, 0, 0, 0, PAGE]))

    def test_a_full_host_pool_demotes_the_batch_to_make_room(self):
        self._check([3, 3, 3, 1], host_pages=3)


class TestQueuedOwnerWalkPythonCore(_QueuedOwnerWalkOnTrees, CustomTestCase):
    tree_core_backend = "python"


class TestQueuedOwnerWalkRustCore(_QueuedOwnerWalkOnTrees, CustomTestCase):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
