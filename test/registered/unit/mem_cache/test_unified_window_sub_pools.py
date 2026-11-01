"""Windows of different lengths are separate sub-pools named by their length.

A plan may declare several windows, each a sub-pool named by its own length.
Its prefix tree slot is named by its sub-pool. Storage, sizing, the allocator,
the prefix tree and PD peers hold any number of windows; host stores carry one
further window and refuse more where they read it.
"""

import random
import unittest
from array import array
from types import SimpleNamespace

import torch
from test_unified_chain_realize import _live_bytes, _page_bytes

from sglang.srt.mem_cache.kv.alloc.pricing import feasible, first_plan
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    MLAGeometry,
    WindowLayers,
    kv_layer_decls,
    window_sub_pool_name,
)
from sglang.srt.mem_cache.kv.plan import (
    SubPoolExternal,
    plan_kv_storage,
    tree_slot_ids,
)
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-small")

_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16)
_MLA = MLAGeometry(kv_lora_rank=16, qk_rope_head_dim=8, kv_cache_dtype=torch.bfloat16)


def _decls(windows, geometry=_MHA):
    return kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=geometry,
        windows=[
            WindowLayers(layer_ids=(i + 1,), geometry=geometry, window=w)
            for i, w in enumerate(windows)
        ],
    )


def _plan(windows, geometry=_MHA):
    return plan_kv_storage(_decls(windows, geometry), page_size=4)


def _storage(windows, *, geometry=_MHA, device="cpu", lazy=True):
    setup_allocator_context()
    plan = _plan(windows, geometry)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=128,
            window_tokens=64,
            state_slots=0,
            max_running_requests=3,
            context_len=64,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device=device,
        start_layer=0,
        end_layer=len(windows) + 1,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=64,
        lazy_compaction=lazy,
    )
    return build_kv_storage(plan, sized, ctx)


class TestWindowNames(CustomTestCase):
    def test_each_window_is_named_by_its_own_length(self):
        self.assertEqual(window_sub_pool_name(16), "window_16")
        self.assertEqual(window_sub_pool_name(4096), "window_4096")
        # A window whose length the model does not state.
        self.assertEqual(window_sub_pool_name(None), "window")
        plan = _plan((64, 16, 8))
        self.assertEqual(
            {s.name for s in plan.sub_pools},
            {"full", "window_64", "window_16", "window_8"},
        )
        self.assertEqual(
            [s.name for s in plan.windows()], ["window_64", "window_16", "window_8"]
        )
        self.assertEqual(plan.primary_window().name, "window_64")
        # A window's name does not depend on the other windows.
        self.assertEqual(_plan((16,)).primary_window().name, "window_16")

    def test_external_names_follow_sub_pools_and_window_slots(self):
        # A window's prefix tree slot is named by its sub-pool. Host stores
        # name window entries by window slot and keep two; PD peers carry every
        # window as the window state type, told apart by sub-pool.
        first = SubPoolExternal(
            host_pool="swa", tree_component="WINDOW_16", metrics="swa", wire="swa"
        )
        self.assertEqual(_plan((16,)).sub_pool("window_16").external, first)
        two = _plan((16, 64))
        self.assertEqual(
            two.sub_pool("window_64").external,
            SubPoolExternal(
                host_pool="swa", tree_component="WINDOW_64", metrics="swa", wire="swa"
            ),
        )
        self.assertEqual(
            two.sub_pool("window_16").external,
            SubPoolExternal(
                host_pool="auxiliary_swa",
                tree_component="WINDOW_16",
                metrics="swa",
                wire="swa",
            ),
        )
        slots = two.tree_slots()
        self.assertEqual(
            [slot.component for slot in slots], ["FULL", "WINDOW_64", "WINDOW_16"]
        )
        self.assertEqual(
            [slot.host_pool for slot in slots], ["kv", "swa", "auxiliary_swa"]
        )
        self.assertEqual([slot.window for slot in slots][1:], [64, 16])
        # The tree's first window slot plays the window role.
        self.assertEqual(
            [ct.name for ct in tree_slot_ids(slots).values()],
            ["FULL", "SWA", "WINDOW_16"],
        )
        self.assertEqual(
            [(e.sub_pool, e.state_type) for e in two.wire_entries()],
            [("window_16", "swa"), ("window_64", "swa")],
        )
        # A prefix tree and PD peers keep every window.
        three = _plan((64, 16, 8))
        self.assertEqual(
            [(slot.component, slot.window) for slot in three.tree_slots()],
            [("FULL", None), ("WINDOW_64", 64), ("WINDOW_16", 16), ("WINDOW_8", 8)],
        )
        self.assertEqual(three.sub_pool("window_8").external.host_pool, "window_8")
        # Peers list entries of one state type by sub-pool name, whatever order
        # the plan holds the windows in.
        self.assertEqual(
            [(e.sub_pool, e.state_type) for e in three.wire_entries()],
            [("window_16", "swa"), ("window_64", "swa"), ("window_8", "swa")],
        )
        self.assertEqual(_plan((8, 64, 16)).wire_entries(), three.wire_entries())


class TestDraftWindowFusesByLifetime(CustomTestCase):
    def test_draft_window_rides_in_the_target_window_of_its_length(self):
        """A draft window equal to any target window shares that window's
        entries; one equal to none gets a band of its own, named by its length
        as a target window of that length would be."""
        from sglang.srt.mem_cache.kv.plan import sub_pool_names
        from sglang.srt.mem_cache.layout.fused_draft import (
            DraftKVGeometry,
            DraftKVProfile,
            DraftLayerSet,
            PlacementContext,
            place_fused_draft,
        )

        target_windows = (64, 16)
        decls = _decls(target_windows)
        for draft_window, host, new_band in (
            (64, "window_64", False),
            (16, "window_16", False),
            (8, "window_8", True),
        ):
            with self.subTest(draft_window=draft_window):
                placement = place_fused_draft(
                    profile=DraftKVProfile(
                        num_layers=1,
                        kinds={
                            "window": DraftLayerSet(
                                layer_ids=(0,),
                                window=draft_window,
                                geometry=DraftKVGeometry(
                                    head_num=1, head_dim=8, v_head_dim=8
                                ),
                            )
                        },
                    ),
                    num_runners=1,
                    ctx=PlacementContext(
                        host_names=sub_pool_names(decls),
                        target_windows=target_windows,
                    ),
                    store_dtype=torch.bfloat16,
                ).placement
                self.assertEqual(placement.hosts(), (host,))
                self.assertEqual(host in placement.new_bands, new_band)
                self.assertEqual(placement.host_kind(host).name, "swa")
                plan = plan_kv_storage(decls, page_size=4, fused_draft=placement)
                carriers = [
                    s.name for s in plan.sub_pools if s.spec.draft_region is not None
                ]
                self.assertEqual(carriers, [host])
                self.assertEqual(
                    len(plan.windows()), len(target_windows) + int(new_band)
                )


class _Backend:
    """An attention backend as the runner binds it."""

    def __init__(self, max_window_sub_pools):
        self.max_window_sub_pools = max_window_sub_pools
        self.runner_ids = None


class TestRunnerAddressesEveryWindow(CustomTestCase):
    def _runner(self, storage, req_to_token):
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds

        allocator = storage.allocator
        return RunnerIds(
            req_to_token=req_to_token,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=allocator.get_kvcache(),
            page_size=4,
            device="cpu",
        )

    def test_each_window_writes_and_reads_its_own_rows(self):
        """A runner over several window sub-pools names each one's write ids
        and page table through that sub-pool's own pages."""
        from test_runner_ids import _FakeForwardBatch

        for windows in ((64, 16), (64, 16, 8)):
            with self.subTest(windows=windows):
                storage = _storage(windows)
                allocator = storage.allocator
                names = sorted(window_sub_pool_name(w) for w in windows)
                held = allocator.alloc(8)
                virt = allocator.alloc(8)
                req_to_token = torch.zeros((3, 64), dtype=torch.int32)
                req_to_token[0, :16] = torch.cat([held, virt]).to(torch.int32)
                runner = self._runner(storage, req_to_token)
                self.assertEqual(runner.window_sub_pools, tuple(names))
                batch = _FakeForwardBatch(
                    req_pool_indices=torch.tensor([0]),
                    seq_lens=torch.tensor([16]),
                    seq_lens_cpu=torch.tensor([16]),
                    out_cache_loc=virt,
                )
                plan = runner.own_plan(batch)
                rows = set()
                for name in names:
                    band = allocator.window_allocators[name]
                    pages = band.ids.virtual_to_physical
                    want = pages[virt // 4] * 4 + virt % 4
                    got = plan.write_ids(runner, kind=name)
                    self.assertTrue(torch.equal(got, want), name)
                    table = plan.read_table(kind=name, reader=runner)
                    page_ids = table.ids[table.row_ids[0], :4]
                    want_pages = pages[req_to_token[0, :16:4].long() // 4]
                    self.assertTrue(torch.equal(page_ids.long(), want_pages), name)
                    rows.add((name, tuple(want.tolist())))
                self.assertEqual(len(rows), len(names))

    def test_backends_bind_only_as_many_windows_as_they_serve(self):
        """Backends build window metadata per runner: one that serves fewer
        window sub-pools than the runner's layers read is refused by name."""
        for windows, refused in (((64,), False), ((64, 16), True)):
            with self.subTest(windows=windows):
                runner = self._runner(
                    _storage(windows), torch.zeros((3, 64), dtype=torch.int32)
                )
                backend = _Backend(max_window_sub_pools=1)
                if refused:
                    with self.assertRaisesRegex(
                        NotImplementedError, "_Backend.*window_16|window_16.*_Backend"
                    ):
                        runner.bind_and_verify_backends([backend])
                    self.assertIsNone(backend.runner_ids)
                else:
                    runner.bind_and_verify_backends([backend])
                    self.assertIs(backend.runner_ids, runner)
                wide = _Backend(max_window_sub_pools=len(windows))
                runner.bind_and_verify_backends([wide])
                self.assertIs(wide.runner_ids, runner)


class TestSeveralWindowsAllocate(CustomTestCase):
    def _check(self, device, seed, count):
        rng = random.Random(seed)
        realized = 0
        cases = (((64, 16), _MHA), ((64, 16), _MLA), ((64, 16, 8), _MHA))
        while realized < count:
            windows, geometry = cases[realized % len(cases)]
            storage = _storage(windows, geometry=geometry, device=device)
            allocator = storage.allocator
            self.assertEqual(
                set(allocator.window_allocators),
                {window_sub_pool_name(w) for w in windows},
            )
            raw = allocator.arena.raw
            raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8))
            held = [allocator.alloc(rng.randint(1, 8) * 4) for _ in range(4)]
            for ids in held:
                if ids is not None and rng.random() < 0.4:
                    allocator.free(ids)
            snapshot = allocator.pricing_snapshot()
            demands = {b.name: rng.choice((0, 1, 2, 4)) for b in snapshot.bands}
            if not feasible(snapshot, demands):
                continue
            before = _live_bytes(allocator)
            taken = allocator.realize(first_plan(snapshot, demands))
            realized += 1
            after = _live_bytes(allocator)
            self.assertEqual(before.keys(), after.keys())
            for key, payload in before.items():
                self.assertTrue(torch.equal(payload, after[key]), key)
            for name, pages in demands.items():
                self.assertEqual(0 if name not in taken else taken[name].numel(), pages)
            ranges = _page_bytes(allocator, taken)
            for (_, end, a), (start, _, b) in zip(ranges, ranges[1:]):
                self.assertLessEqual(end, start, (a, b))

    def test_several_windows_build_and_realize_on_cpu(self):
        self._check("cpu", seed=3, count=60)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_several_windows_build_and_realize_on_cuda(self):
        self._check("cuda", seed=4, count=12)


class TestTwoTargetWindowsRetire(CustomTestCase):
    def test_decode_retires_each_window_by_its_length(self):
        from sglang.srt.environ import envs
        from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
        from sglang.srt.mem_cache.base_prefix_cache import (
            EvictParams,
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        long_window, short_window = 16, 8
        with (
            envs.SGLANG_OPT_RELEASE_PREFILL_SWA.override(True),
            envs.SGLANG_SWA_EVICTION_INTERVAL.override(1),
        ):
            storage = _storage((long_window, short_window), lazy=False)
            a = storage.allocator
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=storage.requests,
                    token_to_kv_pool_allocator=a,
                    page_size=4,
                    sliding_window_size=long_window,
                    tree_slots=storage.tree_slots(),
                    tree_core_backend="python",
                )
            )
            prefix = a.alloc(16)
            self.assertIsNotNone(prefix)
            key = RadixKey(array("q", range(16)))
            cache.insert(InsertParams(key=key, value=prefix.clone()))
            node_id = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
            lock = cache.lock(node_id)
            req = SimpleNamespace(
                session=None,
                lock=lock,
                seqlen=17,
                decode_batch_idx=0,
                kv=ReqKvInfo(req_pool_idx=1, cache_protected_len=16),
            )
            storage.requests.req_to_token[1, :16] = prefix
            batch = ScheduleBatch(
                reqs=[req],
                tree_cache=cache,
                req_to_token_pool=storage.requests,
                token_to_kv_pool_allocator=a,
                forward_mode=ForwardMode.DECODE,
            )
            node = cache.tree_core.node_by_id(node_id)
            for frontier in range(20, 49, 4):
                ids = a.alloc(4)
                self.assertIsNotNone(ids, frontier)
                storage.requests.req_to_token[1, frontier - 4 : frontier] = ids
                req.seqlen = frontier + 1
                req.decode_batch_idx = frontier - 16
                batch.maybe_evict_swa()
                short = cache.window_slots()[1]
                cache.evict(
                    EvictParams(swa_num_tokens=80, window_num_tokens={short: 80})
                )
                for ct, window in (
                    (ComponentType.SWA, long_window),
                    (short, short_window),
                ):
                    self.assertEqual(
                        node.component_data[ct].lock_ref,
                        int(req.decode_batch_idx < window),
                        (frontier, ct),
                    )
            for name, window in (
                ("window_16", long_window),
                ("window_8", short_window),
            ):
                self.assertEqual(a.window_allocators[name].allocated_count(), window)
            cache.free_kv_row(req.kv, [(16, 48)])
            cache.unlock(lock)
            cache.evict(
                EvictParams(
                    num_tokens=80,
                    swa_num_tokens=80,
                    window_num_tokens={cache.window_slots()[1]: 80},
                )
            )
            self.assertEqual(storage.accounting_violations(), [])
            self.assertTrue(all(b.allocated_count() == 0 for b in a.members))


class TestThreeWindowTree(CustomTestCase):
    """A prefix tree keeps one slot per window sub-pool: each window's values,
    locks and evictions are its own."""

    def _cache(self):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        storage = _storage((16, 8, 4), lazy=False)
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=storage.requests,
                token_to_kv_pool_allocator=storage.allocator,
                page_size=4,
                sliding_window_size=16,
                tree_slots=storage.tree_slots(),
                tree_core_backend="python",
            )
        )
        return storage, cache

    def test_each_window_slot_matches_locks_and_evicts_alone(self):
        from sglang.srt.mem_cache.base_prefix_cache import (
            EvictParams,
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.components import ComponentType

        storage, cache = self._cache()
        a = storage.allocator
        swa, swa_8, swa_4 = cache.window_slots()
        self.assertEqual(
            [ct.name for ct in (swa, swa_8, swa_4)], ["SWA", "WINDOW_8", "WINDOW_4"]
        )
        self.assertIs(swa, ComponentType.SWA)
        self.assertEqual(
            [(name, ct) for name, ct, _, _ in a.admission_windows(cache)],
            [("window_16", swa), ("window_8", swa_8), ("window_4", swa_4)],
        )
        self.assertEqual(
            [cache.window_retain_size(ct) for ct in (swa, swa_8, swa_4)], [16, 8, 4]
        )

        key = RadixKey(array("q", range(16)))
        cache.insert(InsertParams(key=key, value=a.alloc(16).clone()))
        match = cache.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(match.device_prefix_len, 16)
        node = cache.tree_core.node_by_id(match.last_device_node)
        for ct in (swa, swa_8, swa_4):
            self.assertEqual(cache.window_evictable_size(ct), 16)
            self.assertIsNotNone(node.component_data[ct].value)

        # A lock holds each window slot's own trailing window of the prefix.
        lock = cache.lock(match.last_device_node)
        for ct in (swa, swa_8, swa_4):
            self.assertEqual(node.component_data[ct].lock_ref, 1)
        self.assertEqual(
            [cache.window_evictable_size(ct) for ct in (swa, swa_8, swa_4)],
            [0, 8, 12],
        )
        # Each further window evicts what lies before its own window, alone.
        result = cache.evict(EvictParams(window_num_tokens={swa_8: 16, swa_4: 16}))
        self.assertEqual(result.window_num_tokens_evicted, {swa_8: 8, swa_4: 12})
        self.assertEqual(
            (result.num_tokens_evicted, result.swa_num_tokens_evicted), (0, 0)
        )
        self.assertEqual(
            {b.sub_pool_name: b.allocated_count() for b in a.bands},
            {"full": 16, "window_16": 16, "window_8": 8, "window_4": 4},
        )
        # Every window still holds its trailing tokens: the prefix matches whole.
        self.assertEqual(
            cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len, 16
        )
        cache.unlock(lock)

        cache.evict(
            EvictParams(
                num_tokens=64,
                swa_num_tokens=64,
                window_num_tokens={swa_8: 64, swa_4: 64},
            )
        )
        self.assertTrue(all(b.allocated_count() == 0 for b in a.members))
        self.assertEqual(storage.accounting_violations(), [])

    def test_reclaim_evicts_a_further_window_for_its_own_demand(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        storage, cache = self._cache()
        a = storage.allocator
        _, _, swa_4 = cache.window_slots()
        # Fill the pool with cached prefixes.
        start = 1000
        while (ids := a.alloc(16)) is not None:
            key = RadixKey(array("q", range(start, start + 16)))
            cache.insert(InsertParams(key=key, value=ids.clone()))
            start += 16
        reclaimable = a._reclaimable_units(cache)
        pages = next(
            (
                n
                for n in range(1, 64)
                if not a._units_fit({"window_4": n})
                and a._admits_pages({"window_4": n}, reclaimable)
            ),
            None,
        )
        self.assertIsNotNone(pages)
        full_before = cache.full_evictable_size()
        own_before = cache.window_evictable_size(swa_4)
        self.assertTrue(a.reclaim(cache, {"window_4": pages}))
        # The window's own entries go first and suffice: no token entry goes.
        self.assertLess(cache.window_evictable_size(swa_4), own_before)
        self.assertEqual(cache.full_evictable_size(), full_before)
        self.assertEqual(storage.accounting_violations(), [])

    def test_rust_tree_core_defers_several_windows_to_python(self):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
            _rust_fallback_reason,
        )

        storage = _storage((16, 8, 4), lazy=False)
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=storage.allocator,
            page_size=4,
            tree_slots=storage.tree_slots(),
        )
        self.assertIn("several window slots", _rust_fallback_reason(params))


if __name__ == "__main__":
    unittest.main()
