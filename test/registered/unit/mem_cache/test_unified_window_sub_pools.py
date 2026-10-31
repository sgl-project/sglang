"""Windows of different lengths are separate sub-pools named by their length.

A plan may declare several windows. The longest keeps the name, and the
external names, a single-window model has always used; every other window
takes the next window slot's external names, and a slot past what the prefix
tree and PD peers carry is refused where they read it. Storage, sizing and the
allocator hold any number of windows.
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
from sglang.srt.mem_cache.kv.plan import SubPoolExternal, plan_kv_storage
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
    def test_the_longest_window_keeps_the_single_window_name(self):
        self.assertEqual(window_sub_pool_name(16, (16,)), "swa")
        self.assertEqual(window_sub_pool_name(16, (16, 64)), "swa_16")
        self.assertEqual(window_sub_pool_name(64, (16, 64)), "swa")
        # An unstated window is the longest; with no target window, a window
        # is its own longest.
        self.assertEqual(window_sub_pool_name(64, (None, 64)), "swa_64")
        self.assertEqual(window_sub_pool_name(8, ()), "swa")
        plan = _plan((64, 16, 8))
        self.assertEqual(
            {s.name for s in plan.sub_pools}, {"full", "swa", "swa_16", "swa_8"}
        )
        self.assertEqual([s.name for s in plan.windows()], ["swa", "swa_16", "swa_8"])
        self.assertEqual(plan.primary_window().name, "swa")

    def test_external_names_follow_window_slots(self):
        swa = SubPoolExternal(
            host_pool="swa", tree_component="SWA", metrics="swa", wire="swa"
        )
        auxiliary = SubPoolExternal(
            host_pool="auxiliary_swa",
            tree_component="AUXILIARY_SWA",
            metrics="swa",
            wire="auxiliary_swa",
        )
        # One window: the names every single-window peer and host store use.
        self.assertEqual(_plan((16,)).sub_pool("swa").external, swa)
        two = _plan((16, 64))
        self.assertEqual(two.sub_pool("swa").external, swa)
        self.assertEqual(two.sub_pool("swa_16").external, auxiliary)
        self.assertEqual(
            [slot.component for slot in two.tree_slots()],
            ["FULL", "SWA", "AUXILIARY_SWA"],
        )
        self.assertEqual([slot.window for slot in two.tree_slots()][1:], [64, 16])
        self.assertEqual(
            [e.state_type for e in two.wire_entries()], ["swa", "auxiliary_swa"]
        )
        three = _plan((64, 16, 8))
        with self.assertRaisesRegex(ValueError, "at most 2 windows"):
            three.tree_slots()
        with self.assertRaisesRegex(ValueError, "no state type"):
            three.wire_entries()


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
            (64, "swa", False),
            (16, "swa_16", False),
            (8, "swa_8", True),
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


class TestRunnerReadsOneWindow(CustomTestCase):
    def test_runner_refuses_layers_of_several_windows(self):
        """Backends bind one window's ids per runner: a runner whose layers
        attend through two window sub-pools is refused, not given one window's
        ids for both."""
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds

        for windows, refused in (((64,), False), ((64, 16), True)):
            with self.subTest(windows=windows):
                storage = _storage(windows)
                allocator = storage.allocator
                build = lambda: RunnerIds(
                    req_to_token=torch.zeros((3, 64), dtype=torch.int32),
                    token_to_kv_pool_allocator=allocator,
                    token_to_kv_pool=allocator.get_kvcache(),
                    page_size=4,
                    device="cpu",
                )
                if refused:
                    with self.assertRaisesRegex(NotImplementedError, "swa_16"):
                        build()
                else:
                    self.assertIsNotNone(build().space("swa"))


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
                {window_sub_pool_name(w, windows) for w in windows},
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
                cache.evict(EvictParams(swa_num_tokens=80, auxiliary_swa_num_tokens=80))
                for ct, window in (
                    (ComponentType.SWA, long_window),
                    (ComponentType.AUXILIARY_SWA, short_window),
                ):
                    self.assertEqual(
                        node.component_data[ct].lock_ref,
                        int(req.decode_batch_idx < window),
                        (frontier, ct),
                    )
            for name, window in (("swa", long_window), ("swa_8", short_window)):
                self.assertEqual(a.window_allocators[name].allocated_count(), window)
            cache.free_kv_row(req.kv, [(16, 48)])
            cache.unlock(lock)
            cache.evict(
                EvictParams(
                    num_tokens=80, swa_num_tokens=80, auxiliary_swa_num_tokens=80
                )
            )
            self.assertEqual(storage.accounting_violations(), [])
            self.assertTrue(all(b.allocated_count() == 0 for b in a.members))


if __name__ == "__main__":
    unittest.main()
