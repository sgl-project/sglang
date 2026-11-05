"""Each window sub-pool is named by its own length.

The target's window and a draft's window of another length are sub-pools named
by the window they keep. The names they go by outside this process (prefix
tree component, host pool, PD state type, metrics) are unchanged, and
consumers that ask for "the window" by the sliding-window role find the
target's window whatever it is named.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    WindowLifetime,
    kv_layer_decls,
    primary_window,
    window_sub_pool_name,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage, sub_pool_names
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    PlacementContext,
    place_fused_draft,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16)


def _decls(window):
    return kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_MHA,
        window_layer_ids=(1,),
        window_geometry=_MHA,
        window=window,
    )


def _draft_placement(decls, *, target_window, draft_window):
    return place_fused_draft(
        profile=DraftKVProfile(
            num_layers=1,
            kinds={
                "window": DraftLayerSet(
                    layer_ids=(0,),
                    window=draft_window,
                    geometry=DraftKVGeometry(head_num=1, head_dim=8, v_head_dim=8),
                )
            },
        ),
        num_runners=1,
        ctx=PlacementContext(
            host_names=sub_pool_names(decls), target_window=target_window
        ),
        store_dtype=torch.bfloat16,
    ).placement


def _storage(windows, *, device="cpu", lazy=True):
    """A storage of the target's window ``windows[0]`` and, when given, a fused
    draft's window ``windows[1]`` of another length."""
    target, *drafts = windows
    assert len(drafts) <= 1, "a draft adds at most one window"
    decls = _decls(target)
    placement = (
        _draft_placement(decls, target_window=target, draft_window=drafts[0])
        if drafts
        else None
    )
    return _build(
        plan_kv_storage(decls, page_size=4, fused_draft=placement),
        device=device,
        lazy=lazy,
    )


def _build(plan, *, device="cpu", lazy=True):
    setup_allocator_context()
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
        end_layer=2,
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
        plan = plan_kv_storage(_decls(16), page_size=4)
        self.assertEqual({s.name for s in plan.sub_pools}, {"full", "window_16"})
        self.assertEqual(plan.primary_window().name, "window_16")
        self.assertEqual(sub_pool_names(_decls(16)), ("full", "window_16"))
        self.assertEqual(
            plan_kv_storage(_decls(None), page_size=4).primary_window().name, "window"
        )

    def test_the_longest_window_is_primary(self):
        self.assertIsNone(primary_window(()))
        self.assertEqual(primary_window((16, 64, 8)), 64)
        # A window that states no length counts as longest.
        self.assertIsNone(primary_window((16, None)))

    def test_a_plan_asks_for_its_window_by_rule(self):
        plan = plan_kv_storage(_decls(16), page_size=4)
        with self.assertRaises(AssertionError):
            plan.of_lifetime(WindowLifetime)

    def test_external_names_are_unchanged(self):
        # The target's window keeps the names the prefix tree, host pools, PD
        # peers and metrics use for a model's window.
        plan = plan_kv_storage(_decls(16), page_size=4)
        external = plan.sub_pool("window_16").external
        self.assertEqual(
            (
                external.host_pool,
                external.tree_component,
                external.metrics,
                external.wire,
            ),
            ("swa", "WINDOW_16", "swa", "swa"),
        )
        self.assertEqual(
            [slot.component for slot in plan.tree_slots()], ["FULL", "WINDOW_16"]
        )
        self.assertEqual(
            [(e.sub_pool, e.state_type) for e in plan.wire_entries()],
            [("window_16", "swa")],
        )


class TestDraftWindowNames(CustomTestCase):
    def test_a_draft_window_of_the_target_length_rides_in_the_target_window(self):
        decls = _decls(16)
        placement = _draft_placement(decls, target_window=16, draft_window=16)
        self.assertEqual(placement.hosts(), ("window_16",))
        self.assertEqual(placement.new_bands, {})
        plan = plan_kv_storage(decls, page_size=4, fused_draft=placement)
        self.assertEqual(
            [s.name for s in plan.sub_pools if s.spec.draft_region is not None],
            ["window_16"],
        )

    def test_a_draft_window_of_another_length_is_named_by_it(self):
        decls = _decls(16)
        placement = _draft_placement(decls, target_window=16, draft_window=32)
        self.assertEqual(set(placement.new_bands), {"window_32"})
        plan = plan_kv_storage(decls, page_size=4, fused_draft=placement)
        self.assertEqual(
            {s.name for s in plan.sub_pools}, {"full", "window_16", "window_32"}
        )
        # The target's window stays the primary one; the draft's goes by the
        # second window entry of the host pools and PD peers, and the prefix
        # tree names each window by its sub-pool.
        self.assertEqual(plan.primary_window().name, "window_16")
        external = plan.sub_pool("window_32").external
        self.assertEqual(
            (external.host_pool, external.tree_component, external.wire),
            ("auxiliary_swa", "WINDOW_32", "auxiliary_swa"),
        )
        self.assertEqual(
            plan.sub_pool("window_16").external.tree_component, "WINDOW_16"
        )

    def test_a_draft_window_without_a_target_window_takes_the_window_names(self):
        decls = kv_layer_decls(full_layer_ids=(0,), full_geometry=_MHA)
        placement = _draft_placement(decls, target_window=None, draft_window=32)
        self.assertEqual(set(placement.new_bands), {"window_32"})
        plan = plan_kv_storage(decls, page_size=4, fused_draft=placement)
        self.assertEqual(plan.primary_window().name, "window_32")
        self.assertEqual(
            plan.sub_pool("window_32").external.tree_component, "WINDOW_32"
        )

    def test_a_draft_band_cannot_take_a_target_name(self):
        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftBand,
            FusedDraftPlacement,
        )

        placement = FusedDraftPlacement.from_counts(
            counts={"window_16": [1]},
            regions={
                "window_16": DenseDraftRegion(
                    lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
                )
            },
            new_bands={"window_16": DraftBand(kind="window", window=32)},
        )
        with self.assertRaisesRegex(ValueError, "takes a target sub-pool's name"):
            plan_kv_storage(_decls(16), page_size=4, fused_draft=placement)


class TestWindowRole(CustomTestCase):
    def test_the_sliding_window_role_finds_the_target_window(self):
        storage = _storage((16,))
        router = storage.allocator.get_kvcache().layer_router
        self.assertEqual(router.sub_pool(IdSpaceKind.SLIDING_WINDOW).name, "window_16")
        self.assertIs(router.view(IdSpaceKind.SLIDING_WINDOW), router.view("window_16"))
        self.assertTrue(router.plays(1, IdSpaceKind.SLIDING_WINDOW))
        self.assertFalse(router.plays(0, IdSpaceKind.SLIDING_WINDOW))
        self.assertEqual(router.route(1).sub_pool, "window_16")
        # Id spaces are keyed by sub-pool name only.
        self.assertNotIn(IdSpaceKind.SLIDING_WINDOW, storage.allocator.id_spaces())

    def test_runner_ids_resolve_the_role_to_the_target_window(self):
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds

        storage = _storage((16,))
        alloc = storage.allocator
        runner = RunnerIds(
            req_to_token=storage.requests.req_to_token,
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=alloc.get_kvcache(),
            page_size=4,
            device="cpu",
        )
        self.assertEqual(runner.sub_pool_name(IdSpaceKind.SLIDING_WINDOW), "window_16")
        self.assertIs(
            runner.space(IdSpaceKind.SLIDING_WINDOW), runner.space("window_16")
        )
        self.assertEqual(runner.window_kv_kind, IdSpaceKind.SLIDING_WINDOW)
        ids = alloc.alloc(8)
        torch.testing.assert_close(
            runner.ids_in(IdSpaceKind.SLIDING_WINDOW, ids),
            alloc.id_spaces()["window_16"].ids(ids),
        )

    def test_a_fused_draft_names_its_window_host_by_role(self):
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
        from sglang.srt.mem_cache.unified_draft_pool import routed_draft_pool

        # A draft window equal to the target's rides in the target's window
        # sub-pool; one of another length gets a band of its own. Either way
        # the draft's pool routes its layer by the host's name and maps the
        # sliding-window role to that host, as the target's pool does.
        for draft_window, host in ((16, "window_16"), (32, "window_32")):
            with self.subTest(draft_window=draft_window):
                decls = _decls(16)
                placement = _draft_placement(
                    decls, target_window=16, draft_window=draft_window
                )
                plan = plan_kv_storage(decls, page_size=4, fused_draft=placement)
                storage = _build(plan)
                alloc = storage.allocator
                pool = routed_draft_pool(
                    arena=alloc.arena,
                    plan=plan,
                    owner="full",
                    page_size=4,
                    lanes_by_host={host: {0: 0}},
                )
                router = pool.layer_router
                self.assertEqual([p.name for p in router.sub_pools()], [host])
                self.assertEqual(router.route(0).sub_pool, host)
                self.assertEqual(router.sub_pool(IdSpaceKind.SLIDING_WINDOW).name, host)
                self.assertTrue(router.plays(0, IdSpaceKind.SLIDING_WINDOW))
                runner = RunnerIds(
                    req_to_token=storage.requests.req_to_token,
                    token_to_kv_pool_allocator=alloc,
                    token_to_kv_pool=pool,
                    page_size=4,
                    device="cpu",
                    is_draft_worker=True,
                )
                self.assertTrue(runner.is_fused_draft)
                self.assertEqual(runner.sub_pool_name(IdSpaceKind.SLIDING_WINDOW), host)
                self.assertIs(
                    runner.space(IdSpaceKind.SLIDING_WINDOW), runner.space(host)
                )
                ids = alloc.alloc(8)
                torch.testing.assert_close(
                    runner.ids_in(IdSpaceKind.SLIDING_WINDOW, ids),
                    alloc.id_spaces()[host].ids(ids),
                )


if __name__ == "__main__":
    unittest.main()
