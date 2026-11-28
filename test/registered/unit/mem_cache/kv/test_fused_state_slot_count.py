"""BUG REGRESSION: a draft state fused into every state slot keeps the host's slot count,
and so its running requests, while FULL pays for the wider slots, and never fails a
budget that a plain slot of the same width serves.

Every case enters through `KVCacheConfigurator.config_from_budget` on real storage
plans. Only the model and server context around the solve are stood in for."""

import unittest
from itertools import product
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    StateGeometry,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.layout.fused_draft import (
    DraftBand,
    DraftStateGeometry,
    DraftStateRegion,
    FusedDraftPlacement,
    RunnerLanes,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BUDGET = 128 << 20
RATIO = 0.5
# float32 elements per slot: 4 MiB of host state, 1 MiB of draft state.
HOST_ROWS = 1 << 20
DRAFT_ROWS = 1 << 18
CONTEXT = 4096


def _state_geometry(geometry_cls, rows):
    return geometry_cls(
        conv_state_shapes=((rows,),),
        conv_dtype=torch.float32,
        temporal_state_shape=(0, 0, 0),
        temporal_dtype=torch.float32,
    )


def _plan(
    *,
    host_rows=HOST_ROWS,
    draft_rows=None,
    scratch_tokens=None,
    full_heads=1,
    window=None,
):
    geometry = MHAGeometry(
        head_num=full_heads, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
    )
    return plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=geometry,
            window_layer_ids=() if window is None else (2,),
            window_geometry=geometry,
            window=window,
            state_layer_ids=(1,),
            state_geometry=_state_geometry(StateGeometry, host_rows),
        ),
        page_size=1,
        fused_draft=None
        if draft_rows is None
        else FusedDraftPlacement(
            runners=(RunnerLanes(ranges={"mamba": (0, 1)}),),
            regions={
                "mamba": DraftStateRegion(
                    lane_num=1, state=_state_geometry(DraftStateGeometry, draft_rows)
                )
            },
        ),
        speculative_num_draft_tokens=scratch_tokens,
    )


def _configurator(plan, context_len=CONTEXT):
    window = plan.primary_window()
    configurator = SimpleNamespace(
        is_draft_worker=False,
        _unified_plan=plan,
        _unified_sized=None,
        _unified_window_and_state=lambda: (window is not None, True),
        _unified_request_rows=lambda **_kwargs: (0, 0),
        model_config=SimpleNamespace(
            context_len=context_len,
            sliding_window_size=None if window is None else window.lifetime.window,
        ),
        attn_dp_size=1,
        _apply_token_constraints=lambda value: value,
    )
    for name in ("_unified_config_from_budget", "_unified_budget_for_full_tokens"):
        setattr(
            configurator,
            name,
            MethodType(getattr(KVCacheConfigurator, name), configurator),
        )
    return configurator


def _server_args(**limits):
    args = dict(
        enable_unified_memory=True,
        disable_radix_cache=False,
        max_mamba_cache_size=None,
        max_running_requests=None,
        max_total_tokens=None,
        mamba_full_memory_ratio=RATIO,
    )
    args.update(limits)
    return get_context().override_server_args(**args)


def _solve(plan, *, budget=BUDGET, context_len=CONTEXT, **limits):
    configurator = _configurator(plan, context_len)
    with _server_args(**limits):
        config = KVCacheConfigurator.config_from_budget(configurator, budget)
    sized = configurator._unified_sized
    assert config.max_running_requests == sized.max_running_requests
    assert config.unified_memory_pool_bytes == sized.total_bytes
    return sized


class TestFusedStateSlotCount(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        setup_allocator_context()

    def _assert_fits(self, plan, sized, budget=BUDGET):
        window = plan.primary_window()
        charged = (
            sized.units("mamba") * plan.sub_pool("mamba").spec.entry_bytes()
            + sized.full_tokens * plan.owner().spec.entry_bytes()
            + (0 if window is None else sized.window_tokens * window.spec.entry_bytes())
            + plan.sink_bytes()
        )
        self.assertLessEqual(charged, sized.total_bytes)
        self.assertLessEqual(sized.total_bytes + sized.fixed_bytes, budget)

    def test_a_fused_state_keeps_slots_and_requests_and_full_pays(self):
        """The widened slots are charged in full, and their draft part comes out of
        FULL; the scratch and the other fixed allocations stay at the host's width."""
        for radix, scratch in product((True, False), (None, 8)):
            with self.subTest(radix=radix, scratch=scratch):
                host_plan = _plan(scratch_tokens=scratch)
                fused_plan = _plan(draft_rows=DRAFT_ROWS, scratch_tokens=scratch)
                host = _solve(host_plan, disable_radix_cache=not radix)
                fused = _solve(fused_plan, disable_radix_cache=not radix)

                self.assertGreater(host.max_running_requests, 0)
                self.assertEqual(fused.units("mamba"), host.units("mamba"))
                self.assertEqual(fused.max_running_requests, host.max_running_requests)
                self.assertEqual(fused.fixed, host.fixed)
                self.assertEqual(fused.total_bytes, host.total_bytes)
                # The state sink is one more slot of the same width.
                widening = (fused.units("mamba") + 1) * (
                    fused_plan.sub_pool("mamba").spec.entry_bytes()
                    - host_plan.sub_pool("mamba").spec.entry_bytes()
                )
                self.assertEqual(
                    fused_plan.sink_bytes() - host_plan.sink_bytes(),
                    widening // (fused.units("mamba") + 1),
                )
                self.assertAlmostEqual(
                    host.full_tokens - fused.full_tokens,
                    widening // fused_plan.owner().spec.entry_bytes(),
                    delta=1,
                )
                self._assert_fits(host_plan, host)
                self._assert_fits(fused_plan, fused)

    def test_explicit_limits_bind_at_the_full_slot_width(self):
        fused_plan = _plan(draft_rows=DRAFT_ROWS)
        for limits, slots, running, tokens in (
            (dict(max_mamba_cache_size=4), 4, 1, None),
            (dict(max_running_requests=1), None, 1, None),
            (dict(max_running_requests=2, disable_radix_cache=True), 2, 2, None),
            (dict(max_total_tokens=1 << 16), None, None, 1 << 16),
            # Just below what the full-width count leaves FULL: no slot is added.
            (dict(max_total_tokens=5 << 19), 8, 2, 5 << 19),
        ):
            with self.subTest(limits=limits):
                sized = _solve(fused_plan, **limits)
                if slots is not None:
                    self.assertEqual(sized.units("mamba"), slots)
                if running is not None:
                    self.assertEqual(sized.max_running_requests, running)
                if tokens is not None:
                    self.assertEqual(sized.full_tokens, tokens)
                self.assertGreater(sized.max_running_requests, 0)
                self._assert_fits(fused_plan, sized, budget=BUDGET)

    def test_a_stage_holding_mostly_draft_state_leaves_full_half_its_tokens(self):
        """At the host's count these slots would take more than the whole arena, so
        FULL pays for wider slots only down to half of what a plain slot leaves it."""
        plan = _plan(host_rows=1 << 18, draft_rows=3 << 18)
        wide_plan = _plan(host_rows=4 << 18)
        spec = plan.sub_pool("mamba").spec
        self.assertEqual(
            spec.entry_bytes(), wide_plan.sub_pool("mamba").spec.entry_bytes()
        )
        sized = _solve(plan)
        wide = _solve(wide_plan)
        share = int(sized.total_bytes * RATIO / (1 + RATIO))
        self.assertGreater(
            share // spec.host_entry_bytes() * spec.entry_bytes(), sized.total_bytes
        )
        self.assertGreater(sized.units("mamba"), wide.units("mamba"))
        self.assertGreater(sized.max_running_requests, wide.max_running_requests)
        self.assertGreaterEqual(sized.full_tokens, wide.full_tokens // 2)
        self._assert_fits(plan, sized)

    def test_a_fused_slot_boots_wherever_a_plain_slot_of_its_width_does(self):
        """Extra slots never cost the boot, a slot or a request, and FULL keeps one
        longest request, as does a window band; otherwise the plain slot's count
        stands."""
        # host rows, draft rows, FULL heads, context, window, budgets: the first three
        # are each near the bound of another term of FULL's floor, and the last
        # shares FULL's remainder with a window band.
        kib = [k << 10 for k in range(256, 1024, 96)]
        shapes = (
            (1 << 18, 3 << 18, 1, CONTEXT, None, [m << 20 for m in range(14, 26, 2)]),
            (1 << 10, 3 << 10, 1, CONTEXT, None, kib),
            (1 << 10, 3 << 10, 64, 256, None, [m << 20 for m in range(4, 12)]),
            (1 << 10, 3 << 10, 1, CONTEXT, 128, kib),
        )
        for host_rows, draft_rows, heads, context, window, budgets in shapes:
            fused_plan = _plan(
                host_rows=host_rows,
                draft_rows=draft_rows,
                full_heads=heads,
                window=window,
            )
            wide_plan = _plan(
                host_rows=host_rows + draft_rows, full_heads=heads, window=window
            )
            booted = traded = 0
            for radix, ratio, budget in product((True, False), (0.5, 2, 9), budgets):
                limits = dict(
                    budget=budget,
                    context_len=context,
                    mamba_full_memory_ratio=ratio,
                    disable_radix_cache=not radix,
                )
                with self.subTest(
                    rows=host_rows,
                    window=window,
                    radix=radix,
                    ratio=ratio,
                    budget=budget,
                ):
                    try:
                        wide = _solve(wide_plan, **limits)
                    except RuntimeError:
                        continue
                    booted += 1
                    fused = _solve(fused_plan, **limits)
                    self.assertGreaterEqual(fused.units("mamba"), wide.units("mamba"))
                    self.assertGreaterEqual(
                        fused.max_running_requests, wide.max_running_requests
                    )
                    if fused.units("mamba") > wide.units("mamba"):
                        traded += 1
                        self.assertGreaterEqual(fused.full_tokens, context)
                        if window is not None:
                            self.assertGreaterEqual(
                                fused.window_tokens, min(context, window + 1)
                            )
                    else:
                        self.assertEqual(fused.full_tokens, wide.full_tokens)
                    self._assert_fits(fused_plan, fused, budget=budget)
            self.assertGreater(booted, traded)
            self.assertGreater(traded, 0)

    def test_a_state_band_holding_only_draft_state_prices_its_own_width(self):
        """A host with no recurrent state has no slot count to keep; its draft's own
        band buys slots from the state share at that band's width."""
        plan = plan_kv_storage(
            kv_layer_decls(
                full_layer_ids=(0,),
                full_geometry=MHAGeometry(
                    head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
                ),
            ),
            page_size=1,
            fused_draft=FusedDraftPlacement.from_counts(
                counts={"mamba": [1]},
                regions={
                    "mamba": DraftStateRegion(
                        lane_num=1,
                        state=_state_geometry(DraftStateGeometry, DRAFT_ROWS),
                    )
                },
                new_bands={"mamba": DraftBand(kind="state")},
            ),
        )
        spec = plan.sub_pool("mamba").spec
        self.assertEqual(spec.host_entry_bytes(), 0)
        sized = _solve(plan)
        self.assertEqual(
            sized.units("mamba"),
            int(sized.total_bytes * RATIO / (1 + RATIO)) // spec.entry_bytes(),
        )
        self.assertGreater(sized.max_running_requests, 0)
        self._assert_fits(plan, sized)

    def test_pipeline_stages_agree_on_requests_and_each_fits(self):
        """Each stage prices its own fused slot, keeps its own host's count, and
        admits only the requests every stage can hold."""
        stages = (HOST_ROWS, HOST_ROWS // 2)
        fewest = min(
            _solve(_plan(host_rows=rows, draft_rows=DRAFT_ROWS)).max_running_requests
            for rows in stages
        )
        self.assertGreater(fewest, 0)

        def reduce_with_other_stages(requests, **_kwargs):
            requests.fill_(min(requests.item(), fewest))

        for rank, rows in enumerate(stages):
            with (
                self.subTest(stage=rank),
                get_parallel().override(
                    pp_size=2,
                    pp_rank=rank,
                    world_group=SimpleNamespace(cpu_group=None),
                ),
                patch.object(torch.distributed, "all_reduce", reduce_with_other_stages),
            ):
                host_plan = _plan(host_rows=rows)
                fused_plan = _plan(host_rows=rows, draft_rows=DRAFT_ROWS)
                host = _solve(host_plan)
                fused = _solve(fused_plan)
                self.assertEqual(fused.units("mamba"), host.units("mamba"))
                self.assertEqual(fused.max_running_requests, fewest)
                self._assert_fits(fused_plan, fused)

    def test_the_post_capture_resize_keeps_the_boot_solve(self):
        plan = _plan(draft_rows=DRAFT_ROWS)
        configurator = _configurator(plan)
        with _server_args():
            KVCacheConfigurator.config_from_budget(configurator, BUDGET)
            boot = configurator._unified_sized
            # Fixed allocations are resident by now, so free memory excludes them.
            configurator._unified_config_from_budget(
                boot.total_bytes, cap_tokens=boot.full_tokens
            )
        resized = configurator._unified_sized
        self.assertEqual(resized.units("mamba"), boot.units("mamba"))
        self.assertEqual(resized.full_tokens, boot.full_tokens)
        self.assertEqual(resized.max_running_requests, boot.max_running_requests)
        self._assert_fits(plan, resized)


if __name__ == "__main__":
    unittest.main()
