"""No-relief placement is downward closed and its physical takes are realizable."""

import itertools
import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.alloc.band import FloatBand
from sglang.srt.mem_cache.kv.alloc.pricing import no_relief_plan, sequential_fit
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftBand,
    FusedDraftPlacement,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _build(*, state, lazy, draft_dim, device="cpu"):
    setup_allocator_context()
    placement = FusedDraftPlacement.from_counts(
        counts={"draft_swa": [1]},
        regions={
            "draft_swa": DenseDraftRegion(
                lane_num=1,
                head_num=1,
                head_dim=draft_dim,
                store_dtype=torch.bfloat16,
            )
        },
        new_bands={"draft_swa": DraftBand(kind="window", window=12)},
    )
    return build_unified_bundle(
        UnifiedGeometry(
            chain="tri" if state else "swa",
            full_layer_ids=(0,),
            swa_layer_ids=(1,),
            state_layer_ids=(2,) if state else (),
            page_size=4,
            window=16,
            context_len=32,
            full_tokens=96,
            swa_tokens=48,
            state_slots=8,
            lazy=lazy,
            fused_draft=placement,
            device=device,
        )
    ).token_to_kv_pool_allocator


def _random_history(allocator, rng):
    """Allocate and free requests, sometimes freeing only their draft window,
    so the bands' frontiers and holes end up anywhere."""
    live = []
    for _ in range(rng.randint(2, 7)):
        r = rng.random()
        if r < 0.55 or not live:
            got = allocator.alloc(4 * rng.randint(1, 6))
            if got is not None:
                live.append(got)
        elif r < 0.8:
            ids = live.pop(rng.randrange(len(live)))
            allocator.free_window("draft_swa", ids)
            allocator.free(ids)
        else:
            allocator.free_window("draft_swa", live[rng.randrange(len(live))])


def _load_back(allocator, rng, full, swa, auxiliary_swa, state_slots):
    """Issue a host load's takes in a random band order: the owner's token
    ids, each window's host rows, and state slots. Returns the first refused
    sub-pool (or None) and the window rows reserved, which only a prefix tree
    binds."""
    storage = allocator.storage
    takes = [("full", full), ("swa", swa), ("draft_swa", auxiliary_swa)]
    if state_slots:
        takes.append(("mamba", state_slots))
    rng.shuffle(takes)
    reserved = []
    for name, units in takes:
        if not units:
            continue
        if name == "full":
            got = allocator.full_attn_allocator.alloc(units)
        else:
            take, give_back = storage.host_load_alloc(name)
            got = take(units)
            if got is not None and name != "mamba":
                reserved.append((give_back, got))
        if got is None:
            return name, reserved
    return None, reserved


def _give_back(reserved):
    for give_back, rows in reserved:
        give_back(rows)


def _save(bands):
    saved = []
    for band in bands:
        fields = [
            "watermark_physical",
            "live_page_count",
            "_free_phys_pages",
            "_free_phys_sorted",
        ]
        if isinstance(band, FloatBand):
            fields += ["low_wm_page", "high_wm_page", "_next_hole_end"]
        saved.append((band, {name: getattr(band, name) for name in fields}))
    return saved


def _restore(saved):
    for band, fields in saved:
        for name, value in fields.items():
            setattr(band, name, value)


class TestMultiwindowPlacement(unittest.TestCase):
    def test_chunk_budget_crosses_historical_midpoint_fit_island(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                setup_allocator_context()
                placement = FusedDraftPlacement.from_counts(
                    counts={"draft_swa": [1]},
                    regions={
                        "draft_swa": DenseDraftRegion(
                            lane_num=1,
                            head_num=1,
                            head_dim=352,
                            store_dtype=torch.bfloat16,
                        )
                    },
                    new_bands={"draft_swa": DraftBand(kind="window", window=12)},
                )
                allocator = build_unified_bundle(
                    UnifiedGeometry(
                        chain="tri",
                        full_layer_ids=(0,),
                        swa_layer_ids=(1,),
                        state_layer_ids=(2,),
                        head_dim=32,
                        swa_head_dim=288,
                        page_size=4,
                        window=16,
                        context_len=16,
                        full_tokens=104,
                        swa_tokens=16,
                        state_slots=8,
                        lazy=lazy,
                        fused_draft=placement,
                    )
                ).token_to_kv_pool_allocator
                self.assertIsNotNone(allocator.alloc_state_slots(1))
                # A real midpoint allocation followed by a window expiration
                # leaves one live float next to an empty float. The old greedy
                # fit predicate has an island on this reachable geometry.
                owner = allocator.full_attn_allocator
                resident = owner.alloc(4)
                self.assertIsNotNone(resident)
                for window in allocator._windows:
                    window.alloc_with_virtual(resident[::4] // 4)
                allocator.free_window("draft_swa", resident)
                snapshot = allocator.pricing_snapshot()
                names = (owner.sub_pool_name,) + tuple(
                    window.sub_pool_name for window in allocator._windows
                )
                self.assertEqual(
                    [
                        sequential_fit(
                            snapshot,
                            ((allocator.roles.state, 1),)
                            + tuple((name, n) for name in names),
                        )
                        for n in (2, 3, 4)
                    ],
                    [False, True, False],
                )
                cache = SimpleNamespace(
                    sliding_window_size=16,
                    window_retain_size=lambda component: 12,
                    full_evictable_size=lambda: 0,
                    swa_evictable_size=lambda: 0,
                    auxiliary_swa_evictable_size=lambda: 0,
                    supports_prefix_sharing=lambda: True,
                    supports_mamba=lambda: True,
                    mamba_evictable_size=lambda: 0,
                )
                budget = allocator.create_prefill_budget(cache)
                cap = budget._joint_chunk_cap(
                    max_chunk_tokens=11, chunk_limit=12, state_slots=1
                )
                selected = budget.fit_chunk(
                    extend_input_len=12,
                    max_new_tokens=1,
                    chunk_limit=12,
                    state_slots=1,
                )
                self.assertEqual(cap, 8)
                self.assertEqual(selected, cap)
                # The scheduler can round the selected chunk down. Both the
                # selected shape and every positive smaller page shape retain
                # the same pending-state demand and remain realizable.
                for rounded in range(4, selected + 1, 4):
                    self.assertTrue(
                        budget._fits(
                            rounded + 4,
                            budget.swa_tokens(rounded, 0, chunk_limit=12),
                            auxiliary_swa_tokens=budget.auxiliary_tokens(
                                rounded, 0, chunk_limit=12
                            ),
                            state_slots=1,
                        )
                    )
                self.assertIsNotNone(allocator.alloc_state_slots(1))
                self.assertIsNotNone(allocator.alloc(selected))
                self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_downward_closure_and_physical_realization(self):
        for state, lazy, draft_dim in itertools.product(
            (False, True), (False, True), (8, 24)
        ):
            allocator = _build(state=state, lazy=lazy, draft_dim=draft_dim)
            if state:
                self.assertIsNotNone(allocator.alloc_state_slots(1))
            for shape in ("empty", "live", "holes", "empty_auxiliary"):
                with self.subTest(
                    state=state, lazy=lazy, draft_dim=draft_dim, shape=shape
                ):
                    if shape == "live":
                        resident = allocator.alloc(12)
                        self.assertIsNotNone(resident)
                    elif shape == "holes":
                        allocator.free_window("draft_swa", resident[4:8])
                        allocator.free_swa_segment(resident[4:8], start_pos=4)
                    elif shape == "empty_auxiliary":
                        allocator.free_window("draft_swa", resident[:4])
                        allocator.free_window("draft_swa", resident[8:])
                    snapshot = allocator.pricing_snapshot()
                    order = tuple(
                        b
                        for b in (
                            allocator.mamba_allocator,
                            allocator.full_attn_allocator,
                            *allocator._windows,
                        )
                        if b is not None
                    )
                    names = tuple(b.sub_pool_name for b in order)
                    fits = {}
                    saved = _save(order)
                    for values in itertools.product(range(4), repeat=len(names)):
                        demands = tuple(zip(names, values))
                        plan = no_relief_plan(snapshot, demands)
                        fits[values] = plan is not None
                        if sequential_fit(snapshot, demands):
                            self.assertIsNotNone(plan, demands)
                        if plan is None:
                            continue
                        ranges = []
                        try:
                            for band in order:
                                physical = band.take_planned_physical_pages(
                                    plan.takes[band.sub_pool_name]
                                )
                                self.assertIsNotNone(
                                    physical, (shape, demands, band.sub_pool_name)
                                )
                                self.assertEqual(
                                    physical.numel(), dict(demands)[band.sub_pool_name]
                                )
                                for page in physical.tolist():
                                    ranges.append(
                                        (
                                            page * band.entry_bytes_per_page,
                                            (page + 1) * band.entry_bytes_per_page,
                                        )
                                    )
                            ranges.sort()
                            self.assertTrue(
                                all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
                            )
                        finally:
                            _restore(saved)
                    for values, feasible in fits.items():
                        if feasible:
                            for i, value in enumerate(values):
                                if value:
                                    lower = (*values[:i], value - 1, *values[i + 1 :])
                                    self.assertTrue(
                                        fits[lower], (values, lower, snapshot)
                                    )

    def test_stale_source_geometry_refuses_without_taking(self):
        allocator = _build(state=False, lazy=True, draft_dim=8)
        window = allocator._windows[0]
        plan = no_relief_plan(
            allocator.pricing_snapshot(), ((window.sub_pool_name, 2),)
        )
        self.assertIsNotNone(plan)
        self.assertIsNotNone(window.take_physical_pages(1))
        before = allocator.pricing_snapshot()
        self.assertIsNone(
            window.take_planned_physical_pages(plan.takes[window.sub_pool_name])
        )
        self.assertEqual(allocator.pricing_snapshot(), before)

    def test_state_first_then_token_transaction_after_end_flush(self):
        for lazy, tokens in itertools.product((False, True), (4, 8, 16)):
            with self.subTest(lazy=lazy, tokens=tokens):
                allocator = _build(state=True, lazy=lazy, draft_dim=24)
                slots = allocator.alloc_state_slots(3)
                self.assertIsNotNone(slots)
                allocator.state_slots.free(slots[1:2])
                allocator.mamba_allocator.flush_for_allocation()
                demands = ((allocator.roles.state, 2),) + tuple(
                    (band.sub_pool_name, tokens // allocator.page_size)
                    for band in (allocator.full_attn_allocator, *allocator._windows)
                )
                self.assertIsNotNone(
                    no_relief_plan(allocator.pricing_snapshot(), demands)
                )
                self.assertIsNotNone(allocator.alloc_state_slots(2))
                self.assertIsNotNone(allocator.alloc(tokens))
                self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_host_prefix_scope_keeps_future_takes_and_restores_cancellations(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                allocator = _build(state=True, lazy=lazy, draft_dim=24)
                bootstrap_state = allocator.alloc_state_slots(8)
                self.assertIsNotNone(bootstrap_state)
                resident = allocator.alloc(8)
                self.assertIsNotNone(resident)
                allocator.state_slots.free(bootstrap_state)
                allocator.mamba_allocator.flush_for_allocation()
                allocator.free_window("draft_swa", resident)
                window = allocator.window_allocators["draft_swa"]
                take_host, cancel_host = allocator.storage.host_load_alloc("draft_swa")
                with allocator.scoped_physical_takes(
                    8, 8, auxiliary_swa_tokens=16, state_slots=2
                ):
                    # H2D may allocate an empty float before the pending state
                    # and token work. Its midpoint heuristic must not consume
                    # the space certified for those later allocations.
                    cancelled = take_host(4)
                    self.assertIsNotNone(cancelled)
                    cancel_host(cancelled)
                    restored = take_host(8)
                    self.assertIsNotNone(restored)
                    window.ids.bind(resident[::4] // 4, restored[::4] // 4)
                    self.assertTrue(window.moves_blocked())
                    state = allocator.alloc_state_slots(1)
                    self.assertIsNotNone(state)
                    allocator.state_slots.free(state)
                    # Returning state/host pages restores their unconsumed
                    # reservation without inspecting a tensor's ids on host.
                    self.assertIsNotNone(allocator.alloc_state_slots(2))
                    self.assertIsNotNone(allocator.alloc(8))
                self.assertTrue(
                    all(b._physical_take_scope is None for b in allocator.members)
                )
                self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_load_back_scope_lays_out_what_reclaim_promised(self):
        """Reclaim says a host load fits once the chain relays its window
        bands out; entering the load's scope must make that relayout instead
        of refusing the reservation it was promised. (Several windows reclaim
        for a load as for a preallocation.)"""
        for state, demand in (
            (False, dict(full=0, swa=8, auxiliary_swa=0, state_slots=0)),
            (True, dict(full=0, swa=32, auxiliary_swa=32, state_slots=1)),
        ):
            for lazy in (False, True):
                with self.subTest(state=state, lazy=lazy):
                    allocator = _build(state=state, lazy=lazy, draft_dim=8)
                    for tokens in (8, 20, 20):
                        self.assertIsNotNone(allocator.alloc(tokens))
                    args = (demand["full"], demand["swa"])
                    kwargs = dict(
                        auxiliary_swa_tokens=demand["auxiliary_swa"],
                        state_slots=demand["state_slots"],
                    )
                    pages = allocator._policy._demands(
                        allocator, *args, *kwargs.values()
                    )
                    # Only a relayout serves it: no layout in place does.
                    self.assertIsNone(
                        no_relief_plan(allocator.pricing_snapshot(), pages)
                    )
                    self.assertIsNone(
                        allocator.reclaim_for_prealloc(None, *args, **kwargs)
                    )
                    with allocator.scoped_physical_takes(*args, **kwargs):
                        refused, reserved = _load_back(
                            allocator, random.Random(0), *args, *kwargs.values()
                        )
                    self.assertIsNone(refused)
                    _give_back(reserved)
                    self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_reclaimed_load_back_always_takes(self):
        """Whenever reclaim says a host load is ready, its scope opens and
        every take succeeds, whatever the history and take order."""
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            histories = 0
            for seed in range(60 if device == "cpu" else 15):
                for state, lazy in itertools.product((False, True), (False, True)):
                    rng = random.Random(seed)
                    allocator = _build(
                        state=state, lazy=lazy, draft_dim=8, device=device
                    )
                    _random_history(allocator, rng)
                    full, swa, aux = (4 * rng.randint(0, 8) for _ in range(3))
                    state_slots = rng.randint(0, 2) if state else 0
                    if not (full or swa or aux or state_slots):
                        continue
                    args = (full, swa)
                    kwargs = dict(auxiliary_swa_tokens=aux, state_slots=state_slots)
                    # Nothing is cached, so reclaim is ready exactly when the
                    # demand fits.
                    if not allocator.ensure_capacity(*args, **kwargs):
                        continue
                    self.assertIsNone(
                        allocator.reclaim_for_prealloc(None, *args, **kwargs)
                    )
                    histories += 1
                    with self.subTest(device=device, seed=seed, state=state, lazy=lazy):
                        with allocator.scoped_physical_takes(*args, **kwargs):
                            refused, reserved = _load_back(
                                allocator, rng, full, swa, aux, state_slots
                            )
                        self.assertIsNone(refused)
                        _give_back(reserved)
                        self.assertEqual(allocator.verify_byte_accounting(), [])
            self.assertGreater(histories, 20 if device == "cpu" else 5)

    def test_exception_closes_scope_without_undoing_submitted_work(self):
        allocator = _build(state=False, lazy=True, draft_dim=8)
        window = allocator.window_allocators["draft_swa"]
        with self.assertRaisesRegex(RuntimeError, "after H2D"):
            with allocator.scoped_physical_takes(4, 4, auxiliary_swa_tokens=4):
                self.assertIsNotNone(window.alloc_physical(4))
                raise RuntimeError("after H2D")
        self.assertTrue(all(b._physical_take_scope is None for b in allocator.members))
        self.assertEqual(window.allocated_count(), 4)
        self.assertTrue(window.moves_blocked())


if __name__ == "__main__":
    unittest.main()
