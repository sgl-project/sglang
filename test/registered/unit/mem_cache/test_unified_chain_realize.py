"""A layout the chain planner finds is one the allocator can reach.

Random histories leave holes in every band, pending pages in lazy ends, pins
on some bands, fixed pages in others and network transfers on live pages.
Whenever `feasible` admits a demand, the least rearrangement serving it must
realize: every live page keeps its bytes under its virtual id, every fixed
page stays where it is, no two bands' pages share a byte, and each band takes
exactly its demand. Chains range over two ends, one float between them, and
two window floats (the target's and a fused draft's window of another length)
with or without a state end. Through the allocator's own
relief, unfinished forwards also write some live pages and free others while
they run, and a layout that leaves network rows in place is taken whenever one
exists.
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv.alloc import band as band_module
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.alloc.pricing import feasible, first_plan
from sglang.srt.mem_cache.kv.alloc.signals import NetworkTransfer
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    StateGeometry,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage, sub_pool_names
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_WINDOW,
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    PlacementContext,
    place_fused_draft,
)
from sglang.srt.runtime_context import mamba_cache_chunk_size
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.token_demand import demand_windows, token_demand
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
    hold_band,
    setup_allocator_context,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small")


def _allocator(device, chain, lazy):
    """full + window (two ends), or state + window + full (a float between)."""
    state = dict(
        mamba_layer_ids=[0],
        mamba2_cache_params=SimpleNamespace(
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(1, 4, 8)),
            dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0],
        ),
        max_mamba_cache_size=8,
        model_context_len=256,
        extra_max_context_len=1,
        max_num_reqs=4,
        enable_mamba_extra_buffer=False,
        enable_mamba_extra_buffer_lazy=False,
        disable_overlap_schedule=True,
        sliding_window_size=32,
    )
    bundle = build_test_unified_pools(
        device=device,
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=1,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        full_max_total_num_tokens=48,
        swa_max_total_num_tokens=48,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=lazy,
        **(state if chain == "tri" else {}),
    )
    return bundle.token_to_kv_pool_allocator


_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.float16)


def _draft_window_placement(decls, target_window, draft_window):
    return place_fused_draft(
        profile=DraftKVProfile(
            num_layers=1,
            kinds={
                LAYER_WINDOW: DraftLayerSet(
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
        store_dtype=torch.float16,
    ).placement


def _several_floats(device, windows, state, lazy):
    """full + two window floats (the target's, then a fused draft's of another
    length), with a state end below them or not."""
    first_window = 2 if state else 1
    target, draft = windows
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_MHA,
        window_layer_ids=(first_window,),
        window_geometry=_MHA,
        window=target,
        state_layer_ids=(1,) if state else (),
        state_geometry=(
            StateGeometry(
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(1, 4, 8),
                temporal_dtype=torch.float32,
            )
            if state
            else None
        ),
        checkpoint_interval=mamba_cache_chunk_size() if state else None,
    )
    plan = plan_kv_storage(
        decls,
        page_size=1,
        fused_draft=_draft_window_placement(decls, target, draft),
    )
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=48,
            window_tokens=24,
            state_slots=8 if state else 0,
            max_running_requests=4,
            context_len=64,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device=device,
        start_layer=0,
        end_layer=first_window + 1,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=4,
        model_context_len=64,
        lazy_compaction=lazy,
        state_cache_params=SimpleNamespace(layers=(1,)) if state else None,
    )
    return build_kv_storage(plan, sized, ctx).allocator


def _live_bytes(allocator):
    """Every live page's bytes, keyed by (band, virtual page)."""
    raw = allocator.arena.raw.cpu()
    out = {}
    for band in allocator.bands:
        e = band.entry_bytes_per_page
        p2v = band.ids.physical_to_virtual.cpu().tolist()
        for page, virtual in enumerate(p2v):
            if virtual >= 0 and page >= band.min_page_index:
                out[band.sub_pool_name, virtual] = raw[
                    page * e : (page + 1) * e
                ].clone()
    return out


def _live_pages(band):
    p2v = band.ids.physical_to_virtual.cpu().tolist()
    return [p for p, v in enumerate(p2v) if v >= 0 and p >= band.min_page_index]


def _fixed(allocator):
    """Every fixed page holding a live page, and the virtual page it holds.
    Fixed forbids moving a page; a fixed page that was freed holds nothing to
    keep, and a take may reuse it in place like any hole."""
    out = {}
    for band in allocator.bands:
        p2v = band.ids.physical_to_virtual
        for page in band.fixed_pages():
            virtual = int(p2v[page])
            if virtual >= 0:
                out[band.sub_pool_name, page] = virtual
    return out


def _page_bytes(allocator, taken):
    """Byte ranges of every live page and every page just taken, per band."""
    ranges = []
    for band in allocator.bands:
        e = band.entry_bytes_per_page
        p2v = band.ids.physical_to_virtual.cpu().tolist()
        pages = {p for p, v in enumerate(p2v) if v >= 0 and p >= band.min_page_index}
        if band.sub_pool_name in taken:
            fresh = set(taken[band.sub_pool_name].cpu().tolist())
            assert not fresh & pages, "a take handed out a live page"
            pages |= fresh
        ranges += [(p * e, (p + 1) * e, band.sub_pool_name) for p in pages]
    return sorted(ranges)


class TestChainRealize(CustomTestCase):
    def _assert_fixed_kept(self, allocator, fixed):
        """Every fixed page that held a live page still holds it, still fixed."""
        now = {
            (band.sub_pool_name, page): int(band.ids.physical_to_virtual[page])
            for band in allocator.bands
            for page in band.fixed_pages()
        }
        for key, virtual in fixed.items():
            self.assertEqual(now.get(key), virtual, key)

    def _history(self, rng, allocator):
        held, slots = [], []
        state = allocator.mamba_allocator
        for _ in range(rng.randint(3, 10)):
            got = allocator.alloc(rng.randint(1, 8))
            if got is not None:
                held.append(got)
            if state is not None and rng.random() < 0.5:
                got = allocator.alloc_state_slots(1)
                if got is not None:
                    slots.append(got)
        windows = sorted(getattr(allocator, "window_allocators", None) or ())
        for ids in held:
            choice = rng.random()
            if choice < 0.3:
                allocator.free(ids)
            elif choice < 0.5 and len(windows) > 1:
                allocator.free_window(
                    rng.choice(windows), ids[: rng.randint(1, ids.numel())]
                )
            elif choice < 0.5 and allocator.swa_attn_allocator is not None:
                allocator.free_swa(ids[: rng.randint(1, ids.numel())])
        for slot in slots:
            if rng.random() < 0.4:
                state.free(slot)
        for band in allocator.bands:
            if rng.random() < 0.2:
                hold_band(band)
            elif rng.random() < 0.3:
                live = _live_pages(band)
                if live:
                    band.fix_pages("held", rng.sample(live, min(len(live), 2)))

    def _check(self, device, seed, count):
        rng = random.Random(seed)
        realized = refused = 0
        while realized < count:
            setup_allocator_context()
            chain = rng.choice(("swa", "tri"))
            allocator = _allocator(device, chain, lazy=True)
            raw = allocator.arena.raw
            raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8))
            self._history(rng, allocator)
            snapshot = allocator.pricing_snapshot()
            demands = {
                b.name: rng.choice((0, 1, 2, 4, 8, 16, 24, 32)) for b in snapshot.bands
            }
            if not feasible(snapshot, demands):
                refused += 1
                continue
            growth = {b.name: rng.choice((0, 1, 2)) for b in snapshot.bands}
            before = _live_bytes(allocator)
            fixed = _fixed(allocator)
            taken = allocator.realize(first_plan(snapshot, demands, growth=growth))
            realized += 1
            self._assert_fixed_kept(allocator, fixed)
            after = _live_bytes(allocator)
            self.assertEqual(before.keys(), after.keys())
            for key, payload in before.items():
                self.assertTrue(torch.equal(payload, after[key]), key)
            for name, pages in demands.items():
                self.assertEqual(0 if name not in taken else taken[name].numel(), pages)
            ranges = _page_bytes(allocator, taken)
            for (_, end, a), (start, _, b) in zip(ranges, ranges[1:]):
                self.assertLessEqual(end, start, (a, b))
        self.assertGreater(refused, 0)

    def _check_several_floats(self, device, seed, count):
        # Two window floats between the owner's end and a state end (or
        # none), with transfers on live pages finishing while the
        # rearrangement waits for them.
        rng = random.Random(seed)
        shapes = (((32, 16), False), ((32, 16), True))
        realized = refused = 0
        floats = set()
        while realized < count:
            setup_allocator_context()
            windows, state = shapes[realized % len(shapes)]
            allocator = _several_floats(device, windows, state, True)
            # The engines are done by the time a move must wait for them.
            allocator.set_network_waiter(lambda transfers: None)
            raw = allocator.arena.raw
            raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8))
            self._history(rng, allocator)
            running = []
            for band in allocator.bands:
                live = _live_pages(band)
                if live and rng.random() < 0.4:
                    transfer = NetworkTransfer()
                    running.append(transfer)
                    band.record_transfer(
                        transfer,
                        rng.sample(live, min(len(live), 3)),
                        writes=rng.random() < 0.5,
                        release=transfer,
                        unit_tokens=band.pool_page_size,
                    )
            snapshot = allocator.pricing_snapshot()
            demands = {b.name: rng.choice((0, 1, 2, 4, 8, 16)) for b in snapshot.bands}
            if not feasible(snapshot, demands):
                refused += 1
                continue
            growth = {b.name: rng.choice((0, 1, 2)) for b in snapshot.bands}
            before = _live_bytes(allocator)
            fixed = _fixed(allocator)
            taken = allocator.realize(first_plan(snapshot, demands, growth=growth))
            if device == "cuda":
                torch.cuda.synchronize()
            realized += 1
            floats.add((windows, state))
            self._assert_fixed_kept(allocator, fixed)
            after = _live_bytes(allocator)
            self.assertEqual(before.keys(), after.keys())
            for key, payload in before.items():
                self.assertTrue(torch.equal(payload, after[key]), key)
            for name, pages in demands.items():
                self.assertEqual(0 if name not in taken else taken[name].numel(), pages)
            ranges = _page_bytes(allocator, taken)
            for (_, end, a), (start, _, b) in zip(ranges, ranges[1:]):
                self.assertLessEqual(end, start, (a, b))
        self.assertGreater(refused, 0)
        self.assertEqual(floats, set(shapes))

    def test_growth_follows_the_bands_takes_extend(self):
        # Two chains with the same history keep the same growth averages, read
        # from their plans alone. A take that extends a band adds to its
        # average; one that only reuses its holes leaves the average decaying.
        def history():
            setup_allocator_context()
            allocator = _allocator("cpu", "tri", lazy=True)
            for _ in range(4):
                allocator.alloc_state_slots(1)
            held = allocator.alloc(6)
            return allocator, held

        first, held = history()
        second, _ = history()
        self.assertEqual(first._growth, second._growth)
        state = first.mamba_allocator.sub_pool_name
        owner = first.full_attn_allocator.sub_pool_name
        self.assertGreater(first._growth[state], 0)
        grown = first._growth[owner]
        self.assertGreater(grown, 0)
        first.free(held)
        self.assertIsNotNone(first.alloc(6))
        self.assertEqual(first._growth[owner], grown - grown // 16)

    def test_a_load_reservation_is_fixed_until_submitted_or_cancelled(self):
        # A HiCache load reserves window pages before its copy is submitted;
        # the band compacts around them instead of refusing to move at all.
        setup_allocator_context()
        allocator = _allocator("cpu", "swa", lazy=True)
        window = allocator.swa_attn_allocator
        held = allocator.alloc(12)
        allocator.free_swa(held[2:10])
        tokens = window.alloc_physical(3)
        reserved = set(tokens.tolist())
        self.assertEqual(window.fixed_pages(), reserved)
        before = {p: int(window.ids.physical_to_virtual[p]) for p in reserved}
        self.assertGreater(window.flush_for_allocation(), 0)
        self.assertEqual(
            {p: int(window.ids.physical_to_virtual[p]) for p in reserved}, before
        )
        # Packed as far as the reservation allows, which stays below the frontier.
        live = len(_live_pages(window)) + len(reserved)
        if window.grow_direction == "up":
            self.assertEqual(
                window.watermark_physical,
                max(window.min_page_index + live, max(reserved) + 1),
            )
        else:
            self.assertEqual(
                window.watermark_physical + 1,
                min(window.num_pages - live, min(reserved)),
            )
        window.cancel_physical_reservation(tokens[:1])
        self.assertEqual(window.fixed_pages(), reserved - {int(tokens[0])})
        window.end_load_reservation()
        self.assertEqual(window.fixed_pages(), set())

    def _check_answers_are_takes(self, device, seed, count):
        # The allocator's own answers: `ensure_capacity` is exactly whether the
        # take succeeds, and admission (no cache to evict) never says yes to a
        # take that would fail.
        rng = random.Random(seed)
        outcomes = set()
        for _ in range(count):
            setup_allocator_context()
            chain = rng.choice(("swa", "tri"))
            allocator = _allocator(device, chain, lazy=True)
            self._history(rng, allocator)
            tokens = rng.choice((1, 2, 4, 8, 16, 24, 32))
            if rng.random() < 0.5 or allocator.mamba_allocator is None:
                fits = allocator.ensure_capacity(
                    token_demand(allocator, tokens, tokens)
                )
                admitted = allocator.admits(
                    KVDemand(tokens, demand_windows(allocator, tokens, tokens))
                )
                taken = allocator.alloc(tokens)
            else:
                fits = allocator.ensure_capacity(
                    token_demand(allocator, 0, 0, state_slots=tokens % 4 + 1)
                )
                admitted = fits
                taken = allocator.alloc_state_slots(tokens % 4 + 1)
            self.assertEqual(fits, taken is not None)
            if admitted:
                self.assertIsNotNone(taken)
            outcomes.add(fits)
            self.assertFalse(allocator.verify_byte_accounting())
        self.assertEqual(outcomes, {False, True})

    def _check_transfers_leave_answers(self, device, seed, count):
        # Network transfers in flight read or write physical pages; they order
        # moves and never change pricing, so `ensure_capacity` stays exactly
        # whether the take succeeds while they run.
        rng = random.Random(seed)
        outcomes = set()
        for _ in range(count):
            setup_allocator_context()
            chain = rng.choice(("swa", "tri"))
            allocator = _allocator(device, chain, lazy=True)
            # The engines are done by the time a move must wait for them.
            allocator.set_network_waiter(lambda transfers: None)
            self._history(rng, allocator)
            before = allocator.pricing_snapshot()
            running = []
            for band in allocator.bands:
                live = _live_pages(band)
                if live and rng.random() < 0.7:
                    transfer = NetworkTransfer()
                    running.append(transfer)
                    band.record_transfer(
                        transfer,
                        rng.sample(live, min(len(live), 3)),
                        writes=rng.random() < 0.5,
                        release=transfer,
                        unit_tokens=band.pool_page_size,
                    )
            self.assertEqual(allocator.pricing_snapshot(), before)
            tokens = rng.choice((1, 2, 4, 8, 16, 24, 32))
            fits = allocator.ensure_capacity(token_demand(allocator, tokens, tokens))
            taken = allocator.alloc(tokens)
            self.assertEqual(fits, taken is not None)
            outcomes.add(fits)
            if device == "cuda":
                torch.cuda.synchronize()
            self.assertFalse(allocator.verify_byte_accounting())
        self.assertEqual(outcomes, {False, True})

    def _check_reads_are_takes(self, device, seed, count):
        # Each sub-pool's read is the largest demand of that sub-pool alone
        # the take's own answer admits (a scan finds no larger one, so the
        # reads' search may assume nothing about the answer's shape), taking
        # exactly that much succeeds, and the joint token read is the largest
        # demand every token sub-pool can take together.
        rng = random.Random(seed)
        shared = 0
        for _ in range(count):
            setup_allocator_context()
            chain = rng.choice(("swa", "tri"))
            allocator = _allocator(device, chain, lazy=True)
            self._history(rng, allocator)
            page = allocator.page_size
            units = {}
            for band in allocator.bands:
                name = band.sub_pool_name
                bound = allocator._id_bound(band)
                largest = max(
                    n
                    for n in range(bound + 2)
                    if n == 0 or allocator.units_fit({name: n})
                )
                unit = 1 if band is allocator.mamba_allocator else page
                self.assertEqual(allocator.units_available(name), largest * unit)
                units[name] = largest
            joint = allocator.available_size()
            # Bands share bytes: the joint read can be below every one alone.
            tokens = [
                units[b.sub_pool_name] * page
                for b in allocator.bands
                if b is not allocator.mamba_allocator
            ]
            shared += joint < min(tokens)
            self.assertTrue(
                allocator.ensure_capacity(token_demand(allocator, joint, joint))
            )
            self.assertFalse(
                allocator.ensure_capacity(
                    token_demand(allocator, joint + page, joint + page)
                )
            )
            # Taking exactly a read succeeds, through the allocator's takes: a
            # token take binds the owner and every window together.
            state = allocator.mamba_allocator
            if state is not None and rng.random() < 0.5:
                slots = units[state.sub_pool_name]
                if slots:
                    self.assertIsNotNone(allocator.alloc_state_slots(slots))
            elif joint:
                self.assertIsNotNone(allocator.alloc(joint))
            self.assertFalse(allocator.verify_byte_accounting())
        self.assertGreater(shared, 0)

    def test_reads_are_takes_on_cpu(self):
        self._check_reads_are_takes("cpu", seed=31, count=300)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_reads_are_takes_on_cuda(self):
        self._check_reads_are_takes("cuda", seed=32, count=40)

    def test_transfers_leave_capacity_answers_on_cpu(self):
        self._check_transfers_leave_answers("cpu", seed=21, count=300)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_transfers_leave_capacity_answers_on_cuda(self):
        self._check_transfers_leave_answers("cuda", seed=22, count=60)

    def _check_relief(self, device, seed, count, *, last_resort=False):
        # The allocator's own relief: unfinished forwards write live owner
        # pages and some of them are freed while the forward runs; network
        # transfers read or write live pages. Every feasible demand is taken
        # with live bytes kept, and the network rows stay put whenever a layout
        # around them serves the demand. With ``last_resort``, a layout that
        # did not keep them (the plain plan, as a relief makes when no layout
        # around them exists) first moved the rows, leaving the bytes the sends
        # still read outside the bands: the next relief plans around those too
        # whenever it can, and every plan realizes.
        rng = random.Random(seed)
        side = torch.cuda.Stream() if device == "cuda" else None
        shapes = (((32, 16), False), ((32, 16), True))
        waits = []
        realized = refused = kept_in_place = 0
        while realized < count:
            setup_allocator_context()
            windows, state = shapes[realized % len(shapes)]
            allocator = _several_floats(device, windows, state, lazy=True)
            allocator.set_network_waiter(lambda ts: waits.append(len(ts)))
            raw = allocator.arena.raw
            raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8))
            self._history(rng, allocator)
            owner = allocator.full_attn_allocator
            if side is not None:
                p2v = owner.ids.physical_to_virtual.cpu().tolist()
                # Live owner pages only: the reserved pages below the band
                # (the padding sink) are no request's to free.
                virtual = [
                    v for p, v in enumerate(p2v) if v >= 0 and p >= owner.min_page_index
                ]
                if virtual:
                    done = torch.cuda.Event()
                    with torch.cuda.stream(side):
                        torch.cuda._sleep(rng.randint(0, 2_000_000))
                        done.record()
                    written = rng.sample(virtual, min(len(virtual), 6))
                    ids = torch.tensor(written, dtype=torch.int64, device=device)
                    allocator.set_latest_forward_done_event(done)
                    allocator.set_inflight_forward(done, ids)
                    allocator.free(ids[: rng.randint(0, ids.numel())])
            running = []
            for band in allocator.bands:
                live = _live_pages(band)
                if live and rng.random() < 0.5:
                    transfer = NetworkTransfer()
                    running.append(transfer)
                    band.record_transfer(
                        transfer,
                        rng.sample(live, min(len(live), 2)),
                        writes=rng.random() < 0.5,
                        release=transfer,
                        unit_tokens=band.pool_page_size,
                    )
            if last_resort and running:
                snapshot = allocator.pricing_snapshot()
                earlier = first_plan(
                    snapshot,
                    {b.name: rng.choice((0, 2, 8, 16)) for b in snapshot.bands},
                )
                if earlier is not None:
                    allocator._lay_out(earlier)
                    if device == "cuda":
                        torch.cuda.synchronize()
            snapshot = allocator.pricing_snapshot()
            demands = {b.name: rng.choice((0, 1, 2, 4, 8, 16)) for b in snapshot.bands}
            if not feasible(snapshot, demands):
                refused += 1
                continue
            rows = {}
            for band in allocator.bands:
                pages = band.network_rows()
                p2v = band.ids.physical_to_virtual.cpu().tolist()
                rows[band.sub_pool_name] = {p: p2v[p] for p in pages}
                band.fix_pages("probe", sorted(pages))
            held = first_plan(allocator.held_snapshot(), demands) is not None
            for band in allocator.bands:
                band.unfix_pages("probe")
            before = _live_bytes(allocator)
            fixed = _fixed(allocator)
            waits.clear()
            taken = allocator._plan_and_take(demands)
            if device == "cuda":
                torch.cuda.synchronize()
            realized += 1
            self.assertIsNotNone(taken)
            self._assert_fixed_kept(allocator, fixed)
            after = _live_bytes(allocator)
            self.assertEqual(before.keys(), after.keys())
            for key, payload in before.items():
                self.assertTrue(torch.equal(payload, after[key]), key)
            for name, pages in demands.items():
                self.assertEqual(0 if name not in taken else taken[name].numel(), pages)
            if held:
                kept_in_place += 1
                self.assertEqual(waits, [], "a layout around the rows waited")
                for band in allocator.bands:
                    p2v = band.ids.physical_to_virtual.cpu().tolist()
                    for page, virtual in rows[band.sub_pool_name].items():
                        if virtual >= 0:
                            self.assertEqual(p2v[page], virtual, "a network row moved")
            ranges = _page_bytes(allocator, taken)
            for (_, end, a), (start, _, b) in zip(ranges, ranges[1:]):
                self.assertLessEqual(end, start, (a, b))
        self.assertGreater(refused, 0)
        self.assertGreater(kept_in_place, 0)

    def test_reliefs_keep_bytes_and_network_rows_on_cpu(self):
        self._check_relief("cpu", seed=51, count=300)

    def test_reliefs_after_a_last_resort_layout_on_cpu(self):
        self._check_relief("cpu", seed=52, count=300, last_resort=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_reliefs_after_a_last_resort_layout_on_cuda(self):
        self._check_relief("cuda", seed=52, count=60, last_resort=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_reliefs_keep_bytes_and_network_rows_on_cuda(self):
        self._check_relief("cuda", seed=52, count=60)

    def test_capacity_answers_are_takes_on_cpu(self):
        self._check_answers_are_takes("cpu", seed=11, count=600)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_capacity_answers_are_takes_on_cuda(self):
        self._check_answers_are_takes("cuda", seed=12, count=80)

    def test_feasible_demands_realize_with_several_floats_on_cpu(self):
        self._check_several_floats("cpu", seed=41, count=400)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_feasible_demands_realize_with_several_floats_on_cuda(self):
        self._check_several_floats("cuda", seed=42, count=60)

    def test_feasible_demands_realize_on_cpu(self):
        self._check("cpu", seed=7, count=800)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_feasible_demands_realize_on_cuda(self):
        self._check("cuda", seed=8, count=100)


class TestFrontierTakesBindInOneKernel(CustomTestCase):
    """A planned take that only extends an end's frontier takes and binds its
    pages in the fused kernel (`Band.place_pages`), one launch per band, as an
    unplanned take did; a take that fills holes, or a float's, binds apart."""

    def setUp(self):
        setup_allocator_context()

    def _fused_calls(self):
        calls = []
        real = band_module.alloc_bind_inplace

        def counting(*args, **kwargs):
            calls.append(args[0].numel())
            return real(*args, **kwargs)

        patcher = patch.object(band_module, "alloc_bind_inplace", counting)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def _assert_bound(self, allocator, virtual_pages):
        for band in (
            allocator.full_attn_allocator,
            *allocator.window_allocators.values(),
        ):
            physical = band.ids.virtual_to_physical[virtual_pages]
            self.assertTrue(bool((physical >= 0).all()), band.sub_pool_name)
            self.assertTrue(
                torch.equal(band.ids.physical_to_virtual[physical], virtual_pages),
                band.sub_pool_name,
            )

    def test_frontier_takes_bind_in_one_kernel(self):
        # Two ends: FULL and its window both extend a frontier; a float
        # window (between state and FULL) is taken apart.
        for chain, fused_bands in (("swa", 2), ("tri", 1)):
            with self.subTest(chain=chain):
                allocator = _allocator("cpu", chain, lazy=True)
                calls = self._fused_calls()
                taken = allocator.alloc(8)
                self.assertIsNotNone(taken)
                self.assertEqual(calls, [8] * fused_bands)
                self._assert_bound(allocator, taken.to(torch.int64))

    def test_a_take_that_fills_holes_binds_apart(self):
        allocator = _allocator("cpu", "swa", lazy=True)
        first = allocator.alloc(16)
        (window,) = allocator.window_allocators
        allocator.free_window(window, first[:8])
        calls = self._fused_calls()
        taken = allocator.alloc(4)
        self.assertIsNotNone(taken)
        # The owner extends its frontier; the window reuses its holes.
        self.assertEqual(calls, [4])
        self._assert_bound(allocator, taken.to(torch.int64))


if __name__ == "__main__":
    unittest.main()
