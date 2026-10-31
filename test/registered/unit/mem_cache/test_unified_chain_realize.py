"""A layout the chain planner finds is one the allocator can reach.

Random histories leave holes in every band, pending pages in lazy ends, pins
on some bands and fixed pages in others. Whenever `feasible` admits a demand,
the least rearrangement serving it must realize: every live page keeps its
bytes under its virtual id, every fixed page stays where it is, no two bands'
pages share a byte, and each band takes exactly its demand.
"""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.alloc.pricing import feasible, first_plan
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
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
    """Every fixed page and the virtual page it holds."""
    return {
        (band.sub_pool_name, page): int(band.ids.physical_to_virtual[page])
        for band in allocator.bands
        for page in band.fixed_pages()
    }


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
        for ids in held:
            choice = rng.random()
            if choice < 0.3:
                allocator.free(ids)
            elif choice < 0.5 and allocator.swa_attn_allocator is not None:
                allocator.free_swa(ids[: rng.randint(1, ids.numel())])
        for slot in slots:
            if rng.random() < 0.4:
                state.free(slot)
        for band in allocator.bands:
            if rng.random() < 0.2:
                band.pin_sources = (lambda: True,)
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
            allocator = _allocator(device, chain, lazy=rng.random() < 0.8)
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
            self.assertEqual(_fixed(allocator), fixed)
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
        window.set_hicache_transfer_done_event("load", object())
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
            allocator = _allocator(device, chain, lazy=rng.random() < 0.8)
            self._history(rng, allocator)
            tokens = rng.choice((1, 2, 4, 8, 16, 24, 32))
            if rng.random() < 0.5 or allocator.mamba_allocator is None:
                fits = allocator.ensure_capacity(tokens, tokens)
                admitted = allocator.can_reserve(tokens, tokens)
                taken = allocator.alloc(tokens)
            else:
                fits = allocator.ensure_capacity(0, 0, state_slots=tokens % 4 + 1)
                admitted = fits
                taken = allocator.alloc_state_slots(tokens % 4 + 1)
            self.assertEqual(fits, taken is not None)
            if admitted:
                self.assertIsNotNone(taken)
            outcomes.add(fits)
            self.assertFalse(allocator.verify_byte_accounting())
        self.assertEqual(outcomes, {False, True})

    def test_capacity_answers_are_takes_on_cpu(self):
        self._check_answers_are_takes("cpu", seed=11, count=600)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_capacity_answers_are_takes_on_cuda(self):
        self._check_answers_are_takes("cuda", seed=12, count=80)

    def test_feasible_demands_realize_on_cpu(self):
        self._check("cpu", seed=7, count=800)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_feasible_demands_realize_on_cuda(self):
        self._check("cuda", seed=8, count=100)


if __name__ == "__main__":
    unittest.main()
