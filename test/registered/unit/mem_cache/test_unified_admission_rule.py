"""An admitted demand is a demand the allocator serves.

Admission prices a request's FULL tokens, its tokens in every window and its
state slots against the chain after the cached entries it credits are evicted.
Over random histories that leave holes, pending pages, pinned bands and fixed
pages, whenever admission says yes, evicting exactly the credited entries and
then taking the request's state slots and tokens must succeed.
"""

import random
import unittest
from types import SimpleNamespace

import torch
from test_unified_chain_realize import _allocator, _live_pages

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.token_demand import token_demand
from sglang.test.unified_allocator_fixtures import (
    hold_band,
    setup_allocator_context,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small")


class TestAdmissionIsATake(CustomTestCase):
    def _history(self, rng, allocator, lazy):
        """Leave holes and pending pages; return the cached entries (token ids
        and state slots) the tree could evict."""
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
        kept = []
        for ids in held:
            choice = rng.random()
            if choice < 0.25:
                allocator.free(ids)
            elif choice < 0.45 and allocator.swa_attn_allocator is not None:
                allocator.free_swa(ids[: rng.randint(1, ids.numel())])
                kept.append(ids)
            else:
                kept.append(ids)
        cached = [ids for ids in kept if rng.random() < 0.5]
        cached_slots = [slot for slot in slots if rng.random() < 0.5]
        # Pins and fixed pages only on entries that stay: an evicted page must
        # be free to go.
        protected = set()
        for ids in kept:
            if not any(ids is c for c in cached):
                protected.update(ids.cpu().tolist())
        for band in allocator.bands:
            # A band held where it stands, as before holds needed lazy bands.
            if lazy and rng.random() < 0.15:
                hold_band(band)
            elif rng.random() < 0.3 and band is not state:
                p2v = band.ids.physical_to_virtual.cpu().tolist()
                live = [p for p in _live_pages(band) if p2v[p] in protected]
                if live:
                    band.fix_pages("held", rng.sample(live, min(len(live), 2)))
        return cached, cached_slots

    def _credit(self, allocator, cached, cached_slots):
        """What evicting the cached entries frees in each sub-pool."""
        tokens = (
            torch.cat(cached).cpu() if cached else torch.empty(0, dtype=torch.int64)
        )
        credit = {allocator.full_attn_allocator.sub_pool_name: tokens.numel()}
        for window in allocator._windows:
            bound = window.ids.virtual_to_physical.cpu()[tokens] > 0
            credit[window.sub_pool_name] = int(bound.sum())
        if allocator.mamba_allocator is not None:
            credit[allocator.mamba_allocator.sub_pool_name] = len(cached_slots)
        return credit

    def _check(self, device, seed, count):
        rng = random.Random(seed)
        admitted = refused = 0
        while admitted < count:
            setup_allocator_context()
            lazy = rng.random() < 0.8
            allocator = _allocator(device, rng.choice(("swa", "tri")), lazy=lazy)
            cached, cached_slots = self._history(rng, allocator, lazy)
            tokens = rng.choice((1, 2, 4, 8, 16, 24, 32, 48, 64))
            state_slots = (
                rng.choice((0, 1, 2)) if allocator.mamba_allocator is not None else 0
            )
            if not allocator.admits(
                tokens,
                {w.sub_pool_name: tokens for w in allocator._windows},
                state_slots,
                evictable=self._credit(allocator, cached, cached_slots),
            ):
                refused += 1
                continue
            admitted += 1
            for ids in cached:
                allocator.free(ids)
            for slot in cached_slots:
                allocator.mamba_allocator.free(slot)
            if state_slots:
                self.assertIsNotNone(allocator.alloc_state_slots(state_slots))
            self.assertIsNotNone(allocator.alloc(tokens))
        self.assertGreater(refused, 0)

    def test_cpu(self):
        self._check("cpu", seed=31, count=400)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        self._check("cuda", seed=37, count=60)


def _empty_tree():
    """A prefix tree with nothing evictable."""
    return SimpleNamespace(
        full_evictable_size=lambda: 0,
        swa_evictable_size=lambda: 0,
        window_evictable_size=lambda component: 0,
        mamba_evictable_size=lambda: 0,
        supports_mamba=lambda: True,
        window_retain_size=lambda component: 32,
    )


class TestDecodePreallocationTakesItsStateSlots(CustomTestCase):
    """A decode node's preallocation takes a request's state slot and then its
    tokens from the same bytes. Admission must price them together: checking
    the slot alone and the tokens alone admits requests whose tokens no longer
    fit once the slot is taken."""

    def _queue(self, allocator):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = object.__new__(DecodePreallocQueue)
        queue.tree_cache = _empty_tree()
        queue.token_to_kv_pool_allocator = allocator
        queue.req_to_token_pool = SimpleNamespace(
            holds_state_slots=True,
            enable_mamba_extra_buffer=False,
            mamba_initial_tracking_slots=0,
        )
        return queue

    def _check(self, device):
        req = SimpleNamespace(kv=SimpleNamespace(holds_mamba=False))
        split_admits = 0
        for lazy in (False, True):
            for held_slots in range(8):
                for held_tokens in (0, 4, 8, 16):
                    for tokens in range(8, 80, 4):
                        setup_allocator_context()
                        allocator = _allocator(device, "tri", lazy=lazy)
                        if held_slots:
                            self.assertIsNotNone(
                                allocator.alloc_state_slots(held_slots)
                            )
                        if held_tokens:
                            self.assertIsNotNone(allocator.alloc(held_tokens))
                        queue = self._queue(allocator)
                        slots = queue._pending_state_slots(req)
                        self.assertEqual(slots, 1)
                        budgets = dict(
                            full_allocatable_tokens=10**9,
                            swa_allocatable_tokens=10**9,
                        )
                        alone = queue._prealloc_reservation_fits(
                            token_demand(allocator, tokens, tokens), **budgets
                        )
                        if not queue._prealloc_reservation_fits(
                            token_demand(allocator, tokens, tokens, state_slots=slots),
                            **budgets,
                        ):
                            split_admits += alone
                            continue
                        self.assertIsNotNone(allocator.alloc_state_slots(slots))
                        self.assertIsNotNone(allocator.alloc(tokens))
        # Requests the slot-alone and tokens-alone checks admitted, refused now.
        self.assertGreater(split_admits, 0)

    def test_cpu(self):
        self._check("cpu")


if __name__ == "__main__":
    unittest.main()
