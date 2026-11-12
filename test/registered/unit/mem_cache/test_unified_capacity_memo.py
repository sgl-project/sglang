# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Epoch-memoized capacity views on the allocator chain.

The per-band and composite capacity views are pure functions of a handful of
CPU-resident fields that schedulers read O(queue) times between mutations;
`_CapacityField` descriptors bump `_capacity_epoch` on every rebind, so the
memos invalidate by construction.

The guarded failure mode is a memo serving a STALE value after a mutation the
epoch machinery missed -- a new mutation site writing an uncovered field, or
an in-place write that bypasses `__set__`. Readers cannot detect either one;
stale capacity is silent over-/under-admission, not a crash.
"""

import random
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.token_demand import demand_windows

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _build(lazy: bool):
    from sglang.test.unified_allocator_fixtures import setup_allocator_context

    setup_allocator_context()
    # Function-scope import: the fixture is a TestCase subclass, and a
    # module-scope binding would make pytest collect its tests AGAIN here.
    from test_multi_ended_allocator import TestWindowChainAllocator as _SwaFixture

    inst = _SwaFixture([m for m in dir(_SwaFixture) if m.startswith("test_")][0])
    pool, allocator, kvcache = inst._build()
    allocator.full_attn_allocator.lazy_compaction = lazy
    allocator.swa_attn_allocator.lazy_compaction = lazy
    allocator.lazy_compaction = lazy
    return inst, allocator, kvcache


def _assert_read_fresh(test, allocator):
    """The memoized capacity reads must equal the same reads recomputed with
    every memo dropped (the snapshot and the largest-fit memos)."""
    names = [b.sub_pool_name for b in allocator.bands]
    memoized = (
        allocator.available_size(),
        [allocator.units_available(n) for n in names],
    )
    allocator._fit_memo.clear()
    allocator._snapshot_key = None
    fresh = (
        allocator.available_size(),
        [allocator.units_available(n) for n in names],
    )
    test.assertEqual(memoized, fresh)


class TestCapacityMemoCoherence(unittest.TestCase):
    def _assert_memos_fresh(self, allocator):
        """Every memoized capacity view must equal a fresh recompute."""
        _assert_read_fresh(self, allocator)
        for band in (
            allocator.full_attn_allocator,
            allocator.swa_attn_allocator,
        ):
            self.assertEqual(
                band.available_size(),
                band._available_tokens(),
                msg=f"stale available_size memo on {band.sub_pool_name!r}",
            )
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_memos_track_every_mutation_kind(self):
        for lazy in (True,):
            with self.subTest(lazy_compaction=lazy):
                inst, allocator, kvcache = _build(lazy)
                self._assert_memos_fresh(allocator)

                v1 = inst._alloc(allocator, kvcache, 8)  # both-side bind
                self.assertIsNotNone(v1)
                self._assert_memos_fresh(allocator)

                allocator.free_swa(v1[2:6])  # swa-side tombstones
                self._assert_memos_fresh(allocator)

                inst._free(allocator, kvcache, v1)  # both-side free
                self._assert_memos_fresh(allocator)

                v2 = inst._alloc(allocator, kvcache, 4)
                self.assertIsNotNone(v2)
                allocator.free_group_begin()  # grouped free path
                allocator.free(v2)
                allocator.free_group_end()
                self._assert_memos_fresh(allocator)

                if lazy:
                    allocator.full_attn_allocator._flush(urgent=True)
                    self._assert_memos_fresh(allocator)

                allocator.clear()
                self._assert_memos_fresh(allocator)

    def test_random_op_sequence_value_identity(self):
        """Property: after ANY mutation sequence the memoized views equal
        fresh recomputes. Seeded, so the interleavings are deterministic."""
        rng = random.Random(0xC0FFEE)
        for lazy in (True,):
            with self.subTest(lazy_compaction=lazy):
                inst, allocator, kvcache = _build(lazy)
                live = []
                for step in range(60):
                    op = rng.choice(("alloc", "free", "free_swa", "flush", "clear"))
                    if op == "alloc":
                        v = inst._alloc(allocator, kvcache, rng.choice((1, 2, 4)))
                        if v is not None:
                            live.append(v)
                    elif op == "free" and live:
                        inst._free(allocator, kvcache, live.pop())
                    elif op == "free_swa" and live:
                        v = live[-1]
                        if v.numel() > 1:
                            allocator.free_swa(v[: v.numel() // 2])
                    elif op == "flush":
                        allocator.full_attn_allocator._flush(urgent=True)
                        allocator.swa_attn_allocator._flush(urgent=True)
                    elif op == "clear":
                        allocator.clear()
                        live.clear()
                    self._assert_memos_fresh(allocator)

    def test_bypassing_write_is_caught_by_the_idle_check(self):
        inst, allocator, kvcache = _build(lazy=True)
        v = inst._alloc(allocator, kvcache, 8)
        self.assertIsNotNone(v)
        fa = allocator.full_attn_allocator
        # Prime every memo at the current epoch.
        allocator.available_size()
        fa.available_size()
        # Mutate capacity state BYPASSING the _CapacityField descriptor -- the
        # epoch does not move, so the memos go stale undetectably for readers...
        fa.__dict__["watermark_physical"] = fa.watermark_physical + 2
        # ...but the idle-time coherence check must flag it.
        violations = allocator.verify_byte_accounting()
        self.assertTrue(
            any("stale" in msg for msg in violations),
            msg=f"bypassing write not caught: {violations}",
        )

    def test_float_only_span_move_invalidates_every_memo(self):
        """A hole-free float alloc rebinds NO free-list and has no watermark:
        the span fields are its ONLY capacity state, so unless they are
        `_CapacityField` descriptors the float's own memo AND both neighbours'
        (the span flips transparency, walling off their gaps) keep serving
        pre-move values. Exercised on a hand-wired end+float+end chain so no
        end-pool descriptor write can mask a missing span bump."""
        from test_multi_ended_allocator import TestFloatBand

        inst = TestFloatBand(
            [m for m in dir(TestFloatBand) if m.startswith("test_")][0]
        )
        _pool, sa, fla, da, _kv = inst._build_tri()
        self.assertEqual(fla._hole_pages(), 0)  # hole-free extension path
        self.assertTrue(fla._is_frontier_transparent())

        # Prime every memo while the float is empty/transparent.
        float_cached = fla.available_size()
        low_end_cached = sa.available_size()
        high_end_cached = da.available_size()

        v = fla.alloc(4)  # float-only mutation: span move, no end-pool write
        self.assertIsNotNone(v)
        self.assertFalse(fla._is_frontier_transparent())  # span now opaque

        self.assertEqual(fla.available_size(), fla._available_tokens())
        self.assertEqual(sa.available_size(), sa._available_tokens())
        self.assertEqual(da.available_size(), da._available_tokens())
        # The opaque midpoint span must actually reduce what the neighbours
        # see, i.e. the memos above were not merely re-serving primed values.
        self.assertLess(sa.available_size(), low_end_cached)
        self.assertLess(da.available_size(), high_end_cached)
        self.assertLessEqual(fla.available_size(), float_cached)


class TestTriCapacityMemoCoherence(CustomTestCase):
    """Tri-composite twins of the 2-pool cases: the joint view walks THREE
    bands (mamba end, swa float, full end), so a mutation on ANY of them must
    invalidate the composite memo -- including the two only the tri has, a
    mamba-end state draw and a float span move behind the composite."""

    def _build_tri(self, lazy=True):
        from sglang.test.unified_allocator_fixtures import setup_allocator_context

        setup_allocator_context()
        from test_unified_tri_pool import TestUnifiedTriPool

        inst = TestUnifiedTriPool(
            [m for m in dir(TestUnifiedTriPool) if m.startswith("test_")][0]
        )
        pool, allocator, kvcache, mamba_kv = inst._build(lazy_compaction=lazy)
        return inst, allocator

    def _assert_memos_fresh(self, allocator):
        _assert_read_fresh(self, allocator)
        for band in (
            allocator.full_attn_allocator,
            allocator.swa_attn_allocator,
            allocator.mamba_allocator,
        ):
            self.assertEqual(
                band.available_size(),
                band._available_tokens(),
                msg=f"stale available_size memo on {band.sub_pool_name!r}",
            )
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_joint_capacity_boundary_after_fragmentation(self):
        """Equal FULL/SWA demand must fit exactly at the advertised boundary.

        State allocations move a differently aligned peer, and SWA retirement
        creates reusable holes. The cached joint capacity must still fund an
        actual reservation, while the next page must exceed it.
        """
        from test_unified_tri_pool import TestUnifiedTriPool

        fixture = TestUnifiedTriPool("test_chain_wiring_and_float_swa")
        for page_size in (1, 4, 128):
            for lazy in (True,):
                with self.subTest(page_size=page_size, lazy=lazy):
                    _, allocator, _, _ = fixture._build(
                        n_full=256 * page_size,
                        n_swa=128 * page_size,
                        n_state=32,
                        page_size=page_size,
                        lazy_compaction=lazy,
                    )

                    def check_boundary():
                        capacity = allocator.available_size()
                        self.assertTrue(
                            allocator.admits(
                                KVDemand(
                                    capacity,
                                    demand_windows(allocator, capacity, capacity),
                                )
                            )
                        )
                        # Admission may see room past the advertised capacity
                        # (a relayout opens it); whatever it admits, the take
                        # serves.
                        more = capacity + page_size
                        if allocator.admits(
                            KVDemand(more, demand_windows(allocator, more, more))
                        ):
                            taken = allocator.alloc(more)
                            self.assertIsNotNone(taken)
                            allocator.free(taken)

                    check_boundary()  # empty float is frontier-transparent
                    states = allocator.mamba_allocator.alloc(3)
                    self.assertIsNotNone(states)
                    check_boundary()
                    tokens = allocator.alloc(8 * page_size)
                    self.assertIsNotNone(tokens)
                    check_boundary()  # float now splits the available band
                    allocator.free_swa(tokens[2 * page_size : 5 * page_size])
                    check_boundary()  # interior holes are reusable
                    allocator.free(tokens)
                    check_boundary()  # float parks again
                    allocator.mamba_allocator.free(states)
                    check_boundary()

    def test_memos_track_tri_mutation_kinds(self):
        for lazy in (True,):
            with self.subTest(lazy_compaction=lazy):
                inst, allocator = self._build_tri(lazy)
                ma = allocator.mamba_allocator
                self._assert_memos_fresh(allocator)

                v1 = allocator.alloc(8)  # composite alloc (full + swa bind)
                self.assertIsNotNone(v1)
                self._assert_memos_fresh(allocator)

                s1 = ma.alloc(2)  # mamba end alloc
                self.assertIsNotNone(s1)
                self._assert_memos_fresh(allocator)

                allocator.free_swa(v1[2:6])  # interior float holes
                self._assert_memos_fresh(allocator)

                allocator.free(v1)  # both-side free
                self._assert_memos_fresh(allocator)

                ma.free(s1)  # mamba free
                self._assert_memos_fresh(allocator)

                allocator.clear()
                ma.clear()
                self._assert_memos_fresh(allocator)


# The paging kernels are Triton; neither the epoch nor the memos depend on the
# ids they write, so they are stood in for on CPU.
_PAGING_KERNELS = "sglang.srt.mem_cache.kv.alloc.chain"


class _NoOpKernel:
    def __getitem__(self, grid):
        return self

    def __call__(self, *args, **kwargs):
        pass


@contextmanager
def _stub_paging_kernels():
    with (
        patch(f"{_PAGING_KERNELS}.alloc_extend_kernel", _NoOpKernel()),
        patch(f"{_PAGING_KERNELS}.alloc_decode_kernel", _NoOpKernel()),
    ):
        yield


def _ids(*values):
    return torch.tensor(values, dtype=torch.int64)


class TestPagedStepKeepsTheEpoch(unittest.TestCase):
    def test_a_decode_within_its_pages_keeps_the_capacity_memos(self):
        """BUG REGRESSION. A paged decode that took no new page rebound the
        owner's free virtual ids anyway, which moves the capacity epoch: every
        capacity memo of the chain was dropped and recomputed at each such
        step. A decode within its pages keeps the epoch, and the next capacity
        query is served from the memos; a decode that opens a page moves it."""
        from sglang.test.unified_allocator_fixtures import (
            build_swa_pool,
            setup_allocator_context,
        )

        setup_allocator_context()
        page = 4
        allocator = build_swa_pool(page_size=page)
        owner = allocator.full_attn_allocator
        with _stub_paging_kernels():
            # One request of 2 tokens: its first page holds 2 of 4.
            out = allocator.alloc_extend(
                _ids(0), _ids(0), _ids(2), _ids(2), _ids(-1), 2
            )
            self.assertIsNotNone(out)
            allocator.available_size()
            epoch = owner._chain_capacity_epoch()
            for seq_len in (3, 4):
                out = allocator.alloc_decode(_ids(seq_len), _ids(seq_len), _ids(0))
                self.assertIsNotNone(out)
                self.assertEqual(owner._chain_capacity_epoch(), epoch)
            with (
                patch.object(
                    allocator, "units_fit", wraps=allocator.units_fit
                ) as joint,
                patch.object(
                    owner, "_available_tokens", wraps=owner._available_tokens
                ) as own,
            ):
                allocator.available_size()
                owner.available_size()
            joint.assert_not_called()
            own.assert_not_called()
            # The fifth token opens the request's second page.
            out = allocator.alloc_decode(_ids(5), _ids(5), _ids(0))
            self.assertIsNotNone(out)
            self.assertGreater(owner._chain_capacity_epoch(), epoch)
        self.assertEqual(allocator.verify_byte_accounting(), [])


class TestOneDemandShapeOneSearch(CustomTestCase):
    """Regression: on a chain without a window, `available_size` and
    `full_available_size` share one memo entry, so a read runs one search."""

    def test_the_two_owner_reads_share_one_search(self):
        from sglang.test.unified_allocator_fixtures import (
            UnifiedGeometry,
            build_unified_bundle,
            setup_allocator_context,
        )

        setup_allocator_context()
        allocator = build_unified_bundle(
            UnifiedGeometry(
                chain="full",
                full_layer_ids=(0,),
                full_tokens=256,
                context_len=256,
            )
        ).token_to_kv_pool_allocator
        calls = []
        real = type(allocator).units_fit

        def counting(alloc, demands):
            calls.append(dict(demands))
            return real(alloc, demands)

        with patch.object(type(allocator), "units_fit", counting):
            for _ in range(3):
                allocator.free(allocator.alloc(1))  # moves the epoch
                calls.clear()
                tokens = allocator.available_size()
                searched = len(calls)
                self.assertGreater(searched, 0)
                self.assertEqual(allocator.full_available_size(), tokens)
                self.assertEqual(len(calls), searched)


class TestEndsOnlyCapacityArithmetic(CustomTestCase):
    """On a chain with no float, capacity reads are answered by arithmetic that
    equals the layout search on random chains, and most reads skip the search."""

    def _chain(self, chain):
        from sglang.test.unified_allocator_fixtures import (
            UnifiedGeometry,
            build_unified_bundle,
            setup_allocator_context,
        )

        setup_allocator_context()
        geometry = dict(full_layer_ids=(0,), full_tokens=256, context_len=256)
        if chain == "mamba":
            geometry.update(state_layer_ids=(1,), state_slots=16)
        return build_unified_bundle(
            UnifiedGeometry(chain=chain, page_size=4, **geometry)
        ).token_to_kv_pool_allocator

    def _searched(self, allocator, name):
        allocator._fit_memo.clear()
        with patch.object(type(allocator), "_ends_only_fit", lambda *a: None):
            return allocator.units_available(name)

    def test_the_arithmetic_equals_the_search(self):
        rng = random.Random(20261107)
        answered = asked = 0
        real = None
        for chain in ("full", "mamba"):
            for _ in range(40):
                allocator = self._chain(chain)
                real = type(allocator)._ends_only_fit
                owner = allocator.full_attn_allocator
                state = allocator.mamba_allocator
                held, slots = [], []
                for _ in range(rng.randint(4, 30)):
                    roll = rng.random()
                    if roll < 0.45:
                        ids = allocator.alloc(4 * rng.randint(1, 4))
                        if ids is not None:
                            held.append(ids)
                    elif roll < 0.7 and held:
                        allocator.free(held.pop(rng.randrange(len(held))))
                    elif roll < 0.8 and state is not None:
                        slot = allocator.alloc_state_slots(1)
                        if slot is not None:
                            slots.append(slot)
                    elif roll < 0.85 and slots:
                        state.free(slots.pop(rng.randrange(len(slots))))
                    elif roll < 0.92:
                        # Frees under an unprocessed forward stay pending.
                        steps = allocator.forward_steps
                        steps.forward_launched()
                        steps.forward_recorded(None)
                        if held:
                            allocator.free(held.pop())
                    elif held:
                        page = int(owner.ids.virtual_to_physical[int(held[0][0]) // 4])
                        owner.fix_pages(("test", page), [page])
                names = [owner.sub_pool_name]
                if state is not None:
                    names.append(state.sub_pool_name)
                for name in names:
                    with self.subTest(chain=chain, name=name):
                        allocator._fit_memo.clear()
                        calls = []

                        def counting(alloc, *args, _calls=calls):
                            out = real(alloc, *args)
                            _calls.append(out is not None)
                            return out

                        with patch.object(type(allocator), "_ends_only_fit", counting):
                            fast = allocator.units_available(name)
                        asked += 1
                        answered += bool(calls and calls[0])
                        self.assertEqual(fast, self._searched(allocator, name))
        self.assertGreater(answered, asked // 2)

    def test_joint_reads_of_two_ends_equal_the_search(self):
        """A FULL + window chain whose two bands are ends: the joint read
        (each band takes every token) by arithmetic equals the search."""
        from sglang.test.unified_allocator_fixtures import (
            UnifiedGeometry,
            build_unified_bundle,
            setup_allocator_context,
        )

        rng = random.Random(20261108)
        answered = asked = 0
        for _ in range(40):
            setup_allocator_context()
            allocator = build_unified_bundle(
                UnifiedGeometry(
                    chain="swa",
                    full_layer_ids=(0,),
                    swa_layer_ids=(1,),
                    page_size=4,
                    full_tokens=256,
                    swa_tokens=256,
                    context_len=256,
                )
            ).token_to_kv_pool_allocator
            if any(b.grow_direction == "float" for b in allocator.bands):
                self.skipTest("this chain's window floats")
            real = type(allocator)._ends_only_fit
            window = allocator.swa_attn_allocator
            held = []
            for _ in range(rng.randint(4, 30)):
                roll = rng.random()
                if roll < 0.5:
                    ids = allocator.alloc(4 * rng.randint(1, 4))
                    if ids is not None:
                        held.append(ids)
                elif roll < 0.75 and held:
                    allocator.free(held.pop(rng.randrange(len(held))))
                elif roll < 0.9 and held:
                    allocator.free_window(window.sub_pool_name, held[-1])
                else:
                    steps = allocator.forward_steps
                    steps.forward_launched()
                    steps.forward_recorded(None)
                    if held:
                        allocator.free(held.pop())
            allocator._fit_memo.clear()
            calls = []

            def counting(alloc, *args, _calls=calls):
                out = real(alloc, *args)
                _calls.append(out is not None)
                return out

            with patch.object(type(allocator), "_ends_only_fit", counting):
                fast = allocator.available_size()
            asked += 1
            answered += bool(calls and calls[0])
            allocator._fit_memo.clear()
            with patch.object(type(allocator), "_ends_only_fit", lambda *a: None):
                searched = allocator.available_size()
            self.assertEqual(fast, searched)
        self.assertGreater(answered, asked // 2)


if __name__ == "__main__":
    unittest.main()
