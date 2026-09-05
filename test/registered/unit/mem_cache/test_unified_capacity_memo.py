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

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _build(lazy: bool):
    # Function-scope import: the fixture is a TestCase subclass, and a
    # module-scope binding would make pytest collect its tests AGAIN here.
    from test_multi_ended_allocator import (
        TestUnifiedSWATokenToKVPoolAllocator as _SwaFixture,
    )

    inst = _SwaFixture([m for m in dir(_SwaFixture) if m.startswith("test_")][0])
    pool, allocator, kvcache = inst._build()
    allocator.full_attn_allocator.lazy_compaction = lazy
    allocator.swa_attn_allocator.lazy_compaction = lazy
    allocator.lazy_compaction = lazy
    return inst, allocator, kvcache


class TestCapacityMemoCoherence(unittest.TestCase):
    def _assert_memos_fresh(self, allocator):
        """Every memoized capacity view must equal a fresh recompute."""
        self.assertEqual(
            allocator.available_size(), allocator._compute_available_size()
        )
        for band in (
            allocator.full_attn_allocator,
            allocator.swa_attn_allocator,
        ):
            self.assertEqual(
                band.available_size(),
                band._available_tokens(),
                msg=f"stale available_size memo on {band.sub_pool_name!r}",
            )
            self.assertEqual(
                band.schedulable_available_size(),
                band._available_tokens(
                    extra_gap_bytes=band._peer_drainable_hole_bytes()
                ),
                msg=f"stale schedulable memo on {band.sub_pool_name!r}",
            )
        self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_memos_track_every_mutation_kind(self):
        for lazy in (False, True):
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
        for lazy in (False, True):
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
        inst, allocator, kvcache = _build(lazy=False)
        v = inst._alloc(allocator, kvcache, 8)
        self.assertIsNotNone(v)
        fa = allocator.full_attn_allocator
        # Prime every memo at the current epoch.
        allocator.available_size()
        fa.available_size()
        fa.schedulable_available_size()
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
        from test_multi_ended_allocator import TestFloatMultiEndedAllocator

        inst = TestFloatMultiEndedAllocator(
            [m for m in dir(TestFloatMultiEndedAllocator) if m.startswith("test_")][0]
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


class TestTriCapacityMemoCoherence(unittest.TestCase):
    """Tri-composite twins of the 2-pool cases: the joint view walks THREE
    bands (mamba end, swa float, full end), so a mutation on ANY of them must
    invalidate the composite memo -- including the two only the tri has, a
    mamba-end state draw and a float span move behind the composite."""

    def _build_tri(self, lazy=False):
        from test_unified_tri_pool import TestUnifiedTriPool

        inst = TestUnifiedTriPool(
            [m for m in dir(TestUnifiedTriPool) if m.startswith("test_")][0]
        )
        pool, allocator, kvcache, mamba_kv = inst._build(lazy_compaction=lazy)
        return inst, allocator

    def _assert_memos_fresh(self, allocator):
        self.assertEqual(
            allocator.available_size(), allocator._compute_available_size()
        )
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

    def test_memos_track_tri_mutation_kinds(self):
        for lazy in (False, True):
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


if __name__ == "__main__":
    unittest.main()
