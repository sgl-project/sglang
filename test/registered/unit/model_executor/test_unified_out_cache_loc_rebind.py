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
"""Unified-pool kernel-facing write-loc contract.

`forward_batch.out_cache_loc` is REBOUND to a fresh kernel-facing tensor
exactly once, at ForwardBatch preparation (`apply_unified_kv_loc_rebind`), so
that every downstream consumer — attention backends, model-side pool doors —
sees kernel-facing ids without translating again.

Pinned here:
  1. the rebind itself: a FRESH tensor (the ScheduleBatch's aliased tensor
     stays VIRTUAL for scheduler-thread readers), the contract flag and the
     read-rail callable set, and the ORDER-CRITICAL hybrid-SWA rule (one
     virtual id -> TWO kernel-facing ids: the swa rail must be derived from
     the still-virtual loc BEFORE the full-side rebind replaces it);
  2. `resolve_swa_write_loc`, the single place unified and static SWA pools
     diverge on "what is the swa write loc for this batch".

    python -m pytest test/registered/unit/model_executor/test_unified_out_cache_loc_rebind.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    apply_unified_kv_loc_rebind,
    resolve_swa_write_loc,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DEV = "cpu"


def _make_fb(out_cache_loc, **kw):
    """Minimal ForwardBatch with only the required core fields."""
    n = 0 if out_cache_loc is None else out_cache_loc.shape[0]
    defaults = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=max(n, 1),
        input_ids=torch.zeros(max(n, 1), dtype=torch.int64),
        req_pool_indices=torch.zeros(max(n, 1), dtype=torch.int64),
        seq_lens=torch.ones(max(n, 1), dtype=torch.int64),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=max(n, 1),
    )
    defaults.update(kw)
    return ForwardBatch(**defaults)


def _make_runner(v2p=None, swa_map=None):
    """Fake ModelRunner whose allocator/pool are REAL unified classes (the
    rebind narrows by isinstance), with only the translate methods stubbed.

    `v2p=None` builds a non-unified runner, so the rebind must no-op.
    """
    if v2p is None:
        return SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
        )
    cls = (
        UnifiedSWATokenToKVPoolAllocator
        if swa_map is not None
        else UnifiedMambaTokenToKVPoolAllocator
    )
    allocator = object.__new__(cls)
    # Fresh-tensor gather, like the real translate_kv_loc_dense.
    allocator.translate_kv_loc_dense = lambda t: v2p[t.to(torch.int64)]
    if swa_map is not None:
        allocator.translate_loc_from_full_to_swa = swa_map
    pool = (
        object.__new__(UnifiedSWAKVPool) if swa_map is not None else SimpleNamespace()
    )
    return SimpleNamespace(token_to_kv_pool_allocator=allocator, token_to_kv_pool=pool)


class TestApplyUnifiedKvLocRebind(CustomTestCase):
    def test_rebind_is_fresh_and_sets_contract(self):
        v2p = torch.arange(100, dtype=torch.int64) + 1000  # virtual v -> v+1000
        virtual = torch.tensor([3, 7, 42], dtype=torch.int64)
        virtual_copy = virtual.clone()
        fb = _make_fb(virtual)

        apply_unified_kv_loc_rebind(fb, _make_runner(v2p=v2p))

        # Fresh kernel-facing tensor; the original (ScheduleBatch-aliased)
        # tensor object and content are untouched.
        self.assertIsNot(fb.out_cache_loc, virtual)
        self.assertTrue(torch.equal(fb.out_cache_loc, virtual + 1000))
        self.assertTrue(torch.equal(virtual, virtual_copy))
        self.assertTrue(fb.out_cache_loc_is_physical)
        self.assertIsNotNone(fb._unified_kv_loc_translate)
        # No SWA pool on this runner -> no swa rail.
        self.assertIsNone(fb.swa_out_cache_loc)

    def test_swa_rail_derived_from_virtual_before_rebind(self):
        v2p = torch.arange(100, dtype=torch.int64) * 2  # full: v -> 2v
        swa_inputs = []

        def swa_map(t):
            swa_inputs.append(t.clone())
            return t * 3  # swa: v -> 3v

        virtual = torch.tensor([1, 5, 9], dtype=torch.int64)
        fb = _make_fb(virtual)

        apply_unified_kv_loc_rebind(fb, _make_runner(v2p=v2p, swa_map=swa_map))

        # ORDER-CRITICAL: the swa map must have received the VIRTUAL ids, not
        # the full-side result of the rebind.
        self.assertEqual(len(swa_inputs), 1)
        self.assertTrue(torch.equal(swa_inputs[0], virtual))
        self.assertTrue(torch.equal(fb.swa_out_cache_loc, virtual * 3))
        self.assertEqual(fb.swa_out_cache_loc.dtype, torch.int64)
        self.assertTrue(torch.equal(fb.out_cache_loc, virtual * 2))

    def test_non_unified_is_a_noop(self):
        virtual = torch.tensor([2, 4], dtype=torch.int64)
        fb = _make_fb(virtual)

        apply_unified_kv_loc_rebind(fb, _make_runner())

        self.assertIs(fb.out_cache_loc, virtual)  # identity alias preserved
        self.assertFalse(fb.out_cache_loc_is_physical)
        self.assertIsNone(fb.swa_out_cache_loc)
        self.assertIsNone(fb._unified_kv_loc_translate)

    def test_none_loc_is_a_noop(self):
        fb = _make_fb(None)
        apply_unified_kv_loc_rebind(
            fb, _make_runner(v2p=torch.arange(10, dtype=torch.int64))
        )
        self.assertIsNone(fb.out_cache_loc)
        self.assertFalse(fb.out_cache_loc_is_physical)

    def test_empty_loc_rebinds_and_flags(self):
        # Idle / DP-idle batches: an empty tensor still gets the contract flag,
        # so backend tripwires see a consistent state.
        fb = _make_fb(torch.empty(0, dtype=torch.int64))
        fb.batch_size = 0
        apply_unified_kv_loc_rebind(
            fb, _make_runner(v2p=torch.arange(10, dtype=torch.int64))
        )
        self.assertTrue(fb.out_cache_loc_is_physical)
        self.assertEqual(fb.out_cache_loc.numel(), 0)


class TestResolveSwaWriteLoc(CustomTestCase):
    """The one place the unified and static SWA pools diverge."""

    def test_unified_returns_the_rail(self):
        fb = _make_fb(torch.tensor([10, 20], dtype=torch.int64))
        fb.out_cache_loc_is_physical = True
        fb.swa_out_cache_loc = torch.tensor([1, 2], dtype=torch.int64)
        pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda t: self.fail("must not translate")
        )
        self.assertIs(resolve_swa_write_loc(fb, pool), fb.swa_out_cache_loc)

    def test_unified_without_rail_fails_loud(self):
        fb = _make_fb(torch.tensor([10, 20], dtype=torch.int64))
        fb.out_cache_loc_is_physical = True
        with self.assertRaises(AssertionError):
            resolve_swa_write_loc(fb, SimpleNamespace())

    def test_static_pool_uses_the_static_map(self):
        physical = torch.tensor([10, 20], dtype=torch.int64)
        fb = _make_fb(physical)
        pool = SimpleNamespace(translate_loc_from_full_to_swa=lambda t: t // 10)
        out = resolve_swa_write_loc(fb, pool)
        self.assertTrue(torch.equal(out, torch.tensor([1, 2], dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
