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
"""KVIndexSource — the read-path id-space choke point.

Covers, CPU-only (the builder's pure-torch reference path; GPU parity of the
Triton kernel is a later CUDA CI pin):
  - strict passthrough: a non-unified source returns the SAME req_to_token /
    req_pool_indices objects — zero tensor ops, no copies (the property that
    makes backend re-pointing byte-identical for every non-unified server);
  - static SWA pools keep their legacy full->swa mapping on the view;
  - the canonical table matches the hand formula
        entry[b, c] = clamp(v2p[req_to_token[req[b], c*ps] // ps] * mult, 0)
    over the REAL SWA composite's tables (full AND swa, ps in {1, 4},
    multiplier in {1, 2L}), with the swa table built from VIRTUAL ids;
  - sink routing: dead lanes (seq_len 0), -1 req_to_token entries, and
    tombstoned v2p pages all read entry 0;
  - the capture contract: buffers are zero-filled and idempotent; a refresh
    updates ONLY the live prefix (stale tails and rows beyond bs keep prior
    contents); the returned table is the WHOLE buffer (pointer-stable).

    python -m pytest test/registered/unit/mem_cache/test_kv_index_source.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from test_multi_ended_allocator import _FakeUnifiedSWAKVPool

from sglang.srt.mem_cache.kv_index_source import KVIndexSource
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool

_DEV = "cpu"
_FULL_L = 2
_SWA_L = 3


def _build_composite(ps, full_mult=1, swa_mult=1, n_full_pages=16, n_swa_pages=8):
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=_FULL_L,
        head_num=2,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction="up",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=_SWA_L,
        head_num=2,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction="down",
    )
    n_full, n_swa = n_full_pages * ps, n_swa_pages * ps
    total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full_spec, swa_spec],
        device=_DEV,
        enable_memory_saver=False,
        page_size=ps,
    )
    kvcache = _FakeUnifiedSWAKVPool(pool)
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=kvcache,
        device=_DEV,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        page_size=ps,
        need_sort=False,
        forward_stream=None,
        full_kernel_page_multiplier=full_mult,
        swa_kernel_page_multiplier=swa_mult,
    )
    return allocator


class _FakeSwaTranslatePool:
    """Just the pool-level method KVIndexSource wires for translate_swa."""

    def __init__(self, allocator):
        self._allocator = allocator

    def translate_loc_from_full_to_swa(self, kv_indices, *, out=None):
        return self._allocator.translate_loc_from_full_to_swa(kv_indices, out=out)


def _make_source(allocator, req_to_token, ps):
    # The probe requires that the pool this runner reads/writes IS the one the
    # allocator's ids address, so hand the fake pool back from get_kvcache().
    pool = _FakeSwaTranslatePool(allocator)
    allocator.get_kvcache = lambda: pool
    return KVIndexSource(
        req_to_token=req_to_token,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool,
        page_size=ps,
        device=_DEV,
    )


def _reference_table(req_to_token, req_pool_indices, seq_lens, v2p, mult, ps, width):
    """Independent python derivation of the canonical formula."""
    bs = req_pool_indices.numel()
    out = torch.zeros((bs, width), dtype=torch.int32)
    for b in range(bs):
        req = int(req_pool_indices[b])
        n_pages = -(-int(seq_lens[b]) // ps)
        for c in range(min(n_pages, width)):
            tok = int(req_to_token[req, c * ps])
            page = 0 if tok < 0 else tok // ps
            out[b, c] = max(int(v2p[page]) * mult, 0)
    return out


class TestPassthrough(unittest.TestCase):
    def test_non_unified_returns_same_objects(self):
        """The strict-passthrough property: no copy, no branch, the exact
        tensors backends read today. A regression here (any tensor op on the
        non-unified path) breaks byte-identity for every static-pool server."""
        req_to_token = torch.arange(64, dtype=torch.int64).reshape(4, 16)
        src = KVIndexSource(
            req_to_token=req_to_token,
            token_to_kv_pool_allocator=SimpleNamespace(),  # not a composite
            token_to_kv_pool=SimpleNamespace(),  # not an SWAKVPool
            page_size=1,
            device=_DEV,
        )
        self.assertFalse(src.enabled)
        rows = torch.tensor([2, 0])
        view = src.batch_view(req_pool_indices=rows, seq_lens=torch.tensor([5, 3]))
        self.assertIs(view.table, req_to_token)
        self.assertIs(view.rows, rows)
        self.assertEqual(view.row_stride, req_to_token.stride(0))
        self.assertEqual(view.src_page_size, 1)
        self.assertFalse(view.kernel_facing)
        self.assertIsNone(view.swa_table)
        self.assertIsNone(view.full_to_swa_map)
        # And the translate surface is the identity, not a wrapped copy.
        t = torch.tensor([1, 2, 3])
        self.assertIs(src.translate_full(t), t)

    def test_static_swa_pool_carries_legacy_map(self):
        """Static SWA pools keep the in-kernel full->swa gather mechanism; the
        view must surface their mapping tensor untouched."""
        pool = SWAKVPool.__new__(SWAKVPool)
        mapping = torch.arange(10, dtype=torch.int64)
        pool.full_to_swa_index_mapping = mapping
        src = KVIndexSource(
            req_to_token=torch.zeros((2, 4), dtype=torch.int64),
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=pool,
            page_size=1,
            device=_DEV,
        )
        view = src.batch_view(
            req_pool_indices=torch.tensor([0]), seq_lens=torch.tensor([1])
        )
        self.assertIs(view.full_to_swa_map, mapping)


def _alloc_and_fill(allocator, ps, lens):
    """Allocate per-request virtual runs and write them into a fake
    req_to_token; returns (req_to_token, req_pool_indices, seq_lens)."""
    width = 16 * ps
    req_to_token = torch.full((len(lens), width), -1, dtype=torch.int64)
    for r, n in enumerate(lens):
        n_alloc = -(-n // ps) * ps  # page-aligned virtual run
        v = allocator.alloc(n_alloc)
        assert v is not None
        req_to_token[r, :n] = v[:n]
    return (
        req_to_token,
        torch.arange(len(lens), dtype=torch.int64),
        torch.tensor(lens, dtype=torch.int64),
    )


class TestCanonicalBuild(unittest.TestCase):

    def test_canonical_matches_reference_dense_and_strided(self):
        """The load-bearing formula pin: full AND swa canonical tables equal
        the independent per-element derivation, across page sizes and both
        multiplier regimes (strided=1, dense=2L). The swa table agreeing with
        a formula over VIRTUAL ids is also the never-chained-through-
        full-physical proof."""
        for ps in (1, 4):
            for full_mult, swa_mult in ((1, 1), (2 * _FULL_L, 2 * _SWA_L)):
                allocator = _build_composite(ps, full_mult=full_mult, swa_mult=swa_mult)
                req_to_token, rows, seq_lens = _alloc_and_fill(
                    allocator, ps, lens=[5 * ps, 2 * ps, 3 * ps - 1]
                )
                src = _make_source(allocator, req_to_token, ps)
                self.assertTrue(src.enabled)
                width = 6
                view = src.batch_view(
                    req_pool_indices=rows, seq_lens=seq_lens, max_pages=width
                )
                self.assertTrue(view.kernel_facing)
                self.assertEqual(view.src_page_size, ps)
                self.assertTrue(
                    torch.equal(view.rows, torch.arange(3, dtype=torch.int64))
                )
                want_full = _reference_table(
                    req_to_token,
                    rows,
                    seq_lens,
                    allocator.full_v2p_page_table,
                    full_mult,
                    ps,
                    width,
                )
                want_swa = _reference_table(
                    req_to_token,
                    rows,
                    seq_lens,
                    allocator.swa_v2p_page_table,
                    swa_mult,
                    ps,
                    width,
                )
                self.assertTrue(
                    torch.equal(view.table, want_full),
                    f"full canonical off-formula (ps={ps}, mult={full_mult})",
                )
                self.assertTrue(
                    torch.equal(view.swa_table, want_swa),
                    f"swa canonical off-formula (ps={ps}, mult={swa_mult})",
                )

    def test_sink_routing(self):
        """Dead lanes (seq_len 0), -1 slots inside the live prefix, and
        tombstoned v2p pages must ALL read entry 0 — one wild entry is a
        captured-graph OOB read at replay."""
        ps = 4
        allocator = _build_composite(ps, full_mult=2 * _FULL_L, swa_mult=2 * _SWA_L)
        req_to_token, rows, seq_lens = _alloc_and_fill(
            allocator, ps, lens=[3 * ps, 2 * ps, ps]
        )
        seq_lens[1] = 0  # dead lane
        req_to_token[0, ps] = -1  # unwritten slot inside the live prefix
        # Tombstone row 2's first page on BOTH sides.
        tomb_page = int(req_to_token[2, 0]) // ps
        allocator.full_v2p_page_table[tomb_page] = -1
        allocator.swa_v2p_page_table[tomb_page] = -1
        src = _make_source(allocator, req_to_token, ps)
        view = src.batch_view(req_pool_indices=rows, seq_lens=seq_lens, max_pages=4)
        for table in (view.table, view.swa_table):
            self.assertTrue(bool((table >= 0).all()))
            self.assertTrue(bool((table[1] == 0).all()), "dead lane not sunk")
            self.assertEqual(int(table[0, 1]), 0, "-1 slot not sunk")
            self.assertEqual(int(table[2, 0]), 0, "tombstone not sunk")


class TestVerifyWidenedLengths(unittest.TestCase):
    """Target-verify seam pin. The canonical builder carries no mode
    knowledge: `seq_lens` is a caller-supplied tensor precisely so the
    TARGET_VERIFY metadata builder can pass `seq_lens + draft_len` and get a
    table whose rows cover the draft slots. Pinned here because nothing
    upstream covers verify shapes on the unified pool (the eager triton
    verify branch shipped without any translate at all), and building with
    UN-widened lengths under-covers by exactly the draft pages — the
    historical parked gap this seam closes."""

    def test_widened_lengths_cover_draft_pages(self):
        for ps in (1, 4):
            draft = ps + 1  # crosses a page boundary when ps > 1
            allocator = _build_composite(ps)
            committed = [4 * ps, 2 * ps]
            total = [c + draft for c in committed]
            req_to_token, rows, _ = _alloc_and_fill(allocator, ps, lens=total)
            src = _make_source(allocator, req_to_token, ps)
            widened = torch.tensor(committed, dtype=torch.int64) + draft
            width = -(-max(total) // ps)
            view = src.batch_view(
                req_pool_indices=rows, seq_lens=widened, max_pages=width
            )
            for table, v2p, tag in (
                (view.table, allocator.full_v2p_page_table, "full"),
                (view.swa_table, allocator.swa_v2p_page_table, "swa"),
            ):
                want = _reference_table(req_to_token, rows, widened, v2p, 1, ps, width)
                self.assertTrue(
                    torch.equal(table, want),
                    f"widened {tag} table off-formula (ps={ps})",
                )
            # The draft rows are genuinely load-bearing: an un-widened build
            # leaves their pages unbuilt (the sink), so a verify consumer fed
            # the narrow table reads page 0 for every draft slot.
            narrow = src.batch_view(
                req_pool_indices=rows,
                seq_lens=torch.tensor(committed, dtype=torch.int64),
                max_pages=width,
            )
            self.assertFalse(torch.equal(narrow.table, view.table))
            last_col = -(-total[0] // ps) - 1
            self.assertEqual(int(narrow.table[0, last_col]), 0)
            self.assertNotEqual(int(view.table[0, last_col]), 0)


class TestCaptureContract(unittest.TestCase):
    def test_capture_buffers_zero_filled_idempotent_and_prefix_only(self):
        ps = 4
        allocator = _build_composite(ps, full_mult=2 * _FULL_L, swa_mult=2 * _SWA_L)
        req_to_token = torch.full((4, 16 * ps), -1, dtype=torch.int64)
        v = allocator.alloc(2 * ps)
        req_to_token[1, : 2 * ps] = v
        src = _make_source(allocator, req_to_token, ps)

        src.ensure_capture_buffers(max_bs=4, max_context_len=8 * ps)
        cap = src._cap_full
        self.assertTrue(bool((cap == 0).all()), "capture buffers must start zeroed")
        src.ensure_capture_buffers(max_bs=4, max_context_len=8 * ps)
        self.assertIs(src._cap_full, cap, "ensure_capture_buffers must be idempotent")

        # Poison everything, then refresh a 1-row batch: ONLY its live prefix
        # may change — stale tails and other rows are the fa3 contract.
        cap.fill_(7)
        src._cap_swa.fill_(7)
        view = src.batch_view(
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([2 * ps]),
            captured=True,
        )
        self.assertIs(view.table, cap, "captured view must return the WHOLE buffer")
        want = allocator.full_v2p_page_table[req_to_token[1, ::ps][:2] // ps] * (
            2 * _FULL_L
        )
        self.assertTrue(torch.equal(cap[0, :2], want.to(torch.int32)))
        self.assertTrue(bool((cap[0, 2:] == 7).all()), "stale tail was cleared")
        self.assertTrue(bool((cap[1:] == 7).all()), "rows beyond bs were touched")

    def test_rows_pointer_stable_after_capture_init(self):
        allocator = _build_composite(1)
        src = _make_source(allocator, torch.zeros((2, 8), dtype=torch.int64), 1)
        src.ensure_capture_buffers(max_bs=8, max_context_len=8)
        r1 = src._rows_for(3)
        r2 = src._rows_for(5)
        self.assertEqual(r1.data_ptr(), r2.data_ptr())
        self.assertTrue(torch.equal(r2, torch.arange(5, dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
