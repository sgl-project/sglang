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
"""KVIndexTranslator -- the read-path id translator.

CPU-only: these exercise the builder's pure-torch reference path, not the
Triton kernel.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from test_multi_ended_allocator import _FakeUnifiedSWAKVPool

from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator, KVReadTables
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool

_DEV = "cpu"
_FULL_L = 2
_SWA_L = 3


def _build_composite(ps, collapse=False, n_full_pages=16, n_swa_pages=8):
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
    )
    if collapse:
        # The multiplier-1 arm, where kernel-facing ids ARE the physical ones.
        # No unified sub-pool reports 1 today, so pin the regime here.
        allocator.full_attn_allocator.kernel_page_multiplier = 1
        allocator.swa_attn_allocator.kernel_page_multiplier = 1
    # The fake IS the runner's token_to_kv_pool, and the real UnifiedSWAKVPool
    # carries the pool-level full->swa translate, so the fake must too.
    kvcache.translate_loc_from_full_to_swa = allocator.translate_loc_from_full_to_swa
    return allocator


def _make_source(allocator, req_to_token, ps):
    """The owning runner's source: its token_to_kv_pool IS the allocator's own
    kvcache. (A runner can share the allocator while owning a different pool --
    see TestPoolOwnership.)"""
    return KVIndexTranslator(
        req_to_token=req_to_token,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=allocator.get_kvcache(),
        page_size=ps,
        device=_DEV,
    )


def _reference_table(req_to_token, req_pool_indices, seq_lens, v2p, mult, ps, width):
    """Independent python derivation of the read-table formula."""
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
        """Strict passthrough: no copy, no branch. Any tensor op on the
        non-unified path breaks byte-identity for every static-pool server."""
        req_to_token = torch.arange(64, dtype=torch.int64).reshape(4, 16)
        src = KVIndexTranslator(
            req_to_token=req_to_token,
            token_to_kv_pool_allocator=SimpleNamespace(),  # not a composite
            token_to_kv_pool=SimpleNamespace(),  # not an SWAKVPool
            page_size=1,
            device=_DEV,
        )
        self.assertFalse(src.is_translating)
        rows = torch.tensor([2, 0])
        view = src.build_index_table(
            req_pool_indices=rows, seq_lens=torch.tensor([5, 3])
        )
        self.assertIs(view.ids, req_to_token)
        self.assertIs(view.row_ids, rows)
        self.assertEqual(view.row_stride, req_to_token.stride(0))
        self.assertEqual(view.entry_page_size, 1)
        self.assertFalse(view.is_translated)
        self.assertIsNone(view.sliding_window_ids)
        # And the translate surface is the identity, not a wrapped copy.
        t = torch.tensor([1, 2, 3])
        self.assertIs(src.translate_full_attn_ids(t), t)


def _alloc_and_fill(allocator, ps, lens):
    """Allocate per-request virtual runs and write them into a fake
    req_to_token."""
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


class TestReadTableBuild(unittest.TestCase):
    def test_read_table_matches_reference_across_multipliers(self):
        """Both read tables must equal the independent per-element derivation,
        across page sizes and both multiplier regimes (MLA=1, MHA=2L); the swa
        table agreeing over VIRTUAL ids proves it is never chained through
        full-physical."""
        for ps in (1, 4):
            for collapse in (True, False):
                allocator = _build_composite(ps, collapse=collapse)
                full_mult = allocator.kernel_page_multiplier
                swa_mult = allocator.swa_kernel_page_multiplier
                req_to_token, rows, seq_lens = _alloc_and_fill(
                    allocator, ps, lens=[5 * ps, 2 * ps, 3 * ps - 1]
                )
                src = _make_source(allocator, req_to_token, ps)
                self.assertTrue(src.is_translating)
                width = 6
                view = src.build_index_table(
                    req_pool_indices=rows, seq_lens=seq_lens, max_pages=width
                )
                self.assertTrue(view.is_translated)
                self.assertEqual(view.entry_page_size, ps)
                self.assertTrue(
                    torch.equal(view.row_ids, torch.arange(3, dtype=torch.int64))
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
                    torch.equal(view.ids, want_full),
                    f"full read table off-formula (ps={ps}, mult={full_mult})",
                )
                self.assertTrue(
                    torch.equal(view.sliding_window_ids, want_swa),
                    f"swa read table off-formula (ps={ps}, mult={swa_mult})",
                )

    def test_packed_stream_equals_the_rectangle_it_replaces(self):
        """The two builders must agree element for element:
        packed[indptr[b] + p] == ids[b, p // ps] * ps + p % ps."""
        for ps in (1, 4):
            allocator = _build_composite(ps)
            req_to_token, rows, seq_lens = _alloc_and_fill(
                allocator, ps, lens=[5 * ps, 2 * ps, 3 * ps - 1]
            )
            src = _make_source(allocator, req_to_token, ps)
            view = src.build_index_table(
                req_pool_indices=rows, seq_lens=seq_lens, max_pages=6
            )
            indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
            indptr[1:] = torch.cumsum(seq_lens, dim=0)
            total = int(indptr[-1])
            packed = torch.zeros(total, dtype=torch.int32)
            translated = src.fill_packed_read_stream(
                req_pool_indices=rows,
                seq_lens=seq_lens,
                indptr=indptr,
                total_tokens=total,
                out=packed,
            )
            self.assertTrue(translated)
            for b, n in enumerate(seq_lens.tolist()):
                for pos in range(n):
                    self.assertEqual(
                        int(packed[int(indptr[b]) + pos]),
                        int(view.ids[b, pos // ps]) * ps + pos % ps,
                        f"packed stream off the page table (ps={ps}, b={b}, {pos=})",
                    )

    def test_sink_routing(self):
        """Dead lanes, -1 slots inside the live prefix, and tombstoned v2p
        pages must ALL read entry 0; one wild entry is a captured-graph OOB
        read at replay."""
        ps = 4
        allocator = _build_composite(ps)
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
        view = src.build_index_table(
            req_pool_indices=rows, seq_lens=seq_lens, max_pages=4
        )
        for table in (view.ids, view.sliding_window_ids):
            self.assertTrue(bool((table >= 0).all()))
            self.assertTrue(bool((table[1] == 0).all()), "dead lane not sunk")
            self.assertEqual(int(table[0, 1]), 0, "-1 slot not sunk")
            self.assertEqual(int(table[2, 0]), 0, "tombstone not sunk")


class TestBuildInto(unittest.TestCase):
    """fill_read_table fills a backend-owned padded block table's live prefix with
    FULL-side read-table entries -- the trtllm_mla / flashmla consumption route
    (their rows ARE the read table's rows)."""

    def test_prefix_filled_tail_sentinel_preserved_width_capped(self):
        """The -1 tail sentinel belongs to the backend, and a table padded
        WIDER than the req_to_token page span (trtllm's LCM alignment) must be
        capped rather than trip the builder's width assert."""
        ps = 4
        allocator = _build_composite(ps)
        full_mult = allocator.kernel_page_multiplier
        lens = [5, 2 * ps + 1, 1]
        req_to_token, rows, seq_lens = _alloc_and_fill(allocator, ps, lens=lens)
        src = _make_source(allocator, req_to_token, ps)
        self.assertTrue(src.is_translating)

        width_pages = req_to_token.shape[1] // ps + 3  # wider than the span
        out = torch.full((len(lens), width_pages), -1, dtype=torch.int32)
        src.fill_read_table(out=out, req_pool_indices=rows, seq_lens=seq_lens)

        want = _reference_table(
            req_to_token,
            rows,
            seq_lens,
            allocator.full_v2p_page_table,
            full_mult,
            ps,
            width_pages,
        )
        for b, n in enumerate(lens):
            n_pages = -(-n // ps)
            self.assertTrue(
                torch.equal(out[b, :n_pages], want[b, :n_pages]),
                f"row {b} live prefix off-formula",
            )
            self.assertTrue(
                bool((out[b, n_pages:] == -1).all()),
                f"row {b} tail sentinel clobbered",
            )

    def test_passthrough_source_refuses(self):
        """Callers dispatch on `reads_are_translated`; a passthrough source has
        no v2p to build from and must fail loud, not fill garbage."""
        src = KVIndexTranslator(
            req_to_token=torch.zeros((2, 4), dtype=torch.int64),
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device=_DEV,
        )
        with self.assertRaises(AssertionError):
            src.fill_read_table(
                out=torch.zeros((1, 4), dtype=torch.int32),
                req_pool_indices=torch.tensor([0]),
                seq_lens=torch.tensor([1]),
            )


class TestPoolOwnership(unittest.TestCase):
    """A runner only gets the kernel-facing id space when the pool IT reads and
    writes is the one the allocator's ids address.

    Guarded shape: a runner handed a SHARED allocator while owning a SEPARATE
    KV buffer sized to the allocator's SLOT count. Probing the allocator alone
    reports "unified" for that runner, so its indices would be mapped into the
    composite's kernel-facing space (ids up to num_pages * multiplier) and used
    to address a buffer with only num_slots rows.
    """

    def test_real_factory_bundle_satisfies_the_ownership_identity(self):
        """The guard rests on `allocator.get_kvcache() is token_to_kv_pool`, so
        a factory returning a pool the allocator does not hold would silently
        disable the unified path for EVERY model. Pinned against the real
        factory, not this file's own construction."""
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools

        bundle = init_unified_swa_pools(
            device="cpu",
            kv_cache_dtype=torch.float16,
            head_num=2,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=2,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=1,
            start_layer=0,
            end_layer=4,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            enable_memory_saver=False,
            need_sort=False,
        )
        self.assertIs(
            bundle.token_to_kv_pool_allocator.get_kvcache(),
            bundle.token_to_kv_pool,
        )
        src = KVIndexTranslator(
            req_to_token=torch.zeros((2, 8), dtype=torch.int32, device=_DEV),
            token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
            token_to_kv_pool=bundle.token_to_kv_pool,
            page_size=1,
            device=_DEV,
        )
        self.assertTrue(src.is_translating)

    def test_runner_with_its_own_pool_is_disabled(self):
        """Same allocator, different pool: must stay disabled."""
        alloc = _build_composite(ps=1)
        req_to_token = torch.zeros((2, 8), dtype=torch.int32, device=_DEV)
        own_pool = SimpleNamespace()  # a separate buffer, not the composite's
        src = KVIndexTranslator(
            req_to_token=req_to_token,
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=own_pool,
            page_size=1,
            device=_DEV,
        )
        self.assertFalse(src.is_translating)

    def test_disabled_source_is_strict_passthrough(self):
        """Such a runner must see RAW virtual ids: they index its own pool
        directly, and a translate here is the out-of-bounds bug the ownership
        identity exists to prevent."""
        alloc = _build_composite(ps=1)
        req_to_token = torch.arange(16, dtype=torch.int32, device=_DEV).view(2, 8)
        src = KVIndexTranslator(
            req_to_token=req_to_token,
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device=_DEV,
        )
        rows = torch.tensor([1, 0], dtype=torch.int32, device=_DEV)
        view = src.build_index_table(
            req_pool_indices=rows,
            seq_lens=torch.tensor([3, 2], dtype=torch.int32, device=_DEV),
        )
        # Read table: the EXACT objects a static-pool backend reads today.
        self.assertIs(view.ids, req_to_token)
        self.assertIs(view.row_ids, rows)
        self.assertFalse(view.is_translated)
        # And the token-level surface is the identity, same guard.
        t = torch.tensor([5, 6], dtype=torch.int64, device=_DEV)
        self.assertIs(src.translate_full_attn_ids(t), t)


class TestCaptureContract(unittest.TestCase):
    def test_caller_owned_table_is_returned_whole_and_filled_prefix_only(self):
        ps = 4
        allocator = _build_composite(ps)
        req_to_token = torch.full((4, 16 * ps), -1, dtype=torch.int64)
        v = allocator.alloc(2 * ps)
        req_to_token[1, : 2 * ps] = v
        src = _make_source(allocator, req_to_token, ps)

        # A page-table consumer owns its buffers; entry 0 is the reserved sink
        # in every id space, so zeros are what an unfilled column must read as.
        cap = torch.zeros((4, 8), dtype=torch.int32, device=_DEV)
        cap_swa = torch.zeros((4, 8), dtype=torch.int32, device=_DEV)
        tables = KVReadTables(full=cap, sliding_window=cap_swa)

        # Poison everything, then refresh a 1-row batch: ONLY its live prefix
        # may change -- stale tails and other rows are the fa3 contract.
        cap.fill_(7)
        cap_swa.fill_(7)
        view = src.build_index_table(
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([2 * ps]),
            into=tables,
        )
        self.assertIs(view.ids, cap, "the caller's table comes back WHOLE")
        want = allocator.full_v2p_page_table[req_to_token[1, ::ps][:2] // ps] * (
            2 * _FULL_L
        )
        self.assertTrue(torch.equal(cap[0, :2], want.to(torch.int32)))
        self.assertTrue(bool((cap[0, 2:] == 7).all()), "stale tail was cleared")
        self.assertTrue(bool((cap[1:] == 7).all()), "rows beyond bs were touched")

    def test_row_ids_not_reallocated_across_builds(self):
        """`row_ids` is one arange sized from the request pool and sliced per
        build; a per-build `torch.arange` would be correct but costs an
        allocation and a launch on every replay prep."""
        allocator = _build_composite(1)
        req_to_token, rows, seq_lens = _alloc_and_fill(allocator, 1, lens=[4, 2, 3])
        src = _make_source(allocator, req_to_token, 1)
        self.assertTrue(src.is_translating)
        first = src.build_index_table(
            req_pool_indices=rows[:2], seq_lens=seq_lens[:2], max_pages=4
        )
        second = src.build_index_table(
            req_pool_indices=rows, seq_lens=seq_lens, max_pages=4
        )
        self.assertEqual(first.row_ids.data_ptr(), second.row_ids.data_ptr())
        self.assertTrue(torch.equal(first.row_ids, torch.arange(2, device=_DEV)))
        self.assertTrue(torch.equal(second.row_ids, torch.arange(3, device=_DEV)))
        # Sized to bound any batch the request pool can hold.
        self.assertGreaterEqual(src._rows.numel(), req_to_token.shape[0])


class _FakeForwardBatch:
    """Weakref-able stand-in (SimpleNamespace is not) carrying the fields
    `index_table_for_batch` and `rebind_write_loc` read. `seq_lens_sum`
    defaults to the real sum: it is the signal that the CPU mirror is live,
    and a real ForwardBatch always carries it (None only when gpu_only)."""

    def __init__(
        self,
        *,
        req_pool_indices=None,
        seq_lens=None,
        seq_lens_cpu=None,
        out_cache_loc=None,
        seq_lens_sum=-1,
    ):
        self.req_pool_indices = req_pool_indices
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens_cpu
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = (
            (None if seq_lens is None else int(seq_lens.sum()))
            if seq_lens_sum == -1
            else seq_lens_sum
        )


class TestViewMemo(unittest.TestCase):
    """The eager view is memoized ON THE SOURCE in a single slot keyed by batch
    identity, so per-batch state stays out of the ForwardBatch while one
    metadata build's many consumers still share one table build."""

    def _fb(self, allocator, ps, lens):
        req_to_token, rows, seq_lens = _alloc_and_fill(allocator, ps, lens=lens)
        fb = _FakeForwardBatch(
            req_pool_indices=rows,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
        )
        return fb, req_to_token

    def test_same_batch_returns_the_memoized_view(self):
        ps = 1
        allocator = _build_composite(ps)
        fb, req_to_token = self._fb(allocator, ps, lens=[3, 2])
        src = _make_source(allocator, req_to_token, ps)
        v1 = src.index_table_for_batch(fb)
        v2 = src.index_table_for_batch(fb)
        self.assertIs(v1, v2)

    def test_next_batch_replaces_the_single_slot(self):
        ps = 1
        allocator = _build_composite(ps)
        fb1, req_to_token = self._fb(allocator, ps, lens=[3, 2])
        src = _make_source(allocator, req_to_token, ps)
        v1 = src.index_table_for_batch(fb1)
        fb2 = _FakeForwardBatch(
            req_pool_indices=fb1.req_pool_indices,
            seq_lens=fb1.seq_lens,
            seq_lens_cpu=fb1.seq_lens_cpu,
        )
        v2 = src.index_table_for_batch(fb2)
        self.assertIsNot(v1, v2)
        # Single slot: fb1 no longer matches and rebuilds.
        v1_again = src.index_table_for_batch(fb1)
        self.assertIsNot(v1_again, v1)

    def test_dead_batch_never_matches(self):
        """A garbage-collected batch's slot must not serve a later batch: the
        weakref key goes dead and the build runs fresh."""
        import gc

        ps = 1
        allocator = _build_composite(ps)
        fb1, req_to_token = self._fb(allocator, ps, lens=[3, 2])
        src = _make_source(allocator, req_to_token, ps)
        v1 = src.index_table_for_batch(fb1)
        del fb1
        gc.collect()
        fb2, _ = self._fb(allocator, ps, lens=[2])
        v2 = src.index_table_for_batch(fb2)
        self.assertIsNot(v2, v1)
        self.assertEqual(v2.ids.shape[0], 1)


class TestWriteLoc(unittest.TestCase):
    """The two-phase write contract: `rebind_write_loc` rebinds the full side
    once at ForwardBatch construction, and the sliding-window write loc derives
    POINTWISE from the full-side values -- pads, slices, and fresh copies
    included -- with no handover and no stored per-forward state."""

    def _built(self, ps=1, n=4):
        allocator = _build_composite(ps)
        req_to_token, rows, seq_lens = _alloc_and_fill(allocator, ps, lens=[max(n, 1)])
        src = _make_source(allocator, req_to_token, ps)
        virt = allocator.alloc(-(-n // ps) * ps)[:n]
        want_full = allocator.translate_kv_loc_for_kernel(virt)
        want_swa = allocator.translate_loc_from_full_to_swa(virt)
        return src, allocator, rows, seq_lens, virt, want_full, want_swa

    def _field(self, src, rows, seq_lens, kernel_loc):
        return src.sliding_window_write_loc_for(kernel_loc)

    def test_rebind_translates_full_side_only(self):
        for ps in (1, 4):
            src, _, _, _, virt, want_full, _ = self._built(ps=ps, n=3 * ps)
            keep = virt.clone()
            fb = _FakeForwardBatch(out_cache_loc=virt)
            src.rebind_write_loc(fb)
            # Full side: rebound to a FRESH kernel-facing tensor; the
            # ScheduleBatch's aliased virtual tensor is untouched.
            self.assertIsNot(fb.out_cache_loc, virt)
            self.assertTrue(torch.equal(fb.out_cache_loc, want_full))
            self.assertTrue(torch.equal(virt, keep))

    def test_swa_write_loc_round_trips_from_full_side(self):
        """Derived property: `field(full(t)) == swa(t)` for any virtual run t,
        across page sizes and multipliers."""
        for ps in (1, 4, 64):
            src, _, rows, seq_lens, _, want_full, want_swa = self._built(
                ps=ps, n=3 * ps
            )
            got = self._field(src, rows, seq_lens, want_full)
            self.assertTrue(torch.equal(got, want_swa))

    def test_pad_lanes_derive_to_sink(self):
        """The DP pad appends zeros, and kernel-facing 0 is the reserved
        padding slot in every id space, so pad lanes derive to swa slot 0 with
        no `num_live` bookkeeping."""
        src, _, rows, seq_lens, _, want_full, want_swa = self._built(n=3)
        padded = torch.cat([want_full, want_full.new_zeros(2)])
        got = self._field(src, rows, seq_lens, padded)
        self.assertTrue(torch.equal(got[:3], want_swa))
        self.assertTrue(bool((got[3:] == 0).all()), "pad lanes must land on slot 0")

    def test_tombstoned_swa_page_clamps_to_sink(self):
        src, allocator, rows, seq_lens, virt, want_full, _ = self._built(ps=1, n=2)
        allocator.swa_v2p_page_table[int(virt[0])] = -1
        got = self._field(src, rows, seq_lens, want_full[:1])
        self.assertEqual(int(got[0]), 0)

    def test_static_swa_pool_derives_via_pool_translate(self):
        """Static SWA pools: the field is the pool's own legacy full->swa
        translate, computed at the same build; the rebind stays a no-op."""
        pool = SWAKVPool.__new__(SWAKVPool)
        pool.full_to_swa_index_mapping = torch.arange(10, dtype=torch.int64)
        pool.translate_loc_from_full_to_swa = lambda t: t + 100
        src = KVIndexTranslator(
            req_to_token=torch.zeros((2, 4), dtype=torch.int64),
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=pool,
            page_size=1,
            device=_DEV,
        )
        loc = torch.tensor([5, 6], dtype=torch.int64)
        fb = _FakeForwardBatch(out_cache_loc=loc)
        src.rebind_write_loc(fb)
        self.assertIs(fb.out_cache_loc, loc, "disabled rebind must be a no-op")
        self.assertTrue(torch.equal(src.sliding_window_write_loc_for(loc), loc + 100))

    def test_no_loc_or_no_swa_side_yields_none(self):
        # Unified swa composite, but there is no write loc this forward.
        src, _, rows, seq_lens, _, _, _ = self._built(n=2)
        self.assertIsNone(src.sliding_window_write_loc_for(None))
        # Passthrough on a non-SWA pool: a loc is given, but there is no swa
        # id space to derive into.
        plain = KVIndexTranslator(
            req_to_token=torch.zeros((2, 4), dtype=torch.int64),
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device=_DEV,
        )
        self.assertIsNone(
            plain.sliding_window_write_loc_for(torch.tensor([3], dtype=torch.int64))
        )

    def test_rebind_retires_the_view_memo(self):
        ps = 1
        allocator = _build_composite(ps)
        req_to_token, rows, seq_lens = _alloc_and_fill(allocator, ps, lens=[3, 2])
        src = _make_source(allocator, req_to_token, ps)
        fb = _FakeForwardBatch(
            req_pool_indices=rows, seq_lens=seq_lens, seq_lens_cpu=seq_lens
        )
        v1 = src.index_table_for_batch(fb)
        src.rebind_write_loc(_FakeForwardBatch(out_cache_loc=None))
        v2 = src.index_table_for_batch(fb)
        self.assertIsNot(v2, v1, "rebind starts the next forward: stale views die")


if __name__ == "__main__":
    unittest.main()
