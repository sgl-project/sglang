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
"""KVIndexTranslator — the read-path id-space choke point.

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
    contents); the returned table is the WHOLE buffer (pointer-stable);
  - the eager-view memo: a single source-resident slot keyed by batch
    identity (same batch shares one build; the next batch replaces it; a
    dead batch never matches).

    python -m pytest test/registered/unit/mem_cache/test_kv_index_translator.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from test_multi_ended_allocator import _FakeUnifiedSWAKVPool

from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedSWATokenToKVPoolAllocator,
)
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
        # The multiplier-1 arm: a sub-pool whose views are NOT dense, so its
        # kernel-facing ids are the physical ones. No unified sub-pool is built
        # that way today (the specs all report >1), so pin it here rather than
        # lose the regime -- the table formula has to hold for both.
        allocator.full_attn_allocator.kernel_page_multiplier = 1
        allocator.swa_attn_allocator.kernel_page_multiplier = 1
    # The real UnifiedSWAKVPool carries the pool-level full->swa translate, so
    # the fake must too: it IS the runner's token_to_kv_pool (see _make_source),
    # and a stand-in that lacked the method would force a wrapper object,
    # modelling a pool/allocator split that never occurs for a target runner.
    kvcache.translate_loc_from_full_to_swa = allocator.translate_loc_from_full_to_swa
    return allocator


def _make_source(allocator, req_to_token, ps):
    """The owning runner's source: its token_to_kv_pool IS the allocator's own
    kvcache. (A runner can share the allocator while owning a different pool —
    see TestPoolOwnership.)"""
    return KVIndexTranslator(
        req_to_token=req_to_token,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=allocator.get_kvcache(),
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
                    f"full canonical off-formula (ps={ps}, mult={full_mult})",
                )
                self.assertTrue(
                    torch.equal(view.sliding_window_ids, want_swa),
                    f"swa canonical off-formula (ps={ps}, mult={swa_mult})",
                )

    def test_sink_routing(self):
        """Dead lanes (seq_len 0), -1 slots inside the live prefix, and
        tombstoned v2p pages must ALL read entry 0 — one wild entry is a
        captured-graph OOB read at replay."""
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
    FULL-side canonical entries — the trtllm_mla / flashmla consumption route
    (their rows ARE the canonical rows)."""

    def test_prefix_filled_tail_sentinel_preserved_width_capped(self):
        """Three contracts in one batch: entries equal the canonical formula,
        lanes past each row's live pages keep the backend's -1 sentinel
        (prefix-only — a tail write scatters the trtllm sentinel contract),
        and a table padded WIDER than the req_to_token page span (trtllm's
        LCM alignment) is capped instead of tripping the builder's width
        assert."""
        ps = 4
        allocator = _build_composite(ps)
        full_mult = allocator.kernel_page_multiplier
        lens = [5, 2 * ps + 1, 1]
        req_to_token, rows, seq_lens = _alloc_and_fill(allocator, ps, lens=lens)
        src = _make_source(allocator, req_to_token, ps)
        self.assertTrue(src.is_translating)

        width_pages = req_to_token.shape[1] // ps + 3  # wider than the span
        out = torch.full((len(lens), width_pages), -1, dtype=torch.int32)
        got = src.fill_read_table(out=out, req_pool_indices=rows, seq_lens=seq_lens)
        self.assertIs(got, out)

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
        """Callers dispatch on `enabled`; a passthrough source has no v2p to
        build from and must fail loud, not fill garbage."""
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

    Guarded shape: a runner handed a SHARED allocator (one slot index space,
    one req_to_token) while owning a SEPARATE KV buffer sized to the
    allocator's SLOT count. Probing the allocator alone reports "unified" for
    that runner, so its indices would be mapped into the composite's
    kernel-facing space (kernel-facing ids up to num_pages * multiplier) and then used
    to address a buffer with only num_slots rows — out of bounds on both the
    read gather and the KV store.
    """

    def test_real_factory_bundle_satisfies_the_ownership_identity(self):
        """The guard rests on `allocator.get_kvcache() is token_to_kv_pool`
        holding for a REAL target bundle. If a factory ever returned a pool
        the allocator does not hold, the guard would silently disable the
        unified path for EVERY model — so pin it against the real factory
        rather than against this file's own construction."""
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
        """Consequence of the guard: such a runner must see RAW virtual ids on
        the read rail — they index its own pool directly. A translate here is
        the out-of-bounds bug the ownership identity exists to prevent."""
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
        # Read rail: the EXACT objects a static-pool backend reads today.
        self.assertIs(view.ids, req_to_token)
        self.assertIs(view.row_ids, rows)
        self.assertFalse(view.is_translated)
        # And the token-level surface is the identity, same guard.
        t = torch.tensor([5, 6], dtype=torch.int64, device=_DEV)
        self.assertIs(src.translate_full_attn_ids(t), t)


class TestCaptureContract(unittest.TestCase):
    def test_capture_buffers_zero_filled_idempotent_and_prefix_only(self):
        ps = 4
        allocator = _build_composite(ps)
        req_to_token = torch.full((4, 16 * ps), -1, dtype=torch.int64)
        v = allocator.alloc(2 * ps)
        req_to_token[1, : 2 * ps] = v
        src = _make_source(allocator, req_to_token, ps)

        src.ensure_capture_buffers(max_bs=4, max_context_len=8 * ps)
        cap = src._capture_full_ids
        self.assertTrue(bool((cap == 0).all()), "capture buffers must start zeroed")
        src.ensure_capture_buffers(max_bs=4, max_context_len=8 * ps)
        self.assertIs(
            src._capture_full_ids, cap, "ensure_capture_buffers must be idempotent"
        )

        # Poison everything, then refresh a 1-row batch: ONLY its live prefix
        # may change — stale tails and other rows are the fa3 contract.
        cap.fill_(7)
        src._capture_swa_ids.fill_(7)
        view = src.build_index_table(
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([2 * ps]),
            captured=True,
        )
        self.assertIs(view.ids, cap, "captured view must return the WHOLE buffer")
        want = allocator.full_v2p_page_table[req_to_token[1, ::ps][:2] // ps] * (
            2 * _FULL_L
        )
        self.assertTrue(torch.equal(cap[0, :2], want.to(torch.int32)))
        self.assertTrue(bool((cap[0, 2:] == 7).all()), "stale tail was cleared")
        self.assertTrue(bool((cap[1:] == 7).all()), "rows beyond bs were touched")

    def test_row_ids_not_reallocated_across_builds(self):
        """`row_ids` is a constant arange sized once from the request pool, so
        builds at different batch sizes hand back slices of ONE buffer. A
        per-build `torch.arange` would be correct but would spend an allocation
        and a launch on every replay prep."""
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
    """Weakref-able stand-in (SimpleNamespace is not) carrying the three
    fields `index_table_for_batch` reads."""

    def __init__(self, *, req_pool_indices, seq_lens, seq_lens_cpu):
        self.req_pool_indices = req_pool_indices
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens_cpu


class TestViewMemo(unittest.TestCase):
    """The eager view is memoized ON THE SOURCE in a single slot keyed by
    batch identity — per-batch state stays out of the ForwardBatch (it does
    not scale with the number of id spaces), and one metadata build's many
    consumers still share one table build."""

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


if __name__ == "__main__":
    unittest.main()
