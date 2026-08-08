"""Unit test for strict bit-exact c4 / c4-indexer compress-state capture+restore.

RED spec for "SWA window riding capture of c4/indexer state" (revived from fc18383181 /
b61ce6268a, removed by ab94e91021). Exercises the riding capture/restore helpers
with faked collaborators and real CPU tensors -- no GPU / model required:

  * capture strides by page; one host tile per (rid, B) page boundary
  * the [B-ratio, B) window is packed at tile start (off0=0 contract)
  * captured bytes are byte-identical to the kv_and_score_buffer window
  * capture -> restore round-trip lands byte-exact on the reusing request's ring
  * WITHOUT restore, the reuse-boundary read is stale (the dirty read this fixes)
  * flag OFF (no host pool wired) is a no-op

Until the riding helpers are re-added (Task S.x), the imports below fail and the
whole module errors out at collection time -- that is the intended RED.

Run:
  PYTHONPATH=<worktree>/python python -m pytest \
      test/srt/mem_cache/test_swa_bitexact_compress_state.py -q
"""

import threading
import types
import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4.compress_hip import (
    CompressorHip,
    capture_c4_state_windows_unified,
)
from sglang.srt.mem_cache import memory_pool_host as MPH
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

# Test-only reference oracle, kept for the byte-parity cross-check below.
_REFERENCE = CompressorHip._reference_capture_compress_state_windows


def _capture_via_unified(
    fake_self,
    *,
    kv_and_score_buffer,
    valid_kv_len,
    prefix_len,
    extend_len,
    rid,
    backend,
    stride=1,
    tail_B=-1,
):
    """Drive the production capture (capture_c4_state_windows_unified) through
    the legacy call shape so these tests exercise the real prefill path."""
    kv = kv_and_score_buffer.kv_score
    pre_len = valid_kv_len - extend_len
    pre_state = kv[:pre_len]
    new_tok = kv[pre_len:]
    page = backend.page_size
    tk = backend.token_to_kv_pool
    tk.translate_loc_from_full_to_swa = lambda x: x
    tk._swa_offload_page_stride = stride
    backend.req_to_token_pool = types.SimpleNamespace(
        req_to_token=torch.zeros(
            (int(rid) + 8, int(prefix_len) + int(extend_len) + page + 8),
            dtype=torch.int64,
        )
    )
    state_pool = types.SimpleNamespace(
        translate_from_swa_loc_to_state_loc=lambda x: x,
        get_state_by_state_loc=lambda loc: types.SimpleNamespace(kv_score=pre_state),
    )
    fb = types.SimpleNamespace(
        batch_size=1,
        extend_prefix_lens_cpu=[int(prefix_len)],
        extend_seq_lens_cpu=[int(extend_len)],
        req_pool_indices=torch.tensor([int(rid)], dtype=torch.int64),
    )
    if tail_B is not None and tail_B >= 0:
        fb.orig_seq_lens = torch.tensor([int(tail_B)], dtype=torch.int64)
    capture_c4_state_windows_unified(
        backend=backend,
        state_pool=state_pool,
        kv_score_input=new_tok,
        forward_batch=fb,
        is_indexer=fake_self.is_in_indexer,
        layer_id=fake_self.layer_id,
        ratio=fake_self.ratio,
    )


_CAPTURE = _capture_via_unified


_CAPTURE_DECODE = DeepseekV4HipRadixBackend.capture_compress_state_windows_decode


_TRANSLATE = CompressStatePool.translate_from_swa_loc_to_state_loc


def _fake_host_pool(*, ring_size, slot_bytes, num_pages):
    item_bytes = ring_size * slot_bytes
    host_buf = torch.zeros((num_pages, item_bytes), dtype=torch.uint8)
    counter = {"n": 0}

    def alloc(need):
        assert need == ring_size
        page = counter["n"]
        counter["n"] += 1
        if page >= num_pages:
            return None
        return torch.arange(page * ring_size, page * ring_size + ring_size)

    return types.SimpleNamespace(
        slot_page_size=ring_size,
        item_bytes=item_bytes,
        data_refs=[host_buf],
        _capture_staging={},
        _capture_state_crc={},
        alloc=alloc,
    )


def _fake_backend(host_pool, *, page=256, swa_ring=128, is_indexer=False):
    attr = "_c4_indexer_state_host_pool" if is_indexer else "_c4_state_host_pool"
    pool = types.SimpleNamespace(
        unified_swa_ring_size=swa_ring,
        _c4_state_layer_index={0: 0},
    )
    setattr(pool, attr, host_pool)
    return types.SimpleNamespace(token_to_kv_pool=pool, page_size=page)


def _fake_self(*, ratio=4, is_indexer=False, layer_id=0):
    return types.SimpleNamespace(
        ratio=ratio, is_in_indexer=is_indexer, layer_id=layer_id
    )


class TestCompressStateCapture(unittest.TestCase):
    def _run(
        self,
        *,
        prefix_len,
        extend_len,
        ratio=4,
        page=256,
        swa_ring=128,
        ring_size=8,
        last_dim=16,
        dtype=torch.bfloat16,
        is_indexer=False,
    ):
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()
        hp = _fake_host_pool(ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16)
        backend = _fake_backend(hp, page=page, swa_ring=swa_ring, is_indexer=is_indexer)
        valid_kv_len = (
            prefix_len + extend_len
        )  # empty pre_state for prefix_len%ratio==0
        buf = types.SimpleNamespace(
            kv_score=torch.randint(
                0, 255, (valid_kv_len, last_dim), dtype=torch.int32
            ).to(dtype)
        )
        fs = _fake_self(ratio=ratio, is_indexer=is_indexer)
        _CAPTURE(
            fs,
            kv_and_score_buffer=buf,
            valid_kv_len=valid_kv_len,
            prefix_len=prefix_len,
            extend_len=extend_len,
            rid=7,
            backend=backend,
        )
        return hp, buf, slot_bytes

    def test_capture_windows_byte_exact(self):
        page, ring_size, ratio, last_dim = 256, 8, 4, 16
        hp, buf, slot_bytes = self._run(
            prefix_len=0,
            extend_len=512,
            page=page,
            ring_size=ring_size,
            ratio=ratio,
            last_dim=last_dim,
        )
        # boundaries 256, 512 -> two tiles keyed (7, B)
        self.assertEqual(set(hp._capture_staging), {(7, 256), (7, 512)})
        for B in (256, 512):
            hidx = hp._capture_staging[(7, B)]
            page_row = int(hidx[0].item()) // ring_size
            off0 = 0  # window packed at tile start (off0=0 contract)
            got = hp.data_refs[0][page_row][
                off0 * slot_bytes : (off0 + ratio) * slot_bytes
            ]
            want = (
                buf.kv_score[B - ratio : B].contiguous().view(torch.uint8).reshape(-1)
            )
            self.assertTrue(torch.equal(got, want))
            # trailing slots [ratio, ring_size) never populated by capture
            self.assertTrue(
                torch.equal(
                    hp.data_refs[0][page_row][(off0 + ratio) * slot_bytes :],
                    torch.zeros((ring_size - ratio) * slot_bytes, dtype=torch.uint8),
                )
            )

    def test_no_boundary_no_capture(self):
        # chunk shorter than a page -> no page boundary -> nothing captured
        hp, _, _ = self._run(prefix_len=0, extend_len=100)
        self.assertEqual(hp._capture_staging, {})

    def test_flag_off_noop(self):
        # No host pool wired (bit-exact flag off) -> silent no-op.
        pool = types.SimpleNamespace(
            unified_swa_ring_size=128, _c4_state_layer_index={0: 0}
        )
        backend = types.SimpleNamespace(token_to_kv_pool=pool, page_size=256)
        buf = types.SimpleNamespace(
            kv_score=torch.zeros((512, 16), dtype=torch.bfloat16)
        )
        _CAPTURE(
            _fake_self(),
            kv_and_score_buffer=buf,
            valid_kv_len=512,
            prefix_len=0,
            extend_len=512,
            rid=1,
            backend=backend,
        )  # must not raise

    def test_ratio_128_skipped(self):
        hp, _, _ = self._run(prefix_len=0, extend_len=512, ratio=128)
        self.assertEqual(hp._capture_staging, {})

    def test_indexer_pool_routing(self):
        hp, buf, slot_bytes = self._run(prefix_len=0, extend_len=256, is_indexer=True)
        self.assertEqual(set(hp._capture_staging), {(7, 256)})


class TestUnifiedCaptureEquivalence(unittest.TestCase):
    """The UNIFIED-KV prefill capture helper (``capture_c4_state_windows_unified``,
    wired into ``compressor_v2.forward_unified``) must stage byte-identical tiles
    to the test-only reference oracle ``_reference_capture_compress_state_windows``
    for the same per-request
    ``[pre_kv_state | new tokens]`` buffer. The oracle is fed the already-concatenated
    buffer; unified builds the cat internally from a faked state pool -- both must
    land the same ``(rid, B)`` staging with the same bytes/offsets."""

    def _fake_state_pool(self, pre_state_kv):
        # identity translates; get_state_by_state_loc returns the fixed phantom
        # pre-state (loc-indexed count == pre_state rows) as a KVAndScore-like.
        return types.SimpleNamespace(
            translate_from_swa_loc_to_state_loc=lambda x: x,
            get_state_by_state_loc=lambda loc: types.SimpleNamespace(
                kv_score=pre_state_kv
            ),
        )

    def test_unified_matches_legacy_byte_exact(self):
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()
        prefix_len, extend_len, rid = 0, 512, 7

        # prefix_len%ratio==0 -> compute_state_len_indices gives `ratio` phantom
        # pre-rows (all loc -1); model them as a fixed pre_state tensor.
        pre_state = torch.randint(0, 255, (ratio, last_dim), dtype=torch.int32).to(
            dtype
        )
        new_tok = torch.randint(0, 255, (extend_len, last_dim), dtype=torch.int32).to(
            dtype
        )
        state_buf = torch.cat([pre_state, new_tok], dim=0)  # legacy kv_and_score_buffer

        # --- legacy: fed the pre-built [pre|new] buffer ---
        hp_leg = _fake_host_pool(
            ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16
        )
        be_leg = _fake_backend(hp_leg, page=page, swa_ring=swa_ring)
        _REFERENCE(
            _fake_self(ratio=ratio),
            kv_and_score_buffer=types.SimpleNamespace(kv_score=state_buf),
            valid_kv_len=state_buf.size(0),
            prefix_len=prefix_len,
            extend_len=extend_len,
            rid=rid,
            backend=be_leg,
        )

        # --- unified: builds the cat internally from state pool + raw new tokens ---
        hp_uni = _fake_host_pool(
            ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16
        )
        be_uni = _fake_backend(hp_uni, page=page, swa_ring=swa_ring)
        n = extend_len
        be_uni.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.zeros((rid + 4, n + page), dtype=torch.int64)
        )
        be_uni.token_to_kv_pool.translate_loc_from_full_to_swa = lambda x: x
        fb = types.SimpleNamespace(
            batch_size=1,
            extend_prefix_lens_cpu=[prefix_len],
            extend_seq_lens_cpu=[extend_len],
            req_pool_indices=torch.tensor([rid], dtype=torch.int64),
        )
        capture_c4_state_windows_unified(
            backend=be_uni,
            state_pool=self._fake_state_pool(pre_state),
            kv_score_input=new_tok,
            forward_batch=fb,
            is_indexer=False,
            layer_id=0,
            ratio=ratio,
        )

        self.assertEqual(set(hp_leg._capture_staging), set(hp_uni._capture_staging))
        self.assertEqual(set(hp_uni._capture_staging), {(rid, 256), (rid, 512)})
        self.assertTrue(torch.equal(hp_leg.data_refs[0], hp_uni.data_refs[0]))

    def test_stride_and_tail_gate_only(self):
        """With ``stride>1`` BOTH capture paths must stage a state tile at
        EXACTLY the SWA-carried boundaries: every ``stride``-th page boundary
        plus the true sequence tail page (``orig_seq_lens`` page-aligned) -- and
        nothing else. This is the fix for the orphan-tile staging blow-up: a
        non-stride, non-tail interior boundary must NOT be staged."""
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        stride = 2
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()
        # extend 768 -> boundaries 256(idx1), 512(idx2), 768(idx3). stride=2 keeps
        # idx%2==0 -> 512; tail page (orig=768) forces 768. 256 is dropped.
        prefix_len, extend_len, orig, rid = 0, 768, 768, 7
        expected = {(rid, 512), (rid, 768)}

        pre_state = torch.randint(0, 255, (ratio, last_dim), dtype=torch.int32).to(
            dtype
        )
        new_tok = torch.randint(0, 255, (extend_len, last_dim), dtype=torch.int32).to(
            dtype
        )
        state_buf = torch.cat([pre_state, new_tok], dim=0)
        tail_B = (orig // page) * page

        # --- legacy: caller passes derived stride + tail_B ---
        hp_leg = _fake_host_pool(
            ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16
        )
        be_leg = _fake_backend(hp_leg, page=page, swa_ring=swa_ring)
        _REFERENCE(
            _fake_self(ratio=ratio),
            kv_and_score_buffer=types.SimpleNamespace(kv_score=state_buf),
            valid_kv_len=state_buf.size(0),
            prefix_len=prefix_len,
            extend_len=extend_len,
            rid=rid,
            backend=be_leg,
            stride=stride,
            tail_B=tail_B,
        )
        self.assertEqual(set(hp_leg._capture_staging), expected)

        # --- unified: stride from pool attr, tail from forward_batch.orig_seq_lens ---
        hp_uni = _fake_host_pool(
            ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16
        )
        be_uni = _fake_backend(hp_uni, page=page, swa_ring=swa_ring)
        be_uni.token_to_kv_pool._swa_offload_page_stride = stride
        be_uni.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.zeros((rid + 4, extend_len + page), dtype=torch.int64)
        )
        be_uni.token_to_kv_pool.translate_loc_from_full_to_swa = lambda x: x
        fb = types.SimpleNamespace(
            batch_size=1,
            extend_prefix_lens_cpu=[prefix_len],
            extend_seq_lens_cpu=[extend_len],
            req_pool_indices=torch.tensor([rid], dtype=torch.int64),
            orig_seq_lens=torch.tensor([orig], dtype=torch.int64),
        )
        capture_c4_state_windows_unified(
            backend=be_uni,
            state_pool=self._fake_state_pool(pre_state),
            kv_score_input=new_tok,
            forward_batch=fb,
            is_indexer=False,
            layer_id=0,
            ratio=ratio,
        )
        self.assertEqual(set(hp_uni._capture_staging), expected)
        # byte-exact parity between the two paths at the gated boundaries
        self.assertTrue(torch.equal(hp_leg.data_refs[0], hp_uni.data_refs[0]))

    def _stage_both_paths(self, *, extend_len, orig, stride, bigram, prefix_len=0):
        """Run the oracle and the unified capture over the same input and return
        their staging key sets. They must always agree; the caller asserts on the
        keys themselves."""
        page, swa_ring, ring_size, ratio = 256, 128, 8, 4
        last_dim, dtype = 16, torch.bfloat16
        slot_bytes = last_dim * torch.tensor([], dtype=dtype).element_size()
        rid = 7

        # The overlap prefix compress reads is [cs - (cs%ratio + ratio), cs); the
        # oracle is handed it already concatenated, unified fetches it lazily.
        pre_rows = prefix_len % ratio + ratio
        pre_state = torch.randint(0, 255, (pre_rows, last_dim), dtype=torch.int32).to(
            dtype
        )
        new_tok = torch.randint(0, 255, (extend_len, last_dim), dtype=torch.int32).to(
            dtype
        )
        state_buf = torch.cat([pre_state, new_tok], dim=0)
        tail_B = (orig // page) * page

        hp_leg = _fake_host_pool(
            ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16
        )
        be_leg = _fake_backend(hp_leg, page=page, swa_ring=swa_ring)
        be_leg.token_to_kv_pool._swa_capture_bigram_key = bigram
        _REFERENCE(
            _fake_self(ratio=ratio),
            kv_and_score_buffer=types.SimpleNamespace(kv_score=state_buf),
            valid_kv_len=state_buf.size(0),
            prefix_len=prefix_len,
            extend_len=extend_len,
            rid=rid,
            backend=be_leg,
            stride=stride,
            tail_B=tail_B,
            orig=orig,
        )

        hp_uni = _fake_host_pool(
            ring_size=ring_size, slot_bytes=slot_bytes, num_pages=16
        )
        be_uni = _fake_backend(hp_uni, page=page, swa_ring=swa_ring)
        be_uni.token_to_kv_pool._swa_capture_bigram_key = bigram
        be_uni.token_to_kv_pool._swa_offload_page_stride = stride
        be_uni.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.zeros(
                (rid + 4, prefix_len + extend_len + page), dtype=torch.int64
            )
        )
        be_uni.token_to_kv_pool.translate_loc_from_full_to_swa = lambda x: x
        fb = types.SimpleNamespace(
            batch_size=1,
            extend_prefix_lens_cpu=[prefix_len],
            extend_seq_lens_cpu=[extend_len],
            req_pool_indices=torch.tensor([rid], dtype=torch.int64),
            orig_seq_lens=torch.tensor([orig], dtype=torch.int64),
        )
        capture_c4_state_windows_unified(
            backend=be_uni,
            state_pool=self._fake_state_pool(pre_state),
            kv_score_input=new_tok,
            forward_batch=fb,
            is_indexer=False,
            layer_id=0,
            ratio=ratio,
        )

        leg_keys = set(hp_leg._capture_staging)
        uni_keys = set(hp_uni._capture_staging)
        self.assertEqual(leg_keys, uni_keys, "oracle and unified capture disagree")
        self.assertTrue(torch.equal(hp_leg.data_refs[0], hp_uni.data_refs[0]))
        return leg_keys

    def test_bigram_shifts_page_aligned_tail_down_one_page(self):
        """Under EAGLE the radix key is a bigram view, so an insert of ``n``
        tokens ends the node one page below ``n`` when ``n`` is page-aligned. A
        window keyed at the token boundary is then unclaimable and gets freed
        unused, which collapses the reuse boundary to 0 (#cached-token=0).

        extend=768 with stride=3 leaves exactly one gated boundary, the tail at
        768, so the shift moves the single staged key 768 -> 512 with nothing to
        alias against."""
        rid = 7
        self.assertEqual(
            self._stage_both_paths(extend_len=768, orig=768, stride=3, bigram=False),
            {(rid, 768)},
        )
        self.assertEqual(
            self._stage_both_paths(extend_len=768, orig=768, stride=3, bigram=True),
            {(rid, 512)},
        )

    def test_bigram_leaves_unaligned_tail_alone(self):
        """The shift must key off page alignment, not off EAGLE alone: when ``n``
        is NOT page-aligned the node already ends at ``(n//page)*page`` == the
        tail page, so shifting would push the window one page too low. extend=700
        ends the node at 512, which is where the unshifted tail already lands."""
        for bigram in (False, True):
            with self.subTest(bigram=bigram):
                self.assertEqual(
                    self._stage_both_paths(
                        extend_len=700, orig=700, stride=1, bigram=bigram
                    ),
                    {(7, 256), (7, 512)},
                )

    def test_window_reaching_into_overlap_prefix_is_byte_exact(self):
        """The window can reach back past the chunk's flat-KV start when a tiny
        chunk crosses a page boundary, so it must be stitched from the overlap
        prefix plus the new tokens. cs=254 with a 4-token chunk puts the boundary
        at 256 and the window [252, 256) two rows behind cs -- the only case where
        the prefix is read at all, and where an off-by-one in pre_len would stage
        the wrong bytes. Byte-parity against the oracle is asserted inside."""
        self.assertEqual(
            self._stage_both_paths(
                extend_len=4, orig=258, stride=1, bigram=False, prefix_len=254
            ),
            {(7, 256)},
        )

    def test_bigram_tail_shift_aliases_onto_interior_boundary(self):
        """At stride=1 the shifted tail lands on a boundary that is already staged
        (768 -> 512, and 512 was kept as an interior boundary). Both want the same
        window [508, 512), so the alias is a no-op rather than a boundary whose
        bytes get overwritten with a different window -- but it does mean 768 is
        deliberately absent, since under bigram no node ends there."""
        self.assertEqual(
            self._stage_both_paths(extend_len=768, orig=768, stride=1, bigram=False),
            {(7, 256), (7, 512), (7, 768)},
        )
        self.assertEqual(
            self._stage_both_paths(extend_len=768, orig=768, stride=1, bigram=True),
            {(7, 256), (7, 512)},
        )

    def test_unified_ratio_128_and_no_pool_noop(self):
        hp = _fake_host_pool(ring_size=8, slot_bytes=32, num_pages=4)
        be = _fake_backend(hp, page=256, swa_ring=128)
        be.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.zeros((8, 1024), dtype=torch.int64)
        )
        be.token_to_kv_pool.translate_loc_from_full_to_swa = lambda x: x
        fb = types.SimpleNamespace(
            batch_size=1,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[512],
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
        )
        # ratio != 4 -> no-op
        capture_c4_state_windows_unified(
            backend=be,
            state_pool=self._fake_state_pool(torch.zeros((0, 16))),
            kv_score_input=torch.zeros((512, 16), dtype=torch.bfloat16),
            forward_batch=fb,
            is_indexer=False,
            layer_id=0,
            ratio=128,
        )
        self.assertEqual(hp._capture_staging, {})


class TestSwaStateRoute1Durable(unittest.TestCase):
    """Route1: state pool durable/staging split + promote copy. Exercises the
    real (unbound) DeepSeekV4PagedHostPool alloc/free/clear/promote on a minimal
    CPU instance so the allocator never hands out durable rows and promote moves
    a staged tile into the coupled window row."""

    def _state_pool(
        self, *, layers=2, ring=8, slot_bytes=4, durable_pages=16, slack=8, seed=1
    ):
        p = object.__new__(MPH.DeepSeekV4PagedHostPool)
        p.layout = "layer_first"
        p.slot_page_size = ring
        p.layer_num = layers
        p.item_bytes = ring * slot_bytes
        p.dtype = torch.uint8
        p.size = (durable_pages + slack) * ring
        p._durable_reserve_slots = durable_pages * ring
        p.lock = threading.RLock()
        g = torch.Generator().manual_seed(seed)
        p.kv_buffer = [
            torch.randint(
                0,
                256,
                (durable_pages + slack, ring * slot_bytes),
                dtype=torch.uint8,
                generator=g,
            )
            for _ in range(layers)
        ]
        p.data_refs = p.kv_buffer
        p.clear()
        return p

    def test_clear_reserves_durable(self):
        p = self._state_pool(ring=8, durable_pages=16, slack=8)
        self.assertEqual(p.available_size(), 8 * 8)  # only slack pages
        self.assertGreaterEqual(int(p.free_slots.min().item()), 16 * 8)

    def test_alloc_only_hands_out_staging(self):
        p = self._state_pool(ring=8, durable_pages=16, slack=8)
        for _ in range(8):
            idx = p.alloc(8)
            self.assertIsNotNone(idx)
            self.assertGreaterEqual(int(idx.min().item()), 16 * 8)  # never durable
        self.assertIsNone(p.alloc(8))  # slack exhausted

    def test_free_ignores_durable_rows(self):
        p = self._state_pool(ring=8, durable_pages=16, slack=8)
        before = p.available_size()
        p.free(torch.arange(3 * 8, 3 * 8 + 8))  # durable row 3 -> filtered
        self.assertEqual(p.available_size(), before)
        staged = p.alloc(8)
        p.free(staged)  # staging -> returned
        self.assertEqual(p.available_size(), before)

    def test_promote_copies_tile_and_frees_staged(self):
        p = self._state_pool(ring=8, slot_bytes=4, durable_pages=16, slack=8, seed=5)
        staged = p.alloc(8)
        staged_row = int(staged[0].item()) // 8
        want = [p.data_refs[li][staged_row].clone() for li in range(p.layer_num)]
        avail_before = p.available_size()
        p.promote_captured_page(staged, durable_row=3)
        for li in range(p.layer_num):
            self.assertTrue(torch.equal(p.data_refs[li][3], want[li]))
        self.assertEqual(p.available_size(), avail_before + 8)  # staged returned
        self.assertGreaterEqual(
            int(p.free_slots.min().item()), 16 * 8
        )  # still reserved


class TestSwaStateIndependentPoolL3(unittest.TestCase):
    """Independent-pool L3: the state pool serializes/deserializes its DURABLE
    row addressed by the coupled SWA window page (``_l3_page_size`` == swa_ring,
    NOT the state pool's own ring), and no-ops its controller-driven device<->host
    transfer (``_manual_device_ride``; the manual capture/restore ride owns
    L1<->L2)."""

    def _pool(
        self,
        *,
        layers=2,
        ring=8,
        slot_bytes=4,
        durable_pages=16,
        slack=8,
        swa_ring=128,
        seed=11,
    ):
        p = TestSwaStateRoute1Durable()._state_pool(
            layers=layers,
            ring=ring,
            slot_bytes=slot_bytes,
            durable_pages=durable_pages,
            slack=slack,
            seed=seed,
        )
        p._l3_page_size = swa_ring  # L3 addresses durable row by SWA window page
        p._manual_device_ride = True  # controller device transfer is a no-op
        return p

    def test_get_data_page_addresses_durable_row_by_swa_page(self):
        ring, swa_ring, swa_row = 8, 128, 5
        p = self._pool(ring=ring, swa_ring=swa_ring)
        want = torch.stack(
            [p.data_refs[li][swa_row] for li in range(p.layer_num)]
        ).flatten()
        blob = p.get_data_page(swa_row * swa_ring, flat=True)
        self.assertTrue(torch.equal(blob, want))
        # no packing tail: exactly layer_num * item_bytes
        self.assertEqual(blob.numel(), p.layer_num * p.item_bytes)

    def test_set_from_flat_writes_durable_row_by_swa_page(self):
        ring, swa_ring, swa_row = 8, 128, 9
        p = self._pool(ring=ring, swa_ring=swa_ring)
        blob = torch.randint(0, 256, (p.layer_num * p.item_bytes,), dtype=torch.uint8)
        p.set_from_flat_data_page(swa_row * swa_ring, blob)
        got = torch.stack(
            [p.data_refs[li][swa_row] for li in range(p.layer_num)]
        ).flatten()
        self.assertTrue(torch.equal(got, blob))

    def test_l3_roundtrip_byte_exact_across_pools(self):
        ring, swa_ring = 8, 128
        src = self._pool(ring=ring, swa_ring=swa_ring, seed=1)
        dst = self._pool(ring=ring, swa_ring=swa_ring, seed=2)
        rows = (0, 3, 15)
        for swa_row in rows:
            blob = src.get_data_page(swa_row * swa_ring, flat=True)
            dst.set_from_flat_data_page(swa_row * swa_ring, blob)
        for swa_row in rows:
            for li in range(src.layer_num):
                self.assertTrue(
                    torch.equal(src.data_refs[li][swa_row], dst.data_refs[li][swa_row])
                )

    def test_page_buffer_meta_points_at_durable_row(self):
        ring, swa_ring, swa_row = 8, 128, 4
        p = self._pool(ring=ring, swa_ring=swa_ring)
        idx = torch.arange(swa_row * swa_ring, swa_row * swa_ring + swa_ring)
        ptrs, sizes = p.get_page_buffer_meta(idx)
        self.assertEqual(len(ptrs), p.layer_num)
        for li in range(p.layer_num):
            self.assertEqual(ptrs[li], p.kv_buffer[li][swa_row].data_ptr())
        self.assertEqual(sizes[0], p.item_bytes * p.dtype.itemsize)

    def test_manual_device_ride_backup_and_load_are_noops(self):
        p = self._pool()
        before = [buf.clone() for buf in p.kv_buffer]
        # No device_ptrs / device_pool wired: only a working no-op guard survives.
        p.backup_from_device_all_layer(
            None, torch.arange(0, 8), torch.arange(0, 8), "direct"
        )
        p.load_to_device_per_layer(
            None, torch.arange(0, 8), torch.arange(0, 8), 0, "direct"
        )
        for buf, orig in zip(p.kv_buffer, before):
            self.assertTrue(torch.equal(buf, orig))


if __name__ == "__main__":
    unittest.main()
