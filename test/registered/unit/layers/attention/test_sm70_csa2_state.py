"""CPU tests for SM70 CSA2 D3 static buffers (ring, capacity, sources)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.sm70_csa2 import (
    SWA_ROW_BYTES,
    _FLASH_INDEX_SOURCES,
    _FLASH_KV_SOURCES,
    Sm70Csa2State,
    _gather_swa,
    _is_decode,
    _prefill_swa_view,
    _select_topk,
    _store_prefill_packed_swa,
    _use_packed_csa2,
    compressed_capacity,
    dspark_shared_block_attn,
    dspark_shared_block_valid,
    index_source_for,
    kv_source_for,
    pack_swa_window_from_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCompressedCapacity(CustomTestCase):
    def test_flash_context(self):
        self.assertEqual(compressed_capacity(262144, 1, 8), 262152)
        self.assertEqual(compressed_capacity(262144, 2, 8), 131080)

    def test_multiple_of_block_and_min_rows(self):
        for cap in (1, 7, 128, 4096, 262144):
            for ratio in (1, 2):
                n = compressed_capacity(cap, ratio, 8)
                self.assertEqual(n % 8, 0)
                self.assertGreaterEqual(n, (cap + ratio - 1) // ratio + 1)


class TestRingValid(CustomTestCase):
    def _slots(self, pos: int, w: int = 128) -> torch.Tensor:
        st = Sm70Csa2State(sliding_window=w)
        return st.ring_valid(torch.tensor([pos]))[0]

    def test_short_and_full_window(self):
        self.assertEqual(int(self._slots(0).sum()), 1)
        self.assertTrue(bool(self._slots(0)[0]))
        self.assertEqual(int(self._slots(5).sum()), 6)
        self.assertTrue(bool(self._slots(5)[:6].all()))
        self.assertFalse(bool(self._slots(5)[6:].any()))
        for pos in (127, 130, 100000):
            self.assertTrue(bool(self._slots(pos).all()), pos)

    def test_matches_position_residues(self):
        w = 128
        st = Sm70Csa2State(sliding_window=w)
        for pos in (0, 1, 5, 127, 128, 130, 255, 1000):
            got = st.ring_valid(torch.tensor([pos]))[0]
            expect = torch.zeros(w, dtype=torch.bool)
            lo = max(0, pos - w + 1)
            for p in range(lo, pos + 1):
                expect[p % w] = True
            self.assertTrue(torch.equal(got, expect), pos)


class TestPrefillSwaGather(CustomTestCase):
    def test_scratch_allocated_in_inference_mode_stays_writable(self):
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        backend = SimpleNamespace(max_context_len=64)
        st = m.get_state(backend)
        layer = SimpleNamespace(
            layer_id=3, compress_ratio=0, compressor=None, indexer=None
        )
        topo = {
            "sliding_window": 8,
            "candidate_block_size": 8,
            "candidate_topk_blocks": 4,
            "index_topk": 4,
        }
        with torch.inference_mode():
            m._ensure_buffers(st, layer, topo, torch.device("cpu"), backend)
        scratch = st.swa_scratch[3]
        self.assertFalse(scratch.is_inference())
        scratch[:1].copy_(torch.ones(1, scratch.shape[1], dtype=scratch.dtype))
    def test_gather_matches_absolute_rows(self):
        st = Sm70Csa2State(sliding_window=128)
        st.swa_ring[0] = torch.zeros(128, 4, dtype=torch.int32)
        pos0, t = 180, 12
        for p in range(pos0):
            st.swa_ring[0][p % 128] = p
        packed = torch.arange(pos0, pos0 + t, dtype=torch.int32)[:, None].expand(t, 4)
        buf, origin = _prefill_swa_view(st, 0, pos0, packed)
        self.assertEqual(origin, pos0 - 128)
        positions = torch.arange(pos0, pos0 + t, dtype=torch.int64)
        gathered, valid = _gather_swa(st, buf, origin, positions)
        w = 128
        for i, pos in enumerate(range(pos0, pos0 + t)):
            self.assertTrue(bool(valid[i].all()))
            for s in range(w):
                abs_p = pos - (w - 1) + s
                self.assertEqual(int(gathered[i, s, 0]), abs_p)


class TestSelectTopkFallback(CustomTestCase):
    def test_k4_masks_inf_and_tail(self):
        scores = torch.tensor(
            [[3.0, -float("inf"), 1.0, 4.0, -float("inf"), 9.0]], dtype=torch.float32
        )
        lens = torch.tensor([5], dtype=torch.int32)
        idx = _select_topk(scores, lens, 4)
        self.assertEqual(set(idx[0].tolist()), {3, 0, 2, -1})


class TestFlashSources(CustomTestCase):
    def test_kv_and_index_owners(self):
        kv, ix = _FLASH_KV_SOURCES, _FLASH_INDEX_SOURCES
        self.assertEqual(kv_source_for(5, 2, kv), 2)
        self.assertEqual(kv_source_for(19, 2, kv), 14)
        self.assertEqual(index_source_for(19, 2, ix), 14)
        self.assertEqual(kv_source_for(20, 1, kv), 20)
        self.assertEqual(index_source_for(20, 1, ix), 20)
        self.assertEqual(kv_source_for(27, 1, kv), 20)
        self.assertEqual(index_source_for(27, 1, ix), 24)
        self.assertEqual(kv_source_for(39, 1, kv), 20)
        self.assertEqual(index_source_for(39, 1, ix), 36)


class TestDsparkSwaRingCommit(CustomTestCase):
    def test_commit_ring_writes_chunk_slots(self):
        st = Sm70Csa2State(sliding_window=8)
        st.swa_ring[0] = torch.full((8, 4), 99, dtype=torch.int32)
        packed = torch.arange(5, dtype=torch.int32)[:, None].expand(5, 4).contiguous()
        positions = torch.arange(10, 15)
        _store_prefill_packed_swa(st, 0, 10, positions, packed, commit_ring=True)
        for i, pos in enumerate(range(10, 15)):
            self.assertEqual(int(st.swa_ring[0][pos % 8, 0]), i)
        buf, origin = st.prefill_swa[0]
        self.assertEqual(origin, 2)
        self.assertEqual(int(buf.shape[0]), 13)

    def test_skip_ring_keeps_target_prefix(self):
        st = Sm70Csa2State(sliding_window=8)
        st.swa_ring[0] = torch.arange(8, dtype=torch.int32)[:, None].expand(8, 4).contiguous()
        packed = torch.full((5, 4), 7, dtype=torch.int32)
        positions = torch.arange(10, 15)
        _store_prefill_packed_swa(st, 0, 10, positions, packed, commit_ring=False)
        self.assertTrue(torch.equal(st.swa_ring[0][:, 0], torch.arange(8, dtype=torch.int32)))
        buf, _ = st.prefill_swa[0]
        self.assertTrue(torch.equal(buf[-5:], packed))


class TestPackSwaWindowFromKv(CustomTestCase):
    def test_mask_skips_uncommitted_and_keeps_last_window(self):
        w = 8
        backend = SimpleNamespace(max_context_len=64)
        layer = SimpleNamespace(
            layer_id=0,
            sliding_window=w,
            compress_ratio=0,
            compressor=None,
            indexer=None,
            eps=1e-6,
            qk_rope_head_dim=4,
            kv_norm=SimpleNamespace(weight=torch.ones(8)),
            freqs_cis=torch.ones(32, 4),
        )
        kv = torch.ones(10, 8)
        positions = torch.arange(10)
        mask = torch.tensor(
            [False, True, True, True, False, True, True, True, True, False]
        )

        def fake_pack(x, freqs, rope_dim):
            n = x.shape[0]
            out = torch.zeros(n, SWA_ROW_BYTES, dtype=torch.uint8)
            out[:, 0] = torch.arange(n, dtype=torch.uint8)
            return out

        with patch(
            "sglang.srt.layers.attention.dsv4.sm70_csa2.pack_swa_fp8",
            side_effect=fake_pack,
        ):
            pack_swa_window_from_kv(backend, layer, kv, positions, mask=mask)

        ring = backend._sm70_csa2.swa_ring[0]
        # last W=8 of the 10-token inject: positions 2..9, packed rows 0..7
        # mask[-8:] = T,T,F,T,T,T,T,F → skip pos 4 and 9
        self.assertEqual(int(ring[2, 0]), 0)
        self.assertEqual(int(ring[3, 0]), 1)
        self.assertEqual(int(ring[4, 0]), 0)
        self.assertEqual(int(ring[5, 0]), 3)
        self.assertEqual(int(ring[6, 0]), 4)
        self.assertEqual(int(ring[7, 0]), 5)
        self.assertEqual(int(ring[0, 0]), 6)  # pos 8
        self.assertEqual(int(ring[1, 0]), 0)  # pos 9 masked


class TestDsparkSharedBlockValid(CustomTestCase):
    def test_every_query_sees_full_window_plus_draft_chunk(self):
        st = Sm70Csa2State(sliding_window=8)
        st.swa_ring[0] = torch.arange(8, dtype=torch.int32)[:, None].expand(8, 4).contiguous()
        pos0, t = 10, 5
        packed = torch.arange(100, 105, dtype=torch.int32)[:, None].expand(t, 4).contiguous()
        buf, origin = _prefill_swa_view(st, 0, pos0, packed)
        positions = torch.arange(pos0, pos0 + t, dtype=torch.int64)
        shared_valid = dspark_shared_block_valid(buf, origin)
        causal_rows, causal_valid = _gather_swa(st, buf, origin, positions)

        self.assertEqual(int(shared_valid.shape[0]), int(buf.shape[0]))
        self.assertEqual(int(shared_valid.sum()), int(buf.shape[0]) - max(0, -origin))
        # Query 0's causal window cannot see later MASK KV in the chunk.
        self.assertLess(int(causal_valid[0].sum()), int(shared_valid.sum()))
        self.assertFalse(torch.equal(causal_rows[0], causal_rows[-1]))
        # Shared set is identical for every query: full cat(window, draft).
        self.assertTrue(bool(shared_valid[-t:].all()))

    def test_short_prefix_masks_negative_ring_slots(self):
        st = Sm70Csa2State(sliding_window=8)
        st.swa_ring[0] = torch.zeros(8, 4, dtype=torch.int32)
        packed = torch.ones(3, 4, dtype=torch.int32)
        buf, origin = _prefill_swa_view(st, 0, 2, packed)
        valid = dspark_shared_block_valid(buf, origin)
        abs_pos = origin + torch.arange(buf.shape[0])
        self.assertTrue(torch.equal(valid, abs_pos >= 0))
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[-1]))

    def test_sparse_reads_flag_from_mqa_state_not_just_radix(self):
        st = Sm70Csa2State(sliding_window=8)
        radix = SimpleNamespace()
        mqa = SimpleNamespace(sm70_swa_ring_from_inject=True)
        st.layers[3] = mqa
        self.assertTrue(dspark_shared_block_attn(radix, st, 3))
        self.assertFalse(dspark_shared_block_attn(radix, st, 0))
        radix.sm70_swa_ring_from_inject = True
        self.assertTrue(dspark_shared_block_attn(radix, st, 0))


def _mode(*, decode=False, target_verify=False):
    return SimpleNamespace(
        is_decode=lambda: decode,
        is_target_verify=lambda: target_verify,
    )


class TestPackedCsa2Routing(CustomTestCase):
    def test_decode_t1_uses_packed_kernels(self):
        fb = SimpleNamespace(forward_mode=_mode(decode=True))
        self.assertTrue(_is_decode(fb, 1))
        self.assertTrue(_use_packed_csa2(fb, 1))
        self.assertFalse(_is_decode(fb, 6))
        self.assertFalse(_use_packed_csa2(fb, 6))

    def test_target_verify_uses_packed_kernels_at_t6(self):
        fb = SimpleNamespace(forward_mode=_mode(target_verify=True))
        self.assertFalse(_is_decode(fb, 6))
        self.assertTrue(_use_packed_csa2(fb, 6))
        self.assertTrue(_use_packed_csa2(fb, 1))

    def test_chunked_prefill_stays_on_oracle(self):
        fb = SimpleNamespace(forward_mode=_mode())
        self.assertFalse(_use_packed_csa2(fb, 64))
        self.assertFalse(_is_decode(fb, 1))

    def test_verify_low_ratio_defers_ring_write(self):
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        backend = SimpleNamespace(max_context_len=64)
        st = m.get_state(backend)
        st.swa_ring[0] = torch.arange(8, dtype=torch.int32)[:, None].expand(8, 4).contiguous()
        before = st.swa_ring[0].clone()
        layer = SimpleNamespace(
            layer_id=0, compress_ratio=0, compressor=None, indexer=None
        )
        x = torch.ones(6, 8)
        positions = torch.arange(10, 16)
        with patch.object(m, "_project_swa_kv", return_value=x):
            m._verify_low_ratio_sources(
                backend, layer, x, None, positions, {"index_topk": 4}, st
            )
        self.assertTrue(torch.equal(st.swa_ring[0], before))
        self.assertEqual(tuple(st.verify_kv[0].shape), (6, 8))

    def test_verify_saves_and_restores_cand_ids_per_token(self):
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        backend = SimpleNamespace(max_context_len=64)
        st = m.get_state(backend)
        st.cand_ids = torch.full((1, 4), -1, dtype=torch.int32)
        st.candidate_topk_blocks = 4
        x = torch.ones(2, 4)
        positions = torch.arange(2)

        def write_cand(*_a, **_k):
            st.cand_ids.copy_(torch.tensor([[9, 8, 7, 6]], dtype=torch.int32))
            st.topk[20] = torch.zeros(1, 2, dtype=torch.int32)

        src = SimpleNamespace(
            layer_id=20,
            compress_ratio=1,
            compressor=None,
            indexer=SimpleNamespace(
                is_candidate_source=True, uses_candidates=False, index_topk=2
            ),
        )
        with patch.object(m, "_project_swa_kv", return_value=x), patch.object(
            m, "_decode_compress_and_index", side_effect=write_cand
        ):
            m._verify_low_ratio_sources(
                backend, src, x, None, positions, {"index_topk": 2}, st
            )
        self.assertEqual(st.prefill_cand_ids.tolist(), [[9, 8, 7, 6], [9, 8, 7, 6]])

        seen = []

        def read_cand(*_a, **_k):
            seen.append(st.cand_ids[0].tolist())
            st.topk[21] = torch.zeros(1, 2, dtype=torch.int32)

        later = SimpleNamespace(
            layer_id=21,
            compress_ratio=1,
            compressor=None,
            indexer=SimpleNamespace(
                is_candidate_source=False, uses_candidates=True, index_topk=2
            ),
        )
        st.prefill_cand_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
        with patch.object(m, "_project_swa_kv", return_value=x), patch.object(
            m, "_decode_compress_and_index", side_effect=read_cand
        ):
            m._verify_low_ratio_sources(
                backend, later, x, None, positions, {"index_topk": 2}, st
            )
        self.assertEqual(seen, [[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_verify_sparse_packs_with_mqa_not_radix(self):
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        st = Sm70Csa2State()
        st.verify_kv[0] = torch.ones(2, 4)
        st.swa_ring[0] = torch.zeros(8, 4)
        st.layers[0] = SimpleNamespace(name="mqa", qk_rope_head_dim=64)
        packed_with = []

        def fake_pack(layer, *_a, **_k):
            packed_with.append(getattr(layer, "name", "?"))
            return torch.zeros(2, 4, dtype=torch.uint8)

        q = torch.zeros(2, 2, 4)
        positions = torch.arange(2)
        sink = torch.zeros(2)
        radix = SimpleNamespace(name="radix", layer_id=0, qk_head_dim=512)
        dummy_rows = torch.zeros(2, 8, m.SWA_ROW_BYTES, dtype=torch.uint8)
        dummy_valid = torch.ones(2, 8, dtype=torch.bool)
        before = st.swa_ring[0].clone()
        with patch.object(m, "_pack_verify_swa", side_effect=fake_pack), patch.object(
            m, "_gather_verify_swa", return_value=(dummy_rows, dummy_valid)
        ), patch.object(
            m,
            "unpack_swa_fp8_ue8m0",
            return_value=torch.zeros(2, 8, 512),
        ), patch.object(
            m, "_sparse_attention_rows_aligned", return_value=torch.zeros(2, 2, 4)
        ):
            out = m._verify_sparse(
                q, radix, st, 0, positions, None, None, sink, 1.0, 2
            )
        self.assertEqual(packed_with, ["mqa"])
        self.assertEqual(tuple(out.shape), (2, 2, 4))
        self.assertNotIn(0, st.verify_kv)
        self.assertTrue(torch.equal(st.swa_ring[0], before))

    def test_verify_gather_keeps_committed_window(self):
        """A draft at pos+128 used to overwrite a live ring slot still inside
        the query window. Verify reads committed history plus this block, and
        leaves that slot alone.
        """
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        w = 8
        st = Sm70Csa2State(sliding_window=w)
        st.swa_ring[0] = torch.zeros(w, 1, dtype=torch.int32)
        for p in range(16):
            st.swa_ring[0][p % w, 0] = p
        packed = torch.tensor([[16], [17], [18]], dtype=torch.int32)
        positions = torch.tensor([16, 17, 18])
        rows, valid = m._gather_verify_swa(st, 0, packed, positions)
        self.assertEqual(rows[0, :, 0].tolist(), list(range(9, 17)))
        self.assertTrue(bool(valid[0].all()))
        self.assertEqual(int(st.swa_ring[0][1, 0]), 9)

    def test_commit_keeps_only_the_accepted_prefix(self):
        """commit_lens is the anchor plus accepted drafts. The unaccepted tail
        must not stay in the ring, the ratio-2 pending slot, or a compressed
        pair that included a rejected partner.
        """
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        w = 8
        st = Sm70Csa2State(sliding_window=w)
        st.swa_ring[0] = torch.full((w, 1), 50, dtype=torch.int32)
        st.verify_swa_packed[0] = torch.tensor([[1], [2], [3], [4]], dtype=torch.int32)
        st.verify_positions = torch.tensor([6, 7, 8, 9])
        st.verify_open = torch.ones(1, dtype=torch.int32)
        st.verify_t = torch.tensor([4], dtype=torch.int32)
        st.pending_kv[3] = torch.tensor([9.0])
        st.pending_score[3] = torch.tensor([8.0])
        st.verify_pending_kv_traj[3] = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        st.verify_pending_score_traj[3] = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        # Steps 0 and 1 share row 5 (even then odd). Steps 2 and 3 share row 6.
        table = torch.tensor([[0], [0], [0], [0], [0], [2], [4]], dtype=torch.uint8)
        st.kv_rows[3] = table
        st.verify_row[3] = torch.tensor([5, 5, 6, 6])
        st.verify_kv_orig[3] = torch.tensor([[7], [1], [8], [3]], dtype=torch.uint8)
        backend = SimpleNamespace(_sm70_csa2=st)

        m.sm70_commit_target_verify(backend, torch.tensor([1]), num_positions=4)
        self.assertEqual(int(st.swa_ring[0][6, 0]), 1)
        self.assertEqual(int(st.swa_ring[0][7, 0]), 50)
        self.assertEqual(int(st.swa_ring[0][0, 0]), 50)
        self.assertEqual(int(st.swa_ring[0][1, 0]), 50)
        self.assertEqual(st.pending_kv[3].tolist(), [10.0])
        self.assertEqual(st.pending_score[3].tolist(), [1.0])
        self.assertEqual(int(table[5, 0]), 1)
        self.assertEqual(int(table[6, 0]), 8)
        self.assertEqual(int(st.verify_open[0]), 0)

        st.pending_kv[3].fill_(999)
        m.sm70_commit_target_verify(backend, torch.tensor([4]), num_positions=4)
        self.assertEqual(st.pending_kv[3].tolist(), [999.0])

    def test_verify_pending_does_not_touch_the_live_slot(self):
        """The ratio-2 recurrence during verify runs on a scratch cursor.
        The live slot stays on the last committed token until commit.
        """
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        st = Sm70Csa2State()
        st.pending_kv[0] = torch.tensor([1.0])
        st.pending_score[0] = torch.tensor([2.0])
        prev_kv, prev_score = m._verify_pending_prev(st, 0, 0)
        self.assertEqual(prev_kv.tolist(), [1.0])
        self.assertEqual(prev_score.tolist(), [2.0])
        m._verify_pending_save(
            st, 0, 0, torch.tensor([[3.0]]), torch.tensor([[4.0]])
        )
        self.assertEqual(st.pending_kv[0].tolist(), [1.0])
        self.assertEqual(st.pending_score[0].tolist(), [2.0])
        self.assertEqual(st.verify_pending_kv_traj[0][0].tolist(), [3.0])
        prev_kv, _prev_score = m._verify_pending_prev(st, 0, 1)
        self.assertEqual(prev_kv.tolist(), [3.0])
        self.assertEqual(st.pending_kv[0].tolist(), [1.0])

    def test_pack_verify_swa_leaves_the_live_ring(self):
        from sglang.srt.layers.attention.dsv4 import sm70_csa2 as m

        st = Sm70Csa2State(sliding_window=8)
        st.swa_ring[0] = torch.full((8, m.SWA_ROW_BYTES), 7, dtype=torch.uint8)
        layer = SimpleNamespace(qk_rope_head_dim=64, layer_id=0)
        kv = torch.zeros(3, 4)
        positions = torch.tensor([6, 7, 8])
        marker = torch.arange(3, dtype=torch.uint8)[:, None].expand(3, m.SWA_ROW_BYTES).contiguous()

        def write_stage(dst, _kv, pos, **_k):
            slots = torch.remainder(pos.to(dtype=torch.int64), dst.shape[0])
            for i, slot in enumerate(slots.tolist()):
                dst[int(slot)].fill_(i + 1)

        with patch.object(m, "_attn_glue_on", return_value=True), patch.object(
            m, "_freqs_table", return_value=torch.zeros(1)
        ), patch.object(m, "pack_swa_fp8_at", side_effect=write_stage):
            packed = m._pack_verify_swa(layer, kv, positions, st, 0)
        self.assertTrue(torch.equal(st.swa_ring[0], torch.full_like(st.swa_ring[0], 7)))
        self.assertEqual(packed[:, 0].tolist(), [1, 2, 3])

