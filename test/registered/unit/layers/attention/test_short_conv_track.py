"""Radix mamba-cache track plumbing for ``ShortConvAttnBackend``.

The extend-side snapshot reads a window of conv inputs ending at the
chunk-aligned track position, and the decode-side snapshot copies every layer's
state into its track slot in one launch. Both are indexed arithmetic over the
pool's flattened ``[n_layers * n_slots, ...]`` view, where an off-by-one shows
up as a prefix hit that resumes from the wrong conv window rather than as an
error. These cases pin that arithmetic without a model or a live ModelRunner.
"""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _MockLayerCache:
    conv: List[torch.Tensor]
    temporal: torch.Tensor


def _torch_track_reference(
    state_a,
    state_b,
    src_rows,
    mask_rows,
    dst_rows,
    total_rows,
    check_freed_slots=False,
):
    """CPU stand-in for ``track_mamba_states_if_needed`` (Triton, CUDA only).

    Same contract: for every row ``i < total_rows`` whose mask is set, copy
    ``state[src_rows[i]] -> state[dst_rows[i]]`` in BOTH state tensors, and
    skip rows whose src/dst is negative when ``check_freed_slots``.
    """
    for i in range(total_rows):
        if not bool(mask_rows[i]):
            continue
        src = int(src_rows[i])
        dst = int(dst_rows[i])
        if check_freed_slots and (src < 0 or dst < 0):
            continue
        for state in (state_a, state_b):
            if state[0].numel() == 0:
                continue
            state[dst] = state[src].clone()


class _TrackHarness:
    """Bare ``ShortConvAttnBackend`` wired for the track paths, CPU only.

    ``object.__new__`` keeps this free of a live ModelRunner: the track code
    reads only the attributes set here plus ``mamba_cache_chunk_size`` off the
    server args, which the tests patch on the module.
    """

    def __init__(
        self,
        *,
        num_layers=2,
        num_slots=6,
        num_channels=3,
        hidden_size=4,
        windows=(2, 1),
        extra_buffer=True,
        chunk_size=8,
        track_interval=16,
    ):
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        conv = [
            torch.zeros(num_layers, num_slots, ch, w)
            for ch, w in zip((num_channels, hidden_size), windows)
        ]
        temporal = torch.zeros(num_layers, num_slots, 1, 1, 0)
        self.mamba_cache = SimpleNamespace(conv=conv, temporal=temporal)
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.server_args = SimpleNamespace(
            mamba_cache_chunk_size=chunk_size,
            mamba_track_interval=track_interval,
            speculative_algorithm=None,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = torch.device("cpu")
        backend.enable_unified_memory = False
        backend.conv_states_shape = conv[0].shape
        backend.conv_window_lens = [int(c.shape[-1]) for c in conv]
        backend._cache_indices = None
        backend._cache_indices_buf = None
        backend.forward_metadata = None
        backend._track_conv_indices = None
        backend._track_dst = None
        backend._track_layer_row_base = None
        backend._track_pairs = None
        backend.enable_mamba_extra_buffer = extra_buffer
        if extra_buffer:
            backend._init_track_state(self.server_args, self.mamba_cache)
        self.backend = backend

    def layer_cache(self, layer_idx):
        return _MockLayerCache(
            conv=[c[layer_idx] for c in self.mamba_cache.conv],
            temporal=self.mamba_cache.temporal[layer_idx],
        )


def _track_forward_batch(*, extend_seq_lens, prefix_lens, track_mask, track_indices):
    """Extend ForwardBatch stub carrying exactly the track inputs.

    ``mamba_track_seqlens`` mirrors what the scheduler builds in
    ``_mamba_radix_cache_v2_req_prepare_for_extend``: prefix + extend length
    for tracked rows, -1 for untracked ones.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    seqlens = [
        (p + e) if m else -1
        for p, e, m in zip(prefix_lens, extend_seq_lens, track_mask)
    ]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        mamba_track_seqlens=torch.tensor(seqlens, dtype=torch.int64),
        mamba_track_mask=torch.tensor(track_mask, dtype=torch.bool),
        mamba_track_indices=torch.tensor(track_indices, dtype=torch.int64),
    )


def _query_start_loc(extend_seq_lens):
    out = [0]
    for s in extend_seq_lens:
        out.append(out[-1] + s)
    return torch.tensor(out, dtype=torch.int64)


class TestShortConvTrackIndices(CustomTestCase):
    """``_init_track_conv_indices``: where the extend snapshot reads from.

    The checkpoint is keyed on ``mamba_last_track_seqlen = prefix +
    floor(extend_len / chunk) * chunk``, NOT on the end of the extend, and a
    conv's state at length L is its last ``window`` input rows ending at L. So
    the gather is at ``[qsl_i + aligned - window, qsl_i + aligned)``; off by one
    and every prefix hit resumes from a shifted window.
    """

    def _indices(self, harness, forward_batch, extend_seq_lens):
        from sglang.srt.layers.attention.linear import short_conv_backend

        with unittest.mock.patch.object(
            short_conv_backend,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            return harness.backend._init_track_conv_indices(
                _query_start_loc(extend_seq_lens), forward_batch
            )

    def test_indices_land_on_the_chunk_boundary(self):
        # chunk == 8. Request 0: 20 fresh tokens -> aligned 16, window [14, 16).
        # Request 1: 10 tokens on a 32-token prefix -> aligned 8 within the
        # extend, i.e. flattened [20 + 6, 20 + 8).
        extend = [20, 10]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 32],
            track_mask=[True, True],
            track_indices=[4, 5],
        )
        conv_idx, lag_idx = self._indices(harness, fb, extend)

        self.assertEqual(conv_idx.tolist(), [[14, 15], [26, 27]])
        # The one-token lag entry ends on the same column as the conv window.
        self.assertEqual(lag_idx.tolist(), [[15], [27]])

    def test_untracked_rows_are_dropped(self):
        # mamba_track_mask is False for an extend shorter than one chunk; that
        # row must not appear in the gather at all (its ping-pong slot is not
        # the one the radix cache will read).
        extend = [16, 3]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 0],
            track_mask=[True, False],
            track_indices=[4, 5],
        )
        conv_idx, lag_idx = self._indices(harness, fb, extend)
        self.assertEqual(list(conv_idx.shape), [1, 2])
        self.assertEqual(conv_idx.tolist(), [[14, 15]])
        self.assertEqual(lag_idx.tolist(), [[15]])

    def test_every_entry_shares_the_end_column(self):
        # Both ZAYA1 conv entries snapshot the SAME sequence position; only the
        # window depth differs. If they ever diverge, conv[0] and conv[1] would
        # describe different prefix lengths in one cached node.
        extend = [24]
        harness = _TrackHarness(chunk_size=8, windows=(4, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[3],
        )
        entries = self._indices(harness, fb, extend)
        ends = {int(e[0, -1]) for e in entries}
        self.assertEqual(ends, {23})

    def test_matches_the_single_conv_base_implementation(self):
        # The override must be a faithful generalization: for a model with one
        # conv entry it has to reproduce MambaAttnBackendBase byte-for-byte,
        # or GDN/Mamba2 semantics would have quietly forked.
        import sglang.srt.layers.attention.hybrid_linear_attn_backend as hb
        from sglang.srt.layers.attention.linear import short_conv_backend

        extend = [20, 10]
        harness = _TrackHarness(chunk_size=8, windows=(3,))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 32],
            track_mask=[True, True],
            track_indices=[4, 5],
        )
        qsl = _query_start_loc(extend)
        with unittest.mock.patch.object(
            short_conv_backend,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            mine = harness.backend._init_track_conv_indices(qsl, fb)[0]
        # Both modules bind the derived accessor by name, so patch it in each.
        with unittest.mock.patch.object(
            hb,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            theirs = hb.MambaAttnBackendBase._init_track_conv_indices(
                harness.backend, qsl, fb
            )
        self.assertTrue(torch.equal(mine, theirs))


class TestShortConvTrackExtendSnapshot(CustomTestCase):
    """``track_conv_states_extend``: what actually lands in the track slot.

    ZAYA1 keeps TWO conv entries -- ``conv[0]`` is the conv_qk left padding
    (window == total_padding, over ``qk``) and ``conv[1]`` is the one-token
    ``prev_hs`` lag (window == 1, over ``hidden_states``). Both must be
    snapshotted, from their own input tensor, or a prefix hit restores half a
    state.
    """

    def _prepare(self, harness, fb, extend):
        from sglang.srt.layers.attention.linear import short_conv_backend

        with unittest.mock.patch.object(
            short_conv_backend,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            indices = harness.backend._init_track_conv_indices(
                _query_start_loc(extend), fb
            )
        harness.backend._track_conv_indices = indices
        harness.backend._track_dst = fb.mamba_track_indices[fb.mamba_track_mask]

    def test_both_conv_entries_are_snapshotted(self):
        extend = [20]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[4],
        )
        self._prepare(harness, fb, extend)

        torch.manual_seed(0)
        qk = torch.randn(20, 3)
        hs = torch.randn(20, 4)
        layer_cache = harness.layer_cache(1)
        harness.backend.track_conv_states_extend(tuple(layer_cache.conv), (qk, hs))

        # conv[0] slot 4 == qk rows [14, 16) laid out channel-major.
        self.assertTrue(
            torch.allclose(layer_cache.conv[0][4], qk[14:16].transpose(0, 1))
        )
        # conv[1] slot 4 == the single hidden_states row at the aligned point.
        self.assertTrue(torch.allclose(layer_cache.conv[1][4], hs[15].unsqueeze(-1)))
        # Nothing else moved, and only this layer was touched.
        self.assertEqual(float(layer_cache.conv[0][3].abs().sum()), 0.0)
        self.assertEqual(float(harness.mamba_cache.conv[0][0].abs().sum()), 0.0)

    def test_resumed_prefix_reads_only_this_extends_tokens(self):
        # A request resuming a 32-token cached prefix contributes 10 new
        # tokens; the snapshot must sit at prefix + 8 and gather from THIS
        # request's slice of the flattened batch, never from the neighbour's.
        extend = [20, 10]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 32],
            track_mask=[False, True],
            track_indices=[4, 5],
        )
        self._prepare(harness, fb, extend)

        torch.manual_seed(1)
        qk = torch.randn(30, 3)
        hs = torch.randn(30, 4)
        layer_cache = harness.layer_cache(0)
        harness.backend.track_conv_states_extend(tuple(layer_cache.conv), (qk, hs))

        self.assertTrue(
            torch.allclose(layer_cache.conv[0][5], qk[26:28].transpose(0, 1))
        )
        self.assertTrue(torch.allclose(layer_cache.conv[1][5], hs[27].unsqueeze(-1)))
        # The untracked row's ping-pong slot stays untouched.
        self.assertEqual(float(layer_cache.conv[0][4].abs().sum()), 0.0)

    def test_snapshot_equals_the_state_after_the_aligned_prefix(self):
        # The invariant a prefix hit relies on: the tracked state is what the
        # conv would hold after exactly `mamba_last_track_seqlen` tokens, i.e.
        # that many rows' worth of history, not the end-of-extend state.
        extend = [20]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[4],
        )
        self._prepare(harness, fb, extend)

        torch.manual_seed(2)
        qk = torch.randn(20, 3)
        hs = torch.randn(20, 4)
        layer_cache = harness.layer_cache(0)
        harness.backend.track_conv_states_extend(tuple(layer_cache.conv), (qk, hs))

        aligned = 16
        self.assertTrue(
            torch.allclose(
                layer_cache.conv[0][4], qk[aligned - 2 : aligned].transpose(0, 1)
            )
        )
        self.assertTrue(
            torch.allclose(
                layer_cache.conv[1][4], hs[aligned - 1 : aligned].transpose(0, 1)
            )
        )
        # ... and NOT the end-of-extend state, which is what a plain row copy
        # of the live slot would have given.
        self.assertFalse(
            torch.allclose(layer_cache.conv[0][4], qk[18:20].transpose(0, 1))
        )

    def test_no_track_this_step_is_a_no_op(self):
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        layer_cache = harness.layer_cache(0)
        harness.backend.track_conv_states_extend(
            tuple(layer_cache.conv), (torch.randn(4, 3), torch.randn(4, 4))
        )
        self.assertEqual(float(harness.mamba_cache.conv[0].abs().sum()), 0.0)
        self.assertEqual(float(harness.mamba_cache.conv[1].abs().sum()), 0.0)


class TestShortConvTrackDecode(CustomTestCase):
    """``track_conv_states_decode``: the all-layers row copy.

    Exercised against a torch stand-in for the Triton scatter (the real kernel
    is CUDA-only); what is under test here is the row addressing into the
    flattened ``[n_layers * n_slots, ...]`` pool view and the mask, which is
    where a two-conv-entry model can go wrong.
    """

    @staticmethod
    def _decode_batch(mask):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        return SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            mamba_track_mask=torch.tensor(mask, dtype=torch.bool),
        )

    @staticmethod
    def _run(harness, forward_batch, cache_indices, track_indices, record=None):
        from sglang.srt.layers.attention.linear import short_conv_backend

        backend = harness.backend
        backend._cache_indices = torch.tensor(cache_indices, dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor(track_indices, dtype=torch.int64),
            # The unmutated slot ids (pre-clamp), which the unified-memory
            # branch reads to spot freed-slot tombstones.
            mamba_cache_indices=torch.tensor(cache_indices, dtype=torch.int64),
        )

        def _spy(*args, **kwargs):
            if record is not None:
                record.append((args, kwargs))
            return _torch_track_reference(*args, **kwargs)

        with unittest.mock.patch.object(
            short_conv_backend, "track_mamba_states_if_needed", _spy
        ):
            backend.track_conv_states_decode(forward_batch)

    def test_copies_every_layer_for_both_conv_entries(self):
        harness = _TrackHarness(num_layers=3, num_slots=6, windows=(2, 1))
        conv0, conv1 = harness.mamba_cache.conv
        torch.manual_seed(3)
        conv0.normal_()
        conv1.normal_()
        live = [1, 2]
        track = [4, 5]
        before0 = conv0.clone()
        before1 = conv1.clone()

        self._run(harness, self._decode_batch([True, True]), live, track)

        for layer in range(3):
            for src, dst in zip(live, track):
                self.assertTrue(torch.equal(conv0[layer, dst], before0[layer, src]))
                self.assertTrue(torch.equal(conv1[layer, dst], before1[layer, src]))
                # The live slot itself is left alone.
                self.assertTrue(torch.equal(conv0[layer, src], before0[layer, src]))

    def test_masked_rows_are_not_copied(self):
        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        conv0, _ = harness.mamba_cache.conv
        torch.manual_seed(4)
        conv0.normal_()
        before0 = conv0.clone()

        self._run(harness, self._decode_batch([True, False]), [1, 2], [4, 5])

        for layer in range(2):
            self.assertTrue(torch.equal(conv0[layer, 4], before0[layer, 1]))
            # Row 1 is untracked this step: its ping-pong slot must not move.
            self.assertTrue(torch.equal(conv0[layer, 5], before0[layer, 5]))

    def test_launch_happens_even_with_nothing_to_track(self):
        # The cuda-graph inert-buffer contract. Capture runs with an all-False
        # mask buffer; if the launch were skipped then, the replayed graph
        # would never contain the scatter and every snapshot for the life of
        # that graph would be silently lost.
        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        record = []
        self._run(
            harness,
            self._decode_batch([False, False]),
            [1, 2],
            [0, 0],
            record=record,
        )
        self.assertEqual(len(record), 1)
        args, _ = record[0]
        # Both conv entries ride one launch, over 2 layers x 2 rows.
        self.assertEqual(args[5], 4)
        self.assertEqual(float(harness.mamba_cache.conv[0].abs().sum()), 0.0)

    def test_row_ids_are_layer_major(self):
        harness = _TrackHarness(num_layers=3, num_slots=6, windows=(2, 1))
        record = []
        self._run(
            harness, self._decode_batch([True, True]), [1, 2], [4, 5], record=record
        )
        args, _ = record[0]
        src_rows, mask_rows, dst_rows, total = args[2:6]
        self.assertEqual(src_rows.tolist(), [1, 2, 7, 8, 13, 14])
        self.assertEqual(dst_rows.tolist(), [4, 5, 10, 11, 16, 17])
        self.assertEqual(mask_rows.tolist(), [True] * 6)
        self.assertEqual(total, 6)

    def test_freed_slot_tombstone_is_masked_off(self):
        # The unified pool's virtual->physical translate emits -1 for a freed
        # slot. `layer_base + -1` is a VALID row of the previous layer, so the
        # kernel's own negative-index check cannot save us; the row has to be
        # masked out before the base is added.
        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        harness.backend.enable_unified_memory = True
        record = []
        self._run(
            harness, self._decode_batch([True, True]), [1, 2], [4, -1], record=record
        )
        args, _ = record[0]
        self.assertEqual(args[3].tolist(), [True, False, True, False])

    def test_freed_source_slot_is_masked_off(self):
        # _cache_indices has already clamped its -1s to the scratch slot, so
        # the source tombstone has to be read off the untouched metadata
        # tensor or the snapshot would copy slot 0's garbage.
        from sglang.srt.layers.attention.linear import short_conv_backend

        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        harness.backend.enable_unified_memory = True
        backend = harness.backend
        backend._cache_indices = torch.tensor([0, 2], dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor([4, 5], dtype=torch.int64),
            mamba_cache_indices=torch.tensor([-1, 2], dtype=torch.int64),
        )
        record = []
        with unittest.mock.patch.object(
            short_conv_backend,
            "track_mamba_states_if_needed",
            lambda *a, **k: record.append(a),
        ):
            backend.track_conv_states_decode(self._decode_batch([True, True]))
        self.assertEqual(record[0][3].tolist(), [False, True, False, True])

    def test_extend_mode_does_not_take_the_decode_path(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        record = []
        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mamba_track_mask=torch.tensor([True], dtype=torch.bool),
        )
        self._run(harness, fb, [1], [4], record=record)
        self.assertEqual(record, [])

    @unittest.skipUnless(
        torch.cuda.is_available(), "the real track scatter is a Triton CUDA kernel"
    )
    def test_real_triton_scatter_matches_the_torch_reference(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        harness.mamba_cache = SimpleNamespace(
            conv=[c.cuda() for c in harness.mamba_cache.conv],
            temporal=harness.mamba_cache.temporal.cuda(),
        )
        harness.backend.device = torch.device("cuda")
        harness.backend._init_track_state(harness.server_args, harness.mamba_cache)
        conv0, conv1 = harness.mamba_cache.conv
        torch.manual_seed(5)
        conv0.normal_()
        conv1.normal_()
        expected0 = conv0.clone()
        expected1 = conv1.clone()
        for layer in range(2):
            expected0[layer, 4] = conv0[layer, 1]
            expected1[layer, 4] = conv1[layer, 1]

        backend = harness.backend
        backend._cache_indices = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor([4, 5], dtype=torch.int64, device="cuda")
        )
        backend.track_conv_states_decode(
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                mamba_track_mask=torch.tensor(
                    [True, False], dtype=torch.bool, device="cuda"
                ),
            )
        )
        self.assertTrue(torch.equal(conv0, expected0))
        self.assertTrue(torch.equal(conv1, expected1))


if __name__ == "__main__":
    unittest.main()
