"""
Unit tests for the KDA extra_buffer prefix-cache track/restore layout.

KDA's extra_buffer prefix caching reuses GDN's shared track machinery unchanged,
but KDA's `KimiLinearStateShape` stores state in a transposed layout, so the two
snapshots below must line up bit-exactly with GDN's convention. These tests pin
both halves of that:

Conv window (`TestKDAExtraBufferConvTrack`)
    KDA's raw conv pool `conv[0]` is `(slots, K-1, channel)` -- the last two axes
    swapped vs GDN's `(slots, channel, K-1)`. `KDAAttnBackend.forward_extend`
    snapshots the pre-conv window into the *transposed* view
    `conv[0].transpose(-1, -2)` == `(slots, channel, K-1)`:
      - `__init__` stores `conv_states_shape = (channel, K-1)` (transposed) so the
        shared `_init_track_conv_indices` reads `[-1] == K-1` as the window length.
      - `forward_extend` writes `mixed_qkv_t[:, track_conv_indices].transpose(0, 1)`
        (`[num_rows, channel, K-1]`) into `conv_states[mask_indices]`.
    The transpose must round-trip so a later decode reads back the same window a
    byte-identical GDN-layout snapshot would, with no kernel involvement.

Intermediate SSM state `h` (`TestKDAExtraBufferSSMTrack`)
    The correctness core: at a radix track boundary an *unaligned* extended
    sequence snapshots the per-chunk-boundary intermediate state `h` (not the
    final recurrent state) into `ssm_states`. Two conventions must line up:
      - **Layout.** `chunk_kda` returns `h` as `[1, NT, HV, V, K]`
        (NT = ceil(T/chunk)); `ssm_states` is `[slots, HV, V, K]`.
        `_track_mamba_state_extend` does `h.squeeze(0)` then `ssm_states[dst] =
        h[src]`, so `h.squeeze(0)[src]` must be `[HV, V, K]`.
      - **Source indexing (KDA ceil branch).** `KDAAttnBackend` is NOT a
        `Mamba2AttnBackend`, so `_init_track_ssm_indices` takes the ceil branch
        `num_h_states = (extend_seq_lens - 1) // chunk + 1` and selects chunk
        `src_offset + (len_to_track // chunk)` -- a GDN-style choice the Mamba2
        floor branch would not make.
    If either convention drifts (h loses its batch axis, or KDA is mis-routed
    onto the floor branch), a real decode restores the wrong state and accuracy
    silently collapses -- exactly the failure mode this protects.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


# --------------------------------------------------------------------------- #
# Conv-window snapshot                                                         #
# --------------------------------------------------------------------------- #
def _gdn_layout_snapshot(mixed_qkv_t, track_conv_indices, mask_indices, conv_states):
    """GDN reference: conv_states is native `(slots, channel, K-1)`."""
    to_track = mixed_qkv_t[:, track_conv_indices].transpose(0, 1)
    conv_states[mask_indices] = to_track


def _kda_layout_snapshot(mixed_qkv_t, track_conv_indices, mask_indices, conv_raw):
    """KDA path: conv_raw is `(slots, K-1, channel)`; snapshot into the
    transposed `(slots, channel, K-1)` view exactly as forward_extend does."""
    conv_states = conv_raw.transpose(-1, -2)
    to_track = mixed_qkv_t[:, track_conv_indices].transpose(0, 1)
    conv_states[mask_indices] = to_track


class TestKDAExtraBufferConvTrack(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.channel = 12  # q_dim + k_dim + v_dim (packed)
        self.k_minus_1 = 3  # conv_kernel - 1
        self.seq = 40
        self.num_slots = 8

    def _make_inputs(self):
        # mixed_qkv_t is `[channel, seq]` (post transpose(0, 1) in forward_extend).
        mixed_qkv_t = torch.randn(self.channel, self.seq)
        # Two tracked rows, each a length-(K-1) window of flat positions.
        starts = torch.tensor([10, 25])
        track_conv_indices = starts.unsqueeze(-1) + torch.arange(self.k_minus_1)
        mask_indices = torch.tensor([2, 5])  # physical track-dest slots
        return mixed_qkv_t, track_conv_indices, mask_indices

    def test_transposed_view_matches_gdn_layout(self):
        """The KDA transposed-view snapshot is bit-identical to a native GDN
        `(slots, channel, K-1)` snapshot for the same track rows."""
        mixed_qkv_t, track_conv_indices, mask_indices = self._make_inputs()

        gdn = torch.zeros(self.num_slots, self.channel, self.k_minus_1)
        _gdn_layout_snapshot(mixed_qkv_t, track_conv_indices, mask_indices, gdn)

        kda_raw = torch.zeros(self.num_slots, self.k_minus_1, self.channel)
        _kda_layout_snapshot(mixed_qkv_t, track_conv_indices, mask_indices, kda_raw)

        # Reading the KDA raw pool through the same transpose the kernels use
        # must reproduce the GDN native layout exactly.
        self.assertTrue(torch.equal(gdn, kda_raw.transpose(-1, -2)))

    def test_snapshot_content_is_the_window(self):
        """Each tracked slot holds exactly `mixed_qkv[:, window]`, channel-major."""
        mixed_qkv_t, track_conv_indices, mask_indices = self._make_inputs()

        kda_raw = torch.zeros(self.num_slots, self.k_minus_1, self.channel)
        _kda_layout_snapshot(mixed_qkv_t, track_conv_indices, mask_indices, kda_raw)

        view = kda_raw.transpose(-1, -2)  # (slots, channel, K-1)
        for row, slot in enumerate(mask_indices.tolist()):
            expected = mixed_qkv_t[:, track_conv_indices[row]]  # (channel, K-1)
            self.assertTrue(torch.equal(view[slot], expected))

    def test_untracked_slots_untouched(self):
        """Slots not in mask_indices keep their prior (zero) contents."""
        mixed_qkv_t, track_conv_indices, mask_indices = self._make_inputs()

        kda_raw = torch.zeros(self.num_slots, self.k_minus_1, self.channel)
        _kda_layout_snapshot(mixed_qkv_t, track_conv_indices, mask_indices, kda_raw)

        tracked = set(mask_indices.tolist())
        for slot in range(self.num_slots):
            if slot not in tracked:
                self.assertTrue(
                    torch.equal(kda_raw[slot], torch.zeros_like(kda_raw[slot]))
                )

    def test_restore_round_trip(self):
        """A decode-time gather of a tracked slot (transposed to `(K-1, channel)`
        for causal_conv1d_update) recovers the original window."""
        mixed_qkv_t, track_conv_indices, mask_indices = self._make_inputs()

        kda_raw = torch.zeros(self.num_slots, self.k_minus_1, self.channel)
        _kda_layout_snapshot(mixed_qkv_t, track_conv_indices, mask_indices, kda_raw)

        # Decode reads conv[0] raw `(slots, K-1, channel)` and transposes per-slot.
        for row, slot in enumerate(mask_indices.tolist()):
            restored = kda_raw[slot].transpose(-1, -2)  # (channel, K-1)
            expected = mixed_qkv_t[:, track_conv_indices[row]]
            self.assertTrue(torch.equal(restored, expected))


# --------------------------------------------------------------------------- #
# Intermediate SSM state `h` snapshot                                          #
# --------------------------------------------------------------------------- #
def _ceil_num_h_states(extend_seq_lens, chunk):
    """KDA (non-Mamba2) ceil branch from `_init_track_ssm_indices`."""
    return (extend_seq_lens - 1) // chunk + 1


def _track_ssm_h_src(extend_seq_lens, lens_to_track, track_mask, chunk):
    """Replicate the unaligned-row `track_ssm_h_src` computation (ceil branch).

    Mirrors `MambaAttnBackendBase._init_track_ssm_indices`: per-sequence
    chunk-count prefix sum gives each sequence's base offset into the flat `h`
    chunk axis; the unaligned tracked rows add `len_to_track // chunk`.
    """
    num_h_states = _ceil_num_h_states(extend_seq_lens, chunk)
    src_offset = torch.zeros_like(num_h_states)
    src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

    lens_masked = lens_to_track[track_mask]
    offset_masked = src_offset[track_mask]
    is_aligned = (lens_masked % chunk) == 0
    not_aligned = ~is_aligned
    src = offset_masked[not_aligned] + (lens_masked[not_aligned] // chunk)
    return src, not_aligned


def _track_mamba_state_extend(h, ssm_states, src, dst):
    """The exact copy `_track_mamba_state_extend` performs for the h-src path."""
    h = h.squeeze(0)
    ssm_states[dst] = h[src].to(ssm_states.dtype, copy=False)


class TestKDAExtraBufferSSMTrack(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.chunk = 64  # mamba_cache_chunk_size = max(FLA_CHUNK_SIZE=64, page=1)
        self.HV = 4
        self.V = 8
        self.K = 8
        self.num_slots = 6

    def _make_h(self, num_chunks):
        # KDA chunk_kda returns h as [1, NT, HV, V, K].
        return torch.randn(1, num_chunks, self.HV, self.V, self.K)

    def test_h_layout_squeeze_matches_ssm_slot(self):
        """`h.squeeze(0)[src]` is `[HV, V, K]` and equals the selected chunk
        boundary state, landing bit-exactly in `ssm_states[dst]`."""
        num_chunks = 3
        h = self._make_h(num_chunks)
        ssm_states = torch.zeros(self.num_slots, self.HV, self.V, self.K)

        src = torch.tensor([2])  # last chunk boundary
        dst = torch.tensor([4])  # a track-dest slot
        _track_mamba_state_extend(h, ssm_states, src, dst)

        expected = h.squeeze(0)[2]
        self.assertEqual(tuple(expected.shape), (self.HV, self.V, self.K))
        self.assertTrue(torch.equal(ssm_states[4], expected))

    def test_ceil_branch_source_index(self):
        """KDA's ceil branch selects the boundary state at the largest chunk
        multiple <= the tracked length, and picks a chunk the Mamba2 floor
        branch would not for a single-chunk-plus-remainder sequence."""
        chunk = self.chunk
        # One tracked sequence of 150 tokens; track boundary at prefix 130.
        extend_seq_lens = torch.tensor([150])
        lens_to_track = torch.tensor([130])  # unaligned: 130 % 64 == 2
        track_mask = torch.tensor([True])

        src, not_aligned = _track_ssm_h_src(
            extend_seq_lens, lens_to_track, track_mask, chunk
        )
        # ceil num_h_states for 150 = (150-1)//64 + 1 = 3 chunks (offset 0).
        # src = 0 + 130 // 64 = 2  -> the 3rd boundary state (index 2).
        self.assertTrue(bool(not_aligned.item()))
        self.assertEqual(src.tolist(), [2])

        # The Mamba2 floor branch would size num_h_states = 150 // 64 = 2, so an
        # h of only 2 chunk states -- index 2 would be out of range. The ceil
        # branch is what makes index 2 valid, i.e. KDA must NOT take the floor
        # branch. Assert the ceil count actually covers the selected index.
        num_h_states = _ceil_num_h_states(extend_seq_lens, chunk)
        self.assertEqual(num_h_states.tolist(), [3])
        self.assertTrue(src.item() < num_h_states.sum().item())

    def test_multi_sequence_offset_and_restore(self):
        """Two sequences: per-sequence chunk offsets index the flat `h` axis, and
        each tracked unaligned row restores its own chunk-boundary state."""
        chunk = self.chunk
        # seq0: 150 tokens -> 3 chunk states; seq1: 100 tokens -> 2 chunk states.
        extend_seq_lens = torch.tensor([150, 100])
        lens_to_track = torch.tensor([130, 70])  # both unaligned
        track_mask = torch.tensor([True, True])
        dst = torch.tensor([1, 5])

        src, not_aligned = _track_ssm_h_src(
            extend_seq_lens, lens_to_track, track_mask, chunk
        )
        # seq0 offset 0: src0 = 0 + 130//64 = 2
        # seq1 offset 3: src1 = 3 + 70//64  = 3 + 1 = 4
        self.assertEqual(src.tolist(), [2, 4])

        total_chunks = int(_ceil_num_h_states(extend_seq_lens, chunk).sum())
        self.assertEqual(total_chunks, 5)
        h = self._make_h(total_chunks)
        ssm_states = torch.zeros(self.num_slots, self.HV, self.V, self.K)

        _track_mamba_state_extend(h, ssm_states, src, dst)

        h_sq = h.squeeze(0)
        self.assertTrue(torch.equal(ssm_states[1], h_sq[2]))  # seq0 boundary
        self.assertTrue(torch.equal(ssm_states[5], h_sq[4]))  # seq1 boundary
        # Untouched slots stay zero.
        for slot in (0, 2, 3, 4):
            self.assertTrue(
                torch.equal(ssm_states[slot], torch.zeros(self.HV, self.V, self.K))
            )

    def test_aligned_row_takes_no_h_src(self):
        """A tracked row whose length is a chunk multiple is NOT an h-src row
        (it restores the final recurrent state instead); the h-src set is empty."""
        chunk = self.chunk
        extend_seq_lens = torch.tensor([128])
        lens_to_track = torch.tensor([128])  # aligned: 128 % 64 == 0
        track_mask = torch.tensor([True])

        src, not_aligned = _track_ssm_h_src(
            extend_seq_lens, lens_to_track, track_mask, chunk
        )
        self.assertFalse(bool(not_aligned.item()))
        self.assertEqual(src.numel(), 0)


if __name__ == "__main__":
    unittest.main()
