"""
Unit tests for the KDA extra_buffer conv-window track snapshot layout.

KDA's raw conv pool `conv[0]` is stored as `(slots, K-1, channel)` because
`KimiLinearStateShape` swaps the last two axes relative to GDN's
`(slots, channel, K-1)`. `KDAAttnBackend.forward_extend` snapshots the pre-conv
window into a *transposed* view `conv[0].transpose(-1, -2)` == `(slots, channel,
K-1)`, so it can reuse GDN's shared track machinery unchanged:

  - `__init__` stores `conv_states_shape = (channel, K-1)` (transposed) so the
    shared `_init_track_conv_indices` reads `[-1] == K-1` as the window length.
  - `forward_extend` writes `mixed_qkv_t[:, track_conv_indices].transpose(0, 1)`
    (shape `[num_rows, channel, K-1]`) into `conv_states[mask_indices]`, where
    `conv_states` is the transposed view.

These tests assert the transpose round-trips bit-exactly, so a later decode /
restore reads back the same window that a byte-identical GDN-layout snapshot
would, with no kernel involvement.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
