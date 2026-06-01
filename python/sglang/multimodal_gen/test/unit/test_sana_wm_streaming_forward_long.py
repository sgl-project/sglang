# SPDX-License-Identifier: Apache-2.0
"""S1a-2c tests — streaming `forward_long` assembly on the #26153 SANA-WM DiT.

Built up in stages, each pinned at fp64 / CPU / atol=1e-9:
  Stage 0 — RoPE windowing: `WanRotaryPosEmbed` must produce freqs at GLOBAL
    frame positions for a chunk `[start, end)`, equal to the corresponding slice
    of the full table. This is the prerequisite that keeps a chunk's queries
    aligned with the carried K and the softmax concat-window.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    WanRotaryPosEmbed,
    _slice_rope_to_current_chunk,
)

HEAD_DIM = 112
H, W = 2, 3
S = H * W
T = 6


def _rope():
    return WanRotaryPosEmbed(attention_head_dim=HEAD_DIM, patch_size=(1, 1, 1))


def test_rope_full_equals_range_from_zero():
    rope = _rope()
    dense = rope((T, H, W), torch.device("cpu"))
    ranged = rope(((0, T), H, W), torch.device("cpu"))
    torch.testing.assert_close(ranged, dense, atol=1e-9, rtol=0)


def test_rope_window_equals_slice_of_full():
    rope = _rope()
    start, end = 2, 5
    full = rope((end, H, W), torch.device("cpu"))  # frames 0..end-1
    window = rope(((start, end), H, W), torch.device("cpu"))  # frames start..end-1
    # full is frame-major (1, 1, end*S, D/2); slice frames [start:end].
    torch.testing.assert_close(
        window, full[:, :, start * S : end * S, :], atol=1e-9, rtol=0
    )


def test_rope_frame_index_overrides_to_global_positions():
    rope = _rope()
    start, end = 2, 5
    fidx = torch.arange(start, end)
    by_index = rope((end - start, H, W), torch.device("cpu"), frame_index=fidx)
    by_range = rope(((start, end), H, W), torch.device("cpu"))
    torch.testing.assert_close(by_index, by_range, atol=1e-9, rtol=0)


def test_slice_rope_to_current_chunk_is_noop_when_sized():
    rope = _rope()
    freqs = rope(((1, 4), H, W), torch.device("cpu"))  # 3 frames -> 3*S tokens
    assert _slice_rope_to_current_chunk(freqs, 3 * S) is freqs
    assert _slice_rope_to_current_chunk(None, 3 * S) is None
    # wider table trimmed to the trailing chunk
    wide = rope(((0, 4), H, W), torch.device("cpu"))  # 4 frames
    trimmed = _slice_rope_to_current_chunk(wide, 3 * S)
    torch.testing.assert_close(trimmed, wide[:, :, -3 * S :, :], atol=1e-9, rtol=0)
