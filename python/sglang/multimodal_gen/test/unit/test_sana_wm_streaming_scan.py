# SPDX-License-Identifier: Apache-2.0
"""S1a tests — state-carrying (streaming) GDN scans on the #26153 SANA-WM DiT.

The foundational streaming invariant: running the recurrence chunk-by-chunk while
carrying KV/recurrent state across calls produces bit-comparable output to the
monolithic forward scan over the whole sequence. This is what makes the
autoregressive `forward_long` path correct. FP64, CPU.
"""

from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    _gdn_scan_forward,
    _gdn_scan_forward_stateful,
    _single_path_delta_scan_forward,
    _single_path_delta_scan_forward_stateful,
)

B, H, D, T, S = 1, 2, 4, 6, 3
N = T * S


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64)


def _inputs():
    torch.manual_seed(0)
    q, k, v = _rand(B, H, D, N), _rand(B, H, D, N), _rand(B, H, D, N)
    q_rot, k_rot = _rand(B, H, D, N), _rand(B, H, D, N)
    beta = torch.rand(B, H, T, S, dtype=torch.float64)        # per-(frame,token) gate
    decay = torch.rand(B, H, T, dtype=torch.float64) * 0.5 + 0.4  # per-frame decay in (0,1)
    return q, k, v, q_rot, k_rot, beta, decay


def _frame_slice(x, f0, f1):
    # x is (B, H, D, N=T*S), frame-major (T outer, S inner)
    return x[..., f0 * S : f1 * S].contiguous()


def test_main_stateful_matches_nonstateful_full():
    q, k, v, q_rot, k_rot, beta, decay = _inputs()
    ref = _gdn_scan_forward(q, k, v, q_rot, k_rot, beta, decay)
    out = _gdn_scan_forward_stateful(q, k, v, q_rot, k_rot, beta, decay)
    torch.testing.assert_close(out, ref, atol=1e-9, rtol=0)


@pytest.mark.parametrize("split", [1, 2, 4])
def test_main_chunked_state_carry_matches_full(split):
    q, k, v, q_rot, k_rot, beta, decay = _inputs()
    full = _gdn_scan_forward(q, k, v, q_rot, k_rot, beta, decay)

    out0, (kv, z) = _gdn_scan_forward_stateful(
        _frame_slice(q, 0, split), _frame_slice(k, 0, split), _frame_slice(v, 0, split),
        _frame_slice(q_rot, 0, split), _frame_slice(k_rot, 0, split),
        beta[:, :, :split], decay[:, :, :split], return_state=True,
    )
    out1 = _gdn_scan_forward_stateful(
        _frame_slice(q, split, T), _frame_slice(k, split, T), _frame_slice(v, split, T),
        _frame_slice(q_rot, split, T), _frame_slice(k_rot, split, T),
        beta[:, :, split:], decay[:, :, split:],
        init_state_kv=kv, init_state_z=z,
    )
    chunked = torch.cat([out0, out1], dim=-1)  # concat along N (frame-major)
    torch.testing.assert_close(chunked, full, atol=1e-9, rtol=0)


def test_cam_stateful_matches_nonstateful_full():
    _, _, v, q_rot, k_rot, beta, decay = _inputs()
    ref = _single_path_delta_scan_forward(q_rot, k_rot, v, beta, decay)
    out = _single_path_delta_scan_forward_stateful(q_rot, k_rot, v, beta, decay)
    torch.testing.assert_close(out, ref, atol=1e-9, rtol=0)


@pytest.mark.parametrize("split", [1, 3])
def test_cam_chunked_state_carry_matches_full(split):
    _, _, v, q_rot, k_rot, beta, decay = _inputs()
    full = _single_path_delta_scan_forward(q_rot, k_rot, v, beta, decay)

    out0, kv = _single_path_delta_scan_forward_stateful(
        _frame_slice(q_rot, 0, split), _frame_slice(k_rot, 0, split), _frame_slice(v, 0, split),
        beta[:, :, :split], decay[:, :, :split], return_state=True,
    )
    out1 = _single_path_delta_scan_forward_stateful(
        _frame_slice(q_rot, split, T), _frame_slice(k_rot, split, T), _frame_slice(v, split, T),
        beta[:, :, split:], decay[:, :, split:], init_state_kv=kv,
    )
    chunked = torch.cat([out0, out1], dim=-1)
    torch.testing.assert_close(chunked, full, atol=1e-9, rtol=0)
