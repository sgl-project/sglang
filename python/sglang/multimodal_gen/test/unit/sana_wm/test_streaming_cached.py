# SPDX-License-Identifier: Apache-2.0
"""S1a-2a tests — chunk-causal cached scan combiners for streaming `forward_long`.

`_gdn_scan_cached` / `_single_path_delta_scan_cached` are the per-chunk combiners
the streaming path calls: a FORWARD (inclusive) pass that carries recurrent state
across chunks plus a BACKWARD (exclusive) pass recomputed intra-chunk. A
single chunk with no carried state must equal the validated dense
`_gdn_scan_bidirectional` / `_single_path_delta_scan_bidirectional`.

FP64, CPU.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    BidirectionalGDNUCPESinglePathLiteLA,
    GLUMBConvTemp,
    _gdn_scan_bidirectional,
    _gdn_scan_cached,
    _ShortConvolution,
    _single_path_delta_scan_bidirectional,
    _single_path_delta_scan_cached,
    _temporal_short_conv_cached,
)

B, H, D, T, S = 1, 2, 4, 6, 3
N = T * S
HW = (T, S, 1)  # H_sp * W_sp == S


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64)


def _inputs():
    torch.manual_seed(0)
    q, k, v = _rand(B, H, D, N), _rand(B, H, D, N), _rand(B, H, D, N)
    q_rot, k_rot = _rand(B, H, D, N), _rand(B, H, D, N)
    beta = torch.rand(B, H, T, S, dtype=torch.float64)
    decay = torch.rand(B, H, T, dtype=torch.float64) * 0.5 + 0.4
    return q, k, v, q_rot, k_rot, beta, decay


# --------------------------------------------------------------------------- #
# 1. Reduce-to-bidirectional (uses only #26153 functions)
# --------------------------------------------------------------------------- #


def test_gdn_cached_single_chunk_reduces_to_bidirectional():
    q, k, v, q_rot, k_rot, beta, decay = _inputs()
    ref = _gdn_scan_bidirectional(q, k, v, q_rot, k_rot, beta, decay, HW)
    out, (kv, z) = _gdn_scan_cached(q, k, v, q_rot, k_rot, beta, decay)
    torch.testing.assert_close(out, ref, atol=1e-9, rtol=0)
    assert kv.shape == (B, H, D, D) and z.shape == (B, H, D, 1)


def test_cam_cached_single_chunk_reduces_to_bidirectional():
    _, _, v, q_rot, k_rot, beta, decay = _inputs()
    ref = _single_path_delta_scan_bidirectional(q_rot, k_rot, v, beta, decay, HW)
    out, kv = _single_path_delta_scan_cached(q_rot, k_rot, v, beta, decay)
    torch.testing.assert_close(out, ref, atol=1e-9, rtol=0)
    assert kv.shape == (B, H, D, D)


# --------------------------------------------------------------------------- #
# 2. Cached short conv on K (cache slot 4)
# --------------------------------------------------------------------------- #

C_CONV, KERN = 5, 4


def _conv():
    conv = _ShortConvolution(C_CONV, KERN).double()
    with torch.no_grad():  # randomize off the identity init for a non-trivial filter
        conv.weight.copy_(torch.randn(C_CONV, 1, KERN, dtype=torch.float64))
    return conv


def _nslice(x, f0, f1):
    # x is (B, N=T*S, C), frame-major
    return x[:, f0 * S : f1 * S, :].contiguous()


def test_short_conv_cached_single_chunk_reduces_to_bidirectional():
    torch.manual_seed(1)
    x = torch.randn(B, N, C_CONV, dtype=torch.float64)
    conv = _conv()
    ref = BidirectionalGDNUCPESinglePathLiteLA._temporal_short_conv(
        x, conv, (T, S, 1), bidirectional=True
    )
    out, prefix = _temporal_short_conv_cached(x, conv, (T, S, 1))
    torch.testing.assert_close(out, ref, atol=1e-9, rtol=0)
    assert prefix.shape == (B * S, KERN - 1, C_CONV)


def test_short_conv_cached_forward_continuity_across_chunks():
    # The forward (causal) direction must be chunk-invariant via the prefix carry.
    torch.manual_seed(2)
    x = torch.randn(B, N, C_CONV, dtype=torch.float64)
    conv = _conv()
    whole, _ = _temporal_short_conv_cached(x, conv, (T, S, 1), bidirectional=False)

    split = 2
    o0, p0 = _temporal_short_conv_cached(
        _nslice(x, 0, split), conv, (split, S, 1), bidirectional=False
    )
    o1, _ = _temporal_short_conv_cached(
        _nslice(x, split, T), conv, (T - split, S, 1), prefix=p0, bidirectional=False
    )
    chunked = torch.cat([o0, o1], dim=1)  # reassemble frame-major
    torch.testing.assert_close(chunked, whole, atol=1e-9, rtol=0)


# --------------------------------------------------------------------------- #
# 3. Cached FFN temporal tail (GLUMBConvTemp, cache slot 9)
# --------------------------------------------------------------------------- #

C_FFN, HID = 6, 8
HFFN, WFFN = S, 1  # S spatial tokens per frame


def _ffn():
    m = GLUMBConvTemp(C_FFN, HID, t_kernel_size=3).double().eval()
    with (
        torch.no_grad()
    ):  # zero-init t_conv -> randomize for a non-trivial temporal filter
        m.t_conv.weight.copy_(torch.randn_like(m.t_conv.weight))
    return m


def test_ffn_cached_no_prefix_reduces_to_dense():
    torch.manual_seed(3)
    x = torch.randn(B, N, C_FFN, dtype=torch.float64)
    m = _ffn()
    dense = m(x, (T, HFFN, WFFN))
    cached, tail = m(x, (T, HFFN, WFFN), save_ffn_tail=True)
    torch.testing.assert_close(cached, dense, atol=1e-9, rtol=0)
    assert tail.shape[2] == m.t_conv.kernel_size[0] // 2


def test_ffn_cached_final_chunk_matches_whole():
    # The prefix supplies real left context, so the LAST chunk (right edge = end
    # of sequence, same as the whole pass) matches the whole-sequence output.
    torch.manual_seed(4)
    x = torch.randn(B, N, C_FFN, dtype=torch.float64)
    m = _ffn()
    whole, _ = m(x, (T, HFFN, WFFN), save_ffn_tail=True)

    split = 2
    _, tail0 = m(_nslice(x, 0, split), (split, HFFN, WFFN), save_ffn_tail=True)
    o1, _ = m(
        _nslice(x, split, T),
        (T - split, HFFN, WFFN),
        ffn_tail=tail0,
        save_ffn_tail=True,
    )
    torch.testing.assert_close(whole[:, split * S :, :], o1, atol=1e-9, rtol=0)
