# SPDX-License-Identifier: Apache-2.0
"""S1a-2a tests — chunk-causal cached scan combiners for streaming `forward_long`.

`_gdn_scan_cached` / `_single_path_delta_scan_cached` are the per-chunk combiners
the streaming path calls: a FORWARD (inclusive) pass that carries recurrent state
across chunks plus a BACKWARD (exclusive) pass recomputed intra-chunk. Two
invariants pin them down:

1. **Reduce-to-bidirectional** (#26153-only): a single chunk with no carried
   state must equal the validated dense `_gdn_scan_bidirectional` /
   `_single_path_delta_scan_bidirectional`.
2. **Port-fidelity** (vs the reference `minimal-sanawm`): output AND carried state
   must match `recurrent_gdn_cached` / `recurrent_delta_cached` chunk-for-chunk,
   including a 2-chunk state-carry run.

FP64, CPU.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
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


def _frame_slice(x, f0, f1):
    # x is (B, H, D, N=T*S), frame-major (T outer, S inner)
    return x[..., f0 * S : f1 * S].contiguous()


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
# 2. Port-fidelity vs reference (gold standard; skip if reference absent)
# --------------------------------------------------------------------------- #

_REF_DIR = pathlib.Path("/sgl-workspace/sglang/myUtils/sana/minimal-sanawm")


def _load_reference():
    if not (_REF_DIR / "components.py").exists():
        pytest.skip(f"reference impl not found at {_REF_DIR}")
    if str(_REF_DIR) not in sys.path:
        sys.path.insert(0, str(_REF_DIR))
    try:
        import components as ref  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"reference components.py not importable: {exc}")
    return ref


def test_gdn_cached_matches_reference_single_and_chunked():
    ref = _load_reference()
    q, k, v, q_rot, k_rot, beta, decay = _inputs()

    # Single chunk, no carried state.
    mine, (kv_m, z_m) = _gdn_scan_cached(q, k, v, q_rot, k_rot, beta, decay)
    r_out, kv_r, z_r, _ = ref.recurrent_gdn_cached(
        q, k, v, q_rot, k_rot, beta, decay, eps=1e-6, kv_state=None, z_state=None
    )
    torch.testing.assert_close(mine, r_out, atol=1e-9, rtol=0)
    torch.testing.assert_close(kv_m, kv_r, atol=1e-9, rtol=0)
    torch.testing.assert_close(z_m, z_r, atol=1e-9, rtol=0)

    # Two chunks carrying state — mine vs reference, chunk-for-chunk.
    split = 2

    def chunk(x, f0, f1):
        return _frame_slice(x, f0, f1)

    args0 = (
        chunk(q, 0, split), chunk(k, 0, split), chunk(v, 0, split),
        chunk(q_rot, 0, split), chunk(k_rot, 0, split),
        beta[:, :, :split], decay[:, :, :split],
    )
    args1 = (
        chunk(q, split, T), chunk(k, split, T), chunk(v, split, T),
        chunk(q_rot, split, T), chunk(k_rot, split, T),
        beta[:, :, split:], decay[:, :, split:],
    )

    m0, (mkv, mz) = _gdn_scan_cached(*args0)
    m1, _ = _gdn_scan_cached(*args1, init_state_kv=mkv, init_state_z=mz)

    r0, rkv, rz, _ = ref.recurrent_gdn_cached(*args0, eps=1e-6, kv_state=None, z_state=None)
    r1, _, _, _ = ref.recurrent_gdn_cached(*args1, eps=1e-6, kv_state=rkv, z_state=rz)

    torch.testing.assert_close(m0, r0, atol=1e-9, rtol=0)
    torch.testing.assert_close(m1, r1, atol=1e-9, rtol=0)


def test_cam_cached_matches_reference_single_and_chunked():
    ref = _load_reference()
    _, _, v, q_rot, k_rot, beta, decay = _inputs()

    mine, kv_m = _single_path_delta_scan_cached(q_rot, k_rot, v, beta, decay)
    r_out, kv_r = ref.recurrent_delta_cached(q_rot, k_rot, v, beta, decay, state=None)
    torch.testing.assert_close(mine, r_out, atol=1e-9, rtol=0)
    torch.testing.assert_close(kv_m, kv_r, atol=1e-9, rtol=0)

    split = 3
    a0 = (
        _frame_slice(q_rot, 0, split), _frame_slice(k_rot, 0, split),
        _frame_slice(v, 0, split), beta[:, :, :split], decay[:, :, :split],
    )
    a1 = (
        _frame_slice(q_rot, split, T), _frame_slice(k_rot, split, T),
        _frame_slice(v, split, T), beta[:, :, split:], decay[:, :, split:],
    )

    m0, mkv = _single_path_delta_scan_cached(*a0)
    m1, _ = _single_path_delta_scan_cached(*a1, init_state_kv=mkv)
    r0, rkv = ref.recurrent_delta_cached(*a0, state=None)
    r1, _ = ref.recurrent_delta_cached(*a1, state=rkv)

    torch.testing.assert_close(m0, r0, atol=1e-9, rtol=0)
    torch.testing.assert_close(m1, r1, atol=1e-9, rtol=0)


# --------------------------------------------------------------------------- #
# 3. Cached short conv on K (cache slot 4)
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
# 4. Cached FFN temporal tail (GLUMBConvTemp, cache slot 9)
# --------------------------------------------------------------------------- #

C_FFN, HID = 6, 8
HFFN, WFFN = S, 1  # S spatial tokens per frame


def _ffn():
    m = GLUMBConvTemp(C_FFN, HID, t_kernel_size=3).double().eval()
    with torch.no_grad():  # zero-init t_conv -> randomize for a non-trivial temporal filter
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
        _nslice(x, split, T), (T - split, HFFN, WFFN), ffn_tail=tail0, save_ffn_tail=True
    )
    torch.testing.assert_close(whole[:, split * S :, :], o1, atol=1e-9, rtol=0)
