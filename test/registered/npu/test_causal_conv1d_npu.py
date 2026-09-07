"""Tests for the NPU dispatch of sglang's causal_conv1d entry points.

On NPU, ``sglang.srt.layers.attention.mamba.causal_conv1d`` routes
``causal_conv1d_fn`` / ``causal_conv1d_update`` to sgl_kernel_npu's v2
wrappers (AscendC op fast path + Triton/torch fallbacks), which speak
the NPU MambaPool's window-major conv-state layout
``(slots, width - 1, dim)`` -- see ``_init_npu_conv_state``. These tests
pin that contract for the short-conv hybrid models (LFM2 / LFM2-MoE):

- prefill: packed varlen x of shape (dim, cu_seq) with a window-major
  pool; pad slots must not be read from or written to;
- decode: x of shape (batch, dim) with int64 conv_state_indices
  containing pad entries (the dtype ShortConvAttnBackend hands out);
- cache_seqlens (circular buffer) must raise NotImplementedError;
- the NPU branch actually dispatches to the v2 wrappers (monkeypatched
  recording probe).

All references run on CPU in fp32 (fp32 F.conv1d on NPU can produce
NaNs), matching the sibling tests in sgl_kernel_npu.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")

pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU-only test")

if is_npu():
    import torch_npu  # noqa: F401  makes torch.npu available

    import sglang.srt.layers.attention.mamba.causal_conv1d as cc
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

device = "npu"
torch.manual_seed(17)

DIM, WIDTH = 128, 3  # LFM2 conv_kernel == 3
STATE_LEN = WIDTH - 1
PAD = -1  # PAD_SLOT_ID


def _dev(t, dtype=torch.bfloat16):
    return t.to(dtype).to(device).contiguous()


def _ref_prefill_row(toks, init, weight, bias, silu):
    """toks: (dim, L); init: (dim, STATE_LEN). CPU fp32 reference."""
    virt = torch.cat([init, toks], dim=1)
    out = F.conv1d(
        virt.unsqueeze(0),
        weight.unsqueeze(1),
        bias,
        groups=toks.shape[0],
    )[0]
    if silu:
        out = F.silu(out)
    return out, virt[:, -STATE_LEN:]


def test_npu_dispatch_goes_to_v2_wrappers(monkeypatch):
    """The NPU branch must call the sgl_kernel_npu v2 wrappers."""
    calls = {"fn": 0, "update": 0}
    real_fn, real_update = cc._causal_conv1d_fn_npu, cc._causal_conv1d_update_npu

    def rec_fn(*a, **k):
        calls["fn"] += 1
        return real_fn(*a, **k)

    def rec_update(*a, **k):
        calls["update"] += 1
        return real_update(*a, **k)

    monkeypatch.setattr(cc, "_causal_conv1d_fn_npu", rec_fn)
    monkeypatch.setattr(cc, "_causal_conv1d_update_npu", rec_update)

    x = _dev(torch.randn(DIM, 4) * 0.3)
    w = _dev(torch.randn(DIM, WIDTH) * 0.3)
    pool = _dev(torch.randn(4, STATE_LEN, DIM) * 0.3)
    qsl = torch.tensor([0, 4], device=device, dtype=torch.int32)
    idx = torch.tensor([2], device=device, dtype=torch.int64)
    causal_conv1d_fn(
        x,
        w,
        None,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
        conv_states=pool,
        activation=None,
    )
    causal_conv1d_update(
        _dev(torch.randn(2, DIM) * 0.3),
        pool,
        w,
        None,
        activation=None,
        conv_state_indices=torch.tensor([0, 3], device=device, dtype=torch.int64),
    )
    torch.npu.synchronize()
    assert calls == {"fn": 1, "update": 1}


@pytest.mark.parametrize("silu", [False, True])
def test_npu_prefill_varlen_window_major(silu):
    """causal_conv1d_fn on NPU: packed varlen + window-major pool + pads."""
    lens, starts = [5, 2, 1], [0, 5, 7, 8]
    cu = starts[-1]
    x = _dev(torch.randn(DIM, cu) * 0.3)
    x0 = x.clone()
    pool = _dev(torch.randn(6, STATE_LEN, DIM) * 0.3)
    pool0 = pool.clone()
    w = _dev(torch.randn(DIM, WIDTH) * 0.3)
    bias = _dev(torch.randn(DIM) * 0.3)
    qsl = torch.tensor(starts, device=device, dtype=torch.int32)
    # row 1 is a pad request: its outputs are unspecified, but it must
    # not corrupt any real pool slot.
    idx = torch.tensor([1, PAD, 4], device=device, dtype=torch.int64)
    hi = torch.tensor([True, False, True], device=device, dtype=torch.bool)

    out = causal_conv1d_fn(
        x,
        w,
        bias,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=hi,
        conv_states=pool,
        activation="silu" if silu else None,
    )
    torch.npu.synchronize()
    assert out.shape == (DIM, cu)

    wc, bc = w.cpu().float(), bias.cpu().float()
    for i, slot in [(0, 1), (2, 4)]:
        toks = x0[:, starts[i] : starts[i + 1]].cpu().float()
        init = (
            pool0[slot].cpu().float().t()
            if hi[i]
            else torch.zeros(DIM, STATE_LEN)
        )
        ref_out, ref_final = _ref_prefill_row(toks, init, wc, bc, silu)
        torch.testing.assert_close(
            out[:, starts[i] : starts[i + 1]].cpu().float(),
            ref_out,
            atol=2e-2,
            rtol=2e-2,
        )
        # pool is window-major: stored rows are (STATE_LEN, DIM)
        torch.testing.assert_close(
            pool[slot].cpu().float(), ref_final.t(), atol=2e-2, rtol=2e-2
        )
    # slot 0 is the reserved dummy write target for pads; every slot that
    # is neither claimed nor the dummy must be bit-identical.
    untouched = [s for s in range(6) if s not in (0, 1, 4)]
    torch.testing.assert_close(
        pool[untouched].float(), pool0[untouched].float(), atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("silu", [False, True])
def test_npu_decode_window_major(silu):
    """causal_conv1d_update on NPU: (batch, dim) + int64 indices + pads."""
    batch = 4
    x = _dev(torch.randn(batch, DIM) * 0.3)
    x0 = x.clone()
    pool = _dev(torch.randn(6, STATE_LEN, DIM) * 0.3)
    pool0 = pool.clone()
    w = _dev(torch.randn(DIM, WIDTH) * 0.3)
    bias = _dev(torch.randn(DIM) * 0.3)
    idx = torch.tensor([1, 3, PAD, 5], device=device, dtype=torch.int64)

    out = causal_conv1d_update(
        x, pool, w, bias, activation="silu" if silu else None,
        conv_state_indices=idx,
    )
    torch.npu.synchronize()
    assert out.shape == (batch, DIM)

    wc, bc = w.cpu().float(), bias.cpu().float()
    for row, slot in [(0, 1), (1, 3), (3, 5)]:
        state = pool0[slot].cpu().float().t()  # (DIM, STATE_LEN)
        win = torch.cat([state, x0[row].cpu().float().unsqueeze(1)], dim=1)
        ref = F.conv1d(
            win.unsqueeze(0), wc.unsqueeze(1), bc, groups=DIM
        )[0][:, -1]
        if silu:
            ref = F.silu(ref)
        torch.testing.assert_close(
            out[row].cpu().float(), ref, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            pool[slot].cpu().float(), win[:, -STATE_LEN:].t(),
            atol=2e-2, rtol=2e-2,
        )
    untouched = [s for s in range(6) if s not in (1, 3, 5)]
    torch.testing.assert_close(
        pool[untouched].float(), pool0[untouched].float(), atol=0.0, rtol=0.0
    )


def test_npu_update_rejects_cache_seqlens():
    """Circular-buffer updates are not supported on NPU: fail clearly."""
    x = _dev(torch.randn(2, DIM) * 0.3)
    pool = _dev(torch.randn(6, STATE_LEN, DIM) * 0.3)
    w = _dev(torch.randn(DIM, WIDTH) * 0.3)
    with pytest.raises(NotImplementedError, match="circular-buffer"):
        causal_conv1d_update(
            x,
            pool,
            w,
            None,
            cache_seqlens=torch.ones(2, dtype=torch.int32, device=device),
            conv_state_indices=torch.tensor(
                [0, 1], device=device, dtype=torch.int64
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
