"""Mamba2 target verify with a verify narrower than the snapshot buffer.

The SSU kernels address snapshot ``(slot, step)`` as ``slot * cache_steps +
step``; the accept commit reads ``buffer[slot, step]``. Mamba2 passes the
verify width as ``cache_steps``, so with adaptive speculative steps (verify
width < buffer steps) every slot but slot 0 used to land in another slot's
rows. The dispatcher takes the pitch from the buffer: the snapshots of a
narrow verify must equal those written into a buffer of exactly that width.
"""

import pytest
import torch

from sglang.kernels.ops.mamba.triton_ops import ssu_dispatch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _verify(buffer, inputs, verify_width):
    state, x, dt, A, B, C, D, dt_bias, state_idx, buf_idx = inputs
    out = torch.empty_like(x)
    ssu_dispatch.selective_state_update(
        state, x, dt, A, B, C, D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_idx,
        out=out,
        disable_state_update=True,
        intermediate_states_buffer=buffer,
        cache_steps=verify_width,
        intermediate_state_indices=buf_idx,
    )  # fmt: skip
    return out


def test_narrow_verify_snapshots_land_in_their_own_slot(monkeypatch):
    monkeypatch.setattr(
        ssu_dispatch, "_mamba_ssu_backend", ssu_dispatch.TritonSSUBackend()
    )
    torch.manual_seed(0)
    dev = "cuda"
    reqs, width, buffer_steps, heads, dim, dstate, groups = 3, 3, 8, 4, 16, 16, 1
    state = torch.randn(reqs + 2, heads, dim, dstate, device=dev)
    x = torch.randn(reqs, width, heads, dim, device=dev)
    dt = torch.randn(reqs, width, heads, dim, device=dev)
    A = -torch.rand(heads, dim, dstate, device=dev)
    B = torch.randn(reqs, width, groups, dstate, device=dev)
    C = torch.randn(reqs, width, groups, dstate, device=dev)
    D = torch.randn(heads, dim, device=dev)
    dt_bias = torch.randn(heads, dim, device=dev)
    state_idx = torch.tensor([3, 0, 4], dtype=torch.int32, device=dev)
    buf_idx = torch.arange(reqs, dtype=torch.int32, device=dev)
    inputs = (state, x, dt, A, B, C, D, dt_bias, state_idx, buf_idx)

    wide = torch.zeros(reqs, buffer_steps, heads, dim, dstate, device=dev)
    exact = torch.zeros(reqs, width, heads, dim, dstate, device=dev)
    out_wide = _verify(wide, inputs, width)
    out_exact = _verify(exact, inputs, width)

    torch.testing.assert_close(out_wide, out_exact)
    torch.testing.assert_close(wide[:, :width], exact)
    assert not wide[:, width:].any(), "a slot's snapshots spilled into the next slot"
