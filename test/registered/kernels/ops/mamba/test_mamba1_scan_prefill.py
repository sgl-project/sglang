import os
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.triton_ops import mamba1_scan_prefill
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")

# TRITON_INTERPRET=1 runs the kernel on CPU, so the test also works without a GPU.
_DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else get_device()
_TOL = {
    torch.float32: dict(rtol=3e-4, atol=1e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-2),
    torch.bfloat16: dict(rtol=1e-2, atol=5e-2),
}


def _scan_ref(*, x, dt, A, B, C, D, gate, h0):
    """Mamba-1 recurrence for one sequence, fp32. x/dt/gate (T, dim), B/C (T, n)."""
    x, dt, B, C, gate = (t.float() for t in (x, dt, B, C, gate))
    dt = F.softplus(dt)
    h = h0.float()
    ys = []
    for t in range(x.shape[0]):
        h = torch.exp(dt[t][:, None] * A) * h + (dt[t] * x[t])[:, None] * B[t][None]
        ys.append((h * C[t][None]).sum(-1))
    y = (torch.stack(ys) + x * D.float()) * F.silu(gate)
    return y, h


def _make_inputs(*, seq_lens, dim, dstate, itype, num_slots, seed=0):
    torch.manual_seed(seed)
    total = sum(seq_lens)
    # x and gate are column slices of one projection, as in MambaMixer1, so the
    # kernel sees a row stride of 2 * dim.
    xz = torch.randn(total, 2 * dim, dtype=itype, device=_DEVICE)
    x, gate = xz[:, :dim], xz[:, dim:]
    dt = torch.randn(total, dim, dtype=itype, device=_DEVICE)
    B = torch.randn(total, dstate, dtype=itype, device=_DEVICE)
    C = torch.randn(total, dstate, dtype=itype, device=_DEVICE)
    A = -torch.rand(dim, dstate, device=_DEVICE) - 1.0
    D = torch.randn(dim, device=_DEVICE)
    ssm_state = torch.randn(num_slots, dim, 1, dstate, device=_DEVICE)
    qsl = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
        dtype=torch.int32,
        device=_DEVICE,
    )
    return dict(x=x, dt=dt, A=A, B=B, C=C, D=D, gate=gate), ssm_state, qsl


def _skip_unsupported_device():
    if _DEVICE not in ("cuda", "xpu", "cpu"):
        pytest.skip(f"unsupported device {_DEVICE}")


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 80])
@pytest.mark.parametrize("init_mode", ["none", "mixed"])
def test_varlen_batch_matches_reference(itype, dim, init_mode):
    """Per-sequence outputs and final states of a varlen batch match the
    recurrence; dim 80 exercises the masked channel tail and slots the batch
    does not own stay untouched."""
    _skip_unsupported_device()
    seq_lens = [1, 7, 33, 4]
    dstate = 16
    inputs, ssm_state, qsl = _make_inputs(
        seq_lens=seq_lens, dim=dim, dstate=dstate, itype=itype, num_slots=9
    )
    state_indices = torch.tensor([5, 0, 8, 2], dtype=torch.int32, device=_DEVICE)
    has_initial = (
        None
        if init_mode == "none"
        else torch.tensor([True, False, True, False], device=_DEVICE)
    )
    state_before = ssm_state.clone()
    out = torch.empty_like(inputs["x"], memory_format=torch.contiguous_format)

    mamba1_scan_prefill(
        **inputs,
        out=out,
        ssm_state=ssm_state,
        state_indices=state_indices,
        query_start_loc=qsl,
        has_initial=has_initial,
    )

    for i, slot in enumerate(state_indices.tolist()):
        s, e = int(qsl[i]), int(qsl[i + 1])
        use_init = has_initial is not None and bool(has_initial[i])
        h0 = state_before[slot, :, 0] if use_init else torch.zeros_like(inputs["A"])
        y_ref, h_ref = _scan_ref(
            **{k: v[s:e] for k, v in inputs.items() if k not in ("A", "D")},
            A=inputs["A"],
            D=inputs["D"],
            h0=h0,
        )
        torch.testing.assert_close(out[s:e].float(), y_ref, **_TOL[itype])
        torch.testing.assert_close(ssm_state[slot, :, 0], h_ref, **_TOL[itype])
    untouched = [s for s in range(9) if s not in state_indices.tolist()]
    assert torch.equal(ssm_state[untouched], state_before[untouched])


def test_chunked_prefill_matches_single_pass():
    """Scanning a prompt in two chunks, the second resuming from the state the
    first wrote back, equals scanning it in one pass (chunked-prefill handoff)."""
    _skip_unsupported_device()
    dim, dstate, n1, n2 = 64, 16, 11, 6
    inputs, ssm_state, _ = _make_inputs(
        seq_lens=[n1 + n2], dim=dim, dstate=dstate, itype=torch.float32, num_slots=3
    )
    idx = torch.tensor([1], dtype=torch.int32, device=_DEVICE)

    def run(lo, hi, has_init):
        out = torch.empty(hi - lo, dim, device=_DEVICE)
        part = {k: (v if k in ("A", "D") else v[lo:hi]) for k, v in inputs.items()}
        mamba1_scan_prefill(
            **part,
            out=out,
            ssm_state=ssm_state,
            state_indices=idx,
            query_start_loc=torch.tensor(
                [0, hi - lo], dtype=torch.int32, device=_DEVICE
            ),
            has_initial=torch.tensor([has_init], device=_DEVICE),
        )
        return out, ssm_state[1].clone()

    out_full, h_full = run(0, n1 + n2, False)
    out_a, _ = run(0, n1, False)
    out_b, h_chunked = run(n1, n1 + n2, True)

    torch.testing.assert_close(torch.cat([out_a, out_b]), out_full)
    torch.testing.assert_close(h_chunked, h_full)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
