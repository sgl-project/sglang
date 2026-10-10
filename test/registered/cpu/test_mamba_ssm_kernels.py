import importlib.util
from pathlib import Path

import pytest
import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.triton_ops import mamba_chunk_scan_combined
from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import (
    PAD_SLOT_ID,
    selective_state_update,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu-intel")


def _load_existing_reference(file_name, attr):
    path = Path(__file__).resolve().parents[1] / "layers" / "mamba" / file_name
    spec = importlib.util.spec_from_file_location(f"_ref_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


# PyTorch references used by the existing CUDA/Triton kernel tests
ssd_minimal_discrete = _load_existing_reference(
    "test_mamba_ssm_ssd.py", "ssd_minimal_discrete"
)
selective_state_update_ref = _load_existing_reference(
    "test_mamba_ssm.py", "selective_state_update_ref"
)


def _mamba_scan_reference(x, dt, A, B, C, D, z, dt_bias, initial_states, dt_limit):
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    group_ratio = nheads // ngroups
    if initial_states is None:
        state = torch.zeros(batch, nheads, headdim, B.shape[-1])
    else:
        state = initial_states.float().clone()
    if D is not None and D.dim() == 1:
        D = D[:, None]
    outputs = []
    for token in range(seqlen):
        delta = F.softplus(dt[:, token].float() + dt_bias.float())
        delta = delta.clamp(*dt_limit)
        b_token = B[:, token].float().repeat_interleave(group_ratio, dim=1)
        c_token = C[:, token].repeat_interleave(group_ratio, dim=1)
        decay = torch.exp(delta * A.float()[None, :])[:, :, None, None]
        state = (
            state * decay
            + delta[:, :, None, None]
            * b_token[:, :, None, :]
            * x[:, token].float()[:, :, :, None]
        )
        value = (state * c_token.float()[:, :, None, :]).sum(dim=-1)
        if D is not None:
            value = value + (x[:, token] * D).to(value.dtype)
        if z is not None:
            value = value * F.silu(z[:, token])
        outputs.append(value.to(x.dtype))
    return torch.stack(outputs, dim=1), state.to(C.dtype)


def _assert_close_scaled(actual, expected, dtype):
    # low precision chunked GEMMs accumulate error proportional to the output scale
    factor = 1e-2 if dtype == torch.bfloat16 else 2e-3
    atol = max(factor * expected.float().abs().max().item(), 1e-2)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.03, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "seqlen", "chunk_size", "nheads", "headdim", "dstate", "ngroups"),
    [
        (torch.bfloat16, 13, 4, 4, 16, 32, 2),
        (torch.float16, 19, 8, 8, 16, 64, 4),
    ],
)
@pytest.mark.parametrize("has_z", [False, True])
def test_mamba_chunk_scan_combined_cpu(
    dtype, seqlen, chunk_size, nheads, headdim, dstate, ngroups, has_z
):
    torch.manual_seed(23)
    batch = 2
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype)
    dt = torch.randn(batch, seqlen, nheads, dtype=dtype)
    A = -torch.rand(nheads, dtype=torch.float32) - 0.1
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype)
    C = torch.randn_like(B)
    D = torch.randn(nheads, headdim, dtype=torch.float32)
    z = torch.randn_like(x) if has_z else None
    dt_bias = torch.randn(nheads, dtype=torch.float32) - 3.0
    initial_states = torch.randn(batch, nheads, headdim, dstate, dtype=dtype) * 0.1
    out = torch.empty_like(x)

    final_state = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=True,
        dt_limit=(0.0, 3.0),
        out=out,
        return_final_states=True,
    )
    expected_out, expected_state = _mamba_scan_reference(
        x, dt, A, B, C, D, z, dt_bias, initial_states, (0.0, 3.0)
    )

    _assert_close_scaled(out, expected_out, dtype)
    _assert_close_scaled(final_state, expected_state, dtype)


# (seqlen, chunk_size, nheads, headdim, dstate, ngroups, has_init, D_1d)
SCAN_EDGE_CASES = [
    (1, 64, 4, 64, 128, 1, True, True),  # single token
    (
        5,
        64,
        4,
        24,
        48,
        2,
        False,
        False,
    ),  # shorter than a chunk, dims not multiple of 32
    (128, 64, 6, 32, 64, 3, False, True),  # exact multiple of chunk_size
    (300, 256, 8, 64, 128, 8, True, True),  # chunk 256 with a tail chunk
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    (
        "seqlen",
        "chunk_size",
        "nheads",
        "headdim",
        "dstate",
        "ngroups",
        "has_init",
        "D_1d",
    ),
    SCAN_EDGE_CASES,
)
def test_mamba_chunk_scan_combined_cpu_edge_cases(
    dtype, seqlen, chunk_size, nheads, headdim, dstate, ngroups, has_init, D_1d
):
    torch.manual_seed(7)
    batch = 1
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype)
    dt = torch.randn(batch, seqlen, nheads, dtype=dtype)
    A = -torch.rand(nheads) - 0.1
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype)
    C = torch.randn_like(B)
    D = torch.randn(nheads) if D_1d else torch.randn(nheads, headdim)
    dt_bias = torch.randn(nheads) - 3.0
    initial_states = (
        torch.randn(batch, nheads, headdim, dstate, dtype=dtype) * 0.1
        if has_init
        else None
    )
    out = torch.empty_like(x)

    final_state = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=True,
        out=out,
        return_final_states=True,
    )
    expected_out, expected_state = _mamba_scan_reference(
        x, dt, A, B, C, D, None, dt_bias, initial_states, (0.0, float("inf"))
    )

    _assert_close_scaled(out, expected_out, dtype)
    _assert_close_scaled(final_state, expected_state, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("seqlen", "chunk_size", "nheads", "headdim", "dstate", "ngroups"),
    [
        (256, 64, 8, 64, 128, 2),
        (512, 256, 16, 64, 128, 1),  # Mamba2 / hybrid-attention layer shape
    ],
)
def test_mamba_chunk_scan_combined_cpu_vs_ssd_minimal(
    dtype, seqlen, chunk_size, nheads, headdim, dstate, ngroups
):
    torch.manual_seed(0)
    batch = 1
    A = -torch.exp(torch.rand(nheads))
    dt = F.softplus(torch.randn(batch, seqlen, nheads) - 4).to(dtype)
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype)
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype)
    C = torch.randn_like(B)
    initial_states = torch.randn(batch, nheads, headdim, dstate, dtype=dtype) * 0.1
    out = torch.empty_like(x)

    final_state = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        initial_states=initial_states,
        out=out,
        return_final_states=True,
    )

    ratio = nheads // ngroups
    dt_f = dt.float()
    expected_out, expected_state = ssd_minimal_discrete(
        x.float() * dt_f.unsqueeze(-1),
        A[None, None] * dt_f,
        B.float().repeat_interleave(ratio, dim=2),
        C.float().repeat_interleave(ratio, dim=2),
        chunk_size,
        initial_states=initial_states.float()[:, None],
    )

    atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 5e-3)
    torch.testing.assert_close(out.float(), expected_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        final_state.float(), expected_state, atol=atol, rtol=rtol
    )


def test_mamba_chunk_scan_combined_cpu_fp32_fallback():
    torch.manual_seed(3)
    x = torch.randn(1, 20, 4, 16)
    dt = torch.randn(1, 20, 4)
    A = -torch.rand(4) - 0.1
    B = torch.randn(1, 20, 2, 32)
    C = torch.randn_like(B)
    D = torch.randn(4, 16)
    dt_bias = torch.randn(4) - 3.0
    out = torch.empty_like(x)

    final_state = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        8,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
        return_final_states=True,
    )
    expected_out, expected_state = _mamba_scan_reference(
        x, dt, A, B, C, D, None, dt_bias, None, (0.0, float("inf"))
    )
    torch.testing.assert_close(out, expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-4, atol=1e-4)


def test_mamba_chunk_scan_combined_cpu_varlen_tracks():
    torch.manual_seed(41)
    dtype = torch.bfloat16
    seqlens = [3, 7]
    seqlen = sum(seqlens)
    batch, nheads, headdim, dstate, ngroups = 1, 4, 8, 16, 2
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype)
    dt = torch.randn(batch, seqlen, nheads, dtype=dtype)
    A = -torch.rand(nheads, dtype=torch.float32) - 0.1
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype)
    C = torch.randn_like(B)
    dt_bias = torch.randn(nheads, dtype=torch.float32) - 3.0
    initial_states = torch.randn(2, nheads, headdim, dstate, dtype=dtype) * 0.1
    cu_seqlens = torch.tensor([0, seqlens[0], seqlen], dtype=torch.int32)
    seq_idx = torch.cat([torch.zeros(seqlens[0]), torch.ones(seqlens[1])]).to(
        torch.int32
    )[None]
    track_seq_idx = torch.tensor([0, 1], dtype=torch.int32)
    track_end_locs = torch.tensor([2, 8], dtype=torch.int32)
    out = torch.empty_like(x)

    _, varlen_states, track_states = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        4,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        chunk_offsets=torch.tensor([0, 0], dtype=torch.int32),
        cu_seqlens=cu_seqlens,
        dt_softplus=True,
        out=out,
        return_varlen_states=True,
        return_track_states=True,
        track_seq_idx=track_seq_idx,
        track_end_locs=track_end_locs,
    )

    expected_out = torch.empty_like(x)
    expected_final = []
    expected_track = []
    start = 0
    for sequence, length in enumerate(seqlens):
        end = start + length
        sequence_args = (
            x[:, start:end],
            dt[:, start:end],
            A,
            B[:, start:end],
            C[:, start:end],
            None,
            None,
            dt_bias,
        )
        sequence_out, sequence_final = _mamba_scan_reference(
            *sequence_args,
            initial_states[sequence : sequence + 1],
            (0.0, float("inf")),
        )
        expected_out[:, start:end].copy_(sequence_out)
        expected_final.append(sequence_final[0])
        tracked_length = int(track_end_locs[sequence]) - start
        _, tracked_final = _mamba_scan_reference(
            x[:, start : start + tracked_length],
            dt[:, start : start + tracked_length],
            A,
            B[:, start : start + tracked_length],
            C[:, start : start + tracked_length],
            None,
            None,
            dt_bias,
            initial_states[sequence : sequence + 1],
            (0.0, float("inf")),
        )
        expected_track.append(tracked_final[0])
        start = end

    torch.testing.assert_close(out, expected_out, rtol=0.03, atol=0.06)
    torch.testing.assert_close(
        varlen_states, torch.stack(expected_final), rtol=0.03, atol=0.06
    )
    torch.testing.assert_close(
        track_states, torch.stack(expected_track), rtol=0.03, atol=0.06
    )


def _selective_state_update_reference(
    state, x, dt, A, B, C, D, z, dt_bias, state_batch_indices, disable_state_update
):
    output = torch.full_like(x, 17.0)
    expected_state = state.clone()
    batch, time, nheads, _ = x.shape
    ngroups = B.shape[2]
    for batch_idx in range(batch):
        slot = int(state_batch_indices[batch_idx])
        if slot == PAD_SLOT_ID:
            continue
        working_state = expected_state[slot].float().clone()
        for token in range(time):
            delta = F.softplus(dt[batch_idx, token].float() + dt_bias.float())
            for head in range(nheads):
                group = head // (nheads // ngroups)
                old_state = working_state[head]
                new_state = (
                    old_state * torch.exp(delta[head, :, None] * A[head].float())
                    + delta[head, :, None]
                    * B[batch_idx, token, group].float()[None, :]
                    * x[batch_idx, token, head].float()[:, None]
                )
                working_state[head] = new_state
                value = (new_state * C[batch_idx, token, group].float()).sum(dim=-1)
                value = value + (x[batch_idx, token, head] * D[head]).to(value.dtype)
                if z is not None:
                    value = value * F.silu(z[batch_idx, token, head])
                output[batch_idx, token, head] = value.to(x.dtype)
        if not disable_state_update:
            expected_state[slot].copy_(working_state.to(state.dtype))
    return output, expected_state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("disable_state_update", [False, True])
def test_selective_state_update_cpu(dtype, dstate, disable_state_update):
    torch.manual_seed(31)
    batch, time, nheads, dim, ngroups = 3, 3, 4, 16, 2
    state = torch.randn(6, nheads, dim, dstate, dtype=dtype)
    state_before = state.clone()
    x = torch.randn(batch, time, nheads, dim, dtype=dtype)
    dt = torch.randn_like(x)
    A = -torch.rand(nheads, dim, dstate, dtype=torch.float32) - 0.5
    B = torch.randn(batch, time, ngroups, dstate, dtype=dtype)
    C = torch.randn_like(B)
    D = torch.randn(nheads, dim, dtype=torch.float32)
    z = torch.randn_like(x)
    dt_bias = torch.randn(nheads, dim, dtype=torch.float32) - 3.0
    state_batch_indices = torch.tensor([4, PAD_SLOT_ID, 1], dtype=torch.int32)
    out = torch.full_like(x, 17.0)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        pad_slot_id=PAD_SLOT_ID,
        out=out,
        disable_state_update=disable_state_update,
    )
    expected_out, expected_state = _selective_state_update_reference(
        state_before,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        state_batch_indices,
        disable_state_update,
    )

    atol = 0.08 if dtype == torch.bfloat16 else 0.02
    torch.testing.assert_close(out, expected_out, rtol=0.04, atol=atol)
    torch.testing.assert_close(state, expected_state, rtol=0.02, atol=atol)
    assert torch.equal(out[1], torch.full_like(out[1], 17.0))
    assert torch.equal(state[0], state_before[0])
    assert torch.equal(state[2], state_before[2])
    assert torch.equal(state[3], state_before[3])
    assert torch.equal(state[5], state_before[5])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [torch.float32, None])
@pytest.mark.parametrize(
    ("batch", "nheads", "headdim", "dstate", "ngroups"),
    [
        (1, 8, 64, 128, 1),
        (8, 16, 64, 128, 4),
    ],
)
def test_selective_state_update_cpu_mamba2_layout(
    dtype, state_dtype, batch, nheads, headdim, dstate, ngroups
):
    # same expanded (stride 0) A/dt/dt_bias/D layout as the Mamba2 decode path
    torch.manual_seed(5)
    state_dtype = state_dtype or dtype
    total_slots = 4 * batch
    state = torch.randn(total_slots, nheads, headdim, dstate, dtype=state_dtype)
    state_indices = torch.randperm(total_slots)[:batch].to(torch.int32)
    A = -torch.rand(nheads) - 1.0
    A_d = A[:, None, None].expand(-1, headdim, dstate)
    dt_d = torch.randn(batch, nheads, dtype=dtype)[:, :, None].expand(-1, -1, headdim)
    dt_bias = (torch.rand(nheads) - 4.0)[:, None].expand(-1, headdim)
    D_d = torch.randn(nheads)[:, None].expand(-1, headdim)
    x = torch.randn(batch, nheads, headdim, dtype=dtype)
    B = torch.randn(batch, ngroups, dstate, dtype=dtype)
    C = torch.randn_like(B)
    out = torch.empty_like(x)
    state_ref = state[state_indices].clone()

    selective_state_update(
        state,
        x,
        dt_d,
        A_d,
        B,
        C,
        D_d,
        z=None,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices,
        out=out,
    )
    out_ref = selective_state_update_ref(
        state_ref, x, dt_d, A_d, B, C, D=D_d, dt_bias=dt_bias, dt_softplus=True
    )

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (5e-3, 3e-2)
    torch.testing.assert_close(state[state_indices], state_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_selective_state_update_cpu_mamba1_layout(dtype):
    # Mamba1 decode maps onto (nheads=dim, head_dim=1, ngroups=1) with a z gate
    torch.manual_seed(9)
    batch, dim, dstate = 4, 256, 16
    state = torch.randn(batch, dim, 1, dstate)
    x = torch.randn(batch, dim, 1, dtype=dtype)
    dt = torch.randn(batch, dim, 1, dtype=dtype)
    A = -torch.rand(dim, 1, dstate) - 0.5
    B = torch.randn(batch, 1, dstate, dtype=dtype)
    C = torch.randn_like(B)
    D = torch.randn(dim, 1)
    z = torch.randn_like(x)
    dt_bias = torch.zeros(dim, 1)
    out = torch.empty_like(x)
    state_ref = state.clone()

    selective_state_update(
        state, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True, out=out
    )
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (5e-3, 3e-2)
    torch.testing.assert_close(state, state_ref, rtol=3e-4, atol=1e-3)
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
