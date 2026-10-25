# SPDX-License-Identifier: Apache-2.0

import sys
from itertools import accumulate

import pytest
import torch

from sglang.kernels.ops.mamba.triton_ops import mamba_chunk_scan_combined
from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import chunk_state_varlen
from sglang.kernels.ops.mamba.triton_ops.ssd_combined import (
    _mamba_chunk_scan_combined_fwd,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")


def _sequence(length, *, heads, width, state_size, dtype):
    return (
        torch.randn(1, length, heads, width, device="cuda", dtype=dtype),
        torch.rand(1, length, heads, device="cuda") * 0.1,
        torch.randn(1, length, 2, state_size, device="cuda", dtype=dtype),
        torch.randn(1, length, 2, state_size, device="cuda", dtype=dtype),
        torch.randn(1, length, heads, width, device="cuda", dtype=dtype),
    )


def _scan(
    sequences,
    *,
    A,
    D,
    bias,
    softplus,
    initial,
    state_dtype,
    target,
    with_z,
    track=True,
    cpu_boundaries=True,
):
    lengths = [sequence[0].shape[1] for sequence in sequences]
    boundaries = (0, *accumulate(lengths))
    x, dt, B, C, z = (torch.cat(values, dim=1) for values in zip(*sequences))
    if softplus:
        dt = dt - 4
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    indices = offsets = None
    if initial is not None:
        indices, offsets = Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
            cu, 64, boundaries[-1]
        )
    out = torch.empty_like(x)
    result = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        64,
        D=D,
        z=z if with_z else None,
        dt_bias=bias,
        dt_softplus=softplus,
        initial_states=initial,
        cu_seqlens=cu,
        seq_idx=torch.repeat_interleave(
            torch.arange(len(lengths), device="cuda", dtype=torch.int32),
            cu.diff(),
            output_size=boundaries[-1],
        ).unsqueeze(0),
        chunk_indices=indices,
        chunk_offsets=offsets,
        out=out,
        state_dtype=state_dtype,
        return_varlen_states=True,
        return_track_states=track,
        track_seq_idx=torch.tensor([target], device="cuda") if track else None,
        track_end_locs=torch.tensor([boundaries[target] + 64], device="cuda")
        if track
        else None,
        batch_invariant=True,
        batch_invariant_boundaries=boundaries if cpu_boundaries else None,
    )
    return (out, result[1], result[2]) if track else (out, result, None)


@pytest.mark.parametrize("D_kind", ["none", "head", "element"])
@pytest.mark.parametrize("with_z", [False, True])
@pytest.mark.parametrize(
    "softplus,with_bias", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("with_initial", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_batch_invariant_optional_arguments(
    D_kind, with_z, softplus, with_bias, with_initial, dtype, state_dtype
):
    torch.manual_seed(43413)
    heads, width, state_size = 4, 80 if D_kind == "element" else 32, 16
    inputs = dict(heads=heads, width=width, state_size=state_size, dtype=dtype)
    target = _sequence(129, **inputs)
    neighbor = _sequence(31, **inputs)
    A = -torch.exp(torch.rand(heads, device="cuda"))
    D = (
        None
        if D_kind == "none"
        else torch.randn(
            (heads, width) if D_kind == "element" else (heads,), device="cuda"
        )
    )
    bias = torch.randn(heads, device="cuda") * 0.01 if with_bias else None
    initial = torch.randn(heads, width, state_size, device="cuda", dtype=state_dtype)
    neighbor_initial = torch.randn_like(initial)
    options = dict(
        A=A, D=D, bias=bias, softplus=softplus, state_dtype=state_dtype, with_z=with_z
    )
    reference = _scan(
        [target],
        target=0,
        initial=initial.unsqueeze(0) if with_initial else None,
        **options,
    )
    for position in range(3):
        sequences = [neighbor, neighbor]
        sequences.insert(position, target)
        states = [neighbor_initial, neighbor_initial]
        states.insert(position, initial)
        actual = _scan(
            sequences,
            target=position,
            initial=torch.stack(states) if with_initial else None,
            cpu_boundaries=position != 1,
            **options,
        )
        start = position * 31
        torch.testing.assert_close(
            actual[0][:, start : start + 129], reference[0], atol=0, rtol=0
        )
        torch.testing.assert_close(actual[1][position], reference[1][0], atol=0, rtol=0)
        torch.testing.assert_close(actual[2], reference[2], atol=0, rtol=0)
    untracked = _scan(
        sequences,
        target=position,
        track=False,
        initial=torch.stack(states) if with_initial else None,
        **options,
    )
    torch.testing.assert_close(untracked[0], actual[0], atol=0, rtol=0)
    torch.testing.assert_close(untracked[1], actual[1], atol=0, rtol=0)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 129, 257, 769, 1025])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_initial", [False, True])
def test_batch_invariant_final_scan_state(length, dtype, state_dtype, with_initial):
    """Even a partial last chunk must return the scan's own checkpoint."""
    torch.manual_seed(43413)
    heads, width, state_size = 4, 32, 16
    x, dt, B, C, _ = _sequence(
        length, heads=heads, width=width, state_size=state_size, dtype=dtype
    )
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    initial = (
        torch.randn(1, heads, width, state_size, device="cuda", dtype=state_dtype)
        if with_initial
        else None
    )
    indices = offsets = None
    if with_initial:
        indices, offsets = Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
            cu, 64, length
        )
    result = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        -torch.exp(torch.rand(heads, device="cuda")),
        B,
        C,
        64,
        initial_states=initial,
        seq_idx=torch.zeros(1, length, device="cuda", dtype=torch.int32),
        chunk_indices=indices,
        chunk_offsets=offsets,
        cu_seqlens=cu,
        state_dtype=state_dtype,
        out=torch.empty_like(x),
        use_scan_boundary_states=True,
    )
    final_state, varlen_state = result[4:6]
    torch.testing.assert_close(
        varlen_state, final_state.to(state_dtype), atol=0, rtol=0
    )


@pytest.mark.parametrize("width,state_size", [(5, 19), (80, 128)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_batch_invariant_boundary_state_copies(width, state_size, state_dtype, strided):
    """Copy aligned and final states without assuming contiguous input states."""
    torch.manual_seed(43413)
    heads, length, chunk_size = 3, 129, 64
    storage_width = state_size * 2 if strided else state_size
    chunks = torch.randn(
        3, heads, width, storage_width, device="cuda", dtype=state_dtype
    )
    final = torch.randn(1, heads, width, storage_width, device="cuda")
    if strided:
        chunks = chunks[..., ::2]
        final = final[..., ::2]
    states = chunk_state_varlen(
        torch.zeros(length, 1, state_size, device="cuda", dtype=torch.bfloat16),
        torch.zeros(length, heads, width, device="cuda", dtype=torch.bfloat16),
        torch.zeros(heads, 3, chunk_size, device="cuda"),
        torch.zeros(heads, 3, chunk_size, device="cuda"),
        torch.tensor([0, 64, 99, 128, 129], device="cuda", dtype=torch.int32),
        chunks,
        final_states=final,
    )
    torch.testing.assert_close(states[0], chunks[1], atol=0, rtol=0)
    torch.testing.assert_close(states[1], torch.zeros_like(states[1]), atol=0, rtol=0)
    torch.testing.assert_close(states[2], chunks[2], atol=0, rtol=0)
    torch.testing.assert_close(states[3], final[0].to(state_dtype), atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
