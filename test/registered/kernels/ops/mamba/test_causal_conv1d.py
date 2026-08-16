"""Correctness coverage for the JIT depthwise causal conv1d kernels.

Two layers of checks:

* differential -- the JIT prefill/decode kernels must reproduce the AOT
  ``sgl_kernel`` ops they replace bit for bit, on every dispatch path (width,
  vectorized vs. scalar chunk load, varlen, circular buffer, gathered state).
* reference -- both the outputs and the advanced conv state are compared
  against a ``F.conv1d`` reference, so a shared bug in the two ports cannot
  hide behind the differential check.
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reference implementations adapted from
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/mamba/test_causal_conv1d.py

import sys
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import causal_conv1d_fwd as aot_causal_conv1d_fwd
from sgl_kernel import causal_conv1d_update as aot_causal_conv1d_update

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.mamba.causal_conv1d import (
    causal_conv1d_fwd as jit_causal_conv1d_fwd,
)
from sglang.kernels.ops.mamba.causal_conv1d import (
    causal_conv1d_update as jit_causal_conv1d_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=180, stage="nightly", runner_config="1-gpu-large")

PAD_SLOT_ID = -1
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# 8 covers the vectorized chunk path, 15 the scalar tail, 1025/4096 the
# multi-chunk conv-state stitching, 1/3 the seqlen < width branch.
FWD_SEQLENS = get_ci_test_range([1, 3, 8, 15, 128, 1025, 4096], [3, 15, 1025])


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.view(torch.uint8), expected.view(torch.uint8)
    ), "JIT and AOT outputs differ"


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 3e-4, 1e-3
    if dtype == torch.float16:
        return 3e-3, 5e-3
    return 1e-2, 5e-2


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """x: (batch, dim, seqlen); initial/final states: (batch, dim, width - 1)."""
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
    if final_states_out is not None:
        final_states_out.copy_(final_states)
    else:
        final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out, final_states_out


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """x: (batch, dim, seqlen); conv_state: (batch, dim, state_len)."""
    dtype_in = x.dtype
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def _make_fwd_inputs(dtype, batch, dim, seqlen, width, varlen, seed=0):
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)

    if varlen:
        x = randn(dim, seqlen)
        lengths = [seqlen // batch] * batch
        lengths[-1] += seqlen - sum(lengths)
        query_start_loc = torch.tensor(
            [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
            dtype=torch.int32,
            device=device,
        )
        cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
    else:
        x = randn(batch, dim, seqlen)
        query_start_loc = None
        cache_indices = None
    return {
        "x": x,
        "weight": randn(dim, width),
        "bias": randn(dim),
        "conv_states": randn(batch, dim, width - 1),
        "query_start_loc": query_start_loc,
        "cache_indices": cache_indices,
        "has_initial_state": torch.randint(
            0, 2, (batch,), dtype=torch.bool, device=device, generator=gen
        ),
    }


def _run_fwd(impl, inputs, silu_activation):
    x = inputs["x"].clone()
    conv_states = inputs["conv_states"].clone()
    impl(
        x,
        inputs["weight"],
        inputs["bias"],
        conv_states,
        inputs["query_start_loc"],
        inputs["cache_indices"],
        inputs["has_initial_state"],
        silu_activation,
        PAD_SLOT_ID,
    )
    return x, conv_states


def _make_update_inputs(
    dtype, batch, dim, seqlen, width, state_len, circular, gather, seed=0
):
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)

    entries = batch * 4 if gather else batch
    return {
        "x": randn(batch, dim, seqlen),
        "conv_state": randn(entries, dim, state_len),
        "weight": randn(dim, width),
        "bias": randn(dim),
        "cache_seqlens": (
            torch.randint(
                0, state_len, (batch,), dtype=torch.int32, device=device, generator=gen
            )
            if circular
            else None
        ),
        "conv_state_indices": (
            torch.randperm(entries, device=device, generator=gen)[:batch].to(
                torch.int32
            )
            if gather
            else None
        ),
    }


def _run_update(impl, inputs, silu_activation):
    x = inputs["x"].clone()
    conv_state = inputs["conv_state"].clone()
    impl(
        x,
        conv_state,
        inputs["weight"],
        inputs["bias"],
        silu_activation,
        inputs["cache_seqlens"],
        inputs["conv_state_indices"],
        PAD_SLOT_ID,
    )
    return x, conv_state


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", FWD_SEQLENS)
@pytest.mark.parametrize("silu_activation", [True, False])
def test_causal_conv1d_fwd_is_bit_exact(dtype, width, seqlen, silu_activation):
    """The JIT migration must preserve every output and conv-state bit."""
    inputs = _make_fwd_inputs(dtype, 1, 64, seqlen, width, varlen=False)
    actual = _run_fwd(jit_causal_conv1d_fwd, inputs, silu_activation)
    expected = _run_fwd(aot_causal_conv1d_fwd, inputs, silu_activation)
    _assert_bitwise_equal(actual[0], expected[0])
    _assert_bitwise_equal(actual[1], expected[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", [s for s in FWD_SEQLENS if s >= 8])
@pytest.mark.parametrize("silu_activation", [True, False])
def test_causal_conv1d_fwd_varlen_is_bit_exact(dtype, width, seqlen, silu_activation):
    """The varlen path always takes the scalar (non-vectorized) chunk load."""
    inputs = _make_fwd_inputs(dtype, 4, 64, seqlen, width, varlen=True)
    actual = _run_fwd(jit_causal_conv1d_fwd, inputs, silu_activation)
    expected = _run_fwd(aot_causal_conv1d_fwd, inputs, silu_activation)
    _assert_bitwise_equal(actual[0], expected[0])
    _assert_bitwise_equal(actual[1], expected[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", [1, 2, 5])
@pytest.mark.parametrize("state_len", [1, 8, 16])
@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize("gather", [True, False])
def test_causal_conv1d_update_is_bit_exact(
    dtype, width, seqlen, state_len, circular, gather
):
    """Cover both conv-state layouts (shift buffer / circular buffer)."""
    if state_len < width - 1:
        pytest.skip("conv_state must hold at least width - 1 taps")
    inputs = _make_update_inputs(
        dtype, 3, 2048 + 16, seqlen, width, state_len, circular, gather
    )
    actual = _run_update(jit_causal_conv1d_update, inputs, True)
    expected = _run_update(aot_causal_conv1d_update, inputs, True)
    _assert_bitwise_equal(actual[0], expected[0])
    _assert_bitwise_equal(actual[1], expected[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("seqlen", FWD_SEQLENS)
@pytest.mark.parametrize("has_initial_state", [True, False])
def test_causal_conv1d_fwd_matches_reference(dtype, width, seqlen, has_initial_state):
    """Output and final conv state must match an F.conv1d reference."""
    rtol, atol = _tolerance(dtype)
    inputs = _make_fwd_inputs(dtype, 1, 64, seqlen, width, varlen=False, seed=7)
    inputs["has_initial_state"] = torch.full(
        (1,), has_initial_state, dtype=torch.bool, device="cuda"
    )

    out, conv_states = _run_fwd(jit_causal_conv1d_fwd, inputs, silu_activation=True)
    out_ref, final_states_ref = causal_conv1d_ref(
        inputs["x"].clone(),
        inputs["weight"],
        inputs["bias"],
        initial_states=inputs["conv_states"].clone() if has_initial_state else None,
        activation="silu",
    )

    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(conv_states, final_states_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", [1, 5])
@pytest.mark.parametrize("silu_activation", [True, False])
def test_causal_conv1d_update_matches_reference(dtype, width, seqlen, silu_activation):
    rtol, atol = _tolerance(dtype)
    state_len = width - 1
    inputs = _make_update_inputs(
        dtype, 3, 2048, seqlen, width, state_len, circular=False, gather=False, seed=7
    )
    out, conv_state = _run_update(jit_causal_conv1d_update, inputs, silu_activation)
    conv_state_ref = inputs["conv_state"].clone()
    out_ref = causal_conv1d_update_ref(
        inputs["x"].clone(),
        conv_state_ref,
        inputs["weight"],
        inputs["bias"],
        activation="silu" if silu_activation else None,
    )
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(conv_state, conv_state_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("width", [2, 4])
def test_causal_conv1d_fwd_skips_padded_slots(width):
    """Varlen sequences whose cache index is pad_slot_id are not processed."""
    dtype = torch.bfloat16
    batch, padding, dim, seqlen, entries = 4, 3, 64, 512, 40
    device = "cuda"
    x = torch.randn(dim, seqlen, device=device, dtype=dtype)
    x_before = x.clone()
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    conv_states = torch.randn(entries, dim, width - 1, device=device, dtype=dtype)
    conv_states_before = conv_states.clone()
    # The trailing `padding` sequences are empty, so no output token belongs to
    # them; only their conv-state slots would be touched without the pad check.
    lengths = [seqlen // batch] * batch + [0] * padding
    lengths[batch - 1] += seqlen - sum(lengths)
    query_start_loc = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
        dtype=torch.int32,
        device=device,
    )
    indices = torch.randperm(entries, device=device)[:batch].to(torch.int32)
    padded_indices = torch.cat(
        [
            indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ]
    )
    has_initial_state = torch.zeros(batch + padding, dtype=torch.bool, device=device)

    jit_causal_conv1d_fwd(
        x,
        weight,
        None,
        conv_states,
        query_start_loc,
        padded_indices,
        has_initial_state,
        True,
        PAD_SLOT_ID,
    )

    untouched = torch.ones(entries, dtype=torch.bool, device=device)
    untouched[indices] = False
    assert torch.equal(conv_states[untouched], conv_states_before[untouched])
    assert not torch.equal(x, x_before)


@pytest.mark.parametrize("width", [2, 4])
def test_causal_conv1d_update_skips_padded_slots(width):
    """Slots marked with pad_slot_id must be left untouched."""
    dtype = torch.bfloat16
    batch, padding, dim, entries = 3, 5, 128, 30
    device = "cuda"
    x = torch.randn(batch + padding, dim, 1, device=device, dtype=dtype)
    conv_state = torch.randn(entries, dim, width - 1, device=device, dtype=dtype)
    conv_state_before = conv_state.clone()
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    indices = torch.randperm(entries, device=device)[:batch].to(torch.int32)
    padded_indices = torch.cat(
        [
            indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ]
    )

    jit_causal_conv1d_update(
        x, conv_state, weight, None, True, None, padded_indices, PAD_SLOT_ID
    )

    untouched = torch.ones(entries, dtype=torch.bool, device=device)
    untouched[indices] = False
    assert torch.equal(conv_state[untouched], conv_state_before[untouched])


def test_causal_conv1d_rejects_unsupported_dtype():
    x = torch.ones((1, 8, 4), dtype=torch.int32, device="cuda")
    weight = torch.ones((8, 4), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="Unsupported dtype"):
        jit_causal_conv1d_fwd(x, weight, None, None, None, None, None, True, -1)


def test_causal_conv1d_rejects_unsupported_width():
    x = torch.randn((1, 8, 4), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((8, 5), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception, match="width between 2 and 4"):
        jit_causal_conv1d_fwd(x, weight, None, None, None, None, None, True, -1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
