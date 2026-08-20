"""Correctness coverage for the JIT depthwise causal conv1d kernels.

Two layers. The reference tests are the broad gate: they compare against an
``F.conv1d`` formulation across the full dispatch grid, and depend only on the
kernel this repo builds. The differential tests are a small smoke set proving
the migration is bit-faithful to the AOT ops -- narrow on purpose, since they
compare two independently built binaries whose toolchains nothing pins together
(JIT: c++20 / sm_90a; wheel: c++17 / sm_90 / -DNDEBUG, and the pinned PyPI
release on scheduled runs). A bitwise failure with the reference cases green
points at the build environment, not a numerics regression.
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
WIDTHS = [2, 3, 4]
# 8/128 hit the vectorized chunk load (divisible by both vector widths), 15/1025
# the scalar path, 1025/4096 the multi-chunk conv-state stitching, and 1/3 the
# seqlen < width padding branch.
FWD_SEQLENS = get_ci_test_range([1, 3, 8, 15, 128, 1025, 4096], [3, 15, 1025])
# One seqlen per prefill dispatch path, for the bitwise smoke set.
SMOKE_SEQLENS = [15, 128, 1025]
UPDATE_SEQLENS = [1, 2, 5]
# Larger than the `width - 1` every in-tree caller allocates, to reach the shift
# loop that only runs when state_len > width - 1.
UPDATE_STATE_LENS = [8, 16]


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
        "JIT and AOT outputs differ bit-for-bit. If the reference tests in this "
        "file pass, suspect a JIT-vs-wheel toolchain divergence before a "
        "numerics regression -- see the module docstring."
    )


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


###############################################################################
# Reference coverage -- the broad gate.
###############################################################################


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
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
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seqlen", [s for s in FWD_SEQLENS if s >= 8])
@pytest.mark.parametrize("silu_activation", [True, False])
def test_causal_conv1d_fwd_varlen_matches_reference(
    dtype, width, seqlen, silu_activation
):
    """Varlen prefill, per sequence. The layout serving uses; always scalar load."""
    rtol, atol = _tolerance(dtype)
    batch = 4
    inputs = _make_fwd_inputs(dtype, batch, 64, seqlen, width, varlen=True, seed=11)
    activation = "silu" if silu_activation else None

    out, conv_states = _run_fwd(jit_causal_conv1d_fwd, inputs, silu_activation)

    conv_states_ref = inputs["conv_states"].clone()
    starts = inputs["query_start_loc"].tolist()
    for i in range(batch):
        slot = int(inputs["cache_indices"][i])
        x_s = inputs["x"][:, starts[i] : starts[i + 1]].unsqueeze(0)
        out_ref, _ = causal_conv1d_ref(
            x_s.clone(),
            inputs["weight"],
            inputs["bias"],
            initial_states=(
                conv_states_ref[slot].unsqueeze(0).clone()
                if inputs["has_initial_state"][i]
                else None
            ),
            final_states_out=conv_states_ref[slot].unsqueeze(0),
            activation=activation,
        )
        torch.testing.assert_close(
            out[:, starts[i] : starts[i + 1]].unsqueeze(0),
            out_ref,
            rtol=rtol,
            atol=atol,
        )
    torch.testing.assert_close(conv_states, conv_states_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seqlen", UPDATE_SEQLENS)
@pytest.mark.parametrize("state_len_kind", ["exact", *UPDATE_STATE_LENS])
@pytest.mark.parametrize("circular", [True, False])
def test_causal_conv1d_update_matches_reference(
    dtype, width, seqlen, state_len_kind, circular
):
    """Both conv-state layouts: tail-anchored shift buffer and circular buffer."""
    state_len = width - 1 if state_len_kind == "exact" else state_len_kind
    if circular and seqlen > state_len:
        # The reference advances the ring with `scatter_`, whose behavior for the
        # duplicate indices this produces is unspecified -- it cannot arbitrate.
        pytest.skip("circular reference is ambiguous when seqlen > state_len")
    rtol, atol = _tolerance(dtype)
    inputs = _make_update_inputs(
        dtype, 3, 2048, seqlen, width, state_len, circular, gather=False, seed=7
    )
    out, conv_state = _run_update(jit_causal_conv1d_update, inputs, True)
    conv_state_ref = inputs["conv_state"].clone()
    out_ref = causal_conv1d_update_ref(
        inputs["x"].clone(),
        conv_state_ref,
        inputs["weight"],
        inputs["bias"],
        activation="silu",
        cache_seqlens=inputs["cache_seqlens"],
    )
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(conv_state, conv_state_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seqlen", UPDATE_SEQLENS)
def test_causal_conv1d_update_gather_matches_reference(dtype, width, seqlen):
    """Gathered decode: only the indexed slots advance, and they match the ref."""
    rtol, atol = _tolerance(dtype)
    inputs = _make_update_inputs(
        dtype, 3, 2048, seqlen, width, width - 1, circular=False, gather=True, seed=13
    )
    indices = inputs["conv_state_indices"]
    out, conv_state = _run_update(jit_causal_conv1d_update, inputs, True)

    conv_state_ref = inputs["conv_state"][indices].clone()
    out_ref = causal_conv1d_update_ref(
        inputs["x"].clone(),
        conv_state_ref,
        inputs["weight"],
        inputs["bias"],
        activation="silu",
    )
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        conv_state[indices], conv_state_ref, rtol=rtol, atol=atol
    )

    untouched = torch.ones(conv_state.shape[0], dtype=torch.bool, device="cuda")
    untouched[indices] = False
    assert torch.equal(conv_state[untouched], inputs["conv_state"][untouched])


###############################################################################
# Differential smoke set -- the migration proof.
###############################################################################


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seqlen", SMOKE_SEQLENS)
def test_causal_conv1d_fwd_is_bit_exact(dtype, width, seqlen):
    """One seqlen per prefill dispatch path: vectorized, scalar, multi-chunk."""
    inputs = _make_fwd_inputs(dtype, 1, 64, seqlen, width, varlen=False)
    actual = _run_fwd(jit_causal_conv1d_fwd, inputs, True)
    expected = _run_fwd(aot_causal_conv1d_fwd, inputs, True)
    _assert_bitwise_equal(actual[0], expected[0])
    _assert_bitwise_equal(actual[1], expected[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
def test_causal_conv1d_fwd_varlen_is_bit_exact(dtype, width):
    inputs = _make_fwd_inputs(dtype, 4, 64, 1025, width, varlen=True)
    actual = _run_fwd(jit_causal_conv1d_fwd, inputs, True)
    expected = _run_fwd(aot_causal_conv1d_fwd, inputs, True)
    _assert_bitwise_equal(actual[0], expected[0])
    _assert_bitwise_equal(actual[1], expected[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize("gather", [True, False])
def test_causal_conv1d_update_is_bit_exact(dtype, width, circular, gather):
    """Both state layouts crossed with both slot-addressing modes."""
    inputs = _make_update_inputs(dtype, 3, 2048 + 16, 2, width, 8, circular, gather)
    actual = _run_update(jit_causal_conv1d_update, inputs, True)
    expected = _run_update(aot_causal_conv1d_update, inputs, True)
    _assert_bitwise_equal(actual[0], expected[0])
    _assert_bitwise_equal(actual[1], expected[1])


###############################################################################
# Padding and argument validation.
###############################################################################


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


@pytest.mark.parametrize("bad_dtype", [torch.uint8, torch.int32])
def test_causal_conv1d_rejects_non_bool_has_initial_state(bad_dtype):
    """Nothing normalizes this mask on the way in (unlike `cache_indices`), so a
    wider dtype would silently read the wrong byte per sequence."""
    x = torch.randn((1, 8, 4), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((8, 4), dtype=torch.bfloat16, device="cuda")
    conv_states = torch.zeros((1, 8, 3), dtype=torch.bfloat16, device="cuda")
    has_initial_state = torch.ones((1,), dtype=bad_dtype, device="cuda")
    with pytest.raises(Exception, match="has_initial_state must be a bool tensor"):
        jit_causal_conv1d_fwd(
            x, weight, None, conv_states, None, None, has_initial_state, True, -1
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
