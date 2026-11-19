"""Metal GDN kernels on Apple MPS must match the plain-torch oracle."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.linear.gdn_mps import (
    causal_conv1d_update_mps,
    causal_conv1d_varlen_mps,
    gated_delta_rule_mps,
)
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.kernels import gdn_reference as ref

register_mps_ci(est_time=180, suite="stage-a-unit-test-mps")

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)

DEV = "mps"
DTYPES = [torch.bfloat16, torch.float32]


def _randn(*shape, gen, dtype, scale=1.0):
    return (torch.randn(*shape, generator=gen) * scale).to(dtype)


def _qsl(lens):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)


def _check_untouched(got_pool, original_pool, layer):
    for other in range(original_pool.shape[0]):
        if other != layer:
            assert torch.equal(got_pool[other], original_pool[other])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_conv_decode(dtype, batch):
    gen = torch.Generator().manual_seed(batch)
    channels, width, slots = 64, 4, 24
    x = _randn(batch, channels, gen=gen, dtype=dtype)
    pool = _randn(2, slots, channels, width - 1, gen=gen, dtype=dtype)
    weight = _randn(channels, width, gen=gen, dtype=dtype, scale=0.2)
    bias = _randn(channels, gen=gen, dtype=dtype, scale=0.1)
    indices = torch.arange(batch, dtype=torch.int32) % slots
    indices[1::3] = -1  # padding rows keep their output zero and slots untouched
    expected = pool.clone()
    device_pool = pool.to(DEV)

    y = causal_conv1d_update_mps(
        x.to(DEV), device_pool[1], weight.to(DEV), bias.to(DEV), indices.to(DEV)
    )
    y_ref = ref.causal_conv1d_update_reference(x, expected[1], weight, bias, indices)

    torch.testing.assert_close(y.cpu(), y_ref, atol=2e-2, rtol=2e-2)
    got_pool = device_pool.cpu()
    torch.testing.assert_close(got_pool[1], expected[1], atol=0, rtol=0)
    _check_untouched(got_pool, pool, 1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_conv_varlen(dtype):
    gen = torch.Generator().manual_seed(7)
    channels, width, slots = 64, 4, 8
    # Lengths 1 and 2 are shorter than the K-1 window; one sequence is padding.
    lens = (5, 2, 1, 7)
    indices = torch.tensor([1, 3, 5, -1], dtype=torch.int32)
    initial = torch.tensor([True, False, True, True])
    x = _randn(sum(lens), channels, gen=gen, dtype=dtype)
    pool = _randn(slots, channels, width - 1, gen=gen, dtype=dtype)
    weight = _randn(channels, width, gen=gen, dtype=dtype, scale=0.2)
    bias = _randn(channels, gen=gen, dtype=dtype, scale=0.1)
    expected = pool.clone()
    device_pool = pool.to(DEV)

    y = causal_conv1d_varlen_mps(
        x.to(DEV),
        device_pool,
        weight.to(DEV),
        bias.to(DEV),
        indices.to(DEV),
        _qsl(lens).to(DEV),
        initial.to(DEV),
    )
    y_ref = ref.causal_conv1d_varlen_reference(
        x, expected, weight, bias, indices, _qsl(lens), initial
    )

    torch.testing.assert_close(y.cpu(), y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(device_pool.cpu(), expected, atol=0, rtol=0)


def test_conv_varlen_empty_sequences():
    # Empty sequences emit no windows; their state is kept or cleared by the flag.
    gen = torch.Generator().manual_seed(8)
    channels, width = 16, 4
    lens = (0, 3, 0)
    indices = torch.tensor([0, 2, 4], dtype=torch.int32)
    initial = torch.tensor([False, True, False])
    x = _randn(sum(lens), channels, gen=gen, dtype=torch.float32)
    pool = _randn(6, channels, width - 1, gen=gen, dtype=torch.float32)
    weight = _randn(channels, width, gen=gen, dtype=torch.float32, scale=0.2)
    bias = _randn(channels, gen=gen, dtype=torch.float32, scale=0.1)
    expected = pool.clone()
    device_pool = pool.to(DEV)

    y = causal_conv1d_varlen_mps(
        x.to(DEV),
        device_pool,
        weight.to(DEV),
        bias.to(DEV),
        indices.to(DEV),
        _qsl(lens).to(DEV),
        initial.to(DEV),
    )
    y_ref = ref.causal_conv1d_varlen_reference(
        x, expected, weight, bias, indices, _qsl(lens), initial
    )

    torch.testing.assert_close(y.cpu(), y_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(device_pool.cpu(), expected, atol=0, rtol=0)


def test_conv_varlen_rejects_non_bool_initial_state():
    gen = torch.Generator().manual_seed(9)
    channels, width = 16, 4
    x = _randn(1, channels, gen=gen, dtype=torch.float32)
    pool = _randn(1, channels, width - 1, gen=gen, dtype=torch.float32)
    weight = _randn(channels, width, gen=gen, dtype=torch.float32)

    with pytest.raises(ValueError, match="has_initial_state must be bool"):
        causal_conv1d_varlen_mps(
            x.to(DEV),
            pool.to(DEV),
            weight.to(DEV),
            None,
            torch.tensor([0], dtype=torch.int32, device=DEV),
            torch.tensor([0, 1], dtype=torch.int32, device=DEV),
            torch.tensor([1], dtype=torch.int32, device=DEV),
        )


def _recurrent_inputs(gen, dtype, lens, hk=2, hv=4, dk=128, dv=128, slots=8):
    total = sum(lens)
    q = _randn(total, hk, dk, gen=gen, dtype=dtype)
    k = _randn(total, hk, dk, gen=gen, dtype=dtype)
    v = _randn(total, hv, dv, gen=gen, dtype=dtype)
    a = _randn(total, hv, gen=gen, dtype=dtype, scale=0.5)
    b = _randn(total, hv, gen=gen, dtype=dtype)
    A_log = _randn(hv, gen=gen, dtype=torch.float32, scale=0.5)
    dt_bias = _randn(hv, gen=gen, dtype=torch.float32, scale=0.1)
    pool = _randn(2, slots, hv, dv, dk, gen=gen, dtype=torch.float32, scale=0.1)
    return q, k, v, a, b, A_log, dt_bias, pool


def _run_recurrent(tensors, indices, lens, initial):
    q, k, v, a, b, A_log, dt_bias, pool = tensors
    expected = pool.clone()
    device_pool = pool.to(DEV)
    o = gated_delta_rule_mps(
        q.to(DEV),
        k.to(DEV),
        v.to(DEV),
        a.to(DEV),
        b.to(DEV),
        A_log.to(DEV),
        dt_bias.to(DEV),
        device_pool[1],
        indices.to(DEV),
        _qsl(lens).to(DEV),
        initial.to(DEV),
    )
    o_ref = ref.gated_delta_rule_reference(
        q, k, v, a, b, A_log, dt_bias, expected[1], indices, _qsl(lens), initial
    )
    return o.cpu(), o_ref, device_pool.cpu(), expected


def test_recurrent_rejects_non_bool_initial_state():
    gen = torch.Generator().manual_seed(12)
    tensors = _recurrent_inputs(gen, torch.float32, (1,))
    with pytest.raises(ValueError, match="has_initial_state must be bool"):
        _run_recurrent(
            tensors,
            torch.tensor([0], dtype=torch.int32),
            (1,),
            torch.tensor([1], dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "q_heads,q_dk,state_dk,match",
    [
        (4, 64, 64, "q and k"),
        (2, 128, 64, "q and k"),
        (2, 64, 32, "ssm_states"),
    ],
)
def test_recurrent_rejects_mismatched_geometry(q_heads, q_dk, state_dk, match):
    gen = torch.Generator().manual_seed(13)
    tensors = list(_recurrent_inputs(gen, torch.float32, (1,), dk=64, dv=64))
    tensors[0] = _randn(1, q_heads, q_dk, gen=gen, dtype=torch.float32)
    tensors[-1] = _randn(2, 8, 4, 64, state_dk, gen=gen, dtype=torch.float32, scale=0.1)
    with pytest.raises(ValueError, match=match):
        _run_recurrent(
            tensors,
            torch.tensor([0], dtype=torch.int32),
            (1,),
            torch.tensor([True]),
        )


def _check_recurrent(o, o_ref, got_pool, expected, pool, state_tol):
    torch.testing.assert_close(o, o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got_pool[1], expected[1], atol=state_tol, rtol=state_tol)
    _check_untouched(got_pool, pool, 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_recurrent_decode(dtype, batch):
    gen = torch.Generator().manual_seed(batch + 100)
    tensors = _recurrent_inputs(gen, dtype, [1] * batch, slots=24)
    indices = torch.arange(batch, dtype=torch.int32)
    indices[1::3] = -1
    initial = torch.ones(batch, dtype=torch.bool)
    o, o_ref, got_pool, expected = _run_recurrent(
        tensors, indices, [1] * batch, initial
    )
    _check_recurrent(o, o_ref, got_pool, expected, tensors[-1], 2e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dk,dv", [(64, 64), (64, 128), (128, 100), (128, 128), (256, 64)]
)
def test_recurrent_varlen(dtype, dk, dv):
    gen = torch.Generator().manual_seed(dk + dv)
    lens = (3, 1, 5)
    tensors = _recurrent_inputs(gen, dtype, lens, dk=dk, dv=dv)
    indices = torch.tensor([0, 2, -1], dtype=torch.int32)
    initial = torch.tensor([True, False, True])
    o, o_ref, got_pool, expected = _run_recurrent(tensors, indices, lens, initial)
    _check_recurrent(o, o_ref, got_pool, expected, tensors[-1], 2e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_recurrent_strided_qkv(dtype):
    # q/k/v are split views of the fused projection, as the backend passes them.
    gen = torch.Generator().manual_seed(11)
    lens = (4, 2)
    hk, hv, dk, dv = 2, 4, 128, 128
    total = sum(lens)
    fused = _randn(total, 2 * hk * dk + hv * dv, gen=gen, dtype=dtype)
    q, k, v = fused.split([hk * dk, hk * dk, hv * dv], dim=-1)
    q = q.view(total, hk, dk)
    k = k.view(total, hk, dk)
    v = v.view(total, hv, dv)
    a = _randn(total, hv, gen=gen, dtype=dtype, scale=0.5)
    b = _randn(total, hv, gen=gen, dtype=dtype)
    A_log = _randn(hv, gen=gen, dtype=torch.float32, scale=0.5)
    dt_bias = _randn(hv, gen=gen, dtype=torch.float32, scale=0.1)
    pool = _randn(2, 4, hv, dv, dk, gen=gen, dtype=torch.float32, scale=0.1)
    indices = torch.tensor([1, 3], dtype=torch.int32)
    initial = torch.tensor([True, False])
    o, o_ref, got_pool, expected = _run_recurrent(
        (q, k, v, a, b, A_log, dt_bias, pool), indices, lens, initial
    )
    _check_recurrent(o, o_ref, got_pool, expected, pool, 2e-3)


def test_recurrent_mixed_param_dtypes():
    # Qwen3.5 serves with bf16 activations, fp32 A_log and bf16 dt_bias.
    gen = torch.Generator().manual_seed(23)
    tensors = list(_recurrent_inputs(gen, torch.bfloat16, (3, 1)))
    tensors[6] = tensors[6].to(torch.bfloat16)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    initial = torch.tensor([True, False])
    o, o_ref, got_pool, expected = _run_recurrent(tensors, indices, (3, 1), initial)
    _check_recurrent(o, o_ref, got_pool, expected, tensors[-1], 2e-3)


def test_conv_mixed_dtypes():
    """An fp32 current input must reach the conv before the bf16 window store."""
    gen = torch.Generator().manual_seed(31)
    channels, width, slots = 64, 4, 8
    weight = _randn(channels, width, gen=gen, dtype=torch.bfloat16, scale=0.2)
    bias = _randn(channels, gen=gen, dtype=torch.bfloat16, scale=0.1)
    indices = torch.tensor([0, -1, 2], dtype=torch.int32)
    initial = torch.tensor([True, False, True])
    lens = (3, 1, 2)
    pool = _randn(slots, channels, width - 1, gen=gen, dtype=torch.bfloat16)

    x = _randn(3, channels, gen=gen, dtype=torch.float32)
    expected = pool.clone()
    device_pool = pool.to(DEV)
    y = causal_conv1d_update_mps(
        x.to(DEV), device_pool, weight.to(DEV), bias.to(DEV), indices.to(DEV)
    )
    y_ref = ref.causal_conv1d_update_reference(x, expected, weight, bias, indices)
    torch.testing.assert_close(y.cpu(), y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(device_pool.cpu(), expected, atol=0, rtol=0)

    x = _randn(sum(lens), channels, gen=gen, dtype=torch.float32)
    expected = pool.clone()
    device_pool = pool.to(DEV)
    y = causal_conv1d_varlen_mps(
        x.to(DEV),
        device_pool,
        weight.to(DEV),
        bias.to(DEV),
        indices.to(DEV),
        _qsl(lens).to(DEV),
        initial.to(DEV),
    )
    y_ref = ref.causal_conv1d_varlen_reference(
        x, expected, weight, bias, indices, _qsl(lens), initial
    )
    torch.testing.assert_close(y.cpu(), y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(device_pool.cpu(), expected, atol=0, rtol=0)


def test_recurrent_long_decay():
    # Near-unit decay over 2048 tokens must not drift the fp32 state.
    gen = torch.Generator().manual_seed(2048)
    lens = (2048,)
    tensors = list(
        _recurrent_inputs(gen, torch.float32, lens, hk=1, hv=2, dk=64, dv=64)
    )
    tensors[6] = torch.full((2,), -12.0)  # A_log: exp(A_log) ~ 6e-6
    indices = torch.tensor([0], dtype=torch.int32)
    initial = torch.tensor([False])
    o, o_ref, got_pool, expected = _run_recurrent(tensors, indices, lens, initial)

    state_err = (got_pool[1, 0] - expected[1, 0]).abs().max().item()
    scale = expected[1, 0].abs().max().item()
    print(f"\nlong-decay state max abs err {state_err:.3e} (scale {scale:.3e})")
    torch.testing.assert_close(o, o_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(got_pool[1], expected[1], atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
