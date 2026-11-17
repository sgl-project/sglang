"""AITER's FlyDSL GDR decode must match the Triton GDN packed decode.

Shapes are the per-rank Qwen3.5/3.6 GDN layouts (16 K heads, 64 V heads,
head dim 128) at TP8/TP4/TP2/TP1. The batch sizes cross every V-split rung of
the FlyDSL decode tiling on a 256-CU part.

The two kernels are not bit-identical: Triton rounds beta to the activation
dtype before using it and FlyDSL keeps it in fp32 with fast-math exponentials,
so outputs and states are compared with a tolerance, over several chained
steps so a drift in the state would show up.
"""

import pytest
import torch

from sglang.kernels.ops.attention import gdn_packed_decode_aiter_hip as flydsl_gdn
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_HEAD_DIM = 128
_STEPS = 4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="FlyDSL GDN decode needs a ROCm GPU",
)


@pytest.fixture(autouse=True)
def _select_aiter(monkeypatch):
    monkeypatch.setenv("SGLANG_USE_HIP_GDN_PACKED_DECODE_BACKEND", "aiter")
    if not flydsl_gdn.available(torch.device("cuda")):
        pytest.skip("AITER FlyDSL GDN decode is not available on this device")


def _randn(shape, dtype, generator, scale=1.0):
    return (torch.randn(shape, device="cuda", generator=generator) * scale).to(dtype)


def _make_case(tp, batch, state_dtype, seed, slots_padding):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    heads_k, heads_v = 16 // tp, 64 // tp
    qkv_dim = 2 * heads_k * _HEAD_DIM + heads_v * _HEAD_DIM
    slots = batch + 16

    steps = [
        dict(
            mixed_qkv=_randn((batch, qkv_dim), torch.bfloat16, generator),
            a=_randn((batch, heads_v), torch.bfloat16, generator),
            b=_randn((batch, heads_v), torch.bfloat16, generator),
        )
        for _ in range(_STEPS)
    ]
    A_log = torch.rand(heads_v, device="cuda", generator=generator).log()
    dt_bias = _randn((heads_v,), torch.bfloat16, generator, scale=0.5)
    ssm_states = _randn(
        (slots, heads_v, _HEAD_DIM, _HEAD_DIM), torch.float32, generator, scale=0.05
    ).to(state_dtype)

    # Slots scattered over the pool; the tail rows are graph padding.
    cache_indices = torch.randperm(slots, device="cuda", generator=generator)[:batch]
    cache_indices = cache_indices.to(torch.int32)
    if slots_padding:
        cache_indices[-slots_padding:] = -1
    return steps, A_log, dt_bias, ssm_states, cache_indices, heads_v


def _triton_step(step, A_log, dt_bias, state, cache_indices, heads_v):
    batch = step["mixed_qkv"].shape[0]
    out = step["mixed_qkv"].new_empty(batch, 1, heads_v, _HEAD_DIM)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=step["mixed_qkv"],
        a=step["a"],
        b=step["b"],
        A_log=A_log,
        dt_bias=dt_bias,
        scale=_HEAD_DIM**-0.5,
        initial_state=state,
        out=out,
        ssm_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out


def _flydsl_step(step, A_log, dt_bias, state, cache_indices, heads_v):
    args = dict(
        mixed_qkv=step["mixed_qkv"],
        a=step["a"],
        b=step["b"],
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=state,
        cache_indices=cache_indices,
        num_v_heads=heads_v,
        head_v_dim=_HEAD_DIM,
    )
    assert flydsl_gdn.covered(scale=_HEAD_DIM**-0.5, **args)
    return flydsl_gdn.run(**args)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tp", [8, 4, 2, 1])
@pytest.mark.parametrize("batch,padding", [(1, 0), (8, 0), (24, 3), (64, 0), (256, 5)])
def test_matches_triton_packed_decode(state_dtype, tp, batch, padding):
    steps, A_log, dt_bias, state, cache_indices, heads_v = _make_case(
        tp, batch, state_dtype, seed=tp * 1000 + batch, slots_padding=padding
    )
    ref_state, fly_state = state.clone(), state.clone()
    live = cache_indices >= 0
    live_slots = cache_indices[live].long()
    # A bf16 state is re-rounded every step, so it is allowed one bf16 ulp more.
    atol, rtol = (2e-2, 2e-2) if state_dtype == torch.bfloat16 else (1e-2, 1e-2)

    for step in steps:
        ref_out = _triton_step(step, A_log, dt_bias, ref_state, cache_indices, heads_v)
        fly_out = _flydsl_step(step, A_log, dt_bias, fly_state, cache_indices, heads_v)

        torch.testing.assert_close(
            fly_out[live].float(), ref_out[live].float(), atol=atol, rtol=rtol
        )
        assert torch.all(fly_out[~live] == 0)
        torch.testing.assert_close(
            fly_state[live_slots].float(),
            ref_state[live_slots].float(),
            atol=atol,
            rtol=rtol,
        )

    untouched = torch.ones(state.shape[0], dtype=torch.bool, device="cuda")
    untouched[live_slots] = False
    assert torch.equal(fly_state[untouched], state[untouched])


def test_uncovered_launch_is_rejected():
    steps, A_log, dt_bias, state, cache_indices, heads_v = _make_case(
        8, 8, torch.bfloat16, seed=0, slots_padding=0
    )
    args = dict(
        mixed_qkv=steps[0]["mixed_qkv"],
        a=steps[0]["a"],
        b=steps[0]["b"],
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=state,
        cache_indices=cache_indices,
        num_v_heads=heads_v,
        head_v_dim=_HEAD_DIM,
    )
    assert flydsl_gdn.covered(scale=_HEAD_DIM**-0.5, **args)
    # The kernel bakes in head_k_dim ** -0.5.
    assert not flydsl_gdn.covered(scale=0.5, **args)
    # dt_bias is read in the activation dtype.
    assert not flydsl_gdn.covered(
        scale=_HEAD_DIM**-0.5, **{**args, "dt_bias": dt_bias.float()}
    )
    assert not flydsl_gdn.covered(
        scale=_HEAD_DIM**-0.5, **{**args, "cache_indices": cache_indices.long()}
    )
