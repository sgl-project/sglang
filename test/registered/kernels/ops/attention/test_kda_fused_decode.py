"""Kimi-K3 fused KDA decode must match the existing unfused decode chain.

The fused kernel replaces:

    causal_conv1d_update -> kda_packed_decode -> sigmoid-gated RMSNorm

This file covers the local head layouts used by Kimi-K3 TP8/TP16/TP32:
H = 12/6/3. The H=6 and H=3 cases are the branches added by the fixed-head
dispatch in ``kda_fused_decode.cuh``.
"""

import pytest
import torch
import triton

from sglang.kernels.ops.attention import kda_fused_decode
from sglang.kernels.ops.attention.fla import fused_norm_gate
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="nightly", runner_config="4-gpu-gb300")

_HEAD_DIM = 128
_CONV_STATE_W = 3
_SLOTS = 8
_BATCH = 4


def _randn(shape, dtype, generator, scale=1.0):
    return (torch.randn(shape, device="cuda", generator=generator) * scale).to(dtype)


def _make_case(heads: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * _HEAD_DIM
    conv_dim = 3 * seg

    # Keep magnitudes moderate so fp32 state updates stay in a stable range.
    mixed_qkv = _randn((_BATCH, conv_dim), torch.bfloat16, generator, scale=0.2)
    a = _randn((_BATCH, seg), torch.bfloat16, generator, scale=0.2)
    b = _randn((_BATCH, heads), torch.bfloat16, generator, scale=0.2)
    onorm_g = _randn((_BATCH, seg), torch.bfloat16, generator, scale=0.2)

    conv_states = _randn(
        (_SLOTS, _CONV_STATE_W, conv_dim), torch.bfloat16, generator, scale=0.2
    )
    ssm_states = _randn(
        (_SLOTS, heads, _HEAD_DIM, _HEAD_DIM), torch.float32, generator, scale=0.02
    )
    cache_indices = torch.arange(_BATCH, device="cuda", dtype=torch.int32)

    conv_weights = _randn((conv_dim, 4), torch.float32, generator, scale=0.1)
    conv_bias = _randn((conv_dim,), torch.float32, generator, scale=0.05)
    a_log = _randn((heads,), torch.float32, generator, scale=0.1)
    dt_bias = _randn((seg,), torch.float32, generator, scale=0.1)
    onorm_weight = _randn((_HEAD_DIM,), torch.float32, generator, scale=0.1) + 1.0

    return (
        mixed_qkv,
        a,
        b,
        onorm_g,
        conv_states,
        ssm_states,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    )


def _run_unfused_reference(
    mixed_qkv,
    a,
    b,
    onorm_g,
    conv_states,
    ssm_states,
    cache_indices,
    conv_weights,
    conv_bias,
    a_log,
    dt_bias,
    onorm_weight,
):
    heads = ssm_states.shape[-3]
    qkv = causal_conv1d_update(
        mixed_qkv,
        conv_states.transpose(-1, -2),
        conv_weights,
        conv_bias,
        activation="silu",
        conv_state_indices=cache_indices,
    )
    out = torch.empty(
        (_BATCH, 1, heads, _HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    out, _ = fused_recurrent_kda_packed_decode(
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        _HEAD_DIM**-0.5,
        ssm_states,
        out,
        cache_indices,
        use_qk_l2norm_in_kernel=True,
    )
    ref = rms_norm_gated(
        out,
        onorm_g.view(1, _BATCH, heads, _HEAD_DIM),
        onorm_weight,
        None,
        activation="sigmoid",
        eps=1e-6,
    )
    return ref.transpose(0, 1).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "heads,tp_size",
    [
        pytest.param(3, 32, id="tp32_h3"),
        pytest.param(6, 16, id="tp16_h6"),
        pytest.param(12, 8, id="tp8_h12"),
    ],
)
def test_kda_fused_decode_matches_unfused_chain(heads: int, tp_size: int):
    (
        mixed_qkv,
        a,
        b,
        onorm_g,
        conv_states,
        ssm_states,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    ) = _make_case(heads=heads, seed=20260731 + tp_size)

    conv_ref = conv_states.clone()
    conv_fused = conv_states.clone()
    state_ref = ssm_states.clone()
    state_fused = ssm_states.clone()

    w_q_t, w_k_t, w_v_t = [
        weight.t().contiguous()
        for weight in conv_weights.split(heads * _HEAD_DIM, dim=0)
    ]

    assert kda_fused_decode.covered(
        mixed_qkv,
        a,
        b,
        conv_fused,
        state_fused,
        cache_indices,
        onorm_g,
    )

    ref = _run_unfused_reference(
        mixed_qkv.clone(),
        a,
        b,
        onorm_g,
        conv_ref,
        state_ref,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    )
    fused = kda_fused_decode.kda_fused_decode(
        mixed_qkv.clone(),
        a,
        b,
        conv_fused,
        w_q_t,
        w_k_t,
        w_v_t,
        conv_bias,
        a_log,
        dt_bias,
        onorm_g,
        onorm_weight,
        state_fused,
        cache_indices,
        scale=_HEAD_DIM**-0.5,
        onorm_eps=1e-6,
    )
    torch.cuda.synchronize()

    # JIT log breadcrumb for PR/CI evidence that the fused fixed-head branch ran.
    print(f"K3 fused KDA decode test used fused path: TP{tp_size}, H={heads}")

    torch.testing.assert_close(fused, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_fused, state_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(conv_fused, conv_ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "sm,rows,width,dtype,activation,is_rms,block",
    [
        (103, 1, 128, torch.bfloat16, "sigmoid", True, 8),
        (103, 2048, 128, torch.bfloat16, "sigmoid", True, 8),
        (103, 0, 128, torch.bfloat16, "sigmoid", True, 32),
        (103, 2049, 128, torch.bfloat16, "sigmoid", True, 32),
        (103, 96, 64, torch.bfloat16, "sigmoid", True, 32),
        (103, 96, 128, torch.float32, "sigmoid", True, 32),
        (103, 96, 128, torch.bfloat16, "silu", True, 32),
        (103, 96, 128, torch.bfloat16, "sigmoid", False, 32),
        (100, 96, 128, torch.bfloat16, "sigmoid", True, 32),
        (90, 96, 128, torch.bfloat16, "sigmoid", True, 32),
    ],
)
def test_small_row_dispatch(
    monkeypatch, sm, rows, width, dtype, activation, is_rms, block
):
    monkeypatch.setattr(fused_norm_gate, "get_device_sm", lambda: sm)
    x = torch.empty((rows, width), dtype=dtype, device="meta")
    assert fused_norm_gate._get_gated_norm_block_size(x, activation, is_rms) == block


@pytest.mark.parametrize(
    "rows,with_residual",
    [
        (1, False),
        (7, False),
        (8, False),
        (9, False),
        (96, False),
        (1536, False),
        (2048, False),
        (2049, False),
        (7680, False),
        (96, True),
    ],
)
def test_row_tile_preserves_output_statistics_and_alias(rows, with_residual):
    if not torch.cuda.is_available() or fused_norm_gate.get_device_sm() != 103:
        pytest.skip("SM103 required")
    torch.manual_seed(rows)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    gate_dtype = torch.float32 if with_residual else torch.bfloat16
    g = torch.randn_like(x, dtype=gate_dtype)
    weight = torch.randn(128, device="cuda", dtype=gate_dtype)
    bias = torch.randn_like(weight) if with_residual else None
    residual = torch.randn_like(x, dtype=torch.float32) if with_residual else None
    out_dtype = torch.float32 if with_residual else None
    original_x = x.clone()
    expected = torch.empty_like(x, dtype=out_dtype or x.dtype)
    expected_rstd = torch.empty(rows, device="cuda")
    expected_residual = torch.empty_like(residual) if with_residual else None
    fused_norm_gate.layer_norm_gated_fwd_kernel[(triton.cdiv(rows, 32),)](
        x=original_x,
        g=g,
        y=expected,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=expected_residual,
        mean=None,
        rstd=expected_rstd,
        eps=1e-5,
        T=rows,
        D=128,
        BT=32,
        BD=128,
        ACTIVATION="sigmoid",
        IS_RMS_NORM=True,
        STORE_RESIDUAL_OUT=with_residual,
        HAS_RESIDUAL=with_residual,
        HAS_WEIGHT=True,
        HAS_BIAS=with_residual,
        USE_GDC=True,
        launch_pdl=True,
        num_warps=4,
    )
    output, mean, rstd, residual_out = fused_norm_gate.layer_norm_gated_fwd(
        x,
        g,
        weight,
        bias,
        activation="sigmoid",
        eps=1e-5,
        residual=residual,
        out_dtype=out_dtype,
        is_rms_norm=True,
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(rstd, expected_rstd, atol=0, rtol=0)
    assert mean is None
    if with_residual:
        torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)
        torch.testing.assert_close(x, original_x, atol=0, rtol=0)
        assert output.data_ptr() != x.data_ptr()
    else:
        assert output.data_ptr() == x.data_ptr() == residual_out.data_ptr()
    summed = original_x.float() + residual if with_residual else original_x.float()
    reference_rstd = 1 / torch.sqrt(summed.square().mean(-1) + 1e-5)
    reference = summed * reference_rstd[:, None] * weight.float()
    if bias is not None:
        reference += bias
    reference *= torch.sigmoid(g.float())
    torch.testing.assert_close(output.float(), reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(rstd, reference_rstd, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
