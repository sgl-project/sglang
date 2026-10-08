"""CPU unit tests for the opt-in Cake KDA decode route (``SGLANG_CAKE_ROUTES=kda_decode``).

The adapter's ``supports_*`` predicates and forwarders are mocked, so the tests
pin the engine-side wiring only: route off -> never consulted; route on and
admitted -> the Cake forwarder receives the re-expressed tensors (stacked conv
weight, transposed conv state, ``unique_or_null`` indices mode, batch-outermost
recurrent inputs); route on and rejected / failed closed -> the caller's own
path is kept.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.cake_kernels import _routes
from sglang.kernels.cake_kernels import attention_linear_kda as adapter
from sglang.srt.layers.attention.linear.kernels import kda_flashinfer as kf
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

B, H, D = 3, 12, 128
SEG = H * D
SLOTS = 4


_env_patchers = []


@pytest.fixture(autouse=True)
def _reset_routes():
    _routes.reset_cache_for_tests()
    kf._cake_logged.clear()
    yield
    while _env_patchers:
        _env_patchers.pop().stop()
    _routes.reset_cache_for_tests()
    kf._cake_logged.clear()


def _route(on: bool) -> None:
    """Select (or clear) the ``kda_decode`` route for the rest of the test."""
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    _env_patchers.append(patcher)
    if on:
        os.environ[_routes.ENV_VAR] = "kda_decode"
    else:
        os.environ.pop(_routes.ENV_VAR, None)
    _routes.reset_cache_for_tests()


def _k3_layer(bias=None, lower_bound=-5.0):
    return SimpleNamespace(
        layer_id=7,
        bias=bias,
        lower_bound=lower_bound,
        num_v_heads=H,
        head_v_dim=D,
        head_k_dim=D,
        dt_bias=torch.zeros(SEG, dtype=torch.float32),
        A_log=torch.zeros(1, 1, H, 1, dtype=torch.float32),
    )


def _fused_inputs():
    w = torch.randn(3 * SEG, 4, dtype=torch.float32)
    wt = w.t().contiguous()
    fused_static = (
        wt[:, :SEG].contiguous(),
        wt[:, SEG : 2 * SEG].contiguous(),
        wt[:, 2 * SEG :].contiguous(),
        torch.zeros(3 * SEG, dtype=torch.float32),
        torch.zeros(H, dtype=torch.float32),
        torch.ones(D, dtype=torch.float32),
        1e-5,
    )
    tensors = dict(
        mixed_qkv=torch.randn(B, 3 * SEG).to(torch.bfloat16),
        a=torch.randn(B, SEG).to(torch.bfloat16),
        b=torch.randn(1, B, H).to(torch.bfloat16),
        conv_states=torch.zeros(SLOTS, 3, 3 * SEG, dtype=torch.bfloat16),
        ssm_states=torch.zeros(SLOTS, H, D, D, dtype=torch.float32),
        cache_indices=torch.tensor([1, 2, -1], dtype=torch.int32),
        onorm_gate=torch.randn(B, SEG).to(torch.bfloat16),
    )
    return fused_static, tensors


# ---------------------------------------------------------------- fused decode


def test_fused_route_off_never_consults_adapter():
    _route(False)
    fused_static, t = _fused_inputs()
    with (
        mock.patch.object(adapter, "supports_kda_fused_decode") as sup,
        mock.patch.object(adapter, "fused_kda_decode") as fwd,
    ):
        assert kf.try_cake_fused_decode(_k3_layer(), fused_static, **t) is None
    sup.assert_not_called()
    fwd.assert_not_called()


def test_fused_route_on_admitted_forwards_cake_contract():
    _route(True)
    fused_static, t = _fused_inputs()
    layer = _k3_layer()
    expected_out = torch.zeros(1, B, H, D, dtype=torch.bfloat16)
    with (
        mock.patch.object(
            adapter, "supports_kda_fused_decode", return_value=True
        ) as sup,
        mock.patch.object(
            adapter, "fused_kda_decode", return_value=expected_out
        ) as fwd,
    ):
        out = kf.try_cake_fused_decode(layer, fused_static, **t)
        out2 = kf.try_cake_fused_decode(layer, fused_static, **t)
    assert out is expected_out and out2 is expected_out
    # Admission is memoised per (rows, pool) key; the forwarder runs per call.
    assert sup.call_count == 1
    assert fwd.call_count == 2
    args, kwargs = fwd.call_args
    x, weight, conv_state, raw_gate, raw_beta, a_log, dt_bias, idx, state = args[:9]
    output_gate, norm_weight = args[9], args[10]
    assert x is t["mixed_qkv"]
    assert tuple(weight.shape) == (3, 4, SEG) and weight.dtype == torch.float32
    assert torch.equal(weight[1], fused_static[1])  # k-projection taps
    assert weight is layer._cake_fused_conv_weight  # stacked once, cached
    assert tuple(conv_state.shape) == (SLOTS, 3 * SEG, 3)
    assert conv_state.stride() == (9 * SEG, 1, 3 * SEG)
    assert conv_state.data_ptr() == t["conv_states"].data_ptr()  # a view
    assert tuple(raw_gate.shape) == (1, B, H, D)
    assert raw_gate.data_ptr() == t["a"].data_ptr()
    assert raw_beta is t["b"]
    assert a_log is fused_static[4] and dt_bias is layer.dt_bias
    assert idx is t["cache_indices"]  # already int32 + contiguous: no copy
    assert state is t["ssm_states"]
    assert tuple(output_gate.shape) == (B, H, D)
    assert output_gate.data_ptr() == t["onorm_gate"].data_ptr()
    assert norm_weight is fused_static[5]
    assert kwargs == dict(
        lower_bound=-5.0, norm_eps=1e-5, state_indices_mode="unique_or_null"
    )


def test_fused_route_on_rejected_keeps_engine_path():
    _route(True)
    fused_static, t = _fused_inputs()
    with (
        mock.patch.object(adapter, "supports_kda_fused_decode", return_value=False),
        mock.patch.object(adapter, "fused_kda_decode") as fwd,
    ):
        assert kf.try_cake_fused_decode(_k3_layer(), fused_static, **t) is None
    fwd.assert_not_called()


def test_fused_conv_bias_is_not_admitted():
    _route(True)
    fused_static, t = _fused_inputs()
    layer = _k3_layer(bias=torch.zeros(3 * SEG))
    with mock.patch.object(adapter, "supports_kda_fused_decode") as sup:
        assert kf.try_cake_fused_decode(layer, fused_static, **t) is None
    sup.assert_not_called()
    assert layer._cake_fused_decode_disabled


def test_fused_fail_closed_disables_layer():
    _route(True)
    fused_static, t = _fused_inputs()
    layer = _k3_layer()
    with (
        mock.patch.object(adapter, "supports_kda_fused_decode", return_value=True),
        mock.patch.object(
            adapter, "fused_kda_decode", side_effect=RuntimeError("no variant")
        ) as fwd,
    ):
        assert kf.try_cake_fused_decode(layer, fused_static, **t) is None
        assert layer._cake_fused_decode_disabled
        assert kf.try_cake_fused_decode(layer, fused_static, **t) is None
    assert fwd.call_count == 1


# --------------------------------------------------------------- packed decode


def _packed_inputs():
    return dict(
        qkv=torch.randn(B, 3 * SEG).to(torch.bfloat16),
        a=torch.randn(B, SEG).to(torch.bfloat16),
        b=torch.randn(1, B, H).to(torch.bfloat16),
        ssm_states=torch.zeros(SLOTS, H, D, D, dtype=torch.bfloat16),
        cache_indices=torch.tensor([1, 2, -1], dtype=torch.int32),
    )


def test_packed_route_off_or_unbounded_gate_is_skipped():
    t = _packed_inputs()
    with (
        mock.patch.object(adapter, "supports_kda_packed_decode") as sup,
        mock.patch.object(adapter, "packed_kda_decode") as fwd,
    ):
        _route(False)
        assert kf.try_cake_packed_decode(_k3_layer(), **t) is None
        _route(True)
        # The Cake packed kernel is locked to lower_bound=-5.
        assert kf.try_cake_packed_decode(_k3_layer(lower_bound=None), **t) is None
        assert kf.try_cake_packed_decode(_k3_layer(lower_bound=-3.0), **t) is None
    sup.assert_not_called()
    fwd.assert_not_called()


def test_packed_route_on_admitted_forwards_and_reshapes():
    _route(True)
    t = _packed_inputs()
    layer = _k3_layer()
    result = torch.zeros(B, 1, H, D, dtype=torch.bfloat16)
    with (
        mock.patch.object(
            adapter, "supports_kda_packed_decode", return_value=True
        ) as sup,
        mock.patch.object(adapter, "packed_kda_decode", return_value=result) as fwd,
    ):
        out = kf.try_cake_packed_decode(layer, **t)
        kf.try_cake_packed_decode(layer, **t)
    assert tuple(out.shape) == (1, B, H, D) and out.data_ptr() == result.data_ptr()
    assert sup.call_count == 1 and fwd.call_count == 2
    qkv, a, raw_beta, a_log, dt_bias, state, idx = fwd.call_args.args
    assert qkv is t["qkv"] and a is t["a"] and state is t["ssm_states"]
    assert tuple(raw_beta.shape) == (B, H) and raw_beta.data_ptr() == t["b"].data_ptr()
    assert tuple(a_log.shape) == (H,) and a_log.dtype == torch.float32
    assert a_log is layer._cake_packed_a_log
    assert dt_bias is layer.dt_bias and idx is t["cache_indices"]


def test_packed_route_on_rejected_returns_none():
    _route(True)
    t = _packed_inputs()
    with (
        mock.patch.object(adapter, "supports_kda_packed_decode", return_value=False),
        mock.patch.object(adapter, "packed_kda_decode") as fwd,
    ):
        assert kf.try_cake_packed_decode(_k3_layer(), **t) is None
    fwd.assert_not_called()


# ------------------------------------------------------- recurrent T=1 decode

HR = 4  # small equal-head decode shape


def _kernel(fi_result):
    kernel = kf.FlashInferKDAKernel.__new__(kf.FlashInferKDAKernel)
    kernel._recurrent_kda = mock.Mock(return_value=(fi_result, None))
    kernel._gate_cache = {}
    kernel._verify_idx_cache = {}
    kernel._state_contract_ok = set()
    kernel._cake_decode_admission = {}
    return kernel


def _decode_inputs():
    return dict(
        q=torch.randn(1, B, HR, D).to(torch.bfloat16),
        k=torch.randn(1, B, HR, D).to(torch.bfloat16),
        v=torch.randn(1, B, HR, D).to(torch.bfloat16),
        a=torch.randn(1, B, HR, D).to(torch.bfloat16),
        b=torch.randn(1, B, HR).to(torch.bfloat16),
        A_log=torch.zeros(1, 1, HR, 1, dtype=torch.float32),
        dt_bias=torch.zeros(HR * D, dtype=torch.float32),
        ssm_states=torch.zeros(SLOTS, HR, D, D, dtype=torch.bfloat16),
        cache_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
        query_start_loc=torch.arange(B + 1, dtype=torch.int32),
    )


def test_recurrent_decode_route_off_uses_flashinfer_path():
    _route(False)
    fi_out = torch.zeros(1, B, HR, D, dtype=torch.bfloat16)
    kernel = _kernel(fi_out)
    with (
        mock.patch.object(adapter, "supports_kda_recurrent_decode") as sup,
        mock.patch.object(adapter, "recurrent_kda") as fwd,
    ):
        out = kernel.decode(**_decode_inputs())
    assert out.data_ptr() == fi_out.data_ptr()
    kernel._recurrent_kda.assert_called_once()
    sup.assert_not_called()
    fwd.assert_not_called()


def test_recurrent_decode_route_on_admitted_uses_cake_contract():
    _route(True)
    kernel = _kernel(torch.zeros(1, B, HR, D, dtype=torch.bfloat16))
    cake_out = torch.zeros(B, 1, HR, D, dtype=torch.bfloat16)
    inputs = _decode_inputs()
    with (
        mock.patch.object(
            adapter, "supports_kda_recurrent_decode", return_value=True
        ) as sup,
        mock.patch.object(
            adapter, "recurrent_kda", return_value=(cake_out, None)
        ) as fwd,
    ):
        out = kernel.decode(**inputs)
        kernel.decode(**inputs)
    assert tuple(out.shape) == (1, B, HR, D) and out.data_ptr() == cake_out.data_ptr()
    kernel._recurrent_kda.assert_not_called()
    assert sup.call_count == 1 and fwd.call_count == 2
    args, kwargs = fwd.call_args
    q, k, v, g, beta = args
    for tensor in (q, k, v, g):
        assert tuple(tensor.shape) == (B, 1, HR, D) and tensor.dtype == torch.bfloat16
    assert q.data_ptr() == inputs["q"].data_ptr()  # batch-outermost view, no copy
    assert tuple(beta.shape) == (B, 1, HR)
    # beta is batch-outermost (B, 1, HR); the input b is (1, B, HR).
    assert torch.allclose(
        beta.float(), torch.sigmoid(inputs["b"].float()).transpose(0, 1), atol=1e-2
    )
    assert kwargs["initial_state"] is inputs["ssm_states"]
    assert kwargs["ssm_state_indices"] is inputs["cache_indices"]
    assert "cu_seqlens" not in kwargs
    assert kwargs["lower_bound"] is None and kwargs["use_gate_in_kernel"]
    assert kwargs["A_log"].shape == (HR,) and kwargs["dt_bias"].shape == (HR * D,)
    assert kwargs["output_final_state"] is False


def test_recurrent_decode_route_on_rejected_falls_back():
    _route(True)
    fi_out = torch.zeros(1, B, HR, D, dtype=torch.bfloat16)
    kernel = _kernel(fi_out)
    with (
        mock.patch.object(adapter, "supports_kda_recurrent_decode", return_value=False),
        mock.patch.object(adapter, "recurrent_kda") as fwd,
    ):
        out = kernel.decode(**_decode_inputs())
    assert out.data_ptr() == fi_out.data_ptr()
    kernel._recurrent_kda.assert_called_once()
    fwd.assert_not_called()


def test_recurrent_decode_safe_gate_never_consults_cake():
    _route(True)
    fi_out = torch.zeros(1, B, HR, D, dtype=torch.bfloat16)
    kernel = _kernel(fi_out)
    with mock.patch.object(adapter, "supports_kda_recurrent_decode") as sup:
        kernel.decode(**_decode_inputs(), lower_bound=-5.0)
    sup.assert_not_called()
    kernel._recurrent_kda.assert_called_once()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
