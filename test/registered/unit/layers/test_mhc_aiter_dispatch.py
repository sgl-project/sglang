"""Dispatch unit tests for the HIP AITER MHC routing.

These checks verify which backend is selected by ``_mhc_pre_dispatch`` and
``_mhc_post_dispatch`` without launching a server or touching the AITER/
TileLang numerical kernels.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dummy(shape, dtype):
    return torch.zeros(shape, dtype=dtype)


def _set_hip(monkeypatch, *, hip, aiter, tilelang, gfx95=True):
    monkeypatch.setattr(mhc, "is_hip", lambda: hip)
    monkeypatch.setattr(mhc, "is_gfx95_supported", lambda: gfx95)
    monkeypatch.setattr(mhc.envs.SGLANG_USE_AITER, "get", lambda: aiter)
    monkeypatch.setattr(
        mhc.envs.SGLANG_OPT_USE_TILELANG_MHC_PRE, "get", lambda: tilelang
    )
    monkeypatch.setattr(
        mhc.envs.SGLANG_OPT_USE_TILELANG_MHC_POST, "get", lambda: tilelang
    )


def _pre_args():
    return {
        "residual": _dummy((2, 4, 16), torch.bfloat16),
        "fn": _dummy((4 * 4 + 2 * 4, 64), torch.float32),
        "hc_scale": _dummy((3,), torch.float32),
        "hc_base": _dummy((4 * 4 + 2 * 4,), torch.float32),
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 2,
    }


def _pre_mix_shapes():
    return (
        _dummy((2, 4, 1), torch.float32),
        _dummy((2, 4, 4), torch.float32),
        _dummy((2, 16), torch.bfloat16),
    )


def _post_dummy(residual):
    return _dummy(residual.shape, residual.dtype)


def test_mhc_pre_dispatch_routes_to_aiter_on_hip_with_aiter_enabled(monkeypatch):
    _set_hip(monkeypatch, hip=True, aiter=True, tilelang=True)
    captured = {}

    def fake_aiter(**kwargs):
        captured["kwargs"] = kwargs
        return _pre_mix_shapes()

    monkeypatch.setattr(mhc, "_mhc_pre_aiter", fake_aiter)
    monkeypatch.setattr(
        mhc, "_mhc_pre_torch", lambda **kwargs: pytest.fail("torch fallback used")
    )
    monkeypatch.setattr(mhc, "mhc_pre", lambda **kwargs: pytest.fail("TileLang used"))

    args = _pre_args()
    post_mix, comb_mix, layer_input, norm_fused = mhc._mhc_pre_dispatch(**args)

    assert captured["kwargs"]["residual"] is args["residual"]
    assert norm_fused is False
    assert post_mix.shape == (2, 4, 1)
    assert comb_mix.shape == (2, 4, 4)
    assert layer_input.shape == (2, 16)


@pytest.mark.parametrize(
    "gfx95,aiter_enabled",
    [(True, False), (False, True)],
    ids=["aiter-disabled", "off-gfx95"],
)
def test_mhc_pre_dispatch_keeps_hip_torch_fallback(monkeypatch, gfx95, aiter_enabled):
    _set_hip(
        monkeypatch,
        hip=True,
        gfx95=gfx95,
        aiter=aiter_enabled,
        tilelang=True,
    )
    monkeypatch.setattr(mhc, "_mhc_pre_torch", lambda **kwargs: _pre_mix_shapes())
    monkeypatch.setattr(
        mhc, "_mhc_pre_aiter", lambda **kwargs: pytest.fail("AITER fallback bypassed")
    )
    monkeypatch.setattr(
        mhc, "mhc_pre", lambda **kwargs: pytest.fail("TileLang used on HIP")
    )

    post_mix, comb_mix, layer_input, norm_fused = mhc._mhc_pre_dispatch(**_pre_args())

    assert norm_fused is False
    assert post_mix.shape == (2, 4, 1)
    assert comb_mix.shape == (2, 4, 4)
    assert layer_input.shape == (2, 16)


@pytest.mark.parametrize(
    "hip, aiter_enabled, expected",
    [
        (True, True, "_mhc_post_aiter"),
        (True, False, "_mhc_post_torch"),
    ],
)
def test_mhc_post_dispatch_routes_to_configured_backend(
    monkeypatch, hip, aiter_enabled, expected
):
    _set_hip(monkeypatch, hip=hip, aiter=aiter_enabled, tilelang=True)
    x = _dummy((2, 16), torch.bfloat16)
    residual = _dummy((2, 4, 16), torch.bfloat16)
    post_layer_mix = _dummy((2, 4, 1), torch.float32)
    comb_res_mix = _dummy((2, 4, 4), torch.float32)
    args = (x, residual, post_layer_mix, comb_res_mix)
    for name in ("_mhc_post_aiter", "_mhc_post_torch", "mhc_post"):
        fail_unexpected = lambda *a, _name=name, **k: pytest.fail(f"{_name} called")
        monkeypatch.setattr(mhc, name, fail_unexpected)

    def fake(*called, **kwargs):
        assert called == args
        return _post_dummy(residual)

    monkeypatch.setattr(mhc, expected, fake)
    out = mhc._mhc_post_dispatch(*args)
    assert out is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
