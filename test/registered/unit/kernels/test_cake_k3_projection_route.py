"""CPU unit tests for the opt-in Cake Kimi-K3 FP8 projection route
(``SGLANG_CAKE_ROUTES=kimi_k3_fp8_projection``).

The Cake adapter / facade calls are substituted, so these tests pin the
engine-side wiring in ``sglang.srt.models.kimi_k3_cake_projection``: install is
a no-op with the route off or a non-``FP8_PB_WO`` linear; the wrapper prepares
the weight after ``process_weights_after_loading`` (padding N to 128 rows when
needed) or lazily on the first eager call; ``apply`` / ``apply_into`` take Cake
only for admitted BF16 ``[M, K]`` activations, memoise the per-M admission,
fall back inside CUDA-graph capture for an unwarmed M, and fall back for good
on a FlashInfer host rejection.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.cake_kernels import _routes
from sglang.srt.models import kimi_k3_cake_projection as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

K = 256
N_ALIGNED = 256
N_RAGGED = 192  # 1.5 x 128 rows: needs the transient 128-row padding
M = 8

_env_patchers = []


@pytest.fixture(autouse=True)
def _reset_routes():
    _routes.reset_cache_for_tests()
    yield
    while _env_patchers:
        _env_patchers.pop().stop()
    _routes.reset_cache_for_tests()


def _route(on: bool) -> None:
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    _env_patchers.append(patcher)
    if on:
        os.environ[_routes.ENV_VAR] = mod.ROUTE
    else:
        os.environ.pop(_routes.ENV_VAR, None)
    _routes.reset_cache_for_tests()


class _InnerMethod:
    """Stand-in for Fp8LinearMethod (block quant)."""

    weight_block_size = [128, 128]

    def __init__(self, with_apply_into: bool = True):
        self.apply_calls = 0
        self.apply_into_calls = 0
        self.processed = 0
        if not with_apply_into:
            # emulate a method without apply_into
            self.apply_into = None

    def process_weights_after_loading(self, layer):
        self.processed += 1

    def apply(self, layer, x, bias=None):
        self.apply_calls += 1
        inp = x[0] if isinstance(x, tuple) else x
        return torch.full((inp.shape[0], layer.weight.shape[0]), 7.0)

    def apply_into(self, layer, x, out, bias=None):
        self.apply_into_calls += 1
        out.fill_(7.0)
        return out


def _linear(n: int, k: int = K, with_apply_into: bool = True) -> torch.nn.Module:
    linear = torch.nn.Module()
    linear.weight = torch.nn.Parameter(
        torch.zeros(n, k, dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    linear.weight_scale_inv = torch.nn.Parameter(
        torch.ones(-(-n // 128), k // 128, dtype=torch.float32), requires_grad=False
    )
    linear.quant_method = _InnerMethod(with_apply_into)
    return linear


def _attn(**linears) -> torch.nn.Module:
    attn = torch.nn.Module()
    for name, linear in linears.items():
        setattr(attn, name, linear)
    return attn


def _prepared(n: int, k: int = K):
    return SimpleNamespace(n_valid=n, K=k)


def _fake_prepare_weights(calls):
    def _prepare(weight, scale, n_valid):
        calls.append((weight, scale, n_valid))
        return _prepared(n_valid, weight.shape[1])

    return _prepare


class _Runner:
    def __init__(self, out):
        self.out = out
        self.launched = 0

    def launch(self):
        self.launched += 1
        self.out.fill_(1.0)
        return self.out


def _fake_prepare_projection(calls):
    def _prepare(x, prepared, out, workspace):
        runner = _Runner(out)
        calls.append((x, prepared, out, workspace, runner))
        return runner

    return _prepare


@pytest.fixture
def cake_stubs():
    """Admitting adapter / facade stubs; returns the recorded calls."""
    calls = dict(prepare_weights=[], prepare_projection=[], workspace=[])

    def _alloc(prepared, m):
        calls["workspace"].append(m)
        return ("q", "sf")

    with (
        mock.patch.object(mod, "_supports_weights", return_value=True),
        mock.patch.object(mod, "_supports_projection", return_value=True) as sup,
        mock.patch.object(
            mod,
            "_prepare_weights",
            side_effect=_fake_prepare_weights(calls["prepare_weights"]),
        ),
        mock.patch.object(mod, "_allocate_workspace", side_effect=_alloc),
        mock.patch.object(
            mod,
            "_prepare_projection",
            side_effect=_fake_prepare_projection(calls["prepare_projection"]),
        ),
        mock.patch.object(mod, "_is_capturing", return_value=False),
        mock.patch.object(mod, "_make_launcher", return_value=None),  # legacy runner path unless a test overrides
    ):
        calls["supports_projection"] = sup
        yield calls


# ------------------------------------------------------------------- install


def test_install_route_off_is_noop():
    _route(False)
    linear = _linear(N_ALIGNED)
    inner = linear.quant_method
    assert (
        mod.install_cake_kimi_k3_fp8_projections(_attn(q_b_proj=linear), lambda _: True)
        == []
    )
    assert linear.quant_method is inner


def test_install_skips_non_fp8_pb_wo_and_wraps_the_rest():
    _route(True)
    q_b = _linear(N_ALIGNED)
    o_proj = _linear(N_ALIGNED, with_apply_into=True)
    kv_b = _linear(N_ALIGNED)
    attn = _attn(q_b_proj=q_b, o_proj=o_proj, kv_b_proj=kv_b, g_proj=_linear(N_ALIGNED))
    seen = []

    def predicate(name):
        seen.append(name)
        return name != "layers.0.self_attn.kv_b_proj"

    installed = mod.install_cake_kimi_k3_fp8_projections(
        attn, predicate, prefix="layers.0.self_attn"
    )
    assert installed == ["q_b_proj", "o_proj"]  # PROJECTIONS order, kv_b rejected
    assert isinstance(q_b.quant_method, mod.CakeFp8ProjectionLinearMethod)
    assert isinstance(o_proj.quant_method, mod.CakeFp8ProjectionLinearMethod)
    assert isinstance(kv_b.quant_method, _InnerMethod)  # predicate said no
    assert isinstance(attn.g_proj.quant_method, _InnerMethod)  # not a listed projection
    assert (
        "layers.0.self_attn.q_b_proj" in seen
        and "layers.0.self_attn.g_proj" not in seen
    )
    # Idempotent: a second install does not double-wrap.
    assert (
        mod.install_cake_kimi_k3_fp8_projections(attn, predicate, "layers.0.self_attn")
        == []
    )
    assert q_b.quant_method._inner.__class__ is _InnerMethod
    # Unknown attributes delegate to the wrapped method.
    assert q_b.quant_method.weight_block_size == [128, 128]


def test_apply_into_is_advertised_only_when_the_inner_method_has_it():
    _route(True)
    with_it = _linear(N_ALIGNED, with_apply_into=True)
    without = _linear(N_ALIGNED, with_apply_into=False)
    mod.install_cake_kimi_k3_fp8_projections(
        _attn(q_b_proj=with_it, o_proj=without), lambda _: True
    )
    assert getattr(with_it.quant_method, "apply_into", None) is not None
    assert getattr(without.quant_method, "apply_into", None) is None


# ------------------------------------------------------------------- prepare


def test_process_weights_after_loading_delegates_then_prepares(cake_stubs):
    _route(True)
    linear = _linear(N_RAGGED)
    inner = linear.quant_method
    mod.install_cake_kimi_k3_fp8_projections(_attn(q_b_proj=linear), lambda _: True)
    wrapper = linear.quant_method
    wrapper.process_weights_after_loading(linear)
    assert inner.processed == 1
    assert (
        wrapper.prepared is not None and wrapper.n_valid == N_RAGGED and wrapper.k == K
    )
    ((weight, scale, n_valid),) = cake_stubs["prepare_weights"]
    # Ragged N is padded to 128 rows in a transient copy; n_valid stays N.
    assert tuple(weight.shape) == (256, K) and weight.dtype == torch.float8_e4m3fn
    assert weight is not linear.weight.data
    assert tuple(scale.shape) == (2, K // 128) and n_valid == N_RAGGED


def test_aligned_weight_is_prepared_in_place(cake_stubs):
    _route(True)
    linear = _linear(N_ALIGNED)
    mod.install_cake_kimi_k3_fp8_projections(_attn(o_proj=linear), lambda _: True)
    linear.quant_method.process_weights_after_loading(linear)
    ((weight, _scale, n_valid),) = cake_stubs["prepare_weights"]
    assert weight.data_ptr() == linear.weight.data.data_ptr() and n_valid == N_ALIGNED


def test_unsupported_weight_leaves_wrapper_on_fallback(cake_stubs):
    _route(True)
    linear = _linear(N_ALIGNED)
    mod.install_cake_kimi_k3_fp8_projections(_attn(o_proj=linear), lambda _: True)
    with mock.patch.object(mod, "_supports_weights", return_value=False):
        linear.quant_method.process_weights_after_loading(linear)
    assert linear.quant_method.prepared is None
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    out = linear.quant_method.apply(linear, x)
    assert torch.all(out == 7.0) and linear.quant_method._inner.apply_calls == 1
    assert cake_stubs["prepare_projection"] == []


def test_odd_k_or_bad_scale_is_rejected_before_the_adapter(cake_stubs):
    _route(True)
    linear = _linear(N_ALIGNED, k=K + 64)
    linear.weight_scale_inv = torch.nn.Parameter(
        torch.ones(2, 2, dtype=torch.float32), requires_grad=False
    )
    assert mod.prepare_linear_weight(linear, "q_b_proj") is None
    assert cake_stubs["prepare_weights"] == []


# ----------------------------------------------------------------------- apply


def _installed(cake_stubs, n=N_ALIGNED, with_apply_into=True):
    linear = _linear(n, with_apply_into=with_apply_into)
    mod.install_cake_kimi_k3_fp8_projections(_attn(q_b_proj=linear), lambda _: True)
    linear.quant_method.process_weights_after_loading(linear)
    return linear, linear.quant_method, linear.quant_method._inner


def test_apply_admitted_launches_cake_and_memoises_admission(cake_stubs):
    _route(True)
    linear, wrapper, inner = _installed(cake_stubs)
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    out = wrapper.apply(linear, x)
    out2 = wrapper.apply(linear, x)
    assert tuple(out.shape) == (M, N_ALIGNED) and out.dtype == torch.bfloat16
    assert torch.all(out == 1.0) and torch.all(out2 == 1.0)
    assert inner.apply_calls == 0
    assert cake_stubs["supports_projection"].call_count == 1  # per-M memo
    assert len(cake_stubs["prepare_projection"]) == 2  # runner rebound per call
    assert cake_stubs["workspace"] == [M, M]
    x_bound, prepared, out_bound, workspace, runner = cake_stubs["prepare_projection"][
        0
    ]
    assert x_bound is x and prepared is wrapper.prepared and out_bound is out
    assert workspace == ("q", "sf") and runner.launched == 1
    assert M in wrapper._warm


class _Launcher:
    """Stand-in for FlashInfer's per-weight cached launcher (round 7)."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, out=None):
        if out is None:
            out = torch.empty((x.shape[0], N_ALIGNED), dtype=torch.bfloat16)
        self.calls.append((x, out))
        out.fill_(2.0)
        return out


def test_apply_uses_the_cached_launcher_when_flashinfer_has_it(cake_stubs):
    """With the launcher entry the adapter neither allocates a workspace nor rebuilds a runner per call."""
    _route(True)
    launcher = _Launcher()
    with mock.patch.object(mod, "_make_launcher", return_value=launcher):
        linear, wrapper, inner = _installed(cake_stubs)
        x = torch.zeros(M, K, dtype=torch.bfloat16)
        out = wrapper.apply(linear, x)
        out2 = wrapper.apply(linear, x)
    assert wrapper._launcher is launcher
    assert torch.all(out == 2.0) and torch.all(out2 == 2.0)
    assert inner.apply_calls == 0
    assert len(launcher.calls) == 2 and launcher.calls[0][0] is x and launcher.calls[0][1] is out
    assert cake_stubs["prepare_projection"] == [] and cake_stubs["workspace"] == []
    assert M in wrapper._warm


def test_launcher_host_rejection_falls_back_for_good(cake_stubs):
    _route(True)
    launcher = mock.Mock(side_effect=ValueError("out must be a 4-byte aligned bf16"))
    with mock.patch.object(mod, "_make_launcher", return_value=launcher):
        linear, wrapper, inner = _installed(cake_stubs)
        x = torch.zeros(M, K, dtype=torch.bfloat16)
        wrapper.apply(linear, x)
        wrapper.apply(linear, x)
    assert launcher.call_count == 1 and inner.apply_calls == 2
    assert wrapper._admitted[M] is False


def test_apply_falls_back_for_rejected_or_foreign_inputs(cake_stubs):
    _route(True)
    linear, wrapper, inner = _installed(cake_stubs)
    with mock.patch.object(mod, "_supports_projection", return_value=False):
        out = wrapper.apply(linear, torch.zeros(M, K, dtype=torch.bfloat16))
    assert torch.all(out == 7.0) and inner.apply_calls == 1
    assert wrapper._admitted[M] is False
    # Rejection is memoised too: the same M goes straight to the inner method.
    wrapper.apply(linear, torch.zeros(M, K, dtype=torch.bfloat16))
    assert inner.apply_calls == 2 and cake_stubs["prepare_projection"] == []
    # fp16 / 3-D / pre-quantized tuple / wrong K / bias -> inner method.
    wrapper.apply(linear, torch.zeros(4, K, dtype=torch.float16))
    wrapper.apply(linear, torch.zeros(1, 4, K, dtype=torch.bfloat16))
    wrapper.apply(linear, (torch.zeros(4, K, dtype=torch.bfloat16), torch.ones(1)))
    wrapper.apply(linear, torch.zeros(4, K + 128, dtype=torch.bfloat16))
    wrapper.apply(
        linear, torch.zeros(4, K, dtype=torch.bfloat16), torch.zeros(N_ALIGNED)
    )
    assert inner.apply_calls == 7 and cake_stubs["prepare_projection"] == []


def test_capture_falls_back_until_the_shape_was_warmed(cake_stubs):
    _route(True)
    linear, wrapper, inner = _installed(cake_stubs)
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    with mock.patch.object(mod, "_is_capturing", return_value=True):
        out = wrapper.apply(linear, x)
    assert torch.all(out == 7.0) and inner.apply_calls == 1
    assert cake_stubs["prepare_projection"] == []
    # One eager launch warms M; capture then takes Cake.
    wrapper.apply(linear, x)
    with mock.patch.object(mod, "_is_capturing", return_value=True):
        out = wrapper.apply(linear, x)
    assert torch.all(out == 1.0) and inner.apply_calls == 1
    assert len(cake_stubs["prepare_projection"]) == 2
    # A different M inside capture still falls back (logged once, no raise).
    with mock.patch.object(mod, "_is_capturing", return_value=True):
        out = wrapper.apply(linear, torch.zeros(M + 1, K, dtype=torch.bfloat16))
    assert torch.all(out == 7.0) and inner.apply_calls == 2


def test_flashinfer_host_rejection_falls_back_for_good(cake_stubs):
    _route(True)
    linear, wrapper, inner = _installed(cake_stubs)
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    with mock.patch.object(
        mod, "_prepare_projection", side_effect=ValueError("ldo must be even")
    ) as prep:
        out = wrapper.apply(linear, x)
        out2 = wrapper.apply(linear, x)
    assert torch.all(out == 7.0) and torch.all(out2 == 7.0)
    assert inner.apply_calls == 2 and prep.call_count == 1
    assert wrapper._admitted[M] is False and M not in wrapper._warm


def test_apply_into_uses_caller_output_when_it_fits(cake_stubs):
    _route(True)
    linear, wrapper, inner = _installed(cake_stubs)
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    good = torch.zeros(M, N_ALIGNED + 8, dtype=torch.bfloat16)[:, :N_ALIGNED]
    res = wrapper.apply_into(linear, x, good)
    assert res is good and torch.all(good == 1.0) and inner.apply_into_calls == 0
    assert cake_stubs["prepare_projection"][0][2] is good
    # Odd row stride violates the Cake output contract -> inner apply_into.
    odd = torch.zeros(M, N_ALIGNED + 1, dtype=torch.bfloat16)[:, :N_ALIGNED]
    res = wrapper.apply_into(linear, x, odd)
    assert res is odd and torch.all(odd == 7.0) and inner.apply_into_calls == 1
    # Wrong shape -> inner apply_into.
    bad_shape = torch.zeros(M, N_ALIGNED - 2, dtype=torch.bfloat16)
    wrapper.apply_into(linear, x, bad_shape)
    assert inner.apply_into_calls == 2


def test_lazy_prepare_on_first_eager_call_but_not_inside_capture(cake_stubs):
    _route(True)
    linear = _linear(N_ALIGNED)
    mod.install_cake_kimi_k3_fp8_projections(_attn(q_b_proj=linear), lambda _: True)
    wrapper, inner = linear.quant_method, linear.quant_method._inner
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    # Loader processed the weights before the wrapper existed: first call is
    # inside capture -> fallback, nothing prepared.
    with mock.patch.object(mod, "_is_capturing", return_value=True):
        out = wrapper.apply(linear, x)
    assert torch.all(out == 7.0) and wrapper.prepared is None
    assert cake_stubs["prepare_weights"] == [] and inner.apply_calls == 1
    # First eager call prepares and launches.
    out = wrapper.apply(linear, x)
    assert torch.all(out == 1.0) and wrapper.prepared is not None
    assert len(cake_stubs["prepare_weights"]) == 1 and inner.apply_calls == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
