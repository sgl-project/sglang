"""CPU unit tests for the Cake NVFP4 warp-decode MoE route (``moe_nvfp4_warp_decode``).

The FlashInfer runner and the CUDA-graph capture query are replaced by fakes;
tensors are small CPU tensors. Covered: route off leaves the layer unarmed,
route on + admitted geometry arms a state whose ``run`` launches the fake
runner for ``num_tokens <= 32``, ``num_tokens > 32`` declines (TRT-LLM path),
capture before an eager warmup declines, capture after a warmup launches,
padded routing rows (id -1) are parked on expert 0 with weight 0, and a
runner error disables the state instead of propagating.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.cake_kernels import _routes
from sglang.srt.layers.moe.moe_runner import cake_warp_decode as cwd
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

H, INTER, E, TOP_K = 2048, 512, 256, 8  # Qwen3.5-35B-A3B geometry
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _reset():
    _routes.reset_cache_for_tests()
    cwd.reset_logs_for_tests()
    yield
    _routes.reset_cache_for_tests()
    cwd.reset_logs_for_tests()


class FakeRunner:
    def __init__(self):
        self.calls = []

    def pack_inputs(self, act, weights):
        assert weights.key == "cake"
        return [
            None,
            "workspace",
            act.hidden_states_q,
            act.hidden_states_scale,
            act.ids,
            act.weights,
        ]

    def launch_kwargs_for(self, inputs):
        return {}

    def forward(self, inputs, tactic=-1, **kwargs):
        assert tactic == -1
        self.calls.append(inputs)
        return inputs[0]


class RaisingRunner(FakeRunner):
    def forward(self, inputs, tactic=-1, **kwargs):
        raise RuntimeError("workspace must be prepared before capture")


def _fake_pack(hidden_states_q, hidden_states_scale, topk_ids, topk_weights):
    return SimpleNamespace(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        ids=topk_ids,
        weights=topk_weights,
    )


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(cwd, "_activation_pack", _fake_pack)
    monkeypatch.setattr(
        cwd, "_weight_pack", lambda view: SimpleNamespace(key="cake", view=view)
    )
    monkeypatch.setattr(cwd, "_capturing", lambda: False)


def _state(runner=None):
    return cwd.CakeWarpDecodeMoE(
        layer_id=3,
        hidden_size=H,
        intermediate_size=INTER,
        num_experts=E,
        top_k=TOP_K,
        device=DEVICE,
        runner=runner or FakeRunner(),
        weight_pack=SimpleNamespace(key="cake"),
    )


def _inputs(num_tokens, *, ids=None, weights=None):
    hs_q = torch.zeros((num_tokens, H // 2), dtype=torch.uint8)
    hs_scale = torch.zeros((num_tokens, H // 16), dtype=torch.uint8)
    if ids is None:
        ids = (torch.arange(num_tokens * TOP_K, dtype=torch.int32) % E).view(
            num_tokens, TOP_K
        )
    if weights is None:
        weights = torch.full((num_tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32)
    topk = SimpleNamespace(topk_ids=ids, topk_weights=weights)
    out = torch.empty((num_tokens, H), dtype=torch.bfloat16)
    return hs_q, hs_scale, topk, out


def _layer(**overrides):
    cfg = SimpleNamespace(
        activation="silu",
        is_gated=True,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
        apply_router_weight_on_input=False,
        routed_scaling_factor=None,
        num_fused_shared_experts=0,
    )

    def param(shape, dtype):
        return SimpleNamespace(
            data=torch.zeros(shape, dtype=dtype), shape=shape, device=DEVICE
        )

    layer = SimpleNamespace(
        layer_id=3,
        moe_runner_config=cfg,
        moe_ep_size=1,
        num_experts=E,
        num_local_experts=E,
        top_k=TOP_K,
        intermediate_size_per_partition=INTER,
        w13_weight=param((E, 2 * INTER, H // 2), torch.uint8),
        w13_weight_scale=param((E, 2 * INTER, H // 16), torch.float8_e4m3fn),
        w2_weight=param((E, H, INTER // 2), torch.uint8),
        w2_weight_scale=param((E, H, INTER // 16), torch.float8_e4m3fn),
        g1_scale_c=param((E,), torch.float32),
        g1_alphas=param((E,), torch.float32),
        g2_alphas=param((E,), torch.float32),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def test_route_off_does_not_arm_layer(monkeypatch, fakes):
    monkeypatch.setattr(
        cwd, "_build_runner", mock.Mock(side_effect=AssertionError("not called"))
    )
    monkeypatch.setattr(
        cwd, "_supports", mock.Mock(side_effect=AssertionError("not called"))
    )
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_routes.ENV_VAR, None)
        assert cwd.maybe_create_cake_warp_decode_moe(_layer()) is None


def test_route_on_arms_admitted_geometry(monkeypatch, fakes):
    runner = FakeRunner()
    build = mock.Mock(return_value=runner)
    monkeypatch.setattr(cwd, "_build_runner", build)
    monkeypatch.setattr(cwd, "_supports", lambda **kw: True)
    layer = _layer()
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: cwd.ROUTE}):
        state = cwd.maybe_create_cake_warp_decode_moe(layer)
    assert isinstance(state, cwd.CakeWarpDecodeMoE)
    assert (
        state.hidden_size,
        state.intermediate_size,
        state.num_experts,
        state.top_k,
    ) == (
        H,
        INTER,
        E,
        TOP_K,
    )
    build.assert_called_once_with(
        intermediate_size=INTER, num_experts=E, top_k=TOP_K, device=DEVICE
    )
    # The weight view aliases the layer's TRT-LLM tensors; only gemm1_alpha is new.
    view = state._weight_pack.view
    assert set(view) == {
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm2_weights",
        "gemm2_weights_scale",
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
        "gemm1_alpha",
    }
    for key, tensor in (
        ("gemm1_weights", layer.w13_weight.data),
        ("gemm1_weights_scale", layer.w13_weight_scale.data),
        ("gemm2_weights", layer.w2_weight.data),
        ("gemm2_weights_scale", layer.w2_weight_scale.data),
        ("output1_scale_scalar", layer.g1_scale_c.data),
        ("output1_scale_gate_scalar", layer.g1_alphas.data),
        ("output2_scale_scalar", layer.g2_alphas.data),
    ):
        assert view[key].data_ptr() == tensor.data_ptr(), key
    assert view["gemm1_alpha"].shape == (E,) and bool((view["gemm1_alpha"] == 1).all())


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"moe_ep_size": 2}, "expert parallelism"),
        ({"num_local_experts": E // 2}, "expert parallelism"),
    ],
)
def test_route_on_rejects_expert_parallel_layers(monkeypatch, fakes, override, reason):
    monkeypatch.setattr(
        cwd, "_build_runner", mock.Mock(side_effect=AssertionError("not called"))
    )
    monkeypatch.setattr(cwd, "_supports", lambda **kw: True)
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: cwd.ROUTE}):
        assert cwd.maybe_create_cake_warp_decode_moe(_layer(**override)) is None
    assert cwd._layer_rejection(_layer(**override)).startswith(reason)


def test_route_on_rejects_geometry_outside_table(monkeypatch, fakes):
    monkeypatch.setattr(
        cwd, "_build_runner", mock.Mock(side_effect=AssertionError("not called"))
    )
    monkeypatch.setattr(cwd, "_supports", lambda **kw: False)
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: cwd.ROUTE}):
        assert cwd.maybe_create_cake_warp_decode_moe(_layer()) is None


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
def test_small_batch_runs_cake_runner(fakes, num_tokens):
    runner = FakeRunner()
    state = _state(runner)
    hs_q, hs_scale, topk, out = _inputs(num_tokens)
    assert state.run(hs_q, hs_scale, topk, out) is True
    assert len(runner.calls) == 1
    inputs = runner.calls[0]
    assert inputs[0] is out  # engine output buffer is the kernel's output
    assert inputs[4].dtype == torch.int32 and inputs[5].dtype == torch.bfloat16
    assert torch.equal(inputs[4], topk.topk_ids)
    assert num_tokens in state._warmed


def test_large_batch_takes_trtllm_path(fakes):
    runner = FakeRunner()
    state = _state(runner)
    hs_q, hs_scale, topk, out = _inputs(33)
    assert state.run(hs_q, hs_scale, topk, out) is False
    assert runner.calls == []


def test_capture_before_warmup_falls_back(monkeypatch, fakes):
    runner = FakeRunner()
    state = _state(runner)
    hs_q, hs_scale, topk, out = _inputs(8)
    monkeypatch.setattr(cwd, "_capturing", lambda: True)
    assert state.run(hs_q, hs_scale, topk, out) is False
    assert runner.calls == []
    # An eager warmup of the same num_tokens then admits the captured call,
    # reusing the same static routing buffers (identity the FI receipt is bound to).
    monkeypatch.setattr(cwd, "_capturing", lambda: False)
    assert state.run(hs_q, hs_scale, topk, out) is True
    monkeypatch.setattr(cwd, "_capturing", lambda: True)
    assert state.run(hs_q, hs_scale, topk, out) is True
    assert len(runner.calls) == 2
    assert runner.calls[0][4] is runner.calls[1][4]
    assert runner.calls[0][5] is runner.calls[1][5]
    # A different num_tokens during capture still falls back.
    hs_q16, hs_scale16, topk16, out16 = _inputs(16)
    assert state.run(hs_q16, hs_scale16, topk16, out16) is False
    assert len(runner.calls) == 2


def test_padded_rows_are_parked_on_expert_zero(fakes):
    runner = FakeRunner()
    state = _state(runner)
    ids = (torch.arange(4 * TOP_K, dtype=torch.int32) % E).view(4, TOP_K)
    ids[2:] = -1  # mask_topk_ids fill for rows >= num_token_non_padded
    weights = torch.full((4, TOP_K), 0.125, dtype=torch.float32)
    hs_q, hs_scale, topk, out = _inputs(4, ids=ids, weights=weights)
    assert state.run(hs_q, hs_scale, topk, out) is True
    sent_ids, sent_weights = runner.calls[0][4], runner.calls[0][5]
    assert bool((sent_ids[2:] == 0).all()) and bool((sent_weights[2:] == 0).all())
    assert torch.equal(sent_ids[:2], ids[:2])
    assert bool((sent_weights[:2].float() == 0.125).all())


def test_bypassed_topk_is_materialized(fakes):
    runner = FakeRunner()
    state = _state(runner)
    hs_q, hs_scale, standard, out = _inputs(2)
    seen = []

    def to_standard(layer_id):
        seen.append(layer_id)
        return standard

    bypassed = SimpleNamespace(to_standard=to_standard)
    assert state.run(hs_q, hs_scale, bypassed, out) is True
    assert seen == [3]
    # A TopKOutput with neither routing tensors nor to_standard declines.
    assert state.run(hs_q, hs_scale, SimpleNamespace(), out) is False


def test_per_token_scale_and_wrong_output_fall_back(fakes):
    runner = FakeRunner()
    state = _state(runner)
    hs_q, hs_scale, topk, out = _inputs(4)
    assert state.run(hs_q, hs_scale, topk, out, per_token_scale=torch.ones(4)) is False
    assert state.run(hs_q, hs_scale, topk, out.to(torch.float16)) is False
    assert runner.calls == []


def test_runner_error_disables_state(fakes):
    state = _state(RaisingRunner())
    hs_q, hs_scale, topk, out = _inputs(4)
    assert state.run(hs_q, hs_scale, topk, out) is False
    assert state._disabled_reason is not None
    assert state.run(hs_q, hs_scale, topk, out) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
