"""The AITER runner core's side of the radix-4 fused route+sort: which consumer it declares and what it shows it."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from sglang.srt.layers.moe import route_sort
from sglang.srt.layers.moe.moe_runner import aiter_route_sort
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner_input():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    return AiterRunnerInput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones(topk_ids.shape, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 8, 2)),
        "w2_weight": torch.empty((2, 4, 2)),
        "quant_type": AiterQuantType.PER_1X32,
    }
    kwargs.update(overrides)
    return AiterMoeQuantInfo(**kwargs)


def _install_fake_aiter(monkeypatch, fused_moe):
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="INTERLEAVE")
    )

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)


def _route_sort_runner(monkeypatch, *, moe_ep_size=1, available=True):
    monkeypatch.setattr(aiter_route_sort, "available", lambda: available)
    layer = SimpleNamespace(moe_ep_size=moe_ep_size)
    return AiterRunnerCore(MoeRunnerConfig(activation="silu", layer=layer))


def test_aiter_runner_declares_a_route_sort_consumer_only_without_ep(monkeypatch):
    """AITER drops a registered sort under EP's expert mask, and an AITER without
    register_presorted_moe cannot take one, so only then is there no consumer."""
    runner = _route_sort_runner(monkeypatch)
    assert isinstance(
        runner.route_sort_consumer(), aiter_route_sort.AiterRouteSortConsumer
    )
    assert _route_sort_runner(monkeypatch, moe_ep_size=2).route_sort_consumer() is None
    assert (
        _route_sort_runner(monkeypatch, available=False).route_sort_consumer() is None
    )


def test_aiter_runner_shows_its_staged_route_sort_consumer_its_fused_moe_call(
    monkeypatch,
):
    """The AITER route-sort consumer asks AITER for the config of exactly the
    call its own layer's fused_moe gets."""
    captured = []

    def fused_moe(**kwargs):
        captured.append(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    runner = _route_sort_runner(monkeypatch)
    consumer = runner.route_sort_consumer()
    calls = []
    consumer.observe = calls.append
    # The scope drops the sort no fused_moe consumed through AITER's registrar.
    monkeypatch.setattr(aiter_route_sort, "_registrar", lambda: lambda sort: None)
    for staged in (consumer, aiter_route_sort.AiterRouteSortConsumer()):
        with route_sort.scope(staged):
            runner.run(_runner_input(), _quant_info(), {})
    # Shown its own layer's call only, not another staged layer's.
    assert len(calls) == 1 and calls[0].keys() == captured[0].keys()
    assert all(calls[0][name] is value for name, value in captured[0].items())
