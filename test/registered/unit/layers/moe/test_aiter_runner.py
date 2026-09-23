import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="stage-b-test-cpu-intel")


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


def test_aiter_runner_forwards_no_combine_and_extra_fused_moe_kwargs(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"}),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_aiter_runner_rejects_no_combine_when_fused_moe_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))

    with pytest.raises(NotImplementedError, match="no_combine=True"):
        runner.run(_runner_input(), _quant_info(), running_state={})


def test_aiter_runner_preserves_no_combine_rank_for_empty_input(monkeypatch):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))
    runner_input = _runner_input()
    runner_input.hidden_states = torch.zeros((0, 4), dtype=torch.bfloat16)
    runner_input.topk_ids = torch.zeros((0, 2), dtype=torch.int32)
    runner_input.topk_weights = torch.zeros((0, 2), dtype=torch.float32)

    output = runner.run(runner_input, _quant_info(), running_state={})

    assert output.hidden_states.shape == (0, 2, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---- the ROCm fused front: the gate launch sorts, moe_sorting hands the outputs back ----


def _install_fake_moe_sorting(monkeypatch, moe_sorting):
    _install_fake_aiter(monkeypatch, fused_moe=None)
    sys.modules["aiter.fused_moe"].moe_sorting = moe_sorting
    aiter_runner._install_fused_sorting_override.cache_clear()
    monkeypatch.setattr(aiter_runner, "_fill_padded_rows_pair", lambda *a: None)


def _aiter_moe_sorting_signature(calls):
    """aiter's moe_sorting parameters plus the trailing output newer builds add."""

    def moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size=32,
        expert_mask=None,
        num_local_tokens=None,
        dispatch_policy=0,
        return_local_topk_ids=False,
        accumulate=False,
        flat=False,
        output_aux=False,
        output=None,
    ):
        calls.append("aiter")
        return "aiter"

    return moe_sorting


def _fake_front(monkeypatch, launches):
    """CPU stand-ins for the two ROCm gate launches, recording which one ran (the
    batches below stay within ROCM_GATE_SORT_MAX_TOKENS, the fold-in's row limit)."""
    import sglang.srt.layers.moe.rocm_fused_front as front

    def gate(gating_output, correction_bias, topk, *a, **kw):
        launches.append("gate")
        ids = torch.zeros(gating_output.shape[0], topk, dtype=torch.int32)
        return torch.ones(gating_output.shape[0], topk), ids

    def gate_sort(gating_output, correction_bias, topk, *a, **kw):
        launches.append("gate+sort")
        ids = torch.zeros(gating_output.shape[0], topk, dtype=torch.int32)
        sorted_outputs = tuple(torch.empty(1) for _ in range(5))
        return (torch.ones(gating_output.shape[0], topk), ids, *sorted_outputs)

    monkeypatch.setattr(front, "rocm_router_gate", gate)
    monkeypatch.setattr(front, "rocm_router_gate_sort", gate_sort)
    monkeypatch.setattr(front, "_sort_configs", {})
    monkeypatch.setattr(front, "_pending_sorts", {})
    return front


def _sort(num_tokens, ids, **overrides):
    kwargs = dict(
        topk_ids=ids,
        topk_weights=torch.ones(num_tokens, ids.shape[1], dtype=torch.float32),
        num_experts=8,
        model_dim=16,
        moebuf_dtype=torch.bfloat16,
        block_size=32,
    )
    kwargs.update(overrides)
    return sys.modules["aiter.fused_moe"].moe_sorting(**kwargs)


def test_fused_sorting_hands_the_gate_launch_outputs_to_moe_sorting(monkeypatch):
    """First sight of a router records aiter's sorting arguments; the router's next gate
    launch sorts too, and moe_sorting returns those outputs instead of launching.
    Different arguments are re-sorted and re-recorded."""
    calls, launches = [], []
    _install_fake_moe_sorting(monkeypatch, _aiter_moe_sorting_signature(calls))
    front = _fake_front(monkeypatch, launches)
    fused = []
    monkeypatch.setattr(
        "sglang.kernels.ops.moe.aiter_moe_sorting_fused.fused_aiter_moe_sorting",
        lambda *a, **kw: fused.append((a, kw)) or "fused",
    )
    assert aiter_runner._install_fused_sorting_override()
    logits = torch.zeros(2, 8)
    bias = torch.zeros(8)

    def gate_then_sort(**overrides):
        _, ids = front.gate_partials(logits, bias, 2, False, None, logits, None)
        with aiter_runner._fused_sorting_scope(None):
            return _sort(2, ids, **overrides)

    # first sight: the gate runs alone, moe_sorting sorts in its own fused launch
    assert gate_then_sort() == "fused"
    assert launches == ["gate"] and len(fused) == 1
    # the router's next batch sorts in the gate launch and hands the outputs back
    outputs = gate_then_sort()
    assert launches == ["gate", "gate+sort"]
    assert isinstance(outputs, tuple) and len(outputs) == 5 and len(fused) == 1
    # other sorting arguments: the pending outputs are not served, the new
    # arguments are recorded
    assert gate_then_sort(block_size=64) == "fused"
    assert len(fused) == 2
    assert gate_then_sort(block_size=64)[0] is not None and len(fused) == 2
    assert launches == ["gate", "gate+sort", "gate+sort", "gate+sort"]
    assert calls == []


def test_fused_sorting_defers_to_aiter_and_stops_fusing_when_it_cannot_serve(
    monkeypatch,
):
    """A trailing aiter parameter, a call outside the scope or a form the fused launch
    cannot serve goes to aiter; the last also stops folding the router's sorting into
    its gate launch."""
    calls, launches = [], []
    _install_fake_moe_sorting(monkeypatch, _aiter_moe_sorting_signature(calls))
    front = _fake_front(monkeypatch, launches)
    monkeypatch.setattr(
        "sglang.kernels.ops.moe.aiter_moe_sorting_fused.fused_aiter_moe_sorting",
        lambda *a, **kw: "fused",
    )
    assert aiter_runner._install_fused_sorting_override()
    logits = torch.zeros(2, 8)
    bias = torch.zeros(8)

    def gate():
        return front.gate_partials(logits, bias, 2, False, None, logits, None)[1]

    ids = gate()
    assert _sort(2, ids) == "aiter"  # outside the scope
    with aiter_runner._fused_sorting_scope(None):
        assert _sort(2, ids, output=torch.empty(1)) == "aiter"  # trailing parameter
        assert _sort(2, gate()) == "fused"
        ids = gate()
        assert launches == ["gate", "gate", "gate+sort"]
        # dispatch_policy 1 is not the plain sorting: aiter serves it, and the
        # router's gate launches stop sorting
        assert _sort(2, ids, dispatch_policy=1) == "aiter"
        gate()
        assert launches[-1] == "gate"
    assert calls == ["aiter", "aiter", "aiter"]


def test_fused_sorting_override_needs_aiter_sorting_parameters(monkeypatch):
    """An aiter whose moe_sorting leads with other parameters keeps its own kernel."""

    def moe_sorting(topk_ids, weights, num_experts):
        return "aiter"

    _install_fake_moe_sorting(monkeypatch, moe_sorting)
    assert not aiter_runner._install_fused_sorting_override()
    assert sys.modules["aiter.fused_moe"].moe_sorting is moe_sorting
