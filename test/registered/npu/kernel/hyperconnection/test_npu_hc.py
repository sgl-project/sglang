"""Production HC routing and full module graph integration (not model accuracy)."""

import pytest
import torch
from sgl_kernel_npu.qwen3_8_flash_next import hc

from sglang.srt.layers import hyperconnection
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=120, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


def make_module(monkeypatch, *, model=True, per_branch=True, use_combine=True):
    monkeypatch.setattr(hyperconnection, "_is_npu", True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("npu"))
    cfg = hyperconnection.HyperConnectionConfig(
        hidden_size=2560 if model else 128,
        hc_count=4,
        hc_lowrank=320 if model else 32,
        hc_per_branch_norm=per_branch,
    )
    return hyperconnection.GatedResidual(cfg, use_combine=use_combine).to(
        device="npu", dtype=torch.bfloat16
    )


@pytest.mark.parametrize("rows", [0, 1, 32, 33, 4096, 8193, 16384])
@torch.no_grad()
def test_model_public_routing(monkeypatch, rows):
    module = make_module(monkeypatch)
    x = torch.zeros(rows, 10240, device="npu", dtype=torch.bfloat16)
    calls = []
    for name in ("grouped_norm", "mix", "combine"):
        original = getattr(hc, name)

        def counted(*args, _name=name, _fn=original):
            calls.append(_name)
            return _fn(*args)

        monkeypatch.setattr(hc, name, counted)
    # This GPU path must remain disabled even with transfer_to_npu's is_cuda shim.
    assert not hyperconnection.fused_hc_mix_supported(
        x, module.input_mix_weight_down.weight, module.input_mix_weight_up.weight
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("NPU reached a GPU eligibility check or Torch fallback")

    monkeypatch.setattr(module, "_mix_compute", unexpected)
    monkeypatch.setattr(module, "_combine_compute", unexpected)
    monkeypatch.setattr(hyperconnection, "fused_hc_mix_supported", unexpected)
    # Even eligible-looking CUDA metadata cannot bypass the NPU-first branch.
    module._jit_mix_ok = module._jit_combine_ok = True
    assert not hasattr(module, "_npu_hc_model")
    assert not hasattr(module.hc_norm, "_npu_hc_norm")
    mixed, residuals = module.mix(x)
    out = module.combine(mixed, residuals)
    assert residuals[0] is x
    assert mixed.shape == (rows, 2560) and out.shape == x.shape
    assert calls == ([] if rows == 0 else ["grouped_norm", "mix", "combine"])
    torch.testing.assert_close(out, x, atol=0, rtol=0)


@pytest.mark.parametrize("op", ["grouped_norm", "mix", "combine"])
@torch.no_grad()
def test_kernel_errors_are_not_swallowed(monkeypatch, op):
    module = make_module(monkeypatch)
    x = torch.ones(1, 10240, device="npu", dtype=torch.bfloat16)

    def fail(*args):
        raise ValueError("HC test sentinel")

    monkeypatch.setattr(hc, op, fail)
    with pytest.raises(ValueError, match="HC test sentinel"):
        mixed, residuals = module.mix(x)
        module.combine(mixed, residuals)


@pytest.mark.parametrize("bad", ["dtype", "stride", "rank"])
@torch.no_grad()
def test_model_inputs_rejected_not_fallback(monkeypatch, bad):
    module = make_module(monkeypatch)
    x = torch.ones(2, 10240, device="npu", dtype=torch.bfloat16)
    if bad == "dtype":
        x = x.float()
    elif bad == "stride":
        x = torch.stack((x, x), -1)[..., 0]
    else:
        x = x.view(1, 2, 10240)
    with pytest.raises(ValueError):
        module.mix(x)


@pytest.mark.parametrize("per_branch", [False, True])
@torch.no_grad()
def test_unsupported_configuration_rejected(monkeypatch, per_branch):
    module = make_module(monkeypatch, model=False, per_branch=per_branch)
    x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        module.mix(x)
    block = torch.zeros(4, 128, device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        module.combine(block, (x, x))


@torch.no_grad()
def test_final_mixer_without_combine(monkeypatch):
    module = make_module(monkeypatch, use_combine=False)
    x = torch.randn(33, 10240, device="npu", dtype=torch.bfloat16)
    actual, residuals = module.mix(x)
    expected = hc.mix(
        hc.grouped_norm(x, module.hc_norm.weight, 2560, 1e-6),
        module.input_mix_weight_down.weight,
        module.input_mix_weight_up.weight,
        4,
        2560,
    )
    assert residuals[0] is x
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 32, 33, 4096, 8193])
@torch.no_grad()
def test_module_graph_updates(monkeypatch, rows):
    module = make_module(monkeypatch)
    x = torch.randn(rows, 10240, device="npu", dtype=torch.bfloat16)
    block = torch.randn(rows, 2560, device="npu", dtype=torch.bfloat16)
    tensors = [x, block, *module.parameters()]
    pointers = [v.data_ptr() for v in tensors]

    def run():
        mixed, residuals = module.mix(x)
        return mixed, module.combine(block, residuals)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = run()
    for slot in range(-1, len(tensors)):
        if slot >= 0:
            tensors[slot].mul_(-0.5).add_(0.015625)
        graph.replay()
        torch.npu.synchronize()
        assert [v.data_ptr() for v in tensors] == pointers
        for actual, expected in zip(outputs, run()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
