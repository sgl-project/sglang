# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

pytest.importorskip("humming")
from sglang.srt.layers.quantization.humming import HummingMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_mega_preserves_native_mxfp4_bytes_and_global_expert_count(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from sglang.srt.layers.quantization import humming

    monkeypatch.setattr(
        humming,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(enable_mxfp4_mega=True)),
    )
    layer = torch.nn.Module()
    method = HummingMoEMethod.__new__(HummingMoEMethod)
    method.native_mxfp4_mega = True
    layer.sublayer_configs = {
        "w13": {"shape_n": 128, "shape_k": 128},
        "w2": {"shape_n": 128, "shape_k": 64},
    }
    for name, shape in layer.sublayer_configs.items():
        n, k = shape["shape_n"], shape["shape_k"]
        layer.register_parameter(
            name + "_weight",
            torch.nn.Parameter(
                torch.empty((2, n, k // 2), dtype=torch.uint8), requires_grad=False
            ),
        )
        layer.register_parameter(
            name + "_weight_scale",
            torch.nn.Parameter(
                torch.empty((2, n, k // 32), dtype=torch.uint8), requires_grad=False
            ),
        )
    layer.num_experts = 4
    layer.hidden_size = 128
    layer.intermediate_size_per_partition = 64
    layer.num_local_experts = 2
    assert layer.num_experts == 4
    assert method.native_mxfp4_mega
    captured = {}
    expected = {}
    for name in ("w13", "w2"):
        for suffix in ("weight", "weight_scale"):
            tensor = getattr(layer, f"{name}_{suffix}")
            raw = tensor.view(torch.uint8)
            raw.copy_(torch.arange(raw.numel()).reshape(raw.shape).to(torch.uint8))
            expected[f"{name}_{suffix}"] = raw.clone()

    module_name = "flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.weights"
    module = ModuleType(module_name)

    def preprocess(weights, **kwargs):
        captured.update(
            w13=weights.w13,
            w2=weights.w2,
            w13_scale=weights.w13_scale,
            w2_scale=weights.w2_scale,
        )
        return object(), object()

    module.preprocess_mega_weights = preprocess
    monkeypatch.setitem(sys.modules, module_name, module)
    weights_module = ModuleType("flashinfer.moe_ep.weights")
    weights_module.PrequantizedMoEWeights = SimpleNamespace
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep.weights", weights_module)
    method.process_weights_after_loading(layer)
    for name in ("w13", "w2"):
        n = layer.sublayer_configs[name]["shape_n"]
        k = layer.sublayer_configs[name]["shape_k"]
        assert torch.equal(captured[name], expected[name + "_weight"][:, :n, : k // 2])
        assert torch.equal(
            captured[name + "_scale"],
            expected[name + "_weight_scale"][:, :n, : k // 32],
        )
    assert layer._mega_moe_weights_built
    assert layer.num_experts == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
