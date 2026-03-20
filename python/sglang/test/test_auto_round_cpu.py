import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _prepare_imports():
    repo_python = Path(__file__).resolve().parents[2]
    sglang_root = repo_python / "sglang"

    # Avoid importing sglang/__init__.py (which pulls frontend deps) in this UT.
    pkg_map = {
        "sglang": sglang_root,
        "sglang.srt": sglang_root / "srt",
        "sglang.srt.layers": sglang_root / "srt" / "layers",
        "sglang.srt.layers.quantization": sglang_root / "srt" / "layers" / "quantization",
    }
    for name, path in pkg_map.items():
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod

    # Keep this UT lightweight by stubbing modules that otherwise pull heavy
    # runtime dependencies through unrelated import chains.
    utils_stub = types.ModuleType("sglang.srt.utils")
    utils_stub.is_cpu = lambda: False
    utils_stub.is_npu = lambda: False
    sys.modules["sglang.srt.utils"] = utils_stub

    quant_utils_stub = types.ModuleType("sglang.srt.layers.quantization.utils")

    class _MockScalarTypes:
        uint4 = "uint4"
        uint8 = "uint8"
        uint4b8 = "uint4b8"
        uint8b128 = "uint8b128"

    quant_utils_stub.get_scalar_types = lambda: (type("MockScalarType", (), {}), _MockScalarTypes())  # type: ignore[arg-type]
    sys.modules["sglang.srt.layers.quantization.utils"] = quant_utils_stub

    return importlib.import_module("sglang.srt.layers.quantization.auto_round")


auto_round_mod = _prepare_imports()
AutoRoundConfig = auto_round_mod.AutoRoundConfig
AutoRoundAWQCPULinearMethod = auto_round_mod.AutoRoundAWQCPULinearMethod
AutoRoundGPTQCPULinearMethod = auto_round_mod.AutoRoundGPTQCPULinearMethod


def _pack_rows_dim0_int4(qweight_unpacked: torch.Tensor) -> torch.Tensor:
    # [K, N] -> [K // 8, N], 8 x int4 packed into one int32 per row-group.
    assert qweight_unpacked.dtype in (torch.int32, torch.int64, torch.uint8)
    k, n = qweight_unpacked.shape
    assert k % 8 == 0
    out = torch.zeros((k // 8, n), dtype=torch.int32)
    for i in range(8):
        out |= (qweight_unpacked[i::8, :].to(torch.int32) & 0xF) << (4 * i)
    return out


def _pack_cols_dim1_int4(qzeros_unpacked: torch.Tensor) -> torch.Tensor:
    # [G, N] -> [G, N // 8], 8 x int4 packed into one int32 per col-group.
    assert qzeros_unpacked.dtype in (torch.int32, torch.int64, torch.uint8)
    g, n = qzeros_unpacked.shape
    assert n % 8 == 0
    out = torch.zeros((g, n // 8), dtype=torch.int32)
    for i in range(8):
        out |= (qzeros_unpacked[:, i::8].to(torch.int32) & 0xF) << (4 * i)
    return out


def test_auto_round_get_quant_method_routes_awq(monkeypatch):
    cfg = AutoRoundConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_awq",
        backend="auto",
    )
    sentinel = object()
    monkeypatch.setattr(
        cfg,
        "apply_awq_quant_layer",
        lambda layer, prefix, backend="auto": sentinel,
    )
    assert cfg.get_quant_method(object(), "model.layers.0.mlp.down_proj") is sentinel


def test_auto_round_get_quant_method_routes_gptq(monkeypatch):
    cfg = AutoRoundConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        backend="auto",
    )
    sentinel = object()
    monkeypatch.setattr(
        cfg,
        "apply_gptq_quant_layer",
        lambda layer, prefix, backend="auto": sentinel,
    )
    assert cfg.get_quant_method(object(), "model.layers.0.mlp.down_proj") is sentinel


def test_auto_round_gptq_cpu_fallback_matches_reference():
    method = AutoRoundGPTQCPULinearMethod.__new__(AutoRoundGPTQCPULinearMethod)
    method.quant_config = types.SimpleNamespace(weight_bits=4, group_size=2)
    method.use_v2_format = False  # non-v2: qzeros += 1

    # K=8, N=8 so both packed dims align with int4 pack factor (8).
    k, n, group_size = 8, 8, 2
    num_groups = k // group_size

    # int4 range [0, 15]
    qweight_unpacked = (torch.arange(k * n).reshape(k, n) % 16).to(torch.int32)
    qzeros_unpacked = (torch.arange(num_groups * n).reshape(num_groups, n) % 16).to(
        torch.int32
    )
    scales = torch.linspace(0.1, 1.0, num_groups * n, dtype=torch.float32).reshape(
        num_groups, n
    )

    layer = types.SimpleNamespace()
    layer.qweight = torch.nn.Parameter(
        _pack_rows_dim0_int4(qweight_unpacked), requires_grad=False
    )
    layer.qzeros = torch.nn.Parameter(
        _pack_cols_dim1_int4(qzeros_unpacked), requires_grad=False
    )
    layer.g_idx = torch.nn.Parameter(
        torch.empty((0,), dtype=torch.int32), requires_grad=False
    )
    layer.scales = torch.nn.Parameter(scales, requires_grad=False)

    x = torch.linspace(-1.0, 1.0, 3 * k, dtype=torch.float32).reshape(3, k)
    bias = torch.linspace(-0.5, 0.5, n, dtype=torch.float32)

    out = method.apply(layer, x, bias=bias)

    g_idx = torch.arange(k, dtype=torch.long) // group_size
    # non-v2 GPTQ qzeros has +1 offset in dequant
    qzeros_effective = qzeros_unpacked.to(torch.float32) + 1.0
    scale_zeros = qzeros_effective * scales
    dequant_weight = (
        qweight_unpacked.to(torch.float32) * scales[g_idx] - scale_zeros[g_idx]
    )
    ref = x @ dequant_weight + bias
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_auto_round_awq_cpu_requires_int4_kernels(monkeypatch):
    method = AutoRoundAWQCPULinearMethod.__new__(AutoRoundAWQCPULinearMethod)

    layer = types.SimpleNamespace()
    layer.qweight = torch.nn.Parameter(
        torch.zeros((8, 1), dtype=torch.int32), requires_grad=False
    )
    layer.qzeros = torch.nn.Parameter(
        torch.zeros((1, 1), dtype=torch.int32), requires_grad=False
    )
    layer.scales = torch.nn.Parameter(
        torch.ones((1, 8), dtype=torch.float16), requires_grad=False
    )

    monkeypatch.setattr(auto_round_mod, "_has_int4_cpu_kernels", lambda: False)
    with pytest.raises(RuntimeError, match="int4 CPU kernels"):
        method.process_weights_after_loading(layer)
