import importlib.util
import sys
import types
from pathlib import Path

import torch


def _install_fake_modules():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.hardware_backend.npu.utils",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.base_config",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    npu_utils = sys.modules["sglang.srt.hardware_backend.npu.utils"]
    npu_utils.npu_format_cast = lambda tensor: tensor

    base_config = sys.modules["sglang.srt.layers.quantization.base_config"]
    base_config.FusedMoEMethodBase = object


def _load_npu_fused_moe_module():
    _install_fake_modules()
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_npu_fused_moe_method_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_npu_swiglu_oai_matches_minimax_formula_and_differs_from_standard_swiglu():
    module = _load_npu_fused_moe_module()
    x = torch.tensor(
        [[-2.0, 0.5, 1.0, -3.0], [2.0, 3.0, -0.5, 0.25]],
        dtype=torch.float32,
    )
    alpha = 1.702
    limit = 2.0

    out = module.npu_swiglu_oai(x, alpha=alpha, limit=limit)

    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    expected = gate * torch.sigmoid(gate * alpha) * (up + 1)
    standard_swiglu = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(out, expected)
    assert not torch.allclose(out, standard_swiglu)
