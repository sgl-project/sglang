"""CPU regression tests for unquantized experts in ModelSlim checkpoints.

Execute the production config methods without importing the NPU runtime.
Quantization method/kernel constructors are stand-ins; no device math is tested.
"""

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/layers/quantization/modelslim/modelslim.py"
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


@pytest.fixture
def modelslim(monkeypatch):
    class FusedMoE(torch.nn.Module):
        use_triton_kernels = False
        use_flashinfer_trtllm_moe = False
        use_deep_gemm = False

    for name, attr, value in (
        ("sglang.srt.layers.linear", "LinearBase", type("LinearBase", (), {})),
        ("sglang.srt.layers.moe.fused_moe_triton", "FusedMoE", FusedMoE),
    ):
        module = ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    config = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelSlimConfig"
    )
    methods = {
        "__init__",
        "_quant_prefix_candidates",
        "get_moe_scheme",
        "get_quant_method",
    }
    config.body = [
        n for n in config.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    remap = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "remap_dspark_weight_name_npu"
    )
    namespace = {
        "torch": torch,
        "cast": cast,
        "List": list,
        "logger": Mock(),
        "QuantizationConfig": object,
        "UnquantizedFusedMoEMethod": Mock(),
        "ModelSlimFusedMoEMethod": Mock(),
    }
    for name in (
        "ModelSlimW4A4MXFP4MoE",
        "ModelSlimW4A8MXFP4MoE",
        "ModelSlimW4A4Int4MoE",
        "ModelSlimW4A8Int8MoE",
        "ModelSlimW8A8Int8MoE",
        "ModelSlimMXFP8MoEScheme",
    ):
        namespace[name] = Mock()
    module = ast.Module(body=[tree.body[0], remap, config], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace, FusedMoE


def _entries(prefix, schemes, names=("gate_proj", "up_proj", "down_proj")):
    return {f"{prefix}.0.{name}.weight": scheme for name, scheme in zip(names, schemes)}


@pytest.mark.parametrize("stage", [0, 1])
def test_float_draft_preserves_quantized_target(modelslim, stage):
    ns, layer_type = modelslim
    target = "model.layers.0.mlp.experts"
    draft = f"stages.{stage}.mlp.experts"
    config = ns["ModelSlimConfig"](
        {
            **_entries(target, ["W4A8_DYNAMIC"] * 3),
            **_entries(f"mtp.{stage}.ffn.experts", ["FLOAT"] * 3, ("w1", "w3", "w2")),
        }
    )
    layer = layer_type()
    assert (
        config.get_quant_method(layer, draft)
        is ns["UnquantizedFusedMoEMethod"].return_value
    )
    ns["UnquantizedFusedMoEMethod"].assert_called_once_with(False, False, False)
    assert not hasattr(layer, "w13_scheme")
    assert (
        config.get_quant_method(layer, target)
        is ns["ModelSlimFusedMoEMethod"].return_value
    )
    assert ns["ModelSlimW4A8Int8MoE"].call_count == 2
    assert layer.w13_kernel is layer.w13_scheme.kernel
    assert layer.w2_kernel is layer.w2_scheme.kernel


@pytest.mark.parametrize(
    "prefix", ["stages.0.mlp.experts", "language_model.stages.0.mlp.experts"]
)
@pytest.mark.parametrize(
    "names", [("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")]
)
def test_float_supports_existing_prefix_and_projection_aliases(
    modelslim, prefix, names
):
    ns, layer_type = modelslim
    config = ns["ModelSlimConfig"](_entries(prefix, ["FLOAT"] * 3, names))
    assert (
        config.get_quant_method(layer_type(), "stages.0.mlp.experts")
        is ns["UnquantizedFusedMoEMethod"].return_value
    )


@pytest.mark.parametrize(
    "schemes,error",
    [
        (["FLOAT", "W4A8_DYNAMIC", "FLOAT"], "Mismatched ModelSlim"),
        (["FLOAT", "FLOAT", "W4A8_DYNAMIC"], "Unsupported ModelSlim"),
        (["W4A8_DYNAMIC", "W4A8_DYNAMIC", "FLOAT"], "Unsupported ModelSlim"),
        (["UNKNOWN"] * 3, "Unsupported ModelSlim"),
        (["FLOAT", "FLOAT"], "Missing ModelSlim"),
        ([], "Missing ModelSlim"),
    ],
)
def test_invalid_schemes_do_not_fall_back_to_float(modelslim, schemes, error):
    ns, layer_type = modelslim
    prefix = "stages.0.mlp.experts"
    config = ns["ModelSlimConfig"](_entries(prefix, schemes))
    with pytest.raises(ValueError, match=error):
        config.get_quant_method(layer_type(), prefix)
    ns["UnquantizedFusedMoEMethod"].assert_not_called()
