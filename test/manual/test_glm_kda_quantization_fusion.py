"""CPU checks of exact production eligibility; GPU loading tested separately."""

import ast
import copy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "python/sglang/srt/models/glm5_next.py"
UTILS = ROOT / "python/sglang/srt/layers/quantization/utils.py"
PREFIX = "model.layers.1.self_attn"
NAMES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "b_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
]


def production_gate():
    ns = {"MappingProxyType": MappingProxyType, "_FALLBACK_FUSED_SHARDS": {}}
    for path, name in (
        (UTILS, "_module_path_match"),
        (UTILS, "is_layer_skipped"),
        (MODEL, "_kda_projections_are_unquantized"),
    ):
        nodes = [
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        assert len(nodes) == 1
        node = copy.deepcopy(nodes[0])
        node.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns["_kda_projections_are_unquantized"]


def config(ignored, name="fp8"):
    return SimpleNamespace(
        get_name=lambda: name, ignored_layers=ignored, packed_modules_mapping={}
    )


def test_no_quantization():
    assert production_gate()(None, PREFIX)


@pytest.mark.parametrize("ignored", [NAMES, [PREFIX], [f"{PREFIX}.{x}" for x in NAMES]])
def test_all_original_projections_excluded(ignored):
    assert production_gate()(config(ignored), PREFIX)


@pytest.mark.parametrize("missing", NAMES)
def test_each_partially_quantized_projection_disables_fusion(missing):
    assert not production_gate()(config([x for x in NAMES if x != missing]), PREFIX)


@pytest.mark.parametrize(
    "ignored,name",
    [([], "fp8"), (NAMES, "awq"), ([f"other.{n}" for n in NAMES], "fp8")],
)
def test_unsupported_quantization_stays_unfused(ignored, name):
    assert not production_gate()(config(ignored, name), PREFIX)


def test_fused_linear_cannot_requantize_unrecognized_fused_name():
    calls = [
        n
        for n in ast.walk(ast.parse(MODEL.read_text()))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "MergedColumnParallelRepeatedLinear"
    ]
    assert len(calls) == 1
    quant = next(k.value for k in calls[0].keywords if k.arg == "quant_config")
    assert isinstance(quant, ast.Constant) and quant.value is None


def test_head_sharding_guard_remains():
    assignments = [
        n
        for n in ast.walk(ast.parse(MODEL.read_text()))
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "do_fuse_qkvbfg"
            for t in n.targets
        )
    ]
    assert len(assignments) == 1
    assert "head_shard_size == self.tp_size" in ast.unparse(assignments[0].value)
