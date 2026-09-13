"""CPU gate tests. GPU MoE/loader/endpoint qualification is separate."""

import ast
import copy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "python/sglang/srt/models/glm5_next.py"
UTILS = ROOT / "python/sglang/srt/layers/quantization/utils.py"


def extract(path, name, namespace):
    functions = [
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(functions) == 1
    node = copy.deepcopy(functions[0])
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
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def gate(
    cuda=False,
    aiter=True,
    gfx942=True,
    enabled=True,
    shared=1,
    ep=1,
    deepep=False,
    sm=None,
    quant=None,
    wrapped=False,
):
    ns = dict(
        _is_cuda=cuda,
        _use_aiter=aiter,
        _device_sm=sm,
        is_gfx942_supported=lambda: gfx942,
        envs=SimpleNamespace(
            SGLANG_ROCM_GLM_SHARED_EXPERTS_FUSION=SimpleNamespace(get=lambda: enabled)
        ),
        get_parallel=lambda: SimpleNamespace(moe_ep_size=ep),
        get_moe_a2a_backend=lambda: SimpleNamespace(is_deepep=lambda: deepep),
        MappingProxyType=MappingProxyType,
        _FALLBACK_FUSED_SHARDS={},
    )
    extract(UTILS, "_module_path_match", ns)
    extract(UTILS, "is_layer_skipped", ns)
    config = SimpleNamespace(
        n_shared_experts=shared, num_hidden_layers=3, first_k_dense_replace=1
    )
    if wrapped:
        config = SimpleNamespace(text_config=config)
    return extract(MODEL, "shared_experts_fusion_disable_reason", ns)(
        None, config, quant
    )


def fp8(ignored=(), block=(128, 128), name="fp8", mxfp8=False, fp4_experts=False):
    return SimpleNamespace(
        get_name=lambda: name,
        weight_block_size=list(block),
        ignored_layers=list(ignored),
        packed_modules_mapping={},
        use_mxfp8=mxfp8,
        is_fp4_experts=fp4_experts,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"quant": fp8()},
        {"wrapped": True},
        {"quant": fp8(["model.layers.1.mlp.gate"])},
        {"cuda": True, "enabled": False, "aiter": False, "gfx942": False, "sm": 90},
    ],
)
def test_supported_paths(kwargs):
    assert gate(**kwargs) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aiter": False},
        {"gfx942": False},
        {"enabled": False},
        {"shared": 0},
        {"shared": 2},
        {"ep": 8},
        {"deepep": True},
        {"quant": fp8(name="awq")},
        {"quant": fp8(block=(1, 32), mxfp8=True)},
        {"quant": fp8(fp4_experts=True)},
        {"quant": fp8(["model.layers.1.mlp.shared_experts"])},
        {"quant": fp8(["model.layers.1.mlp.shared_experts.gate_proj"])},
        {"quant": fp8(["model.layers.1.mlp.experts.0.gate_proj"])},
        {"quant": fp8(["model.layers.2.mlp"])},
        {"cuda": True, "sm": 70},
    ],
)
def test_unsupported_paths_keep_unfused(kwargs):
    assert gate(**kwargs) is not None


def test_weight_loader_and_wrapper_read_one_published_decision():
    wrapper = ROOT / "python/sglang/srt/models/deepseek_v2.py"
    for path, method in (
        (MODEL, "determine_num_fused_shared_experts"),
        (wrapper, "__init__"),
    ):
        calls = [
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "is_shared_experts_fusion_disabled"
        ]
        assert calls, (path, method)
