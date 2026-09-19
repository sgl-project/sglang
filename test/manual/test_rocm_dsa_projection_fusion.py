"""CPU-only source-contract tests; no serving-stack or GPU qualification.

Run with: pytest test/manual/test_rocm_dsa_projection_fusion.py

Extract the production methods to avoid importing GPU-only attention modules.
The loader/dequantization and projection math below execute the real method
bodies on CPU. Full Indexer construction, graph replay and serving need GPU CI.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
INDEXER = ROOT / "python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
LOADER = ROOT / "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py"
FP8_UTILS = ROOT / "python/sglang/srt/layers/quantization/fp8_utils.py"


def _tree(path):
    return ast.parse(path.read_text())


def _function(path, name, namespace):
    nodes = [
        n
        for n in ast.walk(_tree(path))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(nodes) == 1, name
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
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _gate(
    cuda=False,
    hip=True,
    gfx942=True,
    enabled=True,
    disabled=False,
    neox=False,
    lora=False,
    quant=None,
):
    init = next(
        n
        for n in _tree(INDEXER).body
        if isinstance(n, ast.ClassDef) and n.name == "Indexer"
    )
    init = next(
        n for n in init.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    names = {"use_dsa_indexer_fusion", "use_dsa_indexer_projection_fusion"}
    assignments = [
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr in names for t in n.targets)
    ]
    assert len(assignments) == 2
    obj = SimpleNamespace()
    ns = dict(
        self=obj,
        _is_cuda=cuda,
        _is_hip=hip,
        is_gfx942_supported=lambda: gfx942,
        is_neox_style=neox,
        quant_config=quant,
        get_lora=lambda: SimpleNamespace(enable_lora=lora),
        envs=SimpleNamespace(
            SGLANG_DISABLE_DSA_INDEXER_FUSION=SimpleNamespace(get=lambda: disabled),
            SGLANG_ROCM_DSA_INDEXER_PROJECTION_FUSION=SimpleNamespace(
                get=lambda: enabled
            ),
        ),
    )
    exec(
        compile(ast.Module(body=assignments, type_ignores=[]), str(INDEXER), "exec"), ns
    )
    return obj.use_dsa_indexer_fusion, obj.use_dsa_indexer_projection_fusion


def _quant(name="fp8", block=(128, 128), mxfp8=False):
    return SimpleNamespace(
        get_name=lambda: name,
        weight_block_size=list(block) if block else None,
        use_mxfp8=mxfp8,
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, (False, True)),
        ({"quant": _quant()}, (False, True)),
        ({"enabled": False}, (False, False)),
        ({"disabled": True}, (False, False)),
        ({"neox": True}, (False, False)),
        ({"lora": True}, (False, False)),
        ({"gfx942": False}, (False, False)),
        ({"hip": False}, (False, False)),
        ({"quant": _quant("awq")}, (False, False)),
        ({"quant": _quant(block=None)}, (False, False)),
        ({"quant": _quant(block=(1, 32), mxfp8=True)}, (False, False)),
        ({"cuda": True, "hip": False, "enabled": False}, (True, True)),
        ({"cuda": True, "hip": False, "disabled": True}, (False, False)),
    ],
)
def test_projection_gate_does_not_enable_cuda_postprocessing(kwargs, expected):
    assert _gate(**kwargs) == expected


def _loader_namespace():
    ns = {"torch": torch, "_clone_if_runai_streamed_tensor": lambda x: x.clone()}
    _function(FP8_UTILS, "block_quant_dequant", ns)
    _function(LOADER, "_get_indexer_weight_block_size", ns)
    _function(LOADER, "_load_fused_indexer_wk", ns)
    return ns


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_checkpoint_shards_pack_and_project(fp8, reverse):
    torch.manual_seed(170)
    n, h, k = 128, 64, 256
    wk = torch.randn(n, k, dtype=torch.bfloat16)
    gates = torch.randn(h, k, dtype=torch.bfloat16)
    expected_wk = wk
    prefix = "model.layers.0.self_attn.indexer."
    entries = [(prefix + "wk.weight", wk), (prefix + "weights_proj.weight", gates)]
    quant = None
    if fp8:
        quant = _quant()
        wk = wk.to(torch.float8_e4m3fn)
        scale = torch.tensor([[0.5, 2.0]])
        expected_wk = (wk.float() * scale.repeat_interleave(128, -1)).bfloat16()
        entries = [
            (prefix + "wk.weight", wk),
            (prefix + "wk.weight_scale_inv", scale),
            (prefix + "weights_proj.weight", gates),
        ]
    if reverse:
        entries.reverse()
    packed = torch.full((n + h, k), float("nan"), dtype=torch.bfloat16)
    params = {prefix + "wk_weights_proj.weight": packed}
    pending = {}
    ns = _loader_namespace()
    for name, weight in entries:
        assert ns["_load_fused_indexer_wk"](name, weight, params, pending, quant)
    assert pending == {}
    torch.testing.assert_close(packed[:n], expected_wk, rtol=0, atol=0)
    torch.testing.assert_close(packed[n:], gates, rtol=0, atol=0)

    x = torch.randn(8, k, dtype=torch.bfloat16)
    obj = SimpleNamespace(
        head_dim=n, n_heads=h, wk_weights_proj=lambda x: (x @ packed.T, None)
    )
    fused = _function(INDEXER, "_fused_k_weights", ns)
    key, raw = fused(obj, x)
    torch.testing.assert_close(key, x @ expected_wk.T, rtol=0, atol=0)
    torch.testing.assert_close(raw, x @ gates.T, rtol=0, atol=0)


def test_loader_falls_through_when_modules_are_separate():
    ns = _loader_namespace()
    assert not ns["_load_fused_indexer_wk"](
        "model.layers.0.self_attn.indexer.wk.weight",
        torch.ones(2, 2, dtype=torch.bfloat16),
        {},
        {},
        None,
    )


def test_projection_only_retains_hadamard_rotation():
    ns = {"rotate_activation": lambda x: ("rotated", x)}
    rotate = _function(INDEXER, "_maybe_rotate", ns)
    obj = SimpleNamespace(
        use_dsa_indexer_fusion=False, use_dsa_indexer_projection_fusion=True
    )
    assert rotate(obj, "input") == ("rotated", "input")


def test_k_only_path_does_not_read_absent_wk():
    ns = dict(
        torch=torch,
        _is_cuda=False,
        _is_hip=True,
        _is_xpu=False,
        rotate_activation=lambda x: x,
    )
    method = _function(INDEXER, "_get_k_bf16", ns)
    key = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    obj = SimpleNamespace(
        use_dsa_indexer_projection_fusion=True,
        head_dim=8,
        rope_head_dim=4,
        _fused_k_weights=lambda x: (key.clone(), None),
        k_norm=lambda x: x,
        rotary_emb=lambda p, q, k: (q, k),
        _update_rope_guarded=lambda destination, source: destination.copy_(source),
    )
    torch.testing.assert_close(method(obj, None, torch.arange(2)), key)


def test_fp8_qscale_gate_uses_projected_values():
    ns = {"torch": torch}
    scale = _function(INDEXER, "_scale_head_gates", ns)
    raw = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    qscale = torch.tensor([[[0.5], [2.0]]])
    obj = SimpleNamespace(n_heads=2, softmax_scale=0.125)
    expected = (raw * 2**-0.5).unsqueeze(-1) * qscale * 0.125
    torch.testing.assert_close(scale(obj, raw, qscale), expected, rtol=0, atol=0)
