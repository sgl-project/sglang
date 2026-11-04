"""Isolated CPU checks of the production Engram rotation functions.

AST extraction executes the real function bodies without importing the complete
SGLang/NPU runtime. Small real safetensors files exercise the sidecar loader.
These tests do not establish NPU kernel or graph-capture correctness.
"""

import ast
import copy
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[4]
# The marker is a stdlib-only runtime no-op; avoid importing all of SGLang.
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ENGRAM = ROOT / "python/sglang/srt/layers/engram.py"
MODEL = ROOT / "python/sglang/srt/models/deepseek_v4.py"
CONFIG = ROOT / "python/sglang/srt/configs/deepseek_v41.py"
ROTATION = {
    "value_projection_rotated": True,
    "value_basis": "quarot_global",
    "key_and_gate_basis": "original",
    "runtime_delta_rotation": False,
}


def _extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = []
    for name in names:
        body = tree.body
        for part in name.split("."):
            node = next(
                item
                for item in body
                if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                and item.name == part
            )
            body = node.body
        nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def production():
    namespace = {
        "torch": torch,
        "_is_npu": True,
        "logger": Mock(),
        "_cuda_kernels": Mock(return_value=False),
        "fused_engram_gate": Mock(side_effect=AssertionError("unexpected fused gate")),
    }
    _extract(ENGRAM, ["engram_gate", "Engram.forward"], namespace)
    _extract(MODEL, ["DeepseekV4ForCausalLM._load_npu_engram_gate_basis"], namespace)
    return namespace


def _basis():
    # Orthogonal but not symmetric: using Q instead of Q.T changes the query.
    # Signed permutations also avoid BF16 rounding obscuring the direction test.
    return torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


def _inputs(dtype, tokens=3):
    x = (torch.arange(tokens * 8).reshape(tokens, 2, 4) % 7 - 3).to(dtype)
    kv = ((torch.arange(tokens * 12).reshape(tokens, 12) % 11 - 5) / 2).to(dtype)
    q_weight = torch.tensor([[0.5, 1.0, 1.5, 2.0], [1.0, 0.5, 2.0, 1.5]])
    k_weight = torch.tensor([[1.0, 1.5, 0.5, 1.0], [0.5, 2.0, 1.0, 1.5]])
    return x, kv, q_weight.to(dtype), k_weight.to(dtype)


def _legacy_gate(x, kv, q_weight, k_weight, eps=1e-6, clamp_value=1e-6):
    """The former torch path, retained as an exact no-basis regression oracle."""
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gate_undoes_query_rotation_but_keeps_rotated_residual(production, dtype):
    original, kv, qw, kw = _inputs(dtype)
    basis = _basis()
    torch.testing.assert_close(basis @ basis.T, torch.eye(4), rtol=0, atol=0)
    assert not torch.equal(basis, basis.T)
    rotated = (original.float() @ basis).to(dtype)
    before = rotated.clone()
    production["_cuda_kernels"].return_value = True
    actual = production["engram_gate"](
        rotated, kv, qw, kw, 1e-6, 1e-6, gate_basis=basis
    )
    # Evaluate the original-basis gate in FP32; value is already in the exported
    # basis and must be added to rotated, not to the recovered original query.
    old = _legacy_gate(original.float(), kv, qw, kw)
    delta = old - original.float()
    expected = (rotated.float() + delta).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert not torch.allclose(actual.float(), old.float(), rtol=0, atol=1e-3)
    wrong_query = rotated.float() @ basis
    wrong_delta = _legacy_gate(wrong_query, kv, qw, kw) - wrong_query
    wrong = (rotated.float() + wrong_delta).to(dtype)
    assert not torch.allclose(actual.float(), wrong.float(), rtol=0, atol=1e-3)
    assert torch.equal(rotated, before)
    production["_cuda_kernels"].assert_not_called()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_zero_value_preserves_rotated_residual_exactly(production, dtype):
    x, kv, qw, kw = _inputs(dtype)
    kv[:, -4:] = 0
    result = production["engram_gate"](x, kv, qw, kw, 1e-6, 1e-6, gate_basis=_basis())
    assert torch.equal(result, x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_no_basis_is_bitwise_legacy_torch_path(production, dtype):
    args = _inputs(dtype)
    actual = production["engram_gate"](*args, 1e-6, 1e-6)
    assert torch.equal(actual, _legacy_gate(*args))
    production["_cuda_kernels"].assert_called_once_with(args[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_no_basis_keeps_fused_cuda_dispatch(production, dtype):
    args = _inputs(dtype)
    ids = torch.tensor([7, 99, 8])
    sentinel = object()
    fused = Mock(return_value=sentinel)
    production["_cuda_kernels"].return_value = True
    production["fused_engram_gate"] = fused
    result = production["engram_gate"](*args, 1e-6, 1e-6, ids, 99)
    assert result is sentinel
    fused.assert_called_once_with(*args, 1e-6, 1e-6, ids, 99)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rotated_gate_masks_images_and_handles_empty_input(production, dtype):
    args = _inputs(dtype)
    ids = torch.tensor([7, 99, 8])
    result = production["engram_gate"](*args, 1e-6, 1e-6, ids, 99, _basis())
    unmasked = production["engram_gate"](*args, 1e-6, 1e-6, gate_basis=_basis())
    assert torch.equal(result[1], args[0][1])
    assert torch.equal(result[[0, 2]], unmasked[[0, 2]])
    empty = production["engram_gate"](
        *_inputs(dtype, tokens=0),
        1e-6,
        1e-6,
        input_ids=torch.empty(0, dtype=torch.long),
        image_token_id=99,
        gate_basis=_basis(),
    )
    assert empty.shape == (0, 2, 4)
    assert empty.dtype == dtype


def _engram_stub(q_weight=None, k_weight=None):
    module = torch.nn.Module()
    module.q_weight = torch.nn.Parameter(
        torch.ones(2, 4) if q_weight is None else q_weight, requires_grad=False
    )
    module.k_weight = torch.nn.Parameter(
        torch.ones(2, 4) if k_weight is None else k_weight, requires_grad=False
    )
    module.embed = SimpleNamespace(staged_by_scheduler=False)
    module.image_token_id = 99
    module.eps = module.clamp_value = 1e-6
    module.embed_rows = Mock(return_value=torch.ones(3, 1, 4))
    return module


def test_engram_forward_passes_basis_and_preserves_empty_collective(production):
    x, kv, qw, kw = _inputs(torch.bfloat16)
    engram = _engram_stub(qw, kw)
    engram.wkv = Mock(return_value=(kv, None))
    gate = Mock(wraps=production["engram_gate"])
    production["engram_gate"] = gate
    basis = _basis()
    ids = torch.tensor([7, 99, 8])
    hashes = torch.zeros(3, 1, dtype=torch.long)
    result = production["forward"](engram, x, hashes, input_ids=ids, gate_basis=basis)
    assert gate.call_args.args[-1] is basis
    assert torch.equal(result[1], x[1])
    engram.embed_rows.reset_mock()
    engram.wkv.reset_mock()
    gate.reset_mock()
    empty = x[:0]
    assert (
        production["forward"](
            engram, empty, hashes[:0], input_ids=ids[:0], gate_basis=basis
        )
        is empty
    )
    engram.embed_rows.assert_called_once()
    engram.wkv.assert_not_called()
    gate.assert_not_called()


def _model(tmp_path, *, quant="modelslim", rotation=ROTATION, engrams=True):
    model = torch.nn.Module()
    layers = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.engram = _engram_stub() if engrams else None
        layers.append(layer)
    model.layers = torch.nn.ModuleList(layers)
    return SimpleNamespace(
        model=model,
        config=SimpleNamespace(
            _name_or_path=str(tmp_path),
            hidden_size=4,
            engram_rotation_config=copy.deepcopy(rotation),
        ),
        quant_config=None if quant is None else SimpleNamespace(get_name=lambda: quant),
    )


def _save_basis(tmp_path, tensor=None, key="global_rotation"):
    path = tmp_path / "optional" / "quarot.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: _basis() if tensor is None else tensor}, str(path))
    return path


@pytest.mark.parametrize(
    "is_npu,quant,rotation,engrams",
    [
        (False, "modelslim", ROTATION, True),  # GPU, even with export metadata.
        (True, None, ROTATION, True),  # Native unquantized NPU/A5 path.
        (True, "fp8", ROTATION, True),  # Native A5 FP8 path.
        (True, "mxfp4", ROTATION, True),  # Native A5 MXFP4 path.
        (True, "modelslim", None, True),
        (True, "modelslim", {}, True),
        (True, "modelslim", ROTATION, False),
    ],
)
def test_loader_unaffected_paths_do_not_read_sidecar(
    production, tmp_path, is_npu, quant, rotation, engrams
):
    production["_is_npu"] = is_npu
    wrapper = _model(tmp_path, quant=quant, rotation=rotation, engrams=engrams)
    with (
        patch.object(Path, "is_file", side_effect=AssertionError("unexpected stat")),
        patch("safetensors.safe_open", side_effect=AssertionError("unexpected read")),
    ):
        production["_load_npu_engram_gate_basis"](wrapper)
    assert not hasattr(wrapper.model, "engram_gate_basis")


@pytest.mark.parametrize("field", list(ROTATION))
@pytest.mark.parametrize("invalid", ["missing", "wrong"])
def test_loader_rejects_incomplete_or_incompatible_rotation(
    production, tmp_path, field, invalid
):
    wrapper = _model(tmp_path)
    if invalid == "missing":
        del wrapper.config.engram_rotation_config[field]
    else:
        wrapper.config.engram_rotation_config[field] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported ModelSlim Engram rotation"):
        production["_load_npu_engram_gate_basis"](wrapper)


def test_loader_rejects_non_mapping_rotation(production, tmp_path):
    wrapper = _model(tmp_path, rotation=["quarot_global"])
    with pytest.raises(ValueError, match="rotation"):
        production["_load_npu_engram_gate_basis"](wrapper)


def test_loader_requires_model_path_and_sidecar(production, tmp_path):
    wrapper = _model(tmp_path)
    with pytest.raises(FileNotFoundError, match="quarot.safetensors"):
        production["_load_npu_engram_gate_basis"](wrapper)
    wrapper.config._name_or_path = ""
    with pytest.raises(ValueError, match="checkpoint path"):
        production["_load_npu_engram_gate_basis"](wrapper)


@pytest.mark.parametrize(
    "tensor,key,message",
    [
        (torch.eye(4), "another_key", "Missing global_rotation"),
        (torch.ones(4, 3), "global_rotation", "floating-point"),
        (torch.ones(4, 4, dtype=torch.int32), "global_rotation", "floating-point"),
        (torch.full((4, 4), float("nan")), "global_rotation", "non-finite"),
        (torch.full((4, 4), float("inf")), "global_rotation", "non-finite"),
    ],
)
def test_loader_rejects_invalid_sidecar(production, tmp_path, tensor, key, message):
    _save_basis(tmp_path, tensor, key)
    wrapper = _model(tmp_path)
    with pytest.raises(ValueError, match=message):
        production["_load_npu_engram_gate_basis"](wrapper)
    assert not hasattr(wrapper.model, "engram_gate_basis")


def test_loader_rejects_meta_destination(production, tmp_path):
    _save_basis(tmp_path)
    wrapper = _model(tmp_path)
    wrapper.model.layers[0].engram.to(device="meta")
    with pytest.raises(ValueError, match="Load Engram weights"):
        production["_load_npu_engram_gate_basis"](wrapper)


def test_valid_sidecar_is_shared_nonpersistent_and_loaded_once(production, tmp_path):
    path = _save_basis(tmp_path, _basis().double())
    wrapper = _model(tmp_path)
    load = production["_load_npu_engram_gate_basis"]
    with patch("safetensors.safe_open", wraps=safe_open) as opened:
        load(wrapper)
        basis = wrapper.model.engram_gate_basis
        address = basis.data_ptr()
        load(wrapper)
        opened.assert_called_once_with(str(path), framework="pt", device="cpu")
    assert wrapper.model.engram_gate_basis is basis
    assert basis.data_ptr() == address
    assert basis.dtype == torch.float32
    assert basis.device == wrapper.model.layers[0].engram.q_weight.device
    assert torch.equal(basis, _basis())
    assert dict(wrapper.model.named_buffers()) == {"engram_gate_basis": basis}
    assert "engram_gate_basis" not in wrapper.model.state_dict()
    assert "engram_gate_basis" in wrapper.model._non_persistent_buffers_set
    gate = Mock(wraps=production["engram_gate"])
    production["engram_gate"] = gate
    x, kv, _, _ = _inputs(torch.bfloat16)
    for layer in wrapper.model.layers:
        layer.engram.wkv = Mock(return_value=(kv, None))
        production["forward"](
            layer.engram,
            x,
            torch.zeros(3, 1, dtype=torch.long),
            input_ids=torch.tensor([7, 8, 9]),
            gate_basis=basis,
        )
        assert "engram_gate_basis" not in dict(layer.engram.named_buffers())
    assert gate.call_count == 2
    assert all(call.args[-1] is basis for call in gate.call_args_list)


def test_model_only_supplies_shared_basis_on_npu():
    tree = ast.parse(MODEL.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "engram"
    ]
    assert len(calls) == 1
    keyword = next(item for item in calls[0].keywords if item.arg == "gate_basis")
    assert isinstance(keyword.value, ast.IfExp)
    assert ast.unparse(keyword.value.test) == "_is_npu"
    expr = compile(ast.Expression(keyword.value), str(MODEL), "eval")
    basis = _basis()
    model = SimpleNamespace(engram_gate_basis=basis)
    assert eval(expr, {"_is_npu": True, "self": model}) is basis
    assert eval(expr, {"_is_npu": False, "self": model}) is None


@pytest.mark.parametrize("layout", ["flat", "nested_text", "top_level_with_text"])
def test_config_normalization_preserves_rotation_metadata(layout):
    namespace = {"PretrainedConfig": type("PretrainedConfig", (), {})}
    _extract(CONFIG, ["_config_dict", "normalize_deepseek_v41_config"], namespace)
    text = {"hidden_size": 4, "engram_rotation_config": copy.deepcopy(ROTATION)}
    if layout == "top_level_with_text":
        source = {"text_config": {"hidden_size": 4}, "engram_rotation_config": ROTATION}
    else:
        source = {"text_config": text} if layout == "nested_text" else text
    before = copy.deepcopy(source)
    normalized = namespace["normalize_deepseek_v41_config"](source)
    assert normalized["engram_rotation_config"] == ROTATION
    assert normalized["hidden_size"] == 4
    assert source == before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
