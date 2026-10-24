"""CPU tensor-layout regressions without importing GPU-only LoRA backends."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def _load_definitions(path, names, namespace, class_name=None):
    """Execute the production helpers while avoiding CUDA import side effects."""
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(
            node for node in tree.body if getattr(node, "name", None) == class_name
        )
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    body += [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        )
    ]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def helpers():
    namespace = {"torch": torch, "re": re, "logger": logging.getLogger(__name__)}
    _load_definitions(SRT / "layers/utils/common.py", {"get_layer_id"}, namespace)
    _load_definitions(SRT / "lora/lora.py", {"_ROUTED_EXPERT_PATTERN"}, namespace)
    _load_definitions(
        SRT / "lora/lora.py",
        {"normalize_gate_up_proj", "_is_non_gated_moe_weight"},
        namespace,
        "LoRAAdapter",
    )
    _load_definitions(
        SRT / "lora/utils.py",
        {"get_normalized_target_modules", "EMBEDDING_NAMES", "DSA_INDEXER_LORA_NAMES"},
        namespace,
    )
    _load_definitions(
        SRT / "lora/lora_manager.py", {"init_lora_shapes"}, namespace, "LoRAManager"
    )
    return SimpleNamespace(**namespace)


@pytest.fixture
def adapter(helpers):
    adapter_type = type(
        "Adapter",
        (),
        {
            "normalize_gate_up_proj": helpers.normalize_gate_up_proj,
            "_is_non_gated_moe_weight": helpers._is_non_gated_moe_weight,
        },
    )
    instance = adapter_type()
    instance._moe_is_gated_by_layer = {0: False, 1: True}
    return instance


@pytest.mark.parametrize("expert", ["experts.0", "experts"])
@pytest.mark.parametrize("factor, shape", [("A", (2, 4)), ("B", (6, 2))])
def test_ungated_up_proj_keeps_single_projection_shape(adapter, expert, factor, shape):
    if expert == "experts":
        shape = (3, *shape)
    weight = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
    name = f"model.layers.0.mlp.{expert}.up_proj.lora_{factor}.weight"
    weights = {name: weight}

    adapter.normalize_gate_up_proj(list(weights), weights)

    merged = name.replace(".up_proj.", ".gate_up_proj.")
    assert set(weights) == {merged}
    assert weights[merged] is weight


@pytest.mark.parametrize(
    "prefix", ["mlp", "mlp.shared_experts", "mlp.shared_experts.0"]
)
def test_dense_and_shared_up_proj_keep_standalone_names(adapter, prefix):
    name = f"model.layers.0.{prefix}.up_proj.lora_A.weight"
    weight = torch.ones(2, 4)
    weights = {name: weight}
    adapter.normalize_gate_up_proj(list(weights), weights)
    assert set(weights) == {name}
    assert weights[name] is weight


@pytest.mark.parametrize("up_first", [False, True])
@pytest.mark.parametrize("empty_up", [False, True])
def test_existing_ungated_gate_proj_exports(adapter, up_first, empty_up):
    gate = "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight"
    up = gate.replace("gate_proj", "up_proj")
    weight = torch.ones(2, 4)
    weights = {gate: weight}
    if empty_up:
        weights[up] = torch.empty(0)
    names = list(weights)
    if up_first:
        names.reverse()

    adapter.normalize_gate_up_proj(names, weights)

    merged = gate.replace("gate_proj", "gate_up_proj")
    assert set(weights) == {merged}
    assert weights[merged] is weight


@pytest.mark.parametrize("up_first", [False, True])
def test_mixed_gated_and_ungated_layers(adapter, up_first):
    ungated = "model.layers.0.mlp.experts.0.up_proj.lora_A.weight"
    gate = "model.layers.1.mlp.experts.0.gate_proj.lora_A.weight"
    up = gate.replace("gate_proj", "up_proj")
    weights = {
        ungated: torch.ones(2, 4),
        gate: torch.ones(2, 4),
        up: torch.full((2, 4), 2.0),
    }
    names = list(weights)
    if up_first:
        names.reverse()

    adapter.normalize_gate_up_proj(names, weights)

    assert weights[ungated.replace("up_proj", "gate_up_proj")].shape == (2, 4)
    torch.testing.assert_close(
        weights[gate.replace("gate_proj", "gate_up_proj")],
        torch.cat((torch.ones(2, 4), torch.full((2, 4), 2.0))),
    )


@pytest.mark.parametrize("explicit_targets", [False, True])
@pytest.mark.parametrize("standalone_up", [False, True])
@pytest.mark.parametrize("targets", [["up_proj", "down_proj"], ["q_proj", "v_proj"]])
def test_target_inference_preserves_standalone_up_proj(
    helpers, explicit_targets, standalone_up, targets
):
    module_names = ["model.layers.0.mlp.experts"]
    if standalone_up:
        module_names.append("model.layers.0.mlp.shared_experts.up_proj")
    manager = SimpleNamespace(
        base_model=SimpleNamespace(
            named_modules=lambda: [(name, None) for name in module_names]
        ),
        configs={"adapter": SimpleNamespace(target_modules=targets, r=2)},
        lora_added_tokens_size=0,
    )
    helpers.init_lora_shapes(
        manager, target_modules=set(targets) if explicit_targets else None
    )
    expected = {"gate_up_proj", "down_proj"} if "up_proj" in targets else {"qkv_proj"}
    if standalone_up and "up_proj" in targets:
        expected.add("up_proj")
    assert manager.target_modules == expected
