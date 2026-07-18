"""Hermetic regression tests for Inkling shared-sink LoRA normalization."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, stage="base-b", runner_config="1-gpu-small")

REPO_ROOT = Path(__file__).resolve().parents[4]
LORA_PATH = REPO_ROOT / "python/sglang/srt/lora/lora.py"


def _load_normalizer_class():
    tree = ast.parse(LORA_PATH.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LoRAAdapter"
    )
    method_names = {"_normalize_shared_expert_moe", "normalize_gate_up_proj"}
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {method.name for method in methods} == method_names
    test_class = ast.ClassDef(
        name="_NormalizerUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    namespace = {"Dict": dict, "re": __import__("re"), "torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[])),
            str(LORA_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[test_class.name]


def _normalizer(num_shared: int = 2, *, text_config: bool = False):
    normalizer = _load_normalizer_class()()
    config = SimpleNamespace(
        architectures=None if text_config else ["InklingForConditionalGeneration"],
        model_type="inkling_text" if text_config else "inkling",
        n_shared_experts=num_shared,
    )
    normalizer.base_hf_config = config
    return normalizer


@pytest.mark.parametrize("text_config", [False, True])
def test_proj_named_shared_sink_factors_gain_the_expert_axis(text_config):
    n, rank, hidden, intermediate = 2, 3, 5, 7
    prefix = "model.layers.0.mlp.shared_experts"
    gate_a = torch.arange(rank * hidden).reshape(rank, hidden)
    gate_b = torch.arange(n * 2 * intermediate * rank).reshape(
        n * 2 * intermediate, rank
    )
    down_a = torch.arange(rank * n * intermediate).reshape(rank, n * intermediate)
    down_b = torch.arange(hidden * rank).reshape(hidden, rank)
    weights = {
        f"{prefix}.gate_up_proj.lora_A.weight": gate_a,
        f"{prefix}.gate_up_proj.lora_B.weight": gate_b,
        f"{prefix}.down_proj.lora_A.weight": down_a,
        f"{prefix}.down_proj.lora_B.weight": down_b,
    }

    normalizer = _normalizer(n, text_config=text_config)
    normalizer._normalize_shared_expert_moe(weights)
    normalizer.normalize_gate_up_proj(list(weights), weights)

    torch.testing.assert_close(
        weights[f"{prefix}.gate_up_proj.lora_A.weight"],
        gate_a.unsqueeze(0).repeat(1, 2, 1),
    )
    torch.testing.assert_close(
        weights[f"{prefix}.gate_up_proj.lora_B.weight"],
        gate_b.reshape(n, 2 * intermediate, rank),
    )
    torch.testing.assert_close(
        weights[f"{prefix}.down_proj.lora_A.weight"],
        down_a.reshape(rank, n, intermediate).transpose(0, 1).contiguous(),
    )
    torch.testing.assert_close(
        weights[f"{prefix}.down_proj.lora_B.weight"], down_b.unsqueeze(0)
    )


def test_named_per_expert_outer_factor_is_not_collapsed_to_shared_outer():
    name = "model.layers.0.mlp.shared_experts.1.gate_up_proj.lora_A.weight"
    weight = torch.arange(15).reshape(3, 5)
    weights = {name: weight}

    _normalizer()._normalize_shared_expert_moe(weights)

    torch.testing.assert_close(weights[name], weight)
    assert weights[name].dim() == 2
