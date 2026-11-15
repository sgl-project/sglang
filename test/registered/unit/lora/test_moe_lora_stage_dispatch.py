"""The token-dense LoRA-A family reaches its executor and writes its bridge.

Only the SM100 shared-outer NVFP4 decode plan selects it, so this real CPU
matmul is its only executor oracle; the other families run on GPU through the
runner with real routes.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.plan import AFamily, BridgeLayout, Site
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/lora/moe"


def _method(path, name, **scope):
    """Execute the production host method without importing CUDA extensions."""
    tree = ast.parse((SOURCE / path).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, method],
        type_ignores=[],
    )
    exec(compile(module, str(SOURCE / path), "exec"), scope)
    return scope[name]


def test_a_family_writes_its_bridge():
    # Regression: NVFP4 shared plans admitted token_dense before it had an executor.
    spec = SimpleNamespace(
        family=AFamily.TOKEN_DENSE,
        site=Site.GATE_UP,
        is_shared_outer=True,
        output_layout=BridgeLayout.TOKEN_MAJOR,
    )
    owner = SimpleNamespace(
        workspace=SimpleNamespace(
            tensor=lambda _, shape, **kw: torch.empty(shape, **kw)
        ),
        lora_delta_dtype=torch.float32,
    )
    routes = SimpleNamespace(
        raw=lambda _: SimpleNamespace(num_tokens=5, num_rows=10, max_loras=3)
    )
    run = _method(
        "runner.py",
        "_run_a",
        torch=torch,
        AFamily=AFamily,
        BridgeLayout=BridgeLayout,
        Site=Site,
    )
    x = torch.arange(5 * 16, dtype=torch.float32).reshape(5, 16) / 80
    weight = torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16) / 384
    result = run(
        owner, SimpleNamespace(for_a=lambda _: {}), spec, x, weight, routes, "a"
    )
    # One bridge plane per resident slot, dense over every token.
    torch.testing.assert_close(result, torch.stack([x @ slot.T for slot in weight]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
