"""Every admitted A/B family reaches an executor and writes its destination.

The token-dense path runs a real CPU matmul. Other families replace only the
GPU kernel boundary; their numerical/routing oracles live in kernel tests.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.plan import AFamily, BFamily, BridgeLayout, Site
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/lora/moe"


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


def _routes():
    raw = SimpleNamespace(num_tokens=5, num_rows=10, max_loras=3)
    aligned = SimpleNamespace(num_tokens=5, num_rows=10, max_loras=3)
    shared = SimpleNamespace(num_tokens=5, num_rows=5, max_loras=3)
    return SimpleNamespace(
        raw=lambda _: raw, aligned=lambda _: aligned, shared_token=shared
    )


@pytest.mark.parametrize("family", list(AFamily))
def test_a_family_writes_its_bridge(family):
    routes = _routes()
    token_major = family in (AFamily.TOKEN_GROUPED, AFamily.TOKEN_DENSE)
    spec = SimpleNamespace(
        family=family,
        site=Site.GATE_UP,
        is_shared_outer=token_major,
        output_layout=BridgeLayout.TOKEN_MAJOR
        if token_major
        else BridgeLayout.PAIR_MAJOR,
    )
    route_for_a = _method("runner.py", "_route_for_a", AFamily=AFamily)
    owner = SimpleNamespace(
        _route_for_a=route_for_a,
        workspace=SimpleNamespace(
            tensor=lambda _, shape, **kw: torch.empty(shape, **kw)
        ),
        lora_delta_dtype=torch.float32,
    )

    def launch(value, expected_route):
        def kernel(input, weight, output, route, **kwargs):
            assert route is expected_route
            output.fill_(value)

        return kernel

    run = _method(
        "runner.py",
        "_run_a",
        torch=torch,
        AFamily=AFamily,
        BridgeLayout=BridgeLayout,
        Site=Site,
        grouped_lora_a=launch(
            11, routes.shared_token if token_major else routes.aligned(False)
        ),
        per_row_lora_a=launch(23, routes.raw(False)),
    )
    x = torch.arange(5 * 16, dtype=torch.float32).reshape(5, 16) / 80
    weight = torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16) / 384
    result = run(
        owner, SimpleNamespace(for_a=lambda _: {}), spec, x, weight, routes, "a"
    )
    if family is AFamily.TOKEN_DENSE:
        # Regression: NVFP4 shared plans admitted token_dense before it had an executor.
        expected = torch.stack([x @ slot.T for slot in weight])
    else:
        expected = torch.full(
            (5 if token_major else 10, 8), 23.0 if family is AFamily.PER_ROW else 11.0
        )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("family", list(BFamily))
def test_b_family_writes_the_destination(family):
    routes = _routes()
    expected_route = (
        routes.raw(False) if family is BFamily.PER_ROW else routes.aligned(False)
    )

    def launch(value):
        def kernel(bridge, weight, destination, routing, **kwargs):
            assert routing is expected_route
            assert kwargs["destination_offsets"] == (0, 3)
            assert kwargs["pair_bridge"]
            destination.fill_(value)

        return kernel

    dispatch = _method(
        "kernels/lora_b.py",
        "run_lora_b",
        grouped_lora_b=launch(11),
        _per_row_lora_b=launch(23),
    )
    run = _method(
        "runner.py", "_run_b", Site=Site, BridgeLayout=BridgeLayout, run_lora_b=dispatch
    )
    owner = SimpleNamespace(
        _route_for_b=_method("runner.py", "_route_for_b", BFamily=BFamily),
        gate_up_slices=2,
    )
    spec = SimpleNamespace(
        family=family,
        site=Site.GATE_UP,
        is_shared_outer=False,
        input_layout=BridgeLayout.PAIR_MAJOR,
    )
    output = torch.full((10, 6), float("nan"))
    run(
        owner,
        SimpleNamespace(for_b=lambda _: {}),
        spec,
        torch.zeros(10, 8),
        torch.zeros(3, 6, 8),
        output,
        routes,
    )
    torch.testing.assert_close(
        output, torch.full_like(output, 23 if family is BFamily.PER_ROW else 11)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
