"""CPU routing/workspace contracts; CUDA tests check route contents and replay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_PYTHON = Path(__file__).resolve().parents[4] / "python"


def _load(name, source):
    spec = importlib.util.spec_from_file_location(name, _PYTHON / source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


_VIEW = _load("_route_view", "sglang/kernels/ops/lora/common/route_view.py")
_TRITON = SimpleNamespace(
    jit=lambda function: function,
    cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
    language=SimpleNamespace(),
)
with patch.dict(
    sys.modules,
    {
        "triton": _TRITON,
        "triton.language": _TRITON.language,
        "sglang.kernels.ops.lora.common.route_view": _VIEW,
        "sglang.kernels.ops.lora.moe.virtual_experts": SimpleNamespace(
            _align_block_size_jit=None
        ),
    },
):
    ROUTING = _load("_route_factory", "sglang/kernels/ops/lora/common/routing.py")
Workspace = _load("_route_workspace", "sglang/srt/lora/workspace.py").LoraWorkspace


@pytest.mark.parametrize("cached", [False, True])
def test_large_route_scratch_is_fresh_or_workspace_backed(cached):
    workspace = Workspace() if cached else None

    def allocate(num_buckets=5):
        return ROUTING._aligned_route_scratch(
            workspace,
            prefix="route",
            num_buckets=num_buckets,
            capacity=64,
            block_size=16,
            device=torch.device("cpu"),
        )

    first = allocate()
    assert first["sorted"].shape == (64,)
    assert first["block_bucket_ids"].shape == (4,)
    assert torch.equal(first["counts"], torch.zeros(5, dtype=torch.int32))
    first["counts"].fill_(7)
    second = allocate()
    assert (first["sorted"].data_ptr() == second["sorted"].data_ptr()) == cached
    assert torch.equal(second["counts"], torch.full((5,), 7 if cached else 0))
    grown = allocate(num_buckets=8)["counts"]
    assert grown.data_ptr() != first["counts"].data_ptr()
    assert not grown.any()


def test_stream_keys_separate_scratch_and_routes(monkeypatch):
    workspace = Workspace()
    device, stream = torch.device("cuda:0"), 11
    mapping = SimpleNamespace(
        device=device,
        is_cuda=True,
        dtype=torch.int32,
        data_ptr=lambda: 1234,
        numel=lambda: 8,
        stride=lambda: (1,),
    )
    monkeypatch.setattr(workspace, "_caller", lambda _: stream)
    monkeypatch.setattr(workspace, "_capturing", lambda _: False)
    empty = torch.empty
    # CPU storage stands in for CUDA storage: this tests keys, not synchronization.
    monkeypatch.setattr(
        torch, "empty", lambda shape, *, dtype, device: empty(shape, dtype=dtype)
    )
    build = Mock(side_effect=object)

    def lookup():
        return workspace.route(mapping, ("sorted", 16), build)

    first = workspace.tensor("rank", (8,), dtype=torch.float32, device=device)
    first_route = lookup()
    assert lookup() is first_route
    stream = 22
    second = workspace.tensor("rank", (8,), dtype=torch.float32, device=device)
    assert second.data_ptr() != first.data_ptr()
    assert lookup() is not first_route
    stream = 11
    assert lookup() is first_route
    assert (
        workspace.tensor("rank", (4,), dtype=torch.float32, device=device).data_ptr()
        == first.data_ptr()
    )
    workspace.begin_forward(graph_mode=True)
    decode = lookup()
    workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
    prefill = lookup()
    assert prefill is not decode
    workspace.routes.clear()
    assert lookup() is not prefill
    assert build.call_count == 5


def test_route_cache_keys_match_mapping_and_options():
    workspace = Workspace()
    slots = torch.arange(32, dtype=torch.int32) % 4
    key, build = ("sorted", 16, 4), Mock(side_effect=object)
    first = workspace.route(slots, key, build)
    assert workspace.route(slots[:], key, build) is first
    build.assert_called_once_with()
    for mapping, options in (
        (slots.clone(), key),
        (slots[:16], key),
        (slots[1:17], key),
        (slots[::2], key),
        (slots.view(torch.float32), key),
        (slots, ("sorted", 32, 4)),
        (slots, ("sorted", 16, 8)),
    ):
        build.reset_mock()
        route = workspace.route(mapping, options, build)
        assert workspace.route(mapping, options, build) is route
        build.assert_called_once_with()


def test_moe_route_bundle_passes_each_route_its_view_groups_and_prefix(monkeypatch):
    from sglang.srt.lora.moe import routing as moe_routing
    from sglang.srt.lora.moe.plan import RouteBuilderFamily, RouteRequirement

    slots = torch.tensor([0, 1, -1], dtype=torch.int32)
    topk = torch.tensor([[0, 2], [1, 0], [-1, 2]], dtype=torch.int32)
    built = {}

    def build(mapping, **kwargs):
        route = object()
        built[route] = mapping, kwargs
        return route

    monkeypatch.setattr(moe_routing, "build_route", build)
    plan = SimpleNamespace(
        route_requirements=lambda: set(RouteRequirement),
        route_builder=RouteBuilderFamily.STANDARD,
    )
    routes = moe_routing.build_moe_routes(
        plan,
        topk_ids=topk,
        token_lora_mapping=slots,
        num_local_experts=3,
        max_loras=2,
        block_size=16,
        workspace=Workspace(),
    )
    for requirement in RouteRequirement:
        name = requirement.name
        field = name.lower().removesuffix("_plan")
        _, args = built[getattr(routes, field)]
        assert args["view"].value == ("raw" if name.startswith("RAW_") else "aligned")
        assert args.get("group_ids") is (None if name == "SHARED_TOKEN_PLAN" else topk)
        assert args.get("groups_per_slot", 1) == (
            3 if name.endswith("PER_EXPERT") else 1
        )
        if name.startswith("ALIGNED_"):
            assert args["tensor_prefix"] == f"route:{field}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
