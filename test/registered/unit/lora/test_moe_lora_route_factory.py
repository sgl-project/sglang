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


@pytest.mark.parametrize(
    "tokens,width,buckets,capacity,view,expected",
    [
        (512, 1, 5, None, "aligned", "small"),
        (513, 1, 5, None, "aligned", "jit"),
        (768, 1, 4, None, "aligned", "small"),
        (769, 1, 4, None, "aligned", "jit"),
        (16384, 1, 2, None, "aligned", "large"),
        (1024, 16, 2, None, "aligned", "large"),
        (1, 1, 8192, None, "aligned", "large"),
        (1, 1, 2, 0, "aligned", "large"),
        (1, 1, 2, 64, "aligned", "large"),
        (16384, 1, 8192, 65536, "raw", None),
    ],
)
def test_route_dispatch(tokens, width, buckets, capacity, view, expected):
    slots = torch.zeros(tokens, dtype=torch.int32)
    groups = torch.zeros((tokens, width), dtype=torch.int32) if width > 1 else None
    with (
        patch.object(ROUTING, "_build_small_route") as small,
        patch.object(ROUTING, "_build_large_route") as large,
        patch.object(ROUTING, "_build_route_bucket_ids"),
        patch.object(
            ROUTING, "_align_block_size_jit", return_value=(None, None, None)
        ) as jit,
    ):
        route = ROUTING.build_route(
            slots,
            group_ids=groups,
            max_loras=buckets,
            block_size=16,
            capacity=capacity,
            view=view,
        )
    for name, builder in (("small", small), ("large", large), ("jit", jit)):
        assert builder.call_count == int(expected == name)
    if expected == "large":
        assert large.call_args.kwargs["capacity"] == capacity
    if view == "raw":
        assert route.token_slots is slots and route.group_ids is groups
        assert route.num_rows == tokens * width
        assert route.maybe_sorted_pair_ids is None


@pytest.mark.parametrize("cached", [False, True])
def test_large_route_scratch_is_fresh_or_workspace_backed(cached):
    workspace = Workspace() if cached else None

    def allocate():
        return ROUTING._aligned_route_scratch(
            workspace,
            prefix="route",
            num_buckets=5,
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


def test_eager_scratch_reuses_capacity_without_reinitializing():
    workspace = Workspace()

    def counts(n):
        return workspace.tensor(
            "counts",
            (n,),
            dtype=torch.int32,
            device="cpu",
            zero_on_first_allocation=True,
        )

    first = counts(8)
    assert not first.any()
    first.fill_(7)
    smaller = counts(4)
    assert smaller.data_ptr() == first.data_ptr()
    assert smaller.tolist() == [7] * 4
    grown = counts(16)
    assert grown.data_ptr() != first.data_ptr()
    assert not grown.any()
    assert counts(8).data_ptr() == grown.data_ptr()


def test_graph_phases_preserve_warmed_storage_and_retire_old_addresses(monkeypatch):
    workspace = Workspace()

    def tensor(n):
        return workspace.tensor("delta", (n, 4), dtype=torch.float32, device="cpu")

    eager = tensor(8)
    workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
    prefill = tensor(8)
    prefill.fill_(3)
    assert tensor(2).data_ptr() == prefill.data_ptr()
    workspace.begin_forward(graph_mode=True)
    decode = tensor(4)
    assert len({eager.data_ptr(), prefill.data_ptr(), decode.data_ptr()}) == 3

    monkeypatch.setattr(Workspace, "_capturing", staticmethod(lambda _: True))
    assert tensor(2).data_ptr() == decode.data_ptr()
    with pytest.raises(RuntimeError, match="not warmed"):
        tensor(8)
    workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
    assert tensor(8).data_ptr() == prefill.data_ptr()
    monkeypatch.setattr(Workspace, "_capturing", staticmethod(lambda _: False))
    grown = tensor(16)
    assert grown.data_ptr() != prefill.data_ptr()
    assert any(old.data_ptr() == prefill.data_ptr() for old in workspace._retired)
    assert torch.equal(prefill, torch.full_like(prefill, 3))
    assert tensor(2).data_ptr() == grown.data_ptr()


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


def test_cpu_parallel_region_is_joined_before_return():
    workspace = Workspace()
    order = []

    def compute():
        order.append("compute")
        return "result"

    assert (
        workspace.run_parallel(
            name="region",
            device=torch.device("cpu"),
            side=lambda: order.append("side"),
            compute=compute,
        )
        == "result"
    )
    assert order == ["side", "compute"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
