"""CPU tests for bounded MoE LoRA workspace retention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

_SOURCE = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/lora/moe/workspace.py"
)
_SPEC = importlib.util.spec_from_file_location("_moe_lora_workspace", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MoeLoraWorkspace = _MODULE.MoeLoraWorkspace


def test_eager_buffers_retain_only_largest_capacity_per_semantic_name():
    workspace = MoeLoraWorkspace()
    workspace.begin_forward(graph_mode=False)

    first = workspace.tensor(
        "rank",
        (8, 4),
        dtype=torch.bfloat16,
        device="cpu",
    )
    smaller = workspace.tensor(
        "rank",
        (2, 4),
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert first.untyped_storage().data_ptr() == smaller.untyped_storage().data_ptr()
    assert first.untyped_storage().nbytes() == 8 * 4 * torch.bfloat16.itemsize

    larger = workspace.tensor(
        "rank",
        (16, 4),
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert larger.numel() == 64
    assert larger.untyped_storage().nbytes() == 16 * 4 * torch.bfloat16.itemsize


def test_graph_buckets_keep_exact_shapes_separate_from_eager_capacity():
    workspace = MoeLoraWorkspace()
    workspace.begin_forward(graph_mode=False)
    workspace.tensor(
        "delta",
        (8,),
        dtype=torch.float32,
        device="cpu",
    )

    workspace.begin_forward(graph_mode=True)
    graph_small = workspace.tensor(
        "delta",
        (4,),
        dtype=torch.float32,
        device="cpu",
    )
    graph_small_again = workspace.tensor(
        "delta",
        (4,),
        dtype=torch.float32,
        device="cpu",
    )
    graph_large = workspace.tensor(
        "delta",
        (16,),
        dtype=torch.float32,
        device="cpu",
    )

    assert graph_small.data_ptr() == graph_small_again.data_ptr()
    assert graph_small.data_ptr() != graph_large.data_ptr()


def test_zero_on_first_allocation_preserves_self_restored_eager_state():
    workspace = MoeLoraWorkspace()
    workspace.begin_forward(graph_mode=False)
    counts = workspace.tensor(
        "route_counts",
        (8,),
        dtype=torch.int32,
        device="cpu",
        zero_on_first_allocation=True,
    )
    assert torch.equal(counts, torch.zeros_like(counts))

    # The route scan restores this invariant on device. A repeated workspace
    # lookup must not enqueue another memset or overwrite that kernel-owned
    # state.
    counts.fill_(7)
    reused = workspace.tensor(
        "route_counts",
        (4,),
        dtype=torch.int32,
        device="cpu",
        zero_on_first_allocation=True,
    )
    assert reused.data_ptr() == counts.data_ptr()
    assert torch.equal(reused, torch.full_like(reused, 7))

    # Growing eager capacity allocates new storage, which does need its one
    # initialization before the first histogram launch.
    grown = workspace.tensor(
        "route_counts",
        (16,),
        dtype=torch.int32,
        device="cpu",
        zero_on_first_allocation=True,
    )
    assert torch.equal(grown, torch.zeros_like(grown))


def test_run_parallel_cpu_preserves_dependency_order_and_compute_result():
    workspace = MoeLoraWorkspace()
    order = []

    def side():
        order.append("side")

    def compute():
        order.append("compute")
        return "result"

    result = workspace.run_parallel(
        name="cpu_order",
        device=torch.device("cpu"),
        compute=compute,
        side=side,
    )

    assert result == "result"
    assert order == ["side", "compute"]


def test_parallel_region_state_fails_closed_when_first_used_during_capture(
    monkeypatch,
):
    workspace = MoeLoraWorkspace()
    monkeypatch.setattr(workspace, "_capturing", lambda _device: True)

    with pytest.raises(
        RuntimeError,
        match="side stream was not created before CUDA capture",
    ):
        workspace.side_stream("cuda:0")

    with pytest.raises(
        RuntimeError,
        match="event was not created before CUDA capture: missing:ready",
    ):
        workspace.event("cuda:0", "missing:ready")


def test_graph_mode_iota_keeps_its_address_when_eager_iota_grows():
    workspace = MoeLoraWorkspace()
    workspace.begin_forward(graph_mode=True)
    captured = workspace.iota(8, "cpu")
    address = captured.data_ptr()

    workspace.begin_forward(graph_mode=False)
    grown = workspace.iota(64, "cpu")
    assert grown.numel() == 64 and int(grown[-1]) == 63

    workspace.begin_forward(graph_mode=True)
    replay = workspace.iota(8, "cpu")
    assert replay.data_ptr() == address
    torch.testing.assert_close(replay, torch.arange(8, dtype=torch.int32))
    assert workspace.iota(16, "cpu").data_ptr() != address


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
