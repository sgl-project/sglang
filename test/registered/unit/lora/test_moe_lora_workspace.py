"""CPU tests for bounded MoE LoRA workspace retention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/lora/workspace.py"
_SPEC = importlib.util.spec_from_file_location("_lora_workspace", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
with patch.dict(sys.modules, {_SPEC.name: _MODULE}):
    _SPEC.loader.exec_module(_MODULE)

LoraWorkspace = _MODULE.LoraWorkspace


def test_eager_buffers_retain_only_largest_capacity_per_semantic_name():
    workspace = LoraWorkspace()
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


def test_graph_buckets_share_one_storage_per_name_apart_from_eager():
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=False)
    eager = workspace.tensor("delta", (8,), dtype=torch.float32, device="cpu")

    # The graph runners capture the largest bucket first; the smaller buckets
    # read a prefix of the same storage, so N buckets cost one buffer.
    workspace.begin_forward(graph_mode=True)
    graph_large = workspace.tensor("delta", (16,), dtype=torch.float32, device="cpu")
    graph_small = workspace.tensor("delta", (4,), dtype=torch.float32, device="cpu")
    assert graph_small.data_ptr() == graph_large.data_ptr()
    assert graph_small.shape == (4,) and graph_large.shape == (16,)
    assert graph_large.data_ptr() != eager.data_ptr()

    # Before any capture the storage may still grow (a larger warm-up).
    grown = workspace.tensor("delta", (32,), dtype=torch.float32, device="cpu")
    assert grown.numel() == 32
    assert (
        workspace.tensor("delta", (4,), dtype=torch.float32, device="cpu").data_ptr()
        == grown.data_ptr()
    )


@pytest.mark.parametrize("is_prefill_graph", [False, True])
def test_graph_storage_growth_retires_the_captured_storage_alive(
    monkeypatch, is_prefill_graph
):
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True, is_prefill_graph=is_prefill_graph)
    captured = workspace.tensor("delta", (16,), dtype=torch.float32, device="cpu")
    captured.fill_(3.0)
    monkeypatch.setattr(LoraWorkspace, "_capturing", staticmethod(lambda device: True))
    # Inside a capture a warmed buffer is served, an unwarmed one refused.
    assert (
        workspace.tensor("delta", (8,), dtype=torch.float32, device="cpu").data_ptr()
        == captured.data_ptr()
    )
    with pytest.raises(RuntimeError, match="not warmed before CUDA capture"):
        workspace.tensor("other", (8,), dtype=torch.float32, device="cpu")
    monkeypatch.setattr(LoraWorkspace, "_capturing", staticmethod(lambda device: False))
    # A later, larger warm-up (a decode plan's wider slab) moves the storage; the
    # graph captured on the old one keeps a live, intact buffer.
    grown = workspace.tensor("delta", (64,), dtype=torch.float32, device="cpu")
    assert grown.data_ptr() != captured.data_ptr()
    assert torch.equal(captured, torch.full((16,), 3.0))
    assert any(r.data_ptr() == captured.data_ptr() for r in workspace._retired)
    assert (
        workspace.tensor("delta", (4,), dtype=torch.float32, device="cpu").data_ptr()
        == grown.data_ptr()
    )


def test_graph_phases_share_bucket_views_but_not_storage():
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
    prefill = workspace.tensor("delta", (2, 8, 4), dtype=torch.float32, device="cpu")
    small = workspace.tensor("delta", (2, 3, 4), dtype=torch.float32, device="cpu")
    prefill_iota = workspace.iota(16, "cpu")
    assert small.data_ptr() == prefill.data_ptr()
    assert small.stride() == (12, 4, 1)
    assert workspace.iota(8, "cpu").data_ptr() == prefill_iota.data_ptr()

    workspace.begin_forward(graph_mode=True)
    decode = workspace.tensor("delta", (2, 3, 4), dtype=torch.float32, device="cpu")
    decode_iota = workspace.iota(8, "cpu")
    assert decode.data_ptr() != prefill.data_ptr()
    assert decode_iota.data_ptr() != prefill_iota.data_ptr()
    assert not workspace._retired
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in (prefill, small, decode)
    }
    assert sum(storages.values()) == (2 * 8 * 4 + 2 * 3 * 4) * 4

    workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
    reused = workspace.tensor("delta", (2, 3, 4), dtype=torch.float32, device="cpu")
    assert reused.data_ptr() == prefill.data_ptr()
    assert workspace.iota(8, "cpu").data_ptr() == prefill_iota.data_ptr()

    workspace.begin_forward(graph_mode=False, is_prefill_graph=True)
    eager = workspace.tensor("delta", (2, 3, 4), dtype=torch.float32, device="cpu")
    workspace.begin_forward(graph_mode=False)
    reused = workspace.tensor("delta", (2, 3, 4), dtype=torch.float32, device="cpu")
    assert reused.data_ptr() == eager.data_ptr()
    assert eager.data_ptr() not in storages


def test_capture_requires_warmup_in_its_graph_phase(monkeypatch):
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
    workspace.tensor("delta", (16,), dtype=torch.float32, device="cpu")
    workspace.iota(16, "cpu")
    workspace.begin_forward(graph_mode=True)
    monkeypatch.setattr(LoraWorkspace, "_capturing", staticmethod(lambda device: True))
    with pytest.raises(RuntimeError, match="not warmed before CUDA capture"):
        workspace.tensor("delta", (8,), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="not warmed before CUDA capture"):
        workspace.iota(8, "cpu")


def test_zero_on_first_allocation_preserves_self_restored_eager_state():
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=False)
    counts = workspace.tensor(
        "route_counts",
        (8,),
        dtype=torch.int32,
        device="cpu",
        zero_on_first_allocation=True,
    )
    assert torch.equal(counts, torch.zeros_like(counts))

    # Repeated lookup must not clear state owned by the device route scan.
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
    workspace = LoraWorkspace()
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


def test_parallel_region_state_is_created_on_first_use_even_during_capture(
    monkeypatch,
):
    """Caller-keyed streams/events may be created during capture.
    The side stream must differ from the caller despite CUDA stream-pool reuse.
    """
    if not torch.cuda.is_available():
        pytest.skip("side streams need CUDA")
    workspace = LoraWorkspace()
    monkeypatch.setattr(workspace, "_capturing", lambda _device: True)
    device = torch.device("cuda:0")
    stream = workspace.side_stream(device)
    assert stream.cuda_stream != torch.cuda.current_stream(device).cuda_stream
    assert workspace.side_stream(device) is stream
    event = workspace.event(device, "missing:ready")
    assert workspace.event(device, "missing:ready") is event


@pytest.mark.parametrize("is_prefill_graph", [False, True])
def test_graph_mode_iota_is_one_shared_map_apart_from_eager_growth(
    monkeypatch, is_prefill_graph
):
    workspace = LoraWorkspace()
    workspace.begin_forward(graph_mode=True, is_prefill_graph=is_prefill_graph)
    largest = workspace.iota(16, "cpu")
    address = largest.data_ptr()
    assert workspace.iota(8, "cpu").data_ptr() == address  # a prefix of the same map

    workspace.begin_forward(graph_mode=False)
    grown = workspace.iota(64, "cpu")
    assert grown.numel() == 64 and int(grown[-1]) == 63

    workspace.begin_forward(graph_mode=True, is_prefill_graph=is_prefill_graph)
    replay = workspace.iota(8, "cpu")
    assert replay.data_ptr() == address
    torch.testing.assert_close(replay, torch.arange(8, dtype=torch.int32))
    monkeypatch.setattr(LoraWorkspace, "_capturing", staticmethod(lambda device: True))
    workspace.iota(16, "cpu")
    with pytest.raises(RuntimeError, match="not warmed before CUDA capture"):
        workspace.iota(32, "cpu")
    monkeypatch.setattr(LoraWorkspace, "_capturing", staticmethod(lambda device: False))
    bigger = workspace.iota(32, "cpu")
    assert bigger.data_ptr() != address and int(bigger[-1]) == 31
    torch.testing.assert_close(
        largest, torch.arange(16, dtype=torch.int32)
    )  # retired, intact


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
