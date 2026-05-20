"""Kernel-level tests for the kv-canary spin-wait jitter kernel."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sglang.jit_kernel.kv_canary.jitter import spin_wait_step
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)


_DEVICE = torch.device("cuda")


def _make_cycles(value: int) -> torch.Tensor:
    cycles = torch.zeros(1, dtype=torch.int64, device=_DEVICE)
    cycles[0] = value
    return cycles


def _measure_wall_seconds(
    *, cycles_value: int, warmup: int = 1, runs: int = 3
) -> float:
    cycles = _make_cycles(cycles_value)
    for _ in range(warmup):
        spin_wait_step(cycles=cycles)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        spin_wait_step(cycles=cycles)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) * 1e-3 / runs


def test_spin_wait_zero_cycles_returns_quickly() -> None:
    cycles = _make_cycles(0)
    spin_wait_step(cycles=cycles)
    torch.cuda.synchronize()

    wall = _measure_wall_seconds(cycles_value=0, runs=10)

    assert wall < 200e-6, f"cycles=0 launch should be <200us, got {wall * 1e6:.1f}us"


def test_spin_wait_monotonic_in_cycles() -> None:
    wall_small = _measure_wall_seconds(cycles_value=10_000)
    wall_large = _measure_wall_seconds(cycles_value=1_000_000)

    assert wall_large > wall_small * 5, (
        f"100x cycles should give >>5x wall: small={wall_small * 1e6:.1f}us "
        f"large={wall_large * 1e6:.1f}us"
    )


def test_spin_wait_kernel_loop_present_in_source() -> None:
    """White-box guard against nvcc dead-store elimination silently turning the body into a no-op.

    The kernel source must contain the ``clock64()`` reads and the ``volatile`` sink line; if a
    refactor removes either, the kernel can be optimised away and the fuzzer becomes a launch
    overhead generator. PTX-level inspection is preferred but expensive; checking the source guards
    against the common regression (someone "simplifies" the loop body).
    """
    source_path = (
        Path(__file__).resolve().parents[1] / "csrc" / "kv_canary" / "jitter.cuh"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "clock64()" in source
    assert "volatile" in source


def test_spin_wait_cuda_graph_replay_picks_up_new_cycles() -> None:
    cycles = _make_cycles(1_000)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        spin_wait_step(cycles=cycles)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        spin_wait_step(cycles=cycles)

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    cycles.copy_(torch.tensor([1_000], dtype=torch.int64, device=_DEVICE))
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    wall_small = float(start.elapsed_time(end)) * 1e-3 / 5

    cycles.copy_(torch.tensor([1_000_000], dtype=torch.int64, device=_DEVICE))
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    wall_large = float(start.elapsed_time(end)) * 1e-3 / 5

    assert wall_large > wall_small * 5, (
        f"graph replay must observe new cycles via copy_(): small={wall_small * 1e6:.1f}us "
        f"large={wall_large * 1e6:.1f}us"
    )


def test_spin_wait_rejects_wrong_dtype() -> None:
    bad = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
    with pytest.raises(ValueError, match="int64"):
        spin_wait_step(cycles=bad)


def test_spin_wait_rejects_wrong_shape() -> None:
    bad = torch.zeros(2, dtype=torch.int64, device=_DEVICE)
    with pytest.raises(ValueError, match=r"shape \[1\]"):
        spin_wait_step(cycles=bad)


def test_spin_wait_rejects_cpu_tensor() -> None:
    bad = torch.zeros(1, dtype=torch.int64, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        spin_wait_step(cycles=bad)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
