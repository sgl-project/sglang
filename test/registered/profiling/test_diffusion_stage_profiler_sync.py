"""SGLANG_DIFFUSION_SYNC_STAGE_PROFILING must drain the GPU queue at the
timing start of *stage* records too — otherwise a stage that only launches
kernels (DenoisingStage's tail) leaks its queued work into whichever later
stage blocks first, inflating e.g. DecodingStage readings 2-3x."""

import gzip
import json
import sys
import time

import pytest
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    RequestMetrics,
    StageProfiler,
)
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("requested", [1, 3, -1])
def test_profiler_window_contains_requested_steps(monkeypatch, tmp_path, requested):
    """The trace contains N post-warmup steps, or every step for -1."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    profiler = SGLDiffusionProfiler(
        request_id="window",
        num_steps=requested,
        num_inference_steps=6,
        log_dir=str(tmp_path),
    )
    try:
        for step in range(6):
            with torch.profiler.record_function(f"denoise_step_{step}"):
                torch.ones(8).add_(1)
            profiler.step_denoising_step()
    finally:
        profiler.stop()
    path = next(tmp_path.glob("*.trace.json.gz"))
    with gzip.open(path, "rt") as file:
        events = json.load(file)["traceEvents"]
    actual = [
        e["name"]
        for e in events
        if e.get("ph") == "X" and e.get("name", "").startswith("denoise_step_")
    ]
    indices = range(6) if requested == -1 else range(1, requested + 1)
    assert actual == [f"denoise_step_{step}" for step in indices]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage_entry_sync_excludes_previous_stage_tail(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "1")
    logger = init_logger(__name__)
    metrics = RequestMetrics("stage-sync-test")

    # warm the consumer kernels before measuring stage synchronization
    torch.ones(8, device="cuda").sum().cpu()

    # Calibrate ~0.5 s of queued GPU work.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda._sleep(10_000_000)
    torch.cuda.synchronize()
    cycles = int(10_000_000 / max(time.perf_counter() - t0, 1e-9) * 0.5)

    # Producer stage queues work without awaiting it (a denoise tail).
    with StageProfiler("producer", logger, metrics, perf_dump_path_provided=True):
        torch.cuda._sleep(cycles)
    # Consumer stage's first blocking op used to absorb the producer's tail.
    with StageProfiler("consumer", logger, metrics, perf_dump_path_provided=True):
        torch.ones(8, device="cuda").sum().cpu()

    producer_ms, consumer_ms = metrics.stages["producer"], metrics.stages["consumer"]
    assert producer_ms > 250, (
        f"queued work not attributed to producer: {metrics.stages}"
    )
    assert consumer_ms < 100, f"producer tail leaked into consumer: {metrics.stages}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
