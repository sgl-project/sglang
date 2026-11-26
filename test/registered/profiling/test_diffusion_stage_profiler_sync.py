"""SGLANG_DIFFUSION_SYNC_STAGE_PROFILING must drain the GPU queue at the
timing start of *stage* records too — otherwise a stage that only launches
kernels (DenoisingStage's tail) leaks its queued work into whichever later
stage blocks first, inflating e.g. DecodingStage readings 2-3x."""

import sys
import time

import pytest
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    RequestMetrics,
    StageProfiler,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage_entry_sync_excludes_previous_stage_tail(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "1")
    logger = init_logger(__name__)
    metrics = RequestMetrics("stage-sync-test")

    # warm the consumer kernels before measuring stage synchronization
    torch.ones(8, device="cuda").sum().cpu()

    # Calibrate ~0.5 s of queued GPU work. Warm up first and time a long
    # window: a ~4 ms window was dominated by launch/sync overhead and clock
    # ramp-up, so it underestimated the cycle rate and queued far less work.
    torch.cuda._sleep(10_000_000)
    torch.cuda.synchronize()
    calib_cycles = 200_000_000
    t0 = time.perf_counter()
    torch.cuda._sleep(calib_cycles)
    torch.cuda.synchronize()
    cycles = int(calib_cycles / max(time.perf_counter() - t0, 1e-9) * 0.5)

    # Producer stage queues work without awaiting it (a denoise tail). Time the
    # kernel itself with events so the checks don't depend on the calibration.
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with StageProfiler("producer", logger, metrics, perf_dump_path_provided=True):
        start.record()
        torch.cuda._sleep(cycles)
        end.record()
    # Consumer stage's first blocking op used to absorb the producer's tail.
    with StageProfiler("consumer", logger, metrics, perf_dump_path_provided=True):
        torch.ones(8, device="cuda").sum().cpu()

    kernel_ms = start.elapsed_time(end)
    producer_ms, consumer_ms = metrics.stages["producer"], metrics.stages["consumer"]
    timings = f"{metrics.stages}, producer kernel {kernel_ms:.1f} ms"
    assert kernel_ms > 50, f"too little queued work to measure: {timings}"
    assert producer_ms > 0.8 * kernel_ms, (
        f"queued work not attributed to producer: {timings}"
    )
    assert consumer_ms < 0.2 * kernel_ms, (
        f"producer tail leaked into consumer: {timings}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
