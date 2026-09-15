"""Triton do_bench/do_bench_cudagraph compatible wrapper using flashinfer.testing.bench_gpu_time."""

import numpy as np
from flashinfer.testing import bench_gpu_time


def run_bench(
    fn,
    use_cuda_graph: bool = True,
    quantiles=(0.5, 0.2, 0.8),
    warmup_ms: int = 25,
    rep_ms: int = 100,
):
    """Returns (ms, min_ms, max_ms) or (median,) when quantiles=None."""
    times = bench_gpu_time(
        fn=fn,
        use_cuda_graph=use_cuda_graph,
        dry_run_time_ms=warmup_ms,
        repeat_time_ms=rep_ms,
    )
    if quantiles is None:
        return (float(np.median(times)),)
    return tuple(float(np.percentile(times, q * 100)) for q in quantiles)
