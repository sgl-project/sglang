"""GPU kernel benchmarking utility.

Provides ``run_bench``, a unified wrapper that prefers
``flashinfer.testing.bench_gpu_time`` when available and falls back to
``triton.testing.do_bench`` / ``do_bench_cudagraph`` on platforms where
FlashInfer is not installed (e.g. AMD ROCm).
"""

import numpy as np

try:
    from flashinfer.testing import bench_gpu_time

    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False

try:
    from triton.testing import do_bench, do_bench_cudagraph

    _HAS_TRITON_BENCH = True
except ImportError:
    _HAS_TRITON_BENCH = False


def run_bench(
    fn,
    use_cuda_graph: bool = True,
    quantiles=(0.5, 0.2, 0.8),
    warmup_ms: int = 25,
    rep_ms: int = 100,
):
    """Benchmark *fn* and return timing quantiles in milliseconds.

    Returns ``(median_ms, p20_ms, p80_ms)`` by default, or a single-element
    tuple ``(median_ms,)`` when *quantiles* is ``None``.

    The implementation prefers FlashInfer's ``bench_gpu_time`` for accuracy,
    falling back to Triton's ``do_bench`` / ``do_bench_cudagraph`` when
    FlashInfer is unavailable (common on AMD ROCm).
    """
    if _HAS_FLASHINFER:
        times = bench_gpu_time(
            fn=fn,
            use_cuda_graph=use_cuda_graph,
            dry_run_time_ms=warmup_ms,
            repeat_time_ms=rep_ms,
        )
        if quantiles is None:
            return (float(np.median(times)),)
        return tuple(float(np.percentile(times, q * 100)) for q in quantiles)

    if _HAS_TRITON_BENCH:
        if use_cuda_graph:
            # do_bench_cudagraph(fn, rep, grad_to_none, quantiles, return_mode)
            kwargs = dict(rep=rep_ms)
        else:
            # do_bench(fn, warmup, rep, grad_to_none, quantiles, return_mode)
            kwargs = dict(warmup=warmup_ms, rep=rep_ms)

        bench_fn = do_bench_cudagraph if use_cuda_graph else do_bench
        if quantiles is not None:
            kwargs["quantiles"] = list(quantiles)
        result = bench_fn(fn, **kwargs)
        if quantiles is None:
            # do_bench without quantiles returns a single float
            return (float(result),)
        return tuple(float(v) for v in result)

    raise RuntimeError(
        "No GPU benchmarking backend available. "
        "Install flashinfer (`pip install flashinfer`) or triton (`pip install triton`)."
    )
