"""Common utilities for jit_kernel benchmark files."""

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import triton.testing

from sglang.jit_kernel.mp import multigpu_launch
from sglang.utils import is_in_ci


def multigpu_bench_main(
    name: str,
    file: str,
    num_gpus: Sequence[int],
    main_fn: Callable[[], None],
    *,
    pre_launch_fn: Optional[Callable[[List[int]], None]] = None,
    timeout: Optional[int] = None,
) -> None:
    """cudalib-style multi-GPU benchmark entry point.

    Drop this at the bottom of a benchmark file::

        multigpu_bench_main(
            name=__name__,
            file=__file__,
            num_gpus=range(2, 9),
            main_fn=benchmark.run,
        )

    Mirrors :func:`multigpu_pytest_main` but invokes a caller-supplied function
    instead of pytest. ``main_fn`` is expected to return ``None`` on success;
    any exception propagates as a non-zero exit. Pass ``--num-gpu 2,4`` on the
    command line to override ``num_gpus``.

    ``pre_launch_fn`` (kw-only) runs once in the outer process before any
    torchrun child starts, receiving the runnable world sizes. Use it for
    parallel JIT precompilation so torchrun children hit a warm disk cache.

    ``timeout`` (kw-only, seconds) bounds each per-world-size torchrun
    invocation. Defaults to ``None`` (wait indefinitely) since benchmark sweeps
    can legitimately run long; set it to fail fast on a hung worker.
    """

    def inner() -> int:
        main_fn()
        return 0

    return multigpu_launch(
        name,
        file,
        num_gpus,
        env_key="_IS_BENCH_MULTIGPU_SGLANG_JIT_KERNEL",
        inner=inner,
        kind="benchmark",
        pre_launch_fn=pre_launch_fn,
        timeout=timeout,
    )


# Common constants
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_DEVICE = "cuda"
DEFAULT_QUANTILES = [0.5, 0.2, 0.8]


def create_empty(*shape: int, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE):
    return torch.empty(shape, dtype=dtype, device=device)


def create_random(*shape: int, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE):
    return torch.randn(shape, dtype=dtype, device=device)


def get_benchmark_range(full_range: List, ci_range: List) -> List:
    """Return appropriate benchmark range based on CI environment."""
    return ci_range if is_in_ci() else full_range


def run_benchmark(
    fn: Callable,
    quantiles: Sequence[float] = (),
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Execute benchmark using CUDA graph and return times in microseconds.

    Args:
        fn: Function to benchmark
        quantiles: Quantiles for timing measurements [median, min, max]
        scale: Scale the result down (usually num_layers).

    Returns:
        Tuple of (median_us, max_us, min_us)
    """
    quantiles = list(quantiles or DEFAULT_QUANTILES)
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms / scale, 1000 * max_ms / scale, 1000 * min_ms / scale


def run_benchmark_no_cudagraph(
    fn: Callable,
    quantiles: Sequence[float] = (),
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    quantiles = list(quantiles or DEFAULT_QUANTILES)
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms / scale, 1000 * max_ms / scale, 1000 * min_ms / scale
