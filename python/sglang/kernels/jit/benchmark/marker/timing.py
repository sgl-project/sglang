import contextlib
import math
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
)

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once

from .config import DISABLE_LOG_BANDWIDTH
from .result import BenchResult, Metric

TYPE_LIST = (bool, int, float, str, torch.dtype, torch.device, None.__class__)
_L2_SAFE_RATIO = 5


@cache_once
def _get_benchmark_stream(device_id: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device_id)


def _clone_recursive(in_: Any) -> Any:
    if isinstance(in_, torch.Tensor):
        return in_.clone()
    elif isinstance(in_, (list, tuple)):
        return type(in_)(_clone_recursive(x) for x in in_)
    elif isinstance(in_, dict):
        return {k: _clone_recursive(v) for k, v in in_.items()}
    elif isinstance(in_, TYPE_LIST):
        return in_
    # NOTE: avoid silent error
    raise ValueError(f"unsupported type: {type(in_)}")


def _get_nbytes_recursive(in_: Any) -> int:
    if isinstance(in_, torch.Tensor):
        return in_.nbytes
    elif isinstance(in_, (list, tuple)):
        return sum(_get_nbytes_recursive(x) for x in in_)
    elif isinstance(in_, dict):
        return sum(_get_nbytes_recursive(v) for v in in_.values())
    elif isinstance(in_, TYPE_LIST):
        return 0
    # NOTE: avoid silent error
    raise ValueError(f"unsupported type: {type(in_)}")


def _process_metrics(
    sorted_times: list[float], metrics: tuple[Metric, ...]
) -> list[float]:
    results: list[float] = []
    for metric in metrics:
        if metric == "avg":
            results.append(sum(sorted_times) / len(sorted_times))
        else:
            assert 0 <= metric <= 1, f"invalid metric: {metric}"
            which = min(int(len(sorted_times) * metric), len(sorted_times) - 1)
            results.append(sorted_times[which])
    return results


class _L2FlushBuffer:
    def __init__(self) -> None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        l2_size = current_arch().l2_cache_bytes
        safe_size = int(l2_size * _L2_SAFE_RATIO)
        self.buffer = torch.empty(safe_size // 8, device=device, dtype=torch.int64)

    def flush_l2_cache(self) -> None:
        # NOTE: too much write will be still inflight when the second kernel starts,
        # which may brings noise to the benchmark result, so we only read the buffer
        self.buffer.sum()


def _calculate_rotation_count(nbytes: int, min_rotations: int = 2) -> int:
    """
    Adapted from flashinfer benchmark utility:
    https://github.com/flashinfer-ai/flashinfer/blob/c5a2b06edae4fa2bfd2ae25eed16eb565c70513f/flashinfer/testing/utils.py

    Calculate the number of buffer copies needed to ensure cold L2 cache.

    The function uses conservative thresholds to account for:
    - LRU eviction being gradual (not all data evicted when capacity exceeded)
    - Cache associativity effects (some data may persist in non-conflicting sets)
    - Hardware prefetching behavior

    Returns 1 (no rotation needed) only when tensor size substantially exceeds
    L2 cache, ensuring cache effects are truly negligible.

    Args:
        tensors: List of tensors to consider for rotation (must be on GPU).
        device: Device for L2 cache query (None for current device).
        min_rotations: Minimum number of rotations when rotation is needed.

    Returns:
        Number of buffer copies needed (1 means no rotation needed).
    """
    l2_size = current_arch().l2_cache_bytes
    safe_cache_threshold = l2_size * _L2_SAFE_RATIO

    if nbytes <= 0 or nbytes >= safe_cache_threshold:
        return 1  # No tensors to rotate

    # Conservative formula: ensure between any two uses of the same buffer,
    # we've accessed enough data to fully flush L2 with margin
    # Using safe_cache_threshold ensures we account for all cache effects
    num_rotations = math.ceil(safe_cache_threshold / nbytes) + 1
    return max(min_rotations, num_rotations)


def _do_bench_internal_graph(
    fn: Callable,
    replay_iters: int,
    input_args: Tuple[Any, ...],
    input_kwargs: Dict[str, Any],
    graph_clone_args: Iterable[int],
    graph_clone_kwargs: Iterable[str],
    graph_context: ContextManager,
    sync_multigpu_fn: Callable[[], Any],
) -> List[float]:
    result: List[float] = []
    stream = torch.cuda.current_stream()
    buffer = _L2FlushBuffer()
    # only count the cloned tensors for rotation count
    nbytes = sum(_get_nbytes_recursive(input_args[i]) for i in graph_clone_args)
    nbytes += sum(_get_nbytes_recursive(input_kwargs[k]) for k in graph_clone_kwargs)
    rotate_count = min(_calculate_rotation_count(nbytes), 100)
    loop_count = math.ceil(100 / rotate_count) * rotate_count
    input_args_list = [input_args] * rotate_count
    input_kwargs_list = [input_kwargs] * rotate_count
    graph_clone_args = set(graph_clone_args)
    graph_clone_kwargs = set(graph_clone_kwargs)

    graph = torch.cuda.CUDAGraph()
    # NOTE: we rotate the buffer here to avoid L2 cache effect
    for i in range(1, rotate_count):
        input_args_list[i] = tuple(
            (
                _clone_recursive(input_args[j])
                if j in graph_clone_args
                else input_args[j]
            )
            for j in range(len(input_args))
        )
        input_kwargs_list[i] = dict(
            (k, (_clone_recursive(v) if k in graph_clone_kwargs else v))
            for k, v in input_kwargs.items()
        )
    with graph_context:
        with torch.cuda.graph(graph, stream=stream):
            for i in range(loop_count):
                args = input_args_list[i % rotate_count]
                kwargs = input_kwargs_list[i % rotate_count]
                fn(*args, **kwargs)

    # warm up the graph once
    graph.replay()
    # then replay the graph and measure the time
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    for _ in range(max(replay_iters // loop_count, 10)):
        buffer.flush_l2_cache()
        sync_multigpu_fn()  # sync GPU before each iteration for precise timing
        tic.record(stream)
        graph.replay()
        toc.record(stream)
        stream.synchronize()
        result.append(tic.elapsed_time(toc) / loop_count)
    return result


def do_bench(
    fn: Callable,
    *,
    input_args: Tuple[Any, ...] = (),
    input_kwargs: Dict[str, Any] = {},
    use_cuda_graph: bool = True,
    warmup_iters: int = 20,
    replay_iters: int = 1000,
    metrics: Tuple[Metric, ...] = (0.5, "avg"),
    stream: torch.cuda.Stream | None = None,
    # NOTE: should only clone the read args to avoid L2 cache effect in cuda graph
    graph_clone_args: Iterable[int] | Literal["all"] | None = "all",
    graph_clone_kwargs: Iterable[str] | Literal["all"] | None = "all",
    # NOTE: for memory-bandwidth profiling
    disable_log_bandwidth: bool = DISABLE_LOG_BANDWIDTH,
    memory_args: Iterable[Any] | Literal["all"] | None = "all",
    memory_output: Iterable[Any] | Literal["out"] | None = "out",
    extra_memory_args: Iterable[Any] | None = None,
    extra_memory_footprint: int = 0,
    flops: Optional[float] = None,
    graph_context_fn: Optional[Callable[[], ContextManager]] = None,
    sync_multigpu_fn: Optional[Callable[[], Any]] = None,
    estimated_time_ms: Optional[float] = 1.0,
) -> BenchResult:
    """
    Benchmark a function using CUDA graph or naive loop.
    Adapted from flashinfer benchmark utility:
    https://github.com/flashinfer-ai/flashinfer/blob/c5a2b06edae4fa2bfd2ae25eed16eb565c70513f/flashinfer/testing/utils.py

    :param fn: Function to benchmark
    :param input_args: Positional arguments to pass to the function
    :param input_kwargs: Keyword arguments to pass to the function
    :param use_cuda_graph: Whether to use CUDA graph for benchmarking
    :param warmup_iters: Number of warm-up iterations to run before benchmarking
    :param replay_iters: Number of iterations to run for benchmarking
    :param metrics: Metrics to compute from the timing results (quantiles in [0, 1] or "avg")
    :param stream: CUDA stream to use for benchmarking (if None, a new stream will be created)
    :param graph_clone_args: Indices of input_args to clone for each iteration.
                             Only the read args need to be cloned to avoid L2 cache effect.
    :param graph_clone_kwargs: Keys of input_kwargs to clone for each iteration.
                               Only the read args need to be cloned to avoid L2 cache effect.
    :param disable_log_bandwidth: Whether to disable logging memory bandwidth in the profile report.
    :param memory_args: Optional sequence of arguments to calculate total memory footprint.
                        Used for memory bandwidth estimation in the profile report.
    :param memory_output: Arguments whose output memory should be included in the memory footprint.
    :param extra_memory_args: Additional arguments to consider for memory footprint calculation.
    :param extra_memory_footprint: Additional memory footprint to consider.
                                   This is typically used when the load/store bytes is dynamic.
    :param flops: The number of floating-point operations performed by the benchmark.
                  Used for calculating the achieved computation utilization in the profile report.
    :param graph_context_fn: A callable returning a context manager that wraps the cuda graph capture.
    :param sync_multigpu_fn: A callable to synchronize multiple GPUs before each iteration. For precise
                             benchmark number in multi-GPU benchmark, it should be some synchronization
                             primitive on GPU side (not on CPU side).
    :param estimated_time_ms: Estimated time in milliseconds for the benchmark without CUDA graph.
                              This is typically used to control the benchmark time.
                              Ignored if `use_cuda_graph` is True.
    """
    # first warmup the function
    device_id = torch.cuda.current_device()
    if stream is None:
        stream = _get_benchmark_stream(device_id)
    old_current_stream = torch.cuda.current_stream(device_id)
    result: List[float] = []
    sync_multigpu_fn = sync_multigpu_fn or (lambda: None)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        stream.wait_stream(old_current_stream)
        sync_multigpu_fn()
        for _ in range(warmup_iters):
            fn(*input_args, **input_kwargs)
        if use_cuda_graph:
            # NOTE: by default, reduce all the CPU-side overhead
            if graph_clone_args == "all":
                graph_clone_args = range(len(input_args))
            elif graph_clone_args is None:
                graph_clone_args = []
            if graph_clone_kwargs == "all":
                graph_clone_kwargs = input_kwargs.keys()
            elif graph_clone_kwargs is None:
                graph_clone_kwargs = []
            graph_context = (
                graph_context_fn()
                if graph_context_fn is not None
                else contextlib.nullcontext()
            )
            result = _do_bench_internal_graph(
                fn,
                replay_iters,
                input_args,
                input_kwargs,
                graph_clone_args,
                graph_clone_kwargs,
                graph_context,
                sync_multigpu_fn,
            )
        else:
            # NOTE: no cuda graph, naive loop
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            buffer = _L2FlushBuffer()
            if estimated_time_ms is not None:
                buffer.flush_l2_cache()
                sync_multigpu_fn()
                tic.record(stream)
                fn(*input_args, **input_kwargs)
                toc.record(stream)
                stream.synchronize()
                duration_ms = tic.elapsed_time(toc)
                estimted_iters = int(estimated_time_ms / duration_ms)
                replay_iters = min(replay_iters, estimted_iters)

            for _ in range(max(replay_iters, 10)):
                buffer.flush_l2_cache()
                sync_multigpu_fn()
                tic.record(stream)
                fn(*input_args, **input_kwargs)
                toc.record(stream)
                stream.synchronize()
                result.append(tic.elapsed_time(toc))

    stream.synchronize()
    samples = tuple(sorted(x / 1000 for x in result))  # ms -> s
    result = _process_metrics(list(samples), metrics)
    memory_footprint = None
    if not disable_log_bandwidth:
        if memory_args == "all":
            memory_args = input_args + tuple(input_kwargs.values())
        if memory_output == "out":
            memory_output = fn(*input_args, **input_kwargs)
        memory_footprint = extra_memory_footprint
        memory_footprint += _get_nbytes_recursive(extra_memory_args)
        memory_footprint += _get_nbytes_recursive(memory_args)
        memory_footprint += _get_nbytes_recursive(memory_output)

    return BenchResult(metrics, result, memory_footprint, flops, samples)
