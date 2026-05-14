import itertools
import os
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
)

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.utils import is_in_ci

F = TypeVar("F", bound=Callable[..., "BenchResult"])
Metric: TypeAlias = 'float | Literal["avg"]'

TYPE_LIST = (bool, int, float, str, torch.dtype, torch.device, None.__class__)
DISABLE_LOG_BANDWIDTH = os.environ.get("SGLANG_KERNEL_DISABLE_LOG_BANDWIDTH") == "1"


@cache_once
def _get_stream(device_id: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device_id)


def _clone(in_: Any) -> Any:
    if isinstance(in_, torch.Tensor):
        return in_.clone()
    elif isinstance(in_, (list, tuple)):
        return type(in_)(_clone(x) for x in in_)
    elif isinstance(in_, dict):
        return {k: _clone(v) for k, v in in_.items()}
    elif isinstance(in_, TYPE_LIST):
        return in_
    # NOTE: avoid silent error
    raise ValueError(f"unsupported type: {type(in_)}")


def _get_nbytes(in_: Any) -> int:
    if isinstance(in_, torch.Tensor):
        return in_.nbytes
    elif isinstance(in_, (list, tuple)):
        return sum(_get_nbytes(x) for x in in_)
    elif isinstance(in_, dict):
        return sum(_get_nbytes(v) for v in in_.values())
    elif isinstance(in_, TYPE_LIST):
        return 0
    # NOTE: avoid silent error
    raise ValueError(f"unsupported type: {type(in_)}")


def _process_metrics(times: list[float], metrics: tuple[Metric, ...]) -> list[float]:
    results: list[float] = []
    times = sorted(x / 1000 for x in times)  # convert to seconds and sort
    for metric in metrics:
        if metric == "avg":
            results.append(sum(times) / len(times))
        else:
            assert 0 <= metric <= 1, f"invalid metric: {metric}"
            which = min(int(len(times) * metric), len(times) - 1)
            results.append(times[which])
    return results


class BenchResult(NamedTuple):
    metrics: Tuple[Metric, ...]
    times: List[float]  # in seconds
    memory_footprint: Optional[int]


def bench_one_function(
    fn: Callable,
    *,
    input_args: Tuple[Any, ...] = (),
    input_kwargs: Dict[str, Any] = {},
    use_cuda_graph: bool = True,
    warmup_iters: int = 50,
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
) -> BenchResult:
    """
    Benchmark a function using CUDA graph or naive loop.

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
    """
    # first warmup the function
    device_id = torch.cuda.current_device()
    if stream is None:
        stream = _get_stream(device_id)
    old_current_stream = torch.cuda.current_stream(device_id)
    result: List[float] = []
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        stream.wait_stream(old_current_stream)
        for _ in range(warmup_iters):
            fn(*input_args, **input_kwargs)
        if use_cuda_graph:
            # NOTE: by default, reduce all the CPU-side overhead
            rep_count = 4
            loop_iters = 100
            graph = torch.cuda.CUDAGraph()
            input_args_list = [input_args] * rep_count
            input_kwargs_list = [input_kwargs] * rep_count
            if graph_clone_args == "all":
                graph_clone_args = range(len(input_args))
            elif graph_clone_args is None:
                graph_clone_args = []
            if graph_clone_kwargs == "all":
                graph_clone_kwargs = input_kwargs.keys()
            elif graph_clone_kwargs is None:
                graph_clone_kwargs = []
            graph_clone_args = set(graph_clone_args)
            graph_clone_kwargs = set(graph_clone_kwargs)
            # NOTE: we rotate the buffer here to avoid L2 cache effect
            for i in range(1, rep_count):
                input_args_list[i] = tuple(
                    _clone(input_args[j]) if j in graph_clone_args else input_args[j]
                    for j in range(len(input_args))
                )
                input_kwargs_list[i] = dict(
                    (k, (_clone(v) if k in graph_clone_kwargs else v))
                    for k, v in input_kwargs.items()
                )
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(loop_iters // rep_count):
                    for args, kwargs in zip(input_args_list, input_kwargs_list):
                        fn(*args, **kwargs)
            # warm up the graph
            graph.replay()
            # then replay the graph and measure the time
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            for _ in range(max(replay_iters // loop_iters, 10)):
                tic.record(stream)
                graph.replay()
                toc.record(stream)
                stream.synchronize()
                result.append(tic.elapsed_time(toc) / loop_iters)
        else:
            # NOTE: no cuda graph, naive loop
            empty_tensor = torch.empty(64 * 1024 * 1024, device=f"cuda:{device_id}")
            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            for _ in range(max(replay_iters, 10)):
                empty_tensor.zero_()  # cold the L2 cache
                tic.record(stream)
                fn(*input_args, **input_kwargs)
                toc.record(stream)
                stream.synchronize()
                result.append(tic.elapsed_time(toc))

    stream.synchronize()
    result = _process_metrics(result, metrics)
    memory_footprint = None
    if not disable_log_bandwidth:
        if memory_args == "all":
            memory_args = input_args + tuple(input_kwargs.values())
        if memory_output == "out":
            memory_output = fn(*input_args, **input_kwargs)
        memory_footprint = extra_memory_footprint
        memory_footprint += _get_nbytes(extra_memory_args)
        memory_footprint += _get_nbytes(memory_args)
        memory_footprint += _get_nbytes(memory_output)

    return BenchResult(metrics, result, memory_footprint)


class Benchmark(Generic[F]):
    def __init__(self, fn: F, line_arg: str, line_vals: List[str], *, unit: str):
        self.benchmark_fn = fn
        self.benchmark_configs: dict[str, List[Any]] = OrderedDict()
        self.line_arg = line_arg
        self.line_vals = line_vals
        assert unit in ("us", "ms", "s")
        UNIT_SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0}
        self.unit = unit
        self.unit_scale = UNIT_SCALE[unit]

    def run(self) -> None:
        def get_width(s: Any, min_width: int = 10) -> int:
            return max(len(str(s)) + 2, min_width)

        def format_latency(r: float) -> str:
            length = len(str(int(r)))
            if length < 5:
                return f"{r:.4f}"
            # decrease number of the digits
            digits = max(0, 4 - (length - 5))
            return f"{r:.{digits}f}"

        def format_bandwidth(b: float) -> str:
            return f"{b:.2f}"

        results: List[List[float]] = []
        bandwidth_results: List[List[float]] = []
        benchmark_configs = OrderedDict(reversed(self.benchmark_configs.items()))
        should_log_bandwidth = False
        for system in self.line_vals:
            system_list = []
            bandwidth_list = []
            for config in itertools.product(*benchmark_configs.values()):
                config_dict = dict(zip(benchmark_configs.keys(), config))
                config_dict.update({self.line_arg: system})
                result = self.benchmark_fn(**config_dict)
                system_list.append(result.times[0] / self.unit_scale)
                if not DISABLE_LOG_BANDWIDTH and result.memory_footprint is not None:
                    should_log_bandwidth = True
                    memory_giga_bytes = result.memory_footprint / (1024**3)
                    bandwidth = memory_giga_bytes / result.times[0]
                    bandwidth_list.append(bandwidth)
            results.append(system_list)
            bandwidth_results.append(bandwidth_list)

        num_ids = len(results[0])
        id_width = len(str(num_ids - 1))
        args_widths = [
            max(get_width(v) for v in [key] + vals)
            for key, vals in benchmark_configs.items()
        ]
        system_widths = [
            get_width(f"{system}({self.unit})", 15) for system in self.line_vals
        ]
        bandwidth_widths: List[int] = []
        if should_log_bandwidth:
            bandwidth_widths = [
                get_width(f"{system}(GB/s)", 15) for system in self.line_vals
            ]
        # id, args... , system0, system1, ...
        print(" " * id_width, end="")
        for key, width in zip(benchmark_configs.keys(), args_widths):
            print(f"{key:>{width}}", end="")
        for system, width in zip(self.line_vals, system_widths):
            system_name = f"{system}({self.unit})"
            print(f"{system_name:>{width}}", end="")
        if should_log_bandwidth:
            for system, width in zip(self.line_vals, bandwidth_widths):
                system_name = f"{system}(GB/s)"
                print(f"{system_name:>{width}}", end="")
        print()
        for id, config in enumerate(itertools.product(*benchmark_configs.values())):
            print(f"{id:<{id_width}}", end="")
            for arg, width in zip(config, args_widths):
                print(f"{str(arg):>{width}}", end="")
            for system_result, width in zip(results, system_widths):
                print(f"{format_latency(system_result[id]):>{width}}", end="")
            if should_log_bandwidth:
                for bandwidth_result, width in zip(bandwidth_results, bandwidth_widths):
                    print(f"{format_bandwidth(bandwidth_result[id]):>{width}}", end="")
            print()


def mark_benchmark(line_arg: str, line_vals: List[str], *, unit: str = "us"):
    def decorator(fn: F) -> Benchmark[F]:
        return Benchmark(fn, line_arg, line_vals, unit=unit)

    return decorator


def mark_args(name: str, vals: List[Any], ci_vals: Optional[List[Any]] = None):
    def decorator(bench: Benchmark[F]) -> Benchmark[F]:
        if ci_vals is not None and is_in_ci():
            bench.benchmark_configs[name] = ci_vals
        else:
            bench.benchmark_configs[name] = vals
        return bench

    return decorator
