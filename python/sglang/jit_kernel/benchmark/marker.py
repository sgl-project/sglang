import contextlib
import inspect
import itertools
import math
import os
from typing import (
    Any,
    Callable,
    ContextManager,
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

from sglang.kernels._jit import cache_once
from sglang.utils import is_in_ci

F = TypeVar("F", bound=Callable[..., "BenchResult"])
Metric: TypeAlias = "float | Literal['avg']"
BENCH_CONFIG: TypeAlias = "List[Tuple[Tuple[str, ...], List[Tuple[Any, ...]]]]"
UNIT_SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0}
TYPE_LIST = (bool, int, float, str, torch.dtype, torch.device, None.__class__)
DISABLE_LOG_BANDWIDTH = os.environ.get("SGLANG_KERNEL_DISABLE_LOG_BANDWIDTH") == "1"


__all__ = [
    "BenchResult",
    "BenchSkip",
    "Benchmark",
    "benchmark",
    "parametrize",
    "do_bench",
    "skip",
]


class BenchSkip(Exception):
    pass


def skip(reason: str):
    raise BenchSkip(reason)


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


@cache_once
def _get_l2_cache_size() -> int:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.L2_cache_size


_L2_SAFE_RATIO = 5


def _get_flush_l2_buffer() -> torch.Tensor:
    """Get a buffer sized to flush the L2 cache when accessed."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    l2_size = _get_l2_cache_size()
    safe_size = int(l2_size * _L2_SAFE_RATIO)
    return torch.empty(safe_size, device=device, dtype=torch.uint8)


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
    l2_size = _get_l2_cache_size()
    safe_cache_threshold = l2_size * _L2_SAFE_RATIO

    if nbytes <= 0 or nbytes >= safe_cache_threshold:
        return 1  # No tensors to rotate

    # Conservative formula: ensure between any two uses of the same buffer,
    # we've accessed enough data to fully flush L2 with margin
    # Using safe_cache_threshold ensures we account for all cache effects
    num_rotations = math.ceil(safe_cache_threshold / nbytes) + 1
    return max(min_rotations, num_rotations)


class BenchResult(NamedTuple):
    metrics: Tuple[Metric, ...]
    times: List[float]  # in seconds
    memory_footprint: Optional[int]


class Table:
    """Aligned text table with `|` section separators and `=`/`-` rules."""

    SEP = " | "

    def __init__(self) -> None:
        self._headers: List[str] = []
        self._mins: List[int] = []
        self._pads: List[int] = []
        self._aligns: List[str] = []
        self._seps: set = set()
        self._rows: List[List[str]] = []

    @staticmethod
    def format_latency(r: float) -> str:
        if math.isnan(r):
            return "N/A"
        length = len(str(int(r)))
        if length < 5:
            return f"{r:.4f}"
        # decrease number of the digits
        digits = max(0, 4 - (length - 5))
        return f"{r:.{digits}f}"

    @staticmethod
    def format_bandwidth(b: float) -> str:
        if math.isnan(b):
            return "N/A"
        return f"{b:.2f}"

    def col(
        self,
        header: str = "",
        *,
        min_width: int = 10,
        pad: int = 2,
        align: str = ">",
    ) -> None:
        self._headers.append(header)
        self._mins.append(min_width)
        self._pads.append(pad)
        self._aligns.append(align)

    def sep(self) -> None:
        self._seps.add(len(self._headers))

    def row(self, *cells: Any) -> None:
        assert len(cells) == len(self._headers)
        self._rows.append([str(c) for c in cells])

    def print(self) -> None:
        widths = [
            max(max(len(c) + p for c in [h, *(r[i] for r in self._rows)]), mw)
            for i, (h, mw, p) in enumerate(zip(self._headers, self._mins, self._pads))
        ]
        total = sum(widths) + len(self.SEP) * len(self._seps)

        def fmt(cells: List[str]) -> str:
            parts: List[str] = []
            for i, (cell, w, a) in enumerate(zip(cells, widths, self._aligns)):
                if i in self._seps:
                    parts.append(self.SEP)
                parts.append(f"{cell:{a}{w}}")
            return "".join(parts)

        print("=" * total)
        print(fmt(self._headers))
        print("-" * total)
        for r in self._rows:
            print(fmt(r))
        print("=" * total)


class Benchmark(Generic[F]):
    def __init__(self, fn: F, line_arg: str, line_vals: List[Any], *, unit: str):
        assert unit in UNIT_SCALE and len(set(line_vals)) == len(line_vals) > 0
        self._fn = fn
        self._line_arg = line_arg
        self._line_vals = line_vals
        self._unit = unit
        self._configs: BENCH_CONFIG = []
        self._fn_params = inspect.signature(fn).parameters
        self._unit_scale = UNIT_SCALE[unit]
        assert line_arg in self._fn_params, (
            f"line_arg {line_arg!r} is not a parameter of {fn.__name__}; "
            f"available: {list(self._fn_params)}"
        )
        self._seen_args = {line_arg}

    def add_config(self, names: Tuple[str, ...], vals: List[Tuple[Any, ...]]) -> None:
        """Prepend a parametrize axis. Validates that names are real parameters
        of the benchmark fn, and rejects duplicates / collisions with line_arg."""
        assert len(names) > 0, "parametrize: must provide at least one name"
        for name in names:
            assert name in self._fn_params, (
                f"parametrize name {name!r} is not a parameter of "
                f"{self._fn.__name__}; available: {list(self._fn_params)}"
            )
            assert (
                name not in self._seen_args
            ), f"parametrize name {name!r} is already used"
            self._seen_args.add(name)
        self._configs.insert(0, (names, vals))

    def _collect_results(self) -> Tuple[List[List[float]], List[List[float]], bool]:
        axis_names = [n for n, _ in self._configs]
        axis_vals = [v for _, v in self._configs]
        results: List[List[float]] = []
        bandwidth_results: List[List[float]] = []
        should_log_bandwidth = False
        for system in self._line_vals:
            latencies: List[float] = []
            bandwidths: List[float] = []
            for combo in itertools.product(*axis_vals):
                kwargs: Dict[str, Any] = {self._line_arg: system}
                for names, values in zip(axis_names, combo):
                    kwargs.update(zip(names, values))
                try:
                    result = self._fn(**kwargs)
                except BenchSkip:
                    latencies.append(float("nan"))
                    if not DISABLE_LOG_BANDWIDTH:
                        bandwidths.append(float("nan"))
                    continue
                except BaseException:
                    print(f"Benchmark failed at {system}, kwargs =", kwargs)
                    raise
                latencies.append(result.times[0] / self._unit_scale)
                if not DISABLE_LOG_BANDWIDTH and result.memory_footprint is not None:
                    should_log_bandwidth = True
                    bandwidths.append(
                        result.memory_footprint / (1024**3) / result.times[0]
                    )
            results.append(latencies)
            bandwidth_results.append(bandwidths)
        return results, bandwidth_results, should_log_bandwidth

    def run(self) -> None:
        # Pre-check: every required fn param must be covered.
        flat_names = [n for names, _ in self._configs for n in names]
        kinds = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        missing = {
            n
            for n, p in self._fn_params.items()
            if p.default is inspect.Parameter.empty and p.kind in kinds
        } - (set(flat_names) | {self._line_arg})
        assert not missing, (
            f"parameters not parametrized for {self._fn.__name__}: "
            f"{sorted(missing)}"
        )

        results, bandwidths, should_log_bw = self._collect_results()

        table = Table()
        table.col(min_width=0, pad=0, align="<")  # id column (tight, left-aligned)
        for name in flat_names:
            table.col(name)
        table.sep()
        for system in self._line_vals:
            table.col(f"{system}({self._unit})", min_width=15)
        if should_log_bw:
            table.sep()
            for system in self._line_vals:
                table.col(f"{system}(GB/s)", min_width=15)

        axis_vals = [v for _, v in self._configs]
        for row_id, combo in enumerate(itertools.product(*axis_vals)):
            cells: List[Any] = [row_id]
            cells.extend(v for vt in combo for v in vt)
            cells.extend(table.format_latency(r[row_id]) for r in results)
            if should_log_bw:
                cells.extend(table.format_bandwidth(r[row_id]) for r in bandwidths)
            table.row(*cells)

        table.print()


def benchmark(line_arg: str, line_vals: List[Any], *, unit: str = "us"):
    def decorator(fn: F) -> Benchmark[F]:
        return Benchmark(fn, line_arg, line_vals, unit=unit)

    return decorator


def parametrize(names: str, vals: List[Any], ci_vals: Optional[List[Any]] = None):
    """Add a parametrize axis. Pytest-style:

    - Single name:   `parametrize("dim", [1024, 4096])`
    - Multiple names (correlated):
                     `parametrize("h,d", [(1, 64), (2, 128)])`

    For multi-name axes, each value must be a tuple/list of matching length.
    """
    name_tuple = tuple(n.strip() for n in names.split(","))
    assert all(name_tuple), f"parametrize: empty name in {names!r}"
    arity = len(name_tuple)

    def _normalize(vs: List[Any]) -> List[Tuple[Any, ...]]:
        if arity == 1:
            return [(v,) for v in vs]
        out: List[Tuple[Any, ...]] = []
        for v in vs:
            assert isinstance(
                v, (tuple, list)
            ), f"parametrize: multi-name values must be tuples, got {v!r}"
            t = tuple(v)
            assert (
                len(t) == arity
            ), f"parametrize: each value must have length {arity}, got {t!r}"
            out.append(t)
        return out

    def decorator(bench: Benchmark[F]) -> Benchmark[F]:
        chosen = ci_vals if (ci_vals is not None and is_in_ci()) else vals
        bench.add_config(name_tuple, _normalize(chosen))
        return bench

    return decorator


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
    empty_tensor = _get_flush_l2_buffer()
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
        empty_tensor.zero_()  # cold the L2 cache
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
    graph_context_fn: Optional[Callable[[], ContextManager]] = None,
    sync_multigpu_fn: Optional[Callable[[], Any]] = None,
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
    :param graph_context_fn: A callable returning a context manager that wraps the cuda graph capture.
    :param sync_multigpu_fn: A callable to synchronize multiple GPUs before each iteration. For precise
                             benchmark number in multi-GPU benchmark, it should be some synchronization
                             primitive on GPU side (not on CPU side).
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
            empty_tensor = _get_flush_l2_buffer()
            for _ in range(max(replay_iters, 10)):
                empty_tensor.zero_()  # cold the L2 cache
                sync_multigpu_fn()
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
        memory_footprint += _get_nbytes_recursive(extra_memory_args)
        memory_footprint += _get_nbytes_recursive(memory_args)
        memory_footprint += _get_nbytes_recursive(memory_output)

    return BenchResult(metrics, result, memory_footprint)
