import argparse
import contextlib
import inspect
import itertools
import json
import math
import os
import zlib
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

from sglang.jit_kernel.benchmark import hardware
from sglang.jit_kernel.utils import cache_once
from sglang.utils import is_in_ci

F = TypeVar("F", bound=Callable[..., "BenchResult"])
K = TypeVar("K")
Metric: TypeAlias = "float | Literal['avg']"
BENCH_CONFIG: TypeAlias = "List[Tuple[Tuple[str, ...], List[Tuple[Any, ...]]]]"
UNIT_SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0}
TYPE_LIST = (bool, int, float, str, torch.dtype, torch.device, None.__class__)
DISABLE_LOG_BANDWIDTH = os.environ.get("SGLANG_KERNEL_DISABLE_LOG_BANDWIDTH") == "1"


__all__ = [
    "BenchResult",
    "BenchSkip",
    "Benchmark",
    "Inputs",
    "Kernel",
    "benchmark",
    "kernel",
    "io",
    "main",
    "parametrize",
    "do_bench",
    "skip",
]

DEFAULT_RTOL = 2e-2
DEFAULT_ATOL = 1e-2


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
    """
    l2_size = _get_l2_cache_size()
    safe_cache_threshold = l2_size * _L2_SAFE_RATIO

    if nbytes <= 0 or nbytes >= safe_cache_threshold:
        return 1

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


class Inputs(NamedTuple):
    """One immutable input set for a kernel: positional `args` after the line
    arg, plus `kwargs`. `clone()` deep-copies tensors so each impl/replay reads
    its own buffer (defeats both in-place mutation and L2-cache reuse)."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def clone(self) -> "Inputs":
        return Inputs(
            tuple(_clone_recursive(a) for a in self.args),
            {k: _clone_recursive(v) for k, v in self.kwargs.items()},
        )


def io(*args: Any, **kwargs: Any) -> Inputs:
    """Wrap a kernel's inputs. Return this from a `Kernel.inputs(...)` method."""
    return Inputs(args, kwargs)


def _combo_seed(base: int, kwargs: Dict[str, Any]) -> int:
    """Deterministic per-combo seed: stable across processes (unlike hash())."""
    key = ",".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))
    return (base + zlib.crc32(key.encode())) & 0x7FFFFFFF


class Kernel(Generic[K]):
    """A benchmark with a torch reference for correctness. The user class
    declares three seams; the harness owns seeding, correctness, timing,
    roofline, and reporting:

        inputs(self, <axes...>) -> marker.io(...)   # built once per combo, seeded
        run(self, <line_arg>, *args, **kwargs)      # returns the OUTPUT tensor
        reference = "<one of line_vals>"            # ground truth for assert_close

    `parametrize` axes feed `inputs`; `line_arg`/`line_vals` select impls in `run`.
    """

    def __init__(
        self,
        cls: type,
        *,
        line_arg: str,
        line_vals: List[Any],
        reference: Optional[Any],
        correctness: bool,
        reason: Optional[str],
        flops: Optional[Callable[..., float]],
        flops_dtype: torch.dtype,
        cold: bool,
        rtol: float,
        atol: float,
        unit: str,
    ):
        assert unit in UNIT_SCALE
        assert len(set(line_vals)) == len(line_vals) > 0
        assert hasattr(cls, "inputs") and hasattr(
            cls, "run"
        ), f"{cls.__name__} must define inputs() and run() methods"
        self._cls = cls
        self._name = cls.__name__
        self._instance = cls()
        self._line_arg = line_arg
        self._line_vals = list(line_vals)
        self._unit = unit
        self._unit_scale = UNIT_SCALE[unit]
        self._flops = flops
        self._flops_dtype = flops_dtype
        self._cold = cold
        self.rtol = getattr(self._instance, "rtol", rtol)
        self.atol = getattr(self._instance, "atol", atol)
        self._fn_params = dict(inspect.signature(self._instance.inputs).parameters)
        self._run_params = dict(inspect.signature(self._instance.run).parameters)
        self._configs: BENCH_CONFIG = []
        self._seen_args: set = set()
        assert line_arg in self._run_params, (
            f"line_arg {line_arg!r} is not a parameter of {self._name}.run; "
            f"available: {list(self._run_params)}"
        )
        if correctness:
            assert reference is not None, (
                f"{self._name}: set reference=<one of {self._line_vals}>, "
                f"or correctness=False with reason=..."
            )
            assert (
                reference in self._line_vals
            ), f"{self._name}: reference {reference!r} not in line_vals {self._line_vals}"
        else:
            assert reason, f"{self._name}: correctness=False requires reason=..."
        self._correctness = correctness
        self._reference = reference
        self._reason = reason

    def add_config(self, names: Tuple[str, ...], vals: List[Tuple[Any, ...]]) -> None:
        assert len(names) > 0, "parametrize: must provide at least one name"
        for name in names:
            assert name in self._fn_params, (
                f"parametrize name {name!r} is not a parameter of "
                f"{self._name}.inputs; available: {list(self._fn_params)}"
            )
            assert (
                name not in self._seen_args
            ), f"parametrize name {name!r} is already used"
            self._seen_args.add(name)
        self._configs.insert(0, (names, vals))

    def _combos(self) -> Iterable[Dict[str, Any]]:
        names_per_axis = [ns for ns, _ in self._configs]
        vals_per_axis = [v for _, v in self._configs]
        for combo in itertools.product(*vals_per_axis):
            kwargs: Dict[str, Any] = {}
            for ns, vt in zip(names_per_axis, combo):
                kwargs.update(zip(ns, vt))
            yield kwargs

    def _check_params(self) -> None:
        kinds = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        missing = {
            n
            for n, p in self._fn_params.items()
            if p.default is inspect.Parameter.empty and p.kind in kinds
        } - set(self._seen_args)
        assert (
            not missing
        ), f"inputs() params not parametrized for {self._name}: {sorted(missing)}"

    def _correctness_check(self, base_io: "Inputs", kwargs: Dict[str, Any]) -> int:
        ref_io = base_io.clone()
        ref_out = _clone_recursive(
            self._instance.run(self._reference, *ref_io.args, **ref_io.kwargs)
        )
        checked = 0
        for impl in self._line_vals:
            if impl == self._reference:
                continue
            impl_io = base_io.clone()
            out = self._instance.run(impl, *impl_io.args, **impl_io.kwargs)
            try:
                torch.testing.assert_close(out, ref_out, rtol=self.rtol, atol=self.atol)
            except AssertionError as e:
                raise AssertionError(
                    f"[{self._name}] correctness FAILED: impl={impl!r} vs "
                    f"reference={self._reference!r} at {kwargs}\n{e}"
                ) from None
            checked += 1
        return checked

    def _bench_one(self, impl: Any, base_io: "Inputs", cold: bool) -> BenchResult:
        bio = base_io.clone()
        bench_kwargs = {
            "graph_clone_args": "all" if cold else None,
            "graph_clone_kwargs": "all" if cold else None,
        }
        if hasattr(self._instance, "bench_kwargs"):
            custom_bench_kwargs = self._instance.bench_kwargs(
                impl, *bio.args, **bio.kwargs
            )
            assert isinstance(custom_bench_kwargs, dict), (
                f"{self._name}.bench_kwargs must return a dict of marker.do_bench "
                f"keyword arguments, got {type(custom_bench_kwargs).__name__}"
            )
            bench_kwargs.update(custom_bench_kwargs)
        return do_bench(
            self._instance.run,
            input_args=(impl, *bio.args),
            input_kwargs=bio.kwargs,
            **bench_kwargs,
        )

    def run(
        self,
        *,
        mode: str = "all",
        seed: Optional[int] = None,
        cold: Optional[bool] = None,
        baseline: Optional[str] = None,
        compare: Optional[str] = None,
        tol: float = 0.05,
    ) -> None:
        cold = self._cold if cold is None else cold
        self._check_params()
        rows: List[Tuple[Dict[str, Any], Dict, Dict, Dict]] = []
        n_checked = 0
        for kwargs in self._combos():
            cell_seed = seed if seed is not None else _combo_seed(42, kwargs)
            torch.manual_seed(cell_seed)
            torch.cuda.manual_seed_all(cell_seed)
            base_io = self._instance.inputs(**kwargs)
            assert isinstance(base_io, Inputs), (
                f"{self._name}.inputs must return marker.io(...), "
                f"got {type(base_io).__name__}"
            )
            if self._correctness and mode in ("all", "check"):
                n_checked += self._correctness_check(base_io, kwargs)
            lat: Dict[Any, float] = {}
            gbps: Dict[Any, float] = {}
            tflops: Dict[Any, float] = {}
            if mode in ("all", "bench"):
                for impl in self._line_vals:
                    res = self._bench_one(impl, base_io, cold)
                    lat[impl] = res.times[0] / self._unit_scale
                    if res.memory_footprint:
                        gbps[impl] = res.memory_footprint / (1024**3) / res.times[0]
                    if self._flops is not None:
                        tflops[impl] = self._flops(**kwargs) / res.times[0] / 1e12
            rows.append((kwargs, lat, gbps, tflops))

        self._report(rows, mode, cold, seed, n_checked)
        if baseline:
            self._dump_baseline(rows, baseline)
        if compare:
            self._compare_baseline(rows, compare, tol)

    def _report(
        self,
        rows: List[Tuple[Dict[str, Any], Dict, Dict, Dict]],
        mode: str,
        cold: bool,
        seed: Optional[int],
        n_checked: int,
    ) -> None:
        seed_str = "auto" if seed is None else str(seed)
        if mode == "check":
            print(
                f"[{self._name}] correctness PASSED: {n_checked} comparisons "
                f"across {len(rows)} configs (seed={seed_str}, "
                f"rtol={self.rtol}, atol={self.atol})"
            )
            return

        has_bw = any(r[2] for r in rows)
        has_flops = any(r[3] for r in rows)
        peak_bw = (
            hardware.peak_bandwidth_gbps()
            if has_bw and not DISABLE_LOG_BANDWIDTH
            else None
        )
        peak_tf = hardware.peak_tflops(self._flops_dtype) if has_flops else None

        caption = (
            f"{self._name}  mode={mode}  cache={'cold' if cold else 'warm'}  "
            f"seed={seed_str}"
        )
        if self._correctness and mode == "all":
            caption += "  correctness=OK"
        if peak_bw:
            caption += f"  peak_bw={peak_bw:.0f}GB/s"
        if peak_tf and peak_tf == peak_tf:  # not NaN
            caption += f"  peak={peak_tf:.0f}TF({self._flops_dtype})"
        print(caption)

        ordered_names = [n for ns, _ in self._configs for n in ns]
        table = Table()
        table.col(min_width=0, pad=0, align="<")
        for name in ordered_names:
            table.col(name)
        table.sep()
        for impl in self._line_vals:
            table.col(f"{impl}({self._unit})", min_width=14)
        if peak_bw:
            table.sep()
            for impl in self._line_vals:
                table.col(f"{impl}(%HBM)", min_width=12)
        if peak_tf and peak_tf == peak_tf:
            table.sep()
            for impl in self._line_vals:
                table.col(f"{impl}(%TF)", min_width=12)

        for row_id, (kwargs, lat, gbps, tflops) in enumerate(rows):
            cells: List[Any] = [row_id]
            cells.extend(kwargs[n] for n in ordered_names)
            cells.extend(
                table.format_latency(lat.get(impl, float("nan")))
                for impl in self._line_vals
            )
            if peak_bw:
                cells.extend(
                    table.format_bandwidth(100 * gbps.get(impl, float("nan")) / peak_bw)
                    for impl in self._line_vals
                )
            if peak_tf and peak_tf == peak_tf:
                cells.extend(
                    table.format_bandwidth(
                        100 * tflops.get(impl, float("nan")) / peak_tf
                    )
                    for impl in self._line_vals
                )
            table.row(*cells)
        table.print()

    def _key(self, kwargs: Dict[str, Any]) -> str:
        return ",".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))

    def _dump_baseline(self, rows, path: str) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data[self._name] = {
            self._key(kwargs): {str(k): v for k, v in lat.items()}
            for kwargs, lat, _, _ in rows
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(f"[{self._name}] wrote baseline -> {path}")

    def _compare_baseline(self, rows, path: str, tol: float) -> None:
        with open(path) as f:
            base = json.load(f).get(self._name, {})
        regressions: List[str] = []
        print(f"[{self._name}] compare vs {path} (tol={tol:.0%}):")
        for kwargs, lat, _, _ in rows:
            ref = base.get(self._key(kwargs), {})
            for impl in self._line_vals:
                old, new = ref.get(str(impl)), lat.get(impl)
                if old is None or new is None:
                    continue
                delta = (new - old) / old
                flag = "  REGRESSION" if delta > tol else ""
                if delta > tol:
                    regressions.append(f"{self._key(kwargs)} {impl}: {delta:+.1%}")
                print(
                    f"  {self._key(kwargs)} {impl}: "
                    f"{old:.3f}->{new:.3f}{self._unit} ({delta:+.1%}){flag}"
                )
        if regressions:
            raise SystemExit(
                f"[{self._name}] {len(regressions)} regression(s) over {tol:.0%}:\n  "
                + "\n  ".join(regressions)
            )


def kernel(
    line_arg: str,
    line_vals: List[Any],
    *,
    reference: Optional[Any] = None,
    correctness: bool = True,
    reason: Optional[str] = None,
    flops: Optional[Callable[..., float]] = None,
    flops_dtype: torch.dtype = torch.bfloat16,
    cold: bool = True,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    unit: str = "us",
):
    """Turn a class with `inputs()`/`run()` into a benchmark with correctness.

    Place this decorator closest to the class; stack `@parametrize` above it::

        @marker.parametrize("dim", [1024, 4096])
        @marker.kernel("impl", ["jit", "torch"], reference="torch")
        class MyKernel:
            def inputs(self, dim): ...
            def run(self, impl, x): ...
            def bench_kwargs(self, impl, x): ...  # optional marker.do_bench kwargs
    """

    def decorator(cls: type) -> Kernel:
        return Kernel(
            cls,
            line_arg=line_arg,
            line_vals=line_vals,
            reference=reference,
            correctness=correctness,
            reason=reason,
            flops=flops,
            flops_dtype=flops_dtype,
            cold=cold,
            rtol=rtol,
            atol=atol,
            unit=unit,
        )

    return decorator


def main(*objs: Any) -> None:
    """CLI entry for a bench file: `marker.main(KernelA, KernelB, ...)`.

    Flags: --mode {all,check,bench}  --seed N  --warm  --baseline F  --compare F  --tol X
    Plain `Benchmark` objects (function form) are run in full mode, ignoring flags.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["all", "check", "bench"], default="all")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--warm", action="store_true", help="reuse buffers (L2-hot)")
    p.add_argument("--baseline", default=None, help="dump latencies to JSON")
    p.add_argument("--compare", default=None, help="fail on regression vs JSON")
    p.add_argument("--tol", type=float, default=0.05)
    a = p.parse_args()
    for obj in objs:
        if isinstance(obj, Kernel):
            obj.run(
                mode=a.mode,
                seed=a.seed,
                cold=not a.warm,
                baseline=a.baseline,
                compare=a.compare,
                tol=a.tol,
            )
        else:
            obj.run()
