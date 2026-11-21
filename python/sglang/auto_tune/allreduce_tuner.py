"""All-reduce tuner: time every all-reduce kernel at each message size, keep the fastest.

The server dispatches all-reduces in two stages, and so does the tuner:

- eager (default): each call is launched from Python, as prefill and other
  calls outside a CUDA graph are. A custom all-reduce copies its input into
  the registered staging buffer first, and that copy is part of the time.
- graph (``--graph 1``): the calls are captured into a CUDA graph inside each
  communicator's capture context, and the graph is replayed. A captured
  custom all-reduce reads its input in place, with no copy, which is what a
  decode replay runs.

The eager table goes in ``SGLANG_TUNED_EAGER_CONFIG`` and the graph table in
``SGLANG_TUNED_GRAPH_CONFIG``.

Which backends compete is whatever the SGLang and AITER source scans find
that this tuner can construct. HIP QuickReduce and FlyDSL are included only
when the requested quant is declared in AITER. The result table names each
winner by its full class name.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import logging
import os
import statistics
import time
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from sglang.auto_tune.aiter_comms import (
    DEFAULT_AITER_DIR,
    DEFAULT_SGLANG_DIR,
    OpenerSpec,
    ScannedParams,
    backend_plan,
    format_scan,
    scan_aiter,
    scan_dirs,
    scan_sglang,
    short_name,
    source_name,
    unmatched_skips,
)

if TYPE_CHECKING:
    import torch
    from torch.cuda import CUDAGraph
    from torch.distributed import ProcessGroup

# Codecs the HIP QuickReduce opener can be asked to build. Whether one of
# them competes is decided by the AITER scan, which names FP8 rather than INT8.
QR_QUANT_REGIMES = ("FP", "INT8", "INT6", "INT4")

logger = logging.getLogger(__name__)

_QR_QUANT_ENVS = (
    "ROCM_QUICK_REDUCE_QUANTIZATION",
    "AITER_QUICK_REDUCE_QUANTIZATION",
)

MODES = ("eager", "graph")

DEFAULT_NUM_WARMUP = 20

# The kernels spin-wait on their peers, so a call only completes once every
# rank runs it. Nothing inside a timing loop keeps the ranks together, so they
# are brought back in step every this many calls or replays.
_SYNC_EVERY = 10

# One timed call. The return is the reduced tensor.
StepFn = Callable[[], "torch.Tensor"]
BarrierFn = Callable[[], None]
CaptureFn = Callable[[], ContextManager[None]]
CheckFn = Callable[["torch.Tensor"], Optional[str]]
CallFn = Callable[["torch.Tensor"], Optional["torch.Tensor"]]


class _ConfigDict(TypedDict):
    """One winning row, as written into the JSON."""

    size_bytes: int
    backend: str
    time_us: float
    tp_size: int
    dtype: str
    method: str


class _Measurement(TypedDict):
    """Every backend's time at one message size. None means it was skipped."""

    size_bytes: int
    times_us: Dict[str, Optional[float]]


class AllreduceTuneResult(TypedDict, total=False):
    """The document ``run_allreduce_tuning`` writes. Empty when nothing won."""

    mode: str
    env: str
    tp_size: int
    qr_quant: str
    qr_cast_bf16_to_fp16: int
    fly_algorithm: str
    num_warmups: int
    ignore_acr_max_size: bool
    ignore_scr_max_size: bool
    num_iters: int
    ingraph_calls: Optional[int]
    skip: List[str]
    scandirs: List[str]
    timing: str
    configs: List[_ConfigDict]
    measurements: List[_Measurement]
    path: str


class ReadyParams(ScannedParams):
    """Scan parameters plus the group this process will construct with."""

    cpu_group: "ProcessGroup"
    device: "torch.device"
    algorithm: str


class ReadySpec(TypedDict):
    opener: str
    params: ReadyParams


class _Capturable(Protocol):
    def capture(self) -> ContextManager[None]: ...


class _Closeable(Protocol):
    def close(self) -> None: ...


class _SglangTopo(Protocol):
    """SGLang custom all-reduce link check."""

    world_size: int
    full_nvlink: bool
    use_amd_deterministic_impl: bool


class _CustomProbe(Protocol):
    max_size: int
    world_size: int

    def should_custom_ar(self, tensor: "torch.Tensor") -> bool: ...
    def custom_all_reduce(self, tensor: "torch.Tensor") -> Optional["torch.Tensor"]: ...
    def _all_reduce_impl(
        self, tensor: "torch.Tensor", registered: bool = False
    ) -> "torch.Tensor": ...
    def all_reduce(
        self, tensor: "torch.Tensor", registered_input: bool = False
    ) -> "torch.Tensor": ...


class _QuickComm(Protocol):
    _SUPPORTED_DTYPES: Tuple["torch.dtype", ...]

    def should_quick_allreduce(self, tensor: "torch.Tensor") -> bool: ...
    def quick_all_reduce(self, tensor: "torch.Tensor") -> "torch.Tensor": ...


class _FlyComm(Protocol):
    device: "torch.device"

    def preload(
        self, *, payload_range: Optional[Tuple[int, int]] = None
    ) -> None: ...
    def compile(self, inp: "torch.Tensor", out: "torch.Tensor") -> None: ...
    def allreduce(self, inp: "torch.Tensor", out: "torch.Tensor") -> None: ...


class _OutplaceComm(Protocol):
    def change_state(
        self, enable: Optional[bool] = None
    ) -> ContextManager[None]: ...
    def outplace_all_reduce(
        self, tensor: "torch.Tensor"
    ) -> Optional["torch.Tensor"]: ...


class _SymmComm(Protocol):
    dtype: "torch.dtype"

    def should_torch_symm_mem_allreduce(self, tensor: "torch.Tensor") -> bool: ...


class _AllReduceCall(Protocol):
    def all_reduce(self, tensor: "torch.Tensor") -> Optional["torch.Tensor"]: ...
    def should_custom_ar(self, tensor: "torch.Tensor") -> bool: ...


def _no_capture() -> ContextManager[None]:
    return contextlib.nullcontext()


@dataclasses.dataclass
class AllreduceConfig:
    """Fastest backend at one message size."""

    size_bytes: int
    backend: str
    time_us: float
    tp_size: int
    dtype: str = ""
    # The collective this row timed. The server calls this name instead of
    # guessing from the methods the class defines.
    method: str = ""

    def to_dict(self) -> "_ConfigDict":
        return {
            "size_bytes": self.size_bytes,
            "backend": self.backend,
            "time_us": self.time_us,
            "tp_size": self.tp_size,
            "dtype": self.dtype,
            "method": self.method,
        }


class _Rejected(Exception):
    """The kernel will not run this tensor. The message is the log line."""


@dataclasses.dataclass
class _Backend:
    """One opened all-reduce.

    ``check`` returns why the tensor cannot run, or None. ``call`` runs one
    out-of-place all-reduce and returns its output. ``capture`` is entered
    around a graph capture, the way ``GroupCoordinator.graph_capture`` enters
    each communicator's ``capture()``.
    """

    check: CheckFn
    call: CallFn
    close: Callable[[], None]
    capture: CaptureFn = _no_capture
    prepare: Optional[Callable[[List[int]], None]] = None
    impl: str = ""
    note: str = ""
    method: str = ""


def _normalize_sizes(sizes: List[int]) -> List[int]:
    """The message sizes to time, in bytes. Each one is a bf16 tensor."""
    cleaned = sorted({int(size) for size in sizes})
    bad = [size for size in cleaned if size <= 0 or size % 2]
    if not cleaned or bad:
        raise ValueError(
            "sizes must be positive multiples of 2 bytes, one bf16 tensor "
            f"each. Rejected: {bad or 'none given'}"
        )
    return cleaned


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _reclaim_size_cache(tries: int = 8) -> None:
    """Give this size's cached blocks back to the driver.

    Communicator workspaces stay allocated for the whole sweep, so this does
    not wait for the device to read empty. It stops when ``empty_cache`` no
    longer shrinks the cache, or after ``tries``.
    """
    import gc

    import torch

    gc.collect()
    torch.cuda.synchronize()
    previous: Optional[int] = None
    for _ in range(tries):
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved()
        if reserved <= torch.cuda.memory_allocated() or reserved == previous:
            return
        previous = reserved
        torch.cuda.synchronize()


def _time_eager(
    step: StepFn,
    num_iters: int,
    use_median: bool,
    barrier: BarrierFn,
    sync_every: int,
    num_warmups: int = DEFAULT_NUM_WARMUP,
) -> float:
    """Microseconds per call, launched from Python one call at a time."""
    import torch

    for _ in range(num_warmups):
        step()
    barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        starts[i].record()
        step()
        ends[i].record()
        # Each event pair spans one call, so re-syncing between calls stays
        # out of the measurement.
        if sync_every and (i + 1) % sync_every == 0:
            barrier()
    torch.cuda.synchronize()
    if use_median:
        return statistics.median(
            starts[i].elapsed_time(ends[i]) * 1000 for i in range(num_iters)
        )
    else:
        return statistics.mean(
            starts[i].elapsed_time(ends[i]) * 1000 for i in range(num_iters)
        )


def _replay(
    graph: "CUDAGraph", count: int, barrier: BarrierFn, sync_every: int
) -> None:
    """Replay ``count`` times, letting the ranks drift at most a chunk apart."""
    for i in range(count):
        graph.replay()
        if i + 1 == count or (sync_every and (i + 1) % sync_every == 0):
            barrier()


def _time_graph(
    step: StepFn,
    capture: CaptureFn,
    num_iters: int,
    ingraph_calls: int,
    use_median: bool,
    barrier: BarrierFn,
    sync_every: int,
    num_warmups: int = DEFAULT_NUM_WARMUP,
) -> float:
    """Microseconds per call, from replays of a graph that holds ``ingraph_calls`` calls.

    The calls are recorded back to back, as consecutive layers' all-reduces
    are in a decode graph. ``num_iters`` is the number of timed replays.

    ``barrier`` brings every rank to the same point. It is called the same
    number of times on every rank, so it cannot itself hang.
    """
    import torch

    # Eager warm-up first: it builds kernels and workspaces outside the graph.
    # Inside a capture context a non-captured call is a warm-up that returns
    # zeros, so it would not exercise the kernel.
    for _ in range(num_warmups):
        step()
    barrier()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Exiting ``capture`` registers the captured inputs with the peers. That
    # is a collective, so it runs after the capture ends, on every rank.
    with capture():
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(ingraph_calls):
                step()
    # Registration only agreed on the peers' addresses, it did not wait for
    # them. A rank that replays now hands its kernel an address a peer has yet
    # to publish.
    barrier()

    _replay(graph, num_warmups, barrier, sync_every)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
        # Each event pair spans one replay, so re-syncing between replays
        # stays out of the measurement.
        if sync_every and (i + 1) % sync_every == 0:
            barrier()
    torch.cuda.synchronize()
    samples = (
        starts[i].elapsed_time(ends[i]) * 1000 / ingraph_calls for i in range(num_iters)
    )
    elapsed = statistics.median(samples) if use_median else statistics.mean(samples)
    del graph
    return elapsed


def _time_backend(
    backend: _Backend,
    tensor: "torch.Tensor",
    mode: str,
    num_iters: int,
    ingraph_calls: int,
    use_median: bool,
    barrier: BarrierFn,
    sync_every: int,
    num_warmups: int = DEFAULT_NUM_WARMUP,
) -> float:
    def step() -> "torch.Tensor":
        out = backend.call(tensor)
        if out is None:
            raise _Rejected("the kernel declined the tensor")
        return out

    if mode == "graph":
        return _time_graph(
            step,
            backend.capture,
            num_iters,
            ingraph_calls,
            use_median,
            barrier,
            sync_every,
            num_warmups,
        )
    return _time_eager(
        step, num_iters, use_median, barrier, sync_every, num_warmups
    )


# ---------------------------------------------------------------------------
# Opening backends
# ---------------------------------------------------------------------------


def _reason_from(exc: Exception, what: str) -> str:
    """One line naming the exception, for the skipped column."""
    text = str(exc).splitlines()[0] if str(exc) else ""
    detail = f": {text}" if text else ""
    return f"{what} raised {type(exc).__name__}{detail}"


def _check_backend(backend: _Backend, tensor: "torch.Tensor") -> Optional[str]:
    """Why the tensor cannot run, or None. A raising check counts as a reason.

    A rank that raised here while the others went on into the collectives would
    leave them waiting, so the exception becomes a reason like any other and is
    then agreed across the ranks.
    """
    try:
        return backend.check(tensor)
    except Exception as e:
        return _reason_from(e, "check")


def _agree(ok: bool, cpu_group: "ProcessGroup") -> bool:
    """True only when every rank passes ``ok``.

    A rank that skips a backend while the others enter its collectives leaves
    them waiting forever, so every rank must take the same branch.
    """
    import torch
    import torch.distributed as dist

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(flag.item())


def _shape_reason(
    tensor: "torch.Tensor", supported_dtypes: Sequence["torch.dtype"]
) -> Optional[str]:
    """Why this tensor's dtype or layout is not one the kernel can run."""
    if tensor.dtype not in supported_dtypes:
        return f"dtype {tensor.dtype} is not supported"
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes % 16 != 0:
        return f"{nbytes}B is not a multiple of 16"
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_weak_contiguous,
    )

    if not is_weak_contiguous(tensor):
        return "tensor is not contiguous"
    return None


def _float_dtypes() -> Tuple["torch.dtype", "torch.dtype", "torch.dtype"]:
    import torch

    return (torch.float32, torch.float16, torch.bfloat16)


def _custom_ar_call(
    car: object, tensor: "torch.Tensor", past_cap: bool
) -> Optional["torch.Tensor"]:
    """One call, the way the server runs a custom all-reduce.

    Inside the class's cap that is ``custom_all_reduce``: in a capture context
    on a capturing stream it hands the kernel the input itself, otherwise it
    copies the input into the staging buffer. Past the cap, a class with
    ``_all_reduce_impl`` uses that, and a class with ``all_reduce`` uses that.
    """
    import torch

    custom = cast(_CustomProbe, car)
    if not past_cap:
        return custom.custom_all_reduce(tensor)
    capturing = bool(getattr(car, "_IS_CAPTURING", False)) and (
        torch.cuda.is_current_stream_capturing()
    )
    if hasattr(car, "_all_reduce_impl"):
        return custom._all_reduce_impl(
            tensor, registered=capturing and not bool(getattr(car, "tms_cudagraph", False))
        )
    return custom.all_reduce(
        tensor,
        registered_input=capturing
        and bool(getattr(car, "enable_register_for_capturing", False)),
    )


def _sglang_topology_ok(car: _SglangTopo) -> bool:
    from sglang.srt.utils import is_hip

    if not is_hip():
        return car.world_size == 2 or car.full_nvlink
    return bool(car.use_amd_deterministic_impl or car.full_nvlink)


def _import_class(qualname: str) -> type:
    """Import the class the scan named. The name is ``module.Class``."""
    import importlib

    module_name, _, class_name = qualname.rpartition(".")
    if not module_name or not class_name:
        raise ImportError(qualname)
    return cast(type, getattr(importlib.import_module(module_name), class_name))


def _constructor_parameters(cls: type) -> Dict[str, inspect.Parameter]:
    """Parameters ``cls`` can be called with.

    A subclass such as ``QuickAllReduceInt4`` only names ``algorithm`` and
    ``min_bytes`` and forwards everything else through ``**kwargs`` to
    ``FlyQuickAllReduce``. Those forwarded names are part of the call too.
    """
    merged: Dict[str, inspect.Parameter] = {}
    accepts_var_keyword = False
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            parameters = inspect.signature(init).parameters
        except (TypeError, ValueError):
            continue
        var_keyword = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        )
        if klass is cls:
            merged.update(parameters)
            accepts_var_keyword = var_keyword
            if not var_keyword:
                break
            continue
        if not accepts_var_keyword:
            break
        for name, param in parameters.items():
            if name in merged or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            merged[name] = param
        if not var_keyword:
            break
    return merged


def _construct(
    cls: type,
    cpu_group: "ProcessGroup",
    device: "torch.device",
    algorithm: Optional[str] = None,
) -> object:
    """Build ``cls`` from the group, device, and FlyDSL schedule it asks for."""
    import torch.distributed as dist

    kwargs: Dict[str, Union["ProcessGroup", "torch.device", str, int, bool]] = {}
    for name, param in _constructor_parameters(cls).items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in ("group", "cpu_group"):
            kwargs[name] = cpu_group
        elif name == "algorithm":
            if algorithm is None:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"{cls.__name__} requires {name}")
            else:
                kwargs[name] = algorithm
        elif name == "device":
            kwargs["device"] = device
        elif name == "rank":
            kwargs["rank"] = dist.get_rank(group=cpu_group)
        elif name == "world_size":
            kwargs["world_size"] = dist.get_world_size(group=cpu_group)
        elif name == "enable_register_for_capturing":
            from sglang.srt.environ import envs

            kwargs["enable_register_for_capturing"] = (
                not envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
            )
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f"{cls.__name__} requires {name}")
    return cls(**kwargs)


def _close_comm(comm: object) -> None:
    if comm is not None and hasattr(comm, "close"):
        comm.close()


# A quant-class allreduce accepts any contiguous bf16 payload up to this
# window. The check lives here so a declined size is a skip, not an exception
# in the timed loop.
_FLYDSL_MAX_BYTES = 4 * 1024**3
_FLYDSL_ALIGN = 16
_FLYDSL_WARMUP_BYTES = 1024 * 1024


def _open_scanned(
    qualname: str,
    methods: Sequence[str],
    quant_suffix: str,
    quant: str,
    lift_cap: bool,
    cast_bf16_to_fp16: bool,
    algorithm: str,
    cpu_group: "ProcessGroup",
    device: "torch.device",
) -> Optional[_Backend]:
    """Open the class ``qualname`` and call the collective the scan saw on it."""
    method_names = set(methods)
    comm: Optional[object] = None
    ok = False
    previous: Optional[Dict[str, Optional[str]]] = None
    try:
        cls = _import_class(qualname)
        if "quick_all_reduce" in method_names and not quant_suffix:
            os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = (
                "1" if cast_bf16_to_fp16 else "0"
            )
            previous = {name: os.environ.get(name) for name in _QR_QUANT_ENVS}
            for name in _QR_QUANT_ENVS:
                os.environ[name] = quant
        comm = _construct(cls, cpu_group, device, algorithm=algorithm)
        if quant_suffix and hasattr(comm, "compile"):
            import torch

            warmup = torch.empty(
                _FLYDSL_WARMUP_BYTES // 2, dtype=torch.bfloat16, device=device
            )
            cast(_FlyComm, comm).compile(warmup, torch.empty_like(warmup))
            torch.cuda.synchronize(device)
        if "outplace_all_reduce" in method_names and "custom_all_reduce" not in method_names:
            ok = bool(getattr(comm, "available", False))
        else:
            ok = not getattr(comm, "disabled", False)
    except Exception as e:
        logger.warning("%s failed: %s", qualname, e)
    finally:
        if previous is not None:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    if not _agree(ok, cpu_group) or comm is None:
        _close_comm(comm)
        return None
    return _backend_for(comm, method_names, quant_suffix, lift_cap, qualname)


def _backend_for(
    comm: object,
    methods: set[str],
    quant_suffix: str,
    lift_cap: bool,
    qualname: str,
) -> Optional[_Backend]:
    """Check and call chosen from the methods the scan recorded."""
    capture: CaptureFn = (
        cast(_Capturable, comm).capture if hasattr(comm, "capture") else _no_capture
    )
    close: Callable[[], None] = (
        cast(_Closeable, comm).close if hasattr(comm, "close") else (lambda: None)
    )
    if quant_suffix or (
        "allreduce" in methods and "quick_all_reduce" not in methods
    ):
        return _quant_backend(comm, qualname, capture, close)
    if "quick_all_reduce" in methods:
        return _quick_backend(comm, qualname, capture, close)
    if "custom_all_reduce" in methods:
        return _custom_backend(comm, lift_cap, qualname, capture, close)
    if "outplace_all_reduce" in methods:
        outplace = cast(_OutplaceComm, comm)

        def call_out(tensor: "torch.Tensor") -> Optional["torch.Tensor"]:
            with outplace.change_state(enable=True):
                return outplace.outplace_all_reduce(tensor)

        return _Backend(
            check=lambda tensor: None,
            call=call_out,
            close=close,
            capture=capture,
            impl=qualname,
            method="outplace_all_reduce",
        )
    if "all_reduce" in methods:
        return _all_reduce_backend(comm, qualname, capture, close)
    _close_comm(comm)
    return None


def _custom_backend(
    comm: object,
    lift_cap: bool,
    qualname: str,
    capture: CaptureFn,
    close: Callable[[], None],
) -> _Backend:
    has_past = hasattr(comm, "_all_reduce_impl") or hasattr(comm, "all_reduce")
    if not has_past:
        lift_cap = False
    aiter_past = hasattr(comm, "all_reduce") and not hasattr(comm, "_all_reduce_impl")
    supported = cast(
        Sequence["torch.dtype"],
        getattr(comm, "_SUPPORTED_DTYPES", None) or _float_dtypes(),
    )
    probe = cast(_CustomProbe, comm)
    past_cap_limit = 0
    if has_past:
        past_cap_limit = int(probe.max_size) // 2 if aiter_past else int(probe.max_size)
    past_cap: Dict[int, bool] = {}

    def check(tensor: "torch.Tensor") -> Optional[str]:
        if has_past:
            reason = _shape_reason(tensor, supported)
            if reason:
                return reason
            if not aiter_past and not _sglang_topology_ok(cast(_SglangTopo, comm)):
                return "not supported on this link topology"
        nbytes = tensor.numel() * tensor.element_size()
        if probe.should_custom_ar(tensor):
            past_cap[nbytes] = False
            return None
        if not lift_cap:
            return f"{nbytes}B was declined by should_custom_ar"
        if aiter_past and not (
            probe.world_size == 2 or bool(getattr(comm, "fully_connected", False))
        ):
            return "not supported on this link topology"
        if nbytes > past_cap_limit:
            return f"{nbytes}B does not fit the {past_cap_limit}B buffer"
        past_cap[nbytes] = True
        return None

    def call(tensor: "torch.Tensor") -> Optional["torch.Tensor"]:
        nbytes = tensor.numel() * tensor.element_size()
        return _custom_ar_call(comm, tensor, past_cap.get(nbytes, False))

    note = ""
    if lift_cap:
        note = f"cap lifted up to the {past_cap_limit}B buffer"
    return _Backend(
        check=check,
        call=call,
        close=close,
        capture=capture,
        impl=qualname,
        note=note,
        method="custom_all_reduce",
    )


def _quick_backend(
    comm: object, qualname: str, capture: CaptureFn, close: Callable[[], None]
) -> _Backend:
    quick = cast(_QuickComm, comm)

    def check(tensor: "torch.Tensor") -> Optional[str]:
        if not quick.should_quick_allreduce(tensor):
            reason = _shape_reason(tensor, quick._SUPPORTED_DTYPES)
            nbytes = tensor.numel() * tensor.element_size()
            return reason or f"{nbytes}B is outside the QuickReduce size window"
        return None

    return _Backend(
        check=check,
        call=quick.quick_all_reduce,
        close=close,
        capture=capture,
        impl=qualname,
        method="quick_all_reduce",
    )


def _quant_backend(
    comm: object, qualname: str, capture: CaptureFn, close: Callable[[], None]
) -> _Backend:
    import torch

    floor = int(getattr(comm, "min_bytes", 0))
    fly = cast(_FlyComm, comm)

    def prepare(sizes: List[int]) -> None:
        if not hasattr(comm, "preload"):
            return
        usable = [int(size) for size in sizes if int(size) >= floor]
        if not usable:
            return
        fly.preload(payload_range=(min(usable), max(usable)))
        torch.cuda.synchronize(fly.device if hasattr(comm, "device") else None)

    def check(tensor: "torch.Tensor") -> Optional[str]:
        if tensor.dtype != torch.bfloat16:
            return "quant allreduce supports bf16"
        if not tensor.is_contiguous():
            return "quant allreduce requires a contiguous tensor"
        if tensor.data_ptr() % _FLYDSL_ALIGN:
            return "quant allreduce requires 16-byte alignment"
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes % _FLYDSL_ALIGN:
            return f"{nbytes}B is not a multiple of {_FLYDSL_ALIGN}"
        if nbytes < floor:
            return f"{nbytes}B is below the {floor}B floor"
        if nbytes > _FLYDSL_MAX_BYTES:
            return f"{nbytes}B exceeds the 4GiB window"
        return None

    def call(tensor: "torch.Tensor") -> "torch.Tensor":
        out = torch.empty_like(tensor)
        fly.allreduce(tensor, out)
        return out

    return _Backend(
        check=check,
        call=call,
        close=close,
        capture=capture,
        prepare=prepare,
        impl=qualname,
        method="allreduce",
    )


def _all_reduce_backend(
    comm: object, qualname: str, capture: CaptureFn, close: Callable[[], None]
) -> _Backend:
    def check(tensor: "torch.Tensor") -> Optional[str]:
        if hasattr(comm, "should_torch_symm_mem_allreduce"):
            symm = cast(_SymmComm, comm)
            if not symm.should_torch_symm_mem_allreduce(tensor):
                return _shape_reason(tensor, (symm.dtype,)) or "tensor was declined"
            return None
        plain = cast(_AllReduceCall, comm)
        if hasattr(comm, "should_custom_ar") and not plain.should_custom_ar(tensor):
            nbytes = tensor.numel() * tensor.element_size()
            return f"{nbytes}B was declined by should_custom_ar"
        return None

    return _Backend(
        check=check,
        call=cast(_AllReduceCall, comm).all_reduce,
        close=close,
        capture=capture,
        impl=qualname,
        method="all_reduce",
    )


def _cpu_group(group: "ProcessGroup") -> "ProcessGroup":
    """Custom all-reduce, QuickReduce and PyNccl attach to a gloo group, as in the server."""
    import torch.distributed as dist

    if dist.get_backend(group) != dist.Backend.NCCL:
        return group
    return dist.new_group(backend="gloo")


_OPENERS = {"scanned": _open_scanned}


def _open_backends(
    bound: Dict[str, ReadySpec],
    verbose: bool,
) -> Dict[str, _Backend]:
    """Open every backend once, in the same order on every rank.

    ``bound`` maps the table name to its opener and kwargs. Each torchrun
    process is one GPU and calls this once; the ranks pass the same map, so
    they open the same communicators.
    """
    backends: Dict[str, _Backend] = {}
    for name, spec in bound.items():
        if verbose:
            print(f"  opening {name}", flush=True)
        try:
            opener = _OPENERS[spec["opener"]]
        except KeyError:
            raise ValueError(
                f"{name}: selection opener {spec['opener']!r} has no tuner "
                "implementation"
            ) from None
        params = spec["params"]
        opened = opener(
            qualname=params["qualname"],
            methods=params["methods"],
            quant_suffix=params["quant_suffix"],
            quant=params["quant"],
            lift_cap=params["lift_cap"],
            cast_bf16_to_fp16=params["cast_bf16_to_fp16"],
            algorithm=params["algorithm"],
            cpu_group=params["cpu_group"],
            device=params["device"],
        )
        if opened is None:
            if verbose:
                print(f"  {name}: unavailable", flush=True)
            continue
        backends[name] = opened
        if verbose:
            print(f"  {name}: ready", flush=True)
            if opened.note:
                print(f"  {opened.note}", flush=True)
    return backends


# ---------------------------------------------------------------------------
# Tuning loop
# ---------------------------------------------------------------------------


def tune_allreduce(
    sizes: List[int],
    tp_size: int,
    mode: str = "eager",
    group: Optional["ProcessGroup"] = None,
    num_iters: int = 100,
    ingraph_calls: int = 10,
    verbose: bool = True,
    qr_quant: str = "INT4",
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    fly_algorithm: str = "mesh",
    use_median: bool = False,
    sync_every: int = _SYNC_EVERY,
    aiter_dir: str = DEFAULT_AITER_DIR,
    sglang_dir: str = DEFAULT_SGLANG_DIR,
    num_warmups: int = DEFAULT_NUM_WARMUP,
    skip: Sequence[str] = (),
    scandirs: Sequence[str] = (),
) -> Tuple[List[AllreduceConfig], List[_Measurement]]:
    """Time each backend at each size in ``mode`` and return the fastest per size.

    The AITER tree at ``aiter_dir`` and the SGLang tree at ``sglang_dir`` are
    parsed for their communicator classes, in ``scandirs`` or, when that is
    empty, in ``DEFAULT_SCAN_DIRS``. A class is timed when this tuner
    can construct it. HIP QuickReduce and FlyDSL are timed only when
    ``qr_quant`` is declared in AITER.
    """
    import torch
    import torch.distributed as dist

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if fly_algorithm not in ("mesh", "ring"):
        raise ValueError(
            f"fly_algorithm must be 'mesh' or 'ring', got {fly_algorithm!r}"
        )
    if num_warmups < 0:
        raise ValueError(f"num_warmups must be 0 or more, got {num_warmups}")
    if not dist.is_initialized():
        logger.warning(
            "torch.distributed is not initialized, so all-reduce tuning was "
            "skipped. Launch one process per GPU with torchrun."
        )
        return [], []

    if group is None:
        group = dist.group.WORLD
    assert group is not None
    world_size = dist.get_world_size(group)
    if world_size != tp_size:
        logger.warning("World size %d != TP size %d; using %d", world_size, tp_size, world_size)
        tp_size = world_size

    cpu_group = _cpu_group(group)
    device = torch.device("cuda", torch.cuda.current_device())
    sizes = _normalize_sizes(sizes)
    scan = scan_aiter(aiter_dir, scandirs)
    sglang_scan = scan_sglang(sglang_dir, scandirs)
    plan, skipped, quant_left_out = backend_plan(
        scan,
        sglang_scan,
        qr_quant,
        ignore_acr_max_size=ignore_acr_max_size,
        ignore_scr_max_size=ignore_scr_max_size,
        qr_cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
        skip=skip,
    )
    if verbose:
        print("Scan directories:", flush=True)
        for entry in scan_dirs(scandirs):
            print(f"  {entry}", flush=True)
        print(format_scan(scan, sglang_scan, qr_quant), flush=True)
        print(f"FlyDSL algorithm: {fly_algorithm}", flush=True)
        print("Opened from the scan:", flush=True)
        if plan:
            for name in plan:
                print(f"  {name}", flush=True)
        else:
            print("  (none)", flush=True)
        if quant_left_out:
            print("Left out by quant:", flush=True)
            for name in quant_left_out:
                print(f"  {name}", flush=True)
        if skipped:
            print("Not opened:", flush=True)
            for name in skipped:
                print(f"  {name}", flush=True)
        unknown = unmatched_skips(
            [item.qualname for item in (*scan.found, *sglang_scan.found)],
            skip,
        )
        if unknown:
            print("Skip names matched no scanned class:", flush=True)
            for name in unknown:
                print(f"  {name}", flush=True)
    ready: Dict[str, ReadySpec] = {}
    for name, spec in plan.items():
        params = spec["params"]
        ready[name] = {
            "opener": spec["opener"],
            "params": {
                "qualname": params["qualname"],
                "methods": params["methods"],
                "quant_suffix": params["quant_suffix"],
                "quant": params["quant"],
                "lift_cap": params["lift_cap"],
                "cast_bf16_to_fp16": params["cast_bf16_to_fp16"],
                "cpu_group": cpu_group,
                "device": device,
                "algorithm": fly_algorithm,
            },
        }
    backends = _open_backends(ready, verbose)
    for backend in backends.values():
        if backend.prepare is not None:
            backend.prepare(sizes)
    dtype = torch.bfloat16
    if verbose:
        print(
            "  Backends: "
            + ", ".join(backend.impl or name for name, backend in backends.items()),
            flush=True,
        )
        print(f"  Mode: {mode}", flush=True)
        stat = "median" if use_median else "mean"
        if mode == "graph":
            print(
                f"  Timing: {stat} of {num_iters} replays of a graph with "
                f"{ingraph_calls} captured calls, per call.",
                flush=True,
            )
        else:
            print(
                f"  Timing: {stat} of {num_iters} eager calls after "
                f"{num_warmups} warm-up calls.",
                flush=True,
            )
        print(
            f"  ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16={int(qr_cast_bf16_to_fp16)}",
            flush=True,
        )
        print(f"  Sizes: {sizes}", flush=True)

    results: List[AllreduceConfig] = []
    measurements: List[_Measurement] = []
    total_start = time.perf_counter()

    def barrier() -> None:
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

    try:
        for idx, size_bytes in enumerate(sizes):
            tag = f"  [{idx + 1}/{len(sizes)}] {size_bytes}B"
            tensor = torch.randn(size_bytes // 2, dtype=dtype, device="cuda")

            best_backend: Optional[str] = None
            best_method = ""
            best_time = float("inf")
            row_times: Dict[str, Optional[float]] = {}
            for name, backend in backends.items():
                class_name = backend.impl or name
                barrier()
                reason = _check_backend(backend, tensor)
                # Every rank must agree before any rank enters the collectives.
                t: Optional[float]
                if not _agree(reason is None, cpu_group):
                    reason = reason or "another rank declined the tensor"
                    t = None
                else:
                    try:
                        t = _time_backend(
                            backend,
                            tensor,
                            mode,
                            num_iters,
                            ingraph_calls,
                            use_median,
                            barrier,
                            sync_every,
                            num_warmups,
                        )
                    except _Rejected as e:
                        t = None
                        reason = str(e)
                    except Exception as e:
                        t = None
                        reason = _reason_from(e, "timing")
                barrier()
                if verbose:
                    shown = f"{t:.2f} us" if t is not None else f"skipped: {reason}"
                    print(f"{tag}: {class_name} -> {shown}", flush=True)
                row_times[class_name] = t
                if t is not None and t < best_time:
                    best_time, best_backend, best_method = t, class_name, backend.method

            measurements.append({"size_bytes": size_bytes, "times_us": row_times})
            if best_backend is not None:
                results.append(
                    AllreduceConfig(
                        size_bytes=size_bytes,
                        backend=best_backend,
                        time_us=best_time,
                        tp_size=tp_size,
                        dtype=str(dtype),
                        method=best_method,
                    )
                )
                if verbose:
                    print(
                        f"{tag}: best {best_backend} {best_time:8.2f} us",
                        flush=True,
                    )
            del tensor
            _reclaim_size_cache()
            barrier()
    finally:
        # Close together: a rank that frees its IPC buffers early can leave a
        # peer still using them.
        barrier()
        for name, backend in backends.items():
            try:
                backend.close()
            except Exception as e:
                logger.warning(
                    "closing %s raised %s: %s",
                    name,
                    type(e).__name__,
                    e,
                )

    if verbose:
        print(
            f"\n  All-reduce tuning ({mode}) finished in "
            f"{time.perf_counter() - total_start:.2f}s, "
            f"{len(results)}/{len(sizes)} sizes",
            flush=True,
        )
    return results, measurements


def run_allreduce_tuning(
    sizes: List[int],
    tp_size: int,
    mode: str = "eager",
    group: Optional["ProcessGroup"] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    qr_quant: str = "INT4",
    num_iters: int = 100,
    ingraph_calls: int = 10,
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    fly_algorithm: str = "mesh",
    use_median: bool = False,
    sync_every: int = _SYNC_EVERY,
    aiter_dir: str = DEFAULT_AITER_DIR,
    sglang_dir: str = DEFAULT_SGLANG_DIR,
    num_warmups: int = DEFAULT_NUM_WARMUP,
    skip: Sequence[str] = (),
    scandirs: Sequence[str] = (),
) -> AllreduceTuneResult:
    """Tune, then write ``allreduce_configs_<mode>.json`` into ``output_dir``."""
    if verbose:
        print(f"\n{'=' * 60}\nAll-reduce tuning, {mode} mode\n{'=' * 60}")
        print(f"  TP: {tp_size}")

    results, measurements = tune_allreduce(
        sizes,
        tp_size,
        mode=mode,
        group=group,
        num_iters=num_iters,
        ingraph_calls=ingraph_calls,
        verbose=verbose,
        qr_quant=qr_quant,
        ignore_acr_max_size=ignore_acr_max_size,
        ignore_scr_max_size=ignore_scr_max_size,
        qr_cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
        fly_algorithm=fly_algorithm,
        use_median=use_median,
        sync_every=sync_every,
        aiter_dir=aiter_dir,
        sglang_dir=sglang_dir,
        num_warmups=num_warmups,
        skip=skip,
        scandirs=scandirs,
    )
    if not results:
        if verbose:
            print("  No results.")
        return {}

    output: AllreduceTuneResult = {
        "mode": mode,
        "env": "SGLANG_TUNED_GRAPH_CONFIG" if mode == "graph" else "SGLANG_TUNED_EAGER_CONFIG",
        "tp_size": tp_size,
        "qr_quant": qr_quant,
        "qr_cast_bf16_to_fp16": int(qr_cast_bf16_to_fp16),
        "fly_algorithm": fly_algorithm,
        "num_warmups": int(num_warmups),
        "ignore_acr_max_size": bool(ignore_acr_max_size),
        "ignore_scr_max_size": bool(ignore_scr_max_size),
        "num_iters": int(num_iters),
        "ingraph_calls": int(ingraph_calls) if mode == "graph" else None,
        "skip": list(skip),
        "scandirs": list(scan_dirs(scandirs)),
        "timing": "median" if use_median else "mean",
        "configs": [r.to_dict() for r in results],
        "measurements": measurements,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"allreduce_configs_{mode}.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        output["path"] = path
        if verbose:
            print(f"  Saved to: {path}")
    return output


def _print_comparison(measurements: List[_Measurement], timing: str) -> None:
    """One row per size, one column per backend. A star marks the fastest."""
    names: List[str] = []
    for row in measurements:
        for name in row["times_us"]:
            if name not in names:
                names.append(name)
    if not names:
        return
    shown = [f"{source_name(name)} {short_name(name)}" for name in names]
    width = max(12, *(len(name) for name in shown))
    print(
        f"\nAll-reduce comparison, microseconds per call ({timing}). "
        "skip means the kernel declined the tensor."
    )
    header = f"{'size':>10}  " + "  ".join(f"{name:<{width}}" for name in shown)
    print(header)
    print("-" * len(header))
    for row in measurements:
        times = row["times_us"]
        ranked = [value for value in times.values() if value is not None]
        best = min(ranked) if ranked else None
        cells = []
        for name in names:
            value = times.get(name)
            if value is None:
                cells.append(f"{'skip':<{width}}")
                continue
            text = f"{value:.2f}" + ("*" if value == best else "")
            cells.append(f"{text:<{width}}")
        print(f"{row['size_bytes']:>10}  " + "  ".join(cells))


def print_allreduce_summary(results: AllreduceTuneResult) -> None:
    configs = results.get("configs") or []
    if not configs:
        print("  No all-reduce configs to summarize.")
        return
    mode = results.get("mode", "eager")
    timing = results.get("timing", "mean")
    _print_comparison(results.get("measurements") or [], f"{mode}, {timing}")
    print(f"\n{'=' * 60}\nAll-reduce tuning summary ({mode}, {timing})\n{'=' * 60}")
    for c in configs:
        print(
            f"  {c['size_bytes']:>10}B  {c['backend']}  {c['time_us']:8.2f} us"
        )
    print("=" * 60)
    if results.get("path"):
        print(f"  Use it with {results['env']}={results['path']}")
