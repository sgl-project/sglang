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

Which backends compete comes from the selection ``generate_selection_config``
writes. This module opens each row of that table and does not keep its own
copy of the list.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import statistics
import time
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple

from sglang.auto_tune.generate_selection_config import (
    QR_QUANT_REGIMES,
    build_selection,
    load_selection,
    selection_open_params,
)

logger = logging.getLogger(__name__)

_QR_QUANT_ENVS = (
    "ROCM_QUICK_REDUCE_QUANTIZATION",
    "AITER_QUICK_REDUCE_QUANTIZATION",
)

MODES = ("eager", "graph")

_NUM_WARMUP = 20

# The kernels spin-wait on their peers, so a call only completes once every
# rank runs it. Nothing inside a timing loop keeps the ranks together, so they
# are brought back in step every this many calls or replays.
_SYNC_EVERY = 10


@dataclasses.dataclass
class AllreduceConfig:
    """Fastest backend at one message size."""

    size_bytes: int
    backend: str
    time_us: float
    tp_size: int
    dtype: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


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

    check: Callable[[Any], Optional[str]]
    call: Callable[[Any], Any]
    close: Callable[[], None]
    capture: Callable[[], ContextManager] = contextlib.nullcontext
    prepare: Optional[Callable[[List[int]], None]] = None
    impl: str = ""
    note: str = ""


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
    previous = None
    for _ in range(tries):
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved()
        if reserved <= torch.cuda.memory_allocated() or reserved == previous:
            return
        previous = reserved
        torch.cuda.synchronize()


def _time_eager(
    step,
    num_iters: int,
    use_median: bool,
    barrier: Callable[[], None],
    sync_every: int,
) -> float:
    """Microseconds per call, launched from Python one call at a time."""
    import torch

    for _ in range(_NUM_WARMUP):
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
    graph, count: int, barrier: Callable[[], None], sync_every: int
) -> None:
    """Replay ``count`` times, letting the ranks drift at most a chunk apart."""
    for i in range(count):
        graph.replay()
        if i + 1 == count or (sync_every and (i + 1) % sync_every == 0):
            barrier()


def _time_graph(
    step,
    capture: Callable[[], ContextManager],
    num_iters: int,
    ingraph_calls: int,
    use_median: bool,
    barrier: Callable[[], None],
    sync_every: int,
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
    for _ in range(_NUM_WARMUP):
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

    _replay(graph, _NUM_WARMUP, barrier, sync_every)
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
    tensor,
    mode: str,
    num_iters: int,
    ingraph_calls: int,
    use_median: bool,
    barrier: Callable[[], None],
    sync_every: int,
) -> float:
    def step():
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
        )
    return _time_eager(step, num_iters, use_median, barrier, sync_every)


# ---------------------------------------------------------------------------
# Opening backends
# ---------------------------------------------------------------------------


def _reason_from(exc: Exception, what: str) -> str:
    """One line naming the exception, for the skipped column."""
    text = str(exc).splitlines()[0] if str(exc) else ""
    detail = f": {text}" if text else ""
    return f"{what} raised {type(exc).__name__}{detail}"


def _check_backend(backend: _Backend, tensor) -> Optional[str]:
    """Why the tensor cannot run, or None. A raising check counts as a reason.

    A rank that raised here while the others went on into the collectives would
    leave them waiting, so the exception becomes a reason like any other and is
    then agreed across the ranks.
    """
    try:
        return backend.check(tensor)
    except Exception as e:
        return _reason_from(e, "check")


def _agree(ok: bool, cpu_group) -> bool:
    """True only when every rank passes ``ok``.

    A rank that skips a backend while the others enter its collectives leaves
    them waiting forever, so every rank must take the same branch.
    """
    import torch
    import torch.distributed as dist

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(flag.item())


def _shape_reason(tensor, supported_dtypes) -> Optional[str]:
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


def _float_dtypes():
    import torch

    return (torch.float32, torch.float16, torch.bfloat16)


def _qualname(cls) -> str:
    func = getattr(cls, "func", cls)
    return f"{getattr(func, '__module__', '')}.{getattr(func, '__name__', '')}"


def _custom_ar_call(car, kind: str, tensor, past_cap: bool):
    """One call, the way ``GroupCoordinator._tuned_custom_all_reduce`` makes it.

    Inside the class's cap that is ``custom_all_reduce``: in a capture context
    on a capturing stream it hands the kernel the input itself, otherwise it
    copies the input into the staging buffer. Past the cap the same rule is
    applied without the size test.
    """
    import torch

    if not past_cap:
        return car.custom_all_reduce(tensor)
    capturing = getattr(car, "_IS_CAPTURING", False) and (
        torch.cuda.is_current_stream_capturing()
    )
    if kind == "aiter":
        return car.all_reduce(
            tensor,
            registered_input=capturing
            and getattr(car, "enable_register_for_capturing", False),
        )
    return car._all_reduce_impl(
        tensor, registered=capturing and not getattr(car, "tms_cudagraph", False)
    )


def _sglang_topology_ok(car) -> bool:
    from sglang.srt.utils import is_hip

    if not is_hip():
        return car.world_size == 2 or car.full_nvlink
    return bool(car.use_amd_deterministic_impl or car.full_nvlink)


def _open_custom_allreduce(cpu_group, device, kind: str, lift_cap: bool):
    """SGLang's (``kind="sglang"``) or AITER's (``kind="aiter"``) CustomAllreduce."""
    car = None
    ok = False
    impl = ""
    try:
        if kind == "aiter":
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            from sglang.srt.environ import envs

            impl = _qualname(AiterCustomAllreduce)
            # Same capture setting the server builds it with.
            car = AiterCustomAllreduce(
                group=cpu_group,
                device=device,
                enable_register_for_capturing=(
                    not envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
                ),
            )
        else:
            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            impl = _qualname(CustomAllreduce)
            car = CustomAllreduce(cpu_group, device)
        ok = not getattr(car, "disabled", True)
    except Exception as e:
        logger.warning("%s custom all-reduce failed: %s", kind, e)
    if not _agree(ok, cpu_group):
        if car is not None and hasattr(car, "close"):
            car.close()
        return None

    supported = getattr(car, "_SUPPORTED_DTYPES", None) or _float_dtypes()
    # AITER's two-stage kernel needs twice the input in its pool.
    past_cap_limit = int(car.max_size) // 2 if kind == "aiter" else int(car.max_size)
    past_cap: Dict[int, bool] = {}

    def check(tensor) -> Optional[str]:
        reason = _shape_reason(tensor, supported)
        if reason:
            return reason
        if kind == "sglang" and not _sglang_topology_ok(car):
            return "not supported on this link topology"
        nbytes = tensor.numel() * tensor.element_size()
        if car.should_custom_ar(tensor):
            past_cap[nbytes] = False
            return None
        if not lift_cap:
            return f"{nbytes}B was declined by should_custom_ar"
        if kind == "aiter" and not (
            car.world_size == 2 or getattr(car, "fully_connected", False)
        ):
            return "not supported on this link topology"
        if nbytes > past_cap_limit:
            return f"{nbytes}B does not fit the {past_cap_limit}B buffer"
        past_cap[nbytes] = True
        return None

    def call(tensor):
        nbytes = tensor.numel() * tensor.element_size()
        return _custom_ar_call(car, kind, tensor, past_cap.get(nbytes, False))

    note = ""
    if lift_cap:
        note = f"{kind} cap lifted up to the {past_cap_limit}B buffer"
    return _Backend(
        check=check,
        call=call,
        close=car.close if hasattr(car, "close") else (lambda: None),
        capture=car.capture,
        impl=impl,
        note=note,
    )


def _open_car_v2(cpu_group, device):
    """SGLang's CustomAllReduceV2. A size its heuristic has no plan for is skipped."""
    car = None
    ok = False
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        car = CustomAllReduceV2(group=cpu_group, device=device)
        ok = not getattr(car, "disabled", True)
    except Exception as e:
        logger.warning("custom_allreduce_v2 failed: %s", e)
    if not _agree(ok, cpu_group):
        if car is not None and hasattr(car, "close"):
            car.close()
        return None

    def check(tensor) -> Optional[str]:
        if not car.should_custom_ar(tensor):
            nbytes = tensor.numel() * tensor.element_size()
            return f"{nbytes}B has no custom all-reduce v2 plan"
        return None

    return _Backend(
        check=check,
        call=car.custom_all_reduce,
        close=car.close if hasattr(car, "close") else (lambda: None),
        capture=car.capture,
        impl=_qualname(type(car)),
    )


def _open_nccl(cpu_group, device):
    """PyNccl out-of-place all-reduce, as the server runs an ``nccl`` row."""
    comm = None
    ok = False
    try:
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        comm = PyNcclCommunicator(group=cpu_group, device=device)
        ok = bool(comm.available)
    except Exception as e:
        logger.warning("nccl (PyNccl) failed: %s", e)
    if not _agree(ok, cpu_group):
        return None

    def call(tensor):
        with comm.change_state(enable=True):
            return comm.outplace_all_reduce(tensor)

    return _Backend(
        check=lambda tensor: None,
        call=call,
        close=lambda: None,
        impl=_qualname(type(comm)),
    )


def _open_torch_symm_mem(cpu_group, device):
    comm = None
    ok = False
    try:
        from sglang.srt.distributed.device_communicators.torch_symm_mem import (
            TorchSymmMemCommunicator,
        )

        comm = TorchSymmMemCommunicator(cpu_group, device)
        ok = not comm.disabled
    except Exception as e:
        logger.warning("torch_symm_mem failed: %s", e)
    if not _agree(ok, cpu_group):
        return None

    def check(tensor) -> Optional[str]:
        if not comm.should_torch_symm_mem_allreduce(tensor):
            return _shape_reason(tensor, (comm.dtype,)) or "tensor was declined"
        return None

    return _Backend(check=check, call=comm.all_reduce, close=lambda: None)


def _open_quick_allreduce(quant, cpu_group, device, cast_bf16_to_fp16: bool):
    """QuickReduce at one codec. The codec and the cast are read at construction.

    With ``cast_bf16_to_fp16`` the kernel converts bf16 to fp16 on load and
    back on store, so the timed call includes that conversion.
    """
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = (
        "1" if cast_bf16_to_fp16 else "0"
    )
    previous = {name: os.environ.get(name) for name in _QR_QUANT_ENVS}
    for name in _QR_QUANT_ENVS:
        os.environ[name] = quant
    qr = None
    ok = False
    try:
        from sglang.srt.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qr = QuickAllReduce(cpu_group, device)
        ok = not qr.disabled
    except Exception as e:
        logger.warning("quick_allreduce %s failed: %s", quant, e)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if not _agree(ok, cpu_group):
        if qr is not None:
            qr.close()
        return None

    def check(tensor) -> Optional[str]:
        # The server applies QuickReduce's own window; so does the tuner.
        if not qr.should_quick_allreduce(tensor):
            reason = _shape_reason(tensor, qr._SUPPORTED_DTYPES)
            nbytes = tensor.numel() * tensor.element_size()
            return reason or f"{nbytes}B is outside the QuickReduce size window"
        return None

    # QuickReduce works out of a static IPC buffer and has no capture context.
    return _Backend(
        check=check,
        call=qr.quick_all_reduce,
        close=qr.close,
        impl=_qualname(type(qr)),
    )


# FlyDSL INT4 accepts any contiguous bf16 payload up to this window. The
# check lives here so a declined size is a skip, not an exception in the
# timed loop. This is a different workspace from HIP QuickReduce.
_FLYDSL_MAX_BYTES = 4 * 1024**3
_FLYDSL_ALIGN = 16
_FLYDSL_WARMUP_BYTES = 1024 * 1024


def _open_quick_allreduce_flydsl(cpu_group, device):
    """AITER ``QuickAllReduceInt4``, separate from the HIP QuickReduce row.

    The HIP row opens ``sglang``'s ``QuickAllReduce``. This row opens
    ``aiter.ops.flydsl.QuickAllReduceInt4`` on its own IPC workspace, so a
    bf16 payload is timed on FlyDSL without taking the HIP dispatch. The
    constructor exchanges IPC handles on ``cpu_group``, so every rank calls
    it. The current class JIT-compiles through ``preload``; the older
    ``QuickAllReduceInt4`` used ``compile``. Either one runs here, before
    the timed loop and before a graph capture.
    """
    import torch
    import torch.distributed as dist

    comm = None
    ok = False
    try:
        from aiter.ops.flydsl import QuickAllReduceInt4

        comm = QuickAllReduceInt4(
            group=cpu_group,
            device=device,
            rank=dist.get_rank(group=cpu_group),
            world_size=dist.get_world_size(group=cpu_group),
        )
        # Older QuickAllReduceInt4. Current AITER main aliases that name to
        # FlyQuickAllReduce, which has preload instead.
        if hasattr(comm, "compile"):
            warmup = torch.empty(
                _FLYDSL_WARMUP_BYTES // 2, dtype=torch.bfloat16, device=device
            )
            comm.compile(warmup, torch.empty_like(warmup))
            torch.cuda.synchronize(device)
        ok = True
    except Exception as e:
        logger.warning("quick_allreduce flydsl int4 failed: %s", e)
    if not _agree(ok, cpu_group):
        if comm is not None:
            comm.close()
        return None

    floor = int(getattr(comm, "min_bytes", 0))

    def prepare(sizes: List[int]) -> None:
        if not hasattr(comm, "preload"):
            return
        usable = [int(size) for size in sizes if int(size) >= floor]
        if not usable:
            return
        comm.preload(payload_range=(min(usable), max(usable)))
        torch.cuda.synchronize(device)

    def check(tensor) -> Optional[str]:
        if tensor.dtype != torch.bfloat16:
            return "FlyDSL QuickReduce supports bf16"
        if not tensor.is_contiguous():
            return "FlyDSL QuickReduce requires a contiguous tensor"
        if tensor.data_ptr() % _FLYDSL_ALIGN:
            return "FlyDSL QuickReduce requires 16-byte alignment"
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes % _FLYDSL_ALIGN:
            return f"{nbytes}B is not a multiple of {_FLYDSL_ALIGN}"
        if nbytes < floor:
            return f"{nbytes}B is below the {floor}B FlyDSL floor"
        if nbytes > _FLYDSL_MAX_BYTES:
            return f"{nbytes}B exceeds the FlyDSL 4GiB window"
        return None

    def call(tensor):
        out = torch.empty_like(tensor)
        comm.allreduce(tensor, out)
        return out

    return _Backend(
        check=check,
        call=call,
        close=comm.close,
        prepare=prepare,
        impl=_qualname(type(comm)),
        note="flydsl int4",
    )


def _cpu_group(group):
    """Custom all-reduce, QuickReduce and PyNccl attach to a gloo group, as in the server."""
    import torch.distributed as dist

    if dist.get_backend(group) != dist.Backend.NCCL:
        return group
    return dist.new_group(backend="gloo")


_OPENERS = {
    "custom_allreduce": _open_custom_allreduce,
    "custom_allreduce_v2": _open_car_v2,
    "nccl": _open_nccl,
    "torch_symm_mem": _open_torch_symm_mem,
    "quick_allreduce": _open_quick_allreduce,
    "quick_allreduce_flydsl": _open_quick_allreduce_flydsl,
}


def _open_backends(
    bound: Dict[str, Dict[str, Any]],
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
        opened = opener(**spec["params"])
        if opened is None:
            if verbose:
                print(f"  {name}: unavailable", flush=True)
            continue
        backends[name] = opened
        if verbose:
            suffix = f" ({opened.impl})" if opened.impl else ""
            print(f"  {name}: ready{suffix}", flush=True)
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
    group=None,
    num_iters: int = 100,
    ingraph_calls: int = 10,
    verbose: bool = True,
    qr_quant: str = "INT4",
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    use_median: bool = False,
    sync_every: int = _SYNC_EVERY,
    selection_config: Optional[str] = None,
) -> Tuple[List[AllreduceConfig], List[Dict[str, Any]]]:
    """Time each backend at each size in ``mode`` and return the fastest per size.

    ``selection_config`` is a JSON file from ``generate_selection_config``.
    Without it, the selection is built from the QuickReduce and cap flags.
    """
    import torch
    import torch.distributed as dist

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not dist.is_initialized():
        logger.warning(
            "torch.distributed is not initialized, so all-reduce tuning was "
            "skipped. Launch one process per GPU with torchrun."
        )
        return [], []

    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size != tp_size:
        logger.warning("World size %d != TP size %d; using %d", world_size, tp_size, world_size)
        tp_size = world_size

    cpu_group = _cpu_group(group)
    device = torch.device("cuda", torch.cuda.current_device())
    sizes = _normalize_sizes(sizes)
    if selection_config:
        selection = load_selection(selection_config)
    else:
        selection = build_selection(
            qr_quant,
            ignore_acr_max_size=ignore_acr_max_size,
            ignore_scr_max_size=ignore_scr_max_size,
            qr_cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
        )
    backends = _open_backends(
        selection_open_params(selection, cpu_group, device), verbose
    )
    for backend in backends.values():
        if backend.prepare is not None:
            backend.prepare(sizes)
    dtype = torch.bfloat16
    if verbose:
        if selection_config:
            print(f"  Selection file: {selection_config}", flush=True)
        print(f"  Backends: {', '.join(backends)}", flush=True)
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
                f"{_NUM_WARMUP} warm-up calls.",
                flush=True,
            )
        print(
            f"  ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16={int(qr_cast_bf16_to_fp16)}",
            flush=True,
        )
        print(f"  Sizes: {sizes}", flush=True)

    results: List[AllreduceConfig] = []
    measurements: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

    try:
        for idx, size_bytes in enumerate(sizes):
            tag = f"  [{idx + 1}/{len(sizes)}] {size_bytes}B"
            tensor = torch.randn(size_bytes // 2, dtype=dtype, device="cuda")

            best_backend = None
            best_time = float("inf")
            row_times: Dict[str, Optional[float]] = {}
            for name, backend in backends.items():
                barrier()
                reason = _check_backend(backend, tensor)
                # Every rank must agree before any rank enters the collectives.
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
                    print(f"{tag}: {name} -> {shown}", flush=True)
                row_times[name] = t
                if t is not None and t < best_time:
                    best_time, best_backend = t, name

            measurements.append({"size_bytes": size_bytes, "times_us": row_times})
            if best_backend is not None:
                results.append(
                    AllreduceConfig(
                        size_bytes=size_bytes,
                        backend=best_backend,
                        time_us=best_time,
                        tp_size=tp_size,
                        dtype=str(dtype),
                    )
                )
                if verbose:
                    print(f"{tag}: best {best_backend:24s} {best_time:8.2f} us", flush=True)
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
                logger.warning("closing %s raised %s: %s", name, type(e).__name__, e)

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
    group=None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    qr_quant: str = "INT4",
    num_iters: int = 100,
    ingraph_calls: int = 10,
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    use_median: bool = False,
    sync_every: int = _SYNC_EVERY,
    selection_config: Optional[str] = None,
) -> Dict[str, Any]:
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
        use_median=use_median,
        sync_every=sync_every,
        selection_config=selection_config,
    )
    if not results:
        if verbose:
            print("  No results.")
        return {}

    output = {
        "mode": mode,
        "env": "SGLANG_TUNED_GRAPH_CONFIG" if mode == "graph" else "SGLANG_TUNED_EAGER_CONFIG",
        "tp_size": tp_size,
        "qr_quant": qr_quant,
        "selection_config": selection_config,
        "qr_cast_bf16_to_fp16": int(qr_cast_bf16_to_fp16),
        "ignore_acr_max_size": bool(ignore_acr_max_size),
        "ignore_scr_max_size": bool(ignore_scr_max_size),
        "num_iters": int(num_iters),
        "ingraph_calls": int(ingraph_calls) if mode == "graph" else None,
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


def _print_comparison(measurements: List[Dict[str, Any]], timing: str) -> None:
    """One row per size, one column per backend. A star marks the fastest."""
    names: List[str] = []
    for row in measurements:
        for name in row.get("times_us", {}):
            if name not in names:
                names.append(name)
    if not names:
        return
    width = max(12, *(len(name) for name in names))
    print(
        f"\nAll-reduce comparison, microseconds per call ({timing}). "
        "skip means the kernel declined the tensor."
    )
    header = f"{'size':>10}  " + "  ".join(f"{name:<{width}}" for name in names)
    print(header)
    print("-" * len(header))
    for row in measurements:
        times = row.get("times_us") or {}
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


def print_allreduce_summary(results: Dict[str, Any]) -> None:
    configs = results.get("configs", [])
    if not configs:
        print("  No all-reduce configs to summarize.")
        return
    mode = results.get("mode", "eager")
    timing = results.get("timing", "mean")
    _print_comparison(results.get("measurements", []), f"{mode}, {timing}")
    print(f"\n{'=' * 60}\nAll-reduce tuning summary ({mode}, {timing})\n{'=' * 60}")
    for c in configs:
        print(f"  {c['size_bytes']:>10}B  {c['backend']:24s}  {c['time_us']:8.2f} us")
    print("=" * 60)
    if results.get("path"):
        print(f"  Use it with {results['env']}={results['path']}")
