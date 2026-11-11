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

Backends, under the names the tables use:

- ``sglang_custom_allreduce``: SGLang's CustomAllreduce.
- ``aiter_custom_allreduce``: AITER's CustomAllreduce, the ROCm default.
- ``custom_allreduce_v2``: SGLang's CustomAllReduceV2. On ROCm its graph path
  is the eager kernel recorded as is.
- ``nccl``: PyNccl out-of-place all-reduce (RCCL on ROCm), the call the server
  makes for an ``nccl`` row.
- ``torch_symm_mem``: torch symmetric-memory all-reduce.
- ``quick_allreduce_<codec>``: AMD QuickReduce at one codec.
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

logger = logging.getLogger(__name__)

# Codecs QuickAllReduce reads from ROCM_QUICK_REDUCE_QUANTIZATION. NONE leaves
# the communicator disabled, so it is not a backend.
QR_QUANT_REGIMES = ("FP", "INT8", "INT6", "INT4")
_QR_QUANT_ENVS = (
    "ROCM_QUICK_REDUCE_QUANTIZATION",
    "AITER_QUICK_REDUCE_QUANTIZATION",
)

MODES = ("eager", "graph")

# Reduction width of the GEMM that fills the all-reduce input. Its output
# covers every byte the all-reduce reads, so the collective waits for those
# writes. The width stays modest so the GEMM does not dominate the run; the
# GEMM-only time is subtracted before a winner is chosen.
_PRECEDING_GEMM_K = 256

_NUM_WARMUP = 20


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
    impl: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# Sizes and the preceding GEMM
# ---------------------------------------------------------------------------


def _get_tensor_sizes(model_config: Dict[str, Any]) -> List[int]:
    """Message sizes in bytes for a few token counts of ``[tokens, hidden]``."""
    import torch

    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config.get("shard_intermediate_size", 4 * hidden_size)
    dtype = model_config.get("dtype", torch.bfloat16)
    elem_size = 4 if dtype == torch.float32 else 2
    sizes = [
        hidden_size * elem_size,
        3 * hidden_size * elem_size,
        intermediate_size * elem_size,
        4 * hidden_size * elem_size,
        8 * hidden_size * elem_size,
        16 * hidden_size * elem_size,
    ]
    return sorted(set(sizes))


def _gemm_cols(numel: int, limit: int) -> int:
    """Largest power-of-two column count that divides ``numel`` and fits."""
    cols = 1
    limit = max(1, min(int(limit), int(numel)))
    while cols * 2 <= limit and numel % (cols * 2) == 0:
        cols *= 2
    return cols


def _preceding_gemm(tensor, hidden_size: Optional[int]):
    """Return a callable that writes ``tensor`` with one GEMM, plus its shape.

    The GEMM is queued on the current stream, and the all-reduce after it is
    queued on the same stream with no sync, so it reads the tensor only after
    the GEMM's writes land. In graph mode both are captured.
    """
    import torch

    numel = tensor.numel()
    hidden = int(hidden_size) if hidden_size else 0
    if hidden > 1 and numel % hidden == 0:
        rows, cols = numel // hidden, hidden
    else:
        cols = _gemm_cols(numel, hidden or _PRECEDING_GEMM_K)
        rows = numel // cols
    k = min(_PRECEDING_GEMM_K, max(cols, 1))
    out = tensor.view(rows, cols)
    lhs = torch.randn(rows, k, device=tensor.device, dtype=tensor.dtype)
    rhs = torch.randn(k, cols, device=tensor.device, dtype=tensor.dtype)

    def before():
        torch.mm(lhs, rhs, out=out)

    return before, {"rows": rows, "k": k, "cols": cols}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_eager(step, num_iters: int, use_median: bool) -> float:
    """Microseconds per call, launched from Python one call at a time."""
    import torch

    for _ in range(_NUM_WARMUP):
        step()
    torch.cuda.synchronize()
    if not use_median:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            step()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / num_iters

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        starts[i].record()
        step()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(
        starts[i].elapsed_time(ends[i]) * 1000 for i in range(num_iters)
    )


def _time_graph(
    step,
    capture: Callable[[], ContextManager],
    num_iters: int,
    graph_calls: int,
    use_median: bool,
) -> float:
    """Microseconds per call, from replays of a graph that holds ``graph_calls`` calls.

    The calls are recorded back to back, as consecutive layers' all-reduces
    are in a decode graph. ``num_iters`` is the number of timed replays.
    """
    import torch

    # Eager warm-up first: it builds kernels and workspaces outside the graph.
    # Inside a capture context a non-captured call is a warm-up that returns
    # zeros, so it would not exercise the kernel.
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Exiting ``capture`` registers the captured inputs with the peers. That
    # is a collective, so it runs after the capture ends, on every rank.
    with capture():
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(graph_calls):
                step()
    torch.cuda.synchronize()

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if not use_median:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) * 1000 / (num_iters * graph_calls)
    else:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        for i in range(num_iters):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        elapsed = statistics.median(
            starts[i].elapsed_time(ends[i]) * 1000 / graph_calls
            for i in range(num_iters)
        )
    del graph
    return elapsed


def _time_backend(
    backend: _Backend,
    tensor,
    mode: str,
    num_iters: int,
    graph_calls: int,
    use_median: bool,
    before=None,
) -> float:
    def step():
        if before is not None:
            before()
        out = backend.call(tensor)
        if out is None:
            raise _Rejected("the kernel declined the tensor")
        return out

    if mode == "graph":
        return _time_graph(step, backend.capture, num_iters, graph_calls, use_median)
    return _time_eager(step, num_iters, use_median)


def _time_gemm_only(
    before, mode: str, num_iters: int, graph_calls: int, use_median: bool
) -> float:
    def step():
        before()
        return True

    if mode == "graph":
        return _time_graph(
            step, contextlib.nullcontext, num_iters, graph_calls, use_median
        )
    return _time_eager(step, num_iters, use_median)


# ---------------------------------------------------------------------------
# Opening backends
# ---------------------------------------------------------------------------


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


def _normalize_qr_quants(qr_quants: Optional[List[str]]) -> List[str]:
    selected: List[str] = []
    for quant in qr_quants or ["INT4"]:
        if quant in QR_QUANT_REGIMES and quant not in selected:
            selected.append(quant)
    return selected


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


def _cpu_group(group):
    """Custom all-reduce, QuickReduce and PyNccl attach to a gloo group, as in the server."""
    import torch.distributed as dist

    if dist.get_backend(group) != dist.Backend.NCCL:
        return group
    return dist.new_group(backend="gloo")


def _open_backends(
    qr_quants: List[str],
    cpu_group,
    device,
    verbose: bool,
    ignore_acr_max_size: bool,
    ignore_scr_max_size: bool,
    qr_cast_bf16_to_fp16: bool,
) -> Dict[str, _Backend]:
    """Open every backend once, in the same order on every rank."""
    openers: List[Tuple[str, Callable[[], Optional[_Backend]]]] = [
        (
            "sglang_custom_allreduce",
            lambda: _open_custom_allreduce(
                cpu_group, device, "sglang", lift_cap=ignore_scr_max_size
            ),
        ),
        (
            "aiter_custom_allreduce",
            lambda: _open_custom_allreduce(
                cpu_group, device, "aiter", lift_cap=ignore_acr_max_size
            ),
        ),
        ("custom_allreduce_v2", lambda: _open_car_v2(cpu_group, device)),
        ("nccl", lambda: _open_nccl(cpu_group, device)),
        ("torch_symm_mem", lambda: _open_torch_symm_mem(cpu_group, device)),
    ]
    for quant in qr_quants:
        openers.append(
            (
                f"quick_allreduce_{quant}",
                lambda quant=quant: _open_quick_allreduce(
                    quant, cpu_group, device, qr_cast_bf16_to_fp16
                ),
            )
        )

    backends: Dict[str, _Backend] = {}
    for name, opener in openers:
        if verbose:
            print(f"  opening {name}", flush=True)
        opened = opener()
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
    model_config: Dict[str, Any],
    tp_size: int,
    mode: str = "eager",
    group=None,
    num_iters: int = 100,
    graph_calls: int = 10,
    verbose: bool = True,
    qr_quants: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    use_median: bool = False,
    preceding_gemm: bool = False,
) -> Tuple[List[AllreduceConfig], List[Dict[str, Any]]]:
    """Time each backend at each size in ``mode`` and return the fastest per size."""
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

    qr_quants = _normalize_qr_quants(qr_quants)
    cpu_group = _cpu_group(group)
    device = torch.device("cuda", torch.cuda.current_device())
    sizes = (
        _get_tensor_sizes(model_config)
        if sizes is None
        else sorted({int(size) for size in sizes})
    )
    backends = _open_backends(
        qr_quants,
        cpu_group,
        device,
        verbose,
        ignore_acr_max_size=ignore_acr_max_size,
        ignore_scr_max_size=ignore_scr_max_size,
        qr_cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
    )
    dtype = model_config.get("dtype", torch.bfloat16)
    if verbose:
        print(f"  Backends: {', '.join(backends)}", flush=True)
        print(f"  Mode: {mode}", flush=True)
        stat = "median" if use_median else "mean"
        if mode == "graph":
            print(
                f"  Timing: {stat} of {num_iters} replays of a graph with "
                f"{graph_calls} captured calls, per call.",
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
        if preceding_gemm:
            print(
                "  Each call is a GEMM that writes the all-reduce input, then "
                "the all-reduce, on one stream with no sync. The GEMM-only "
                "time is subtracted.",
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
            elem_size = 4 if dtype == torch.float32 else 2
            tensor = torch.randn(size_bytes // elem_size, dtype=dtype, device="cuda")
            before = None
            gemm_us = None
            shape = None
            if preceding_gemm:
                before, shape = _preceding_gemm(tensor, model_config.get("hidden_size"))
                barrier()
                gemm_us = _time_gemm_only(before, mode, num_iters, graph_calls, use_median)
                barrier()
                if verbose:
                    print(
                        f"{tag}: GEMM [{shape['rows']}, {shape['cols']}] = "
                        f"[{shape['rows']}, {shape['k']}] @ [{shape['k']}, "
                        f"{shape['cols']}] only -> {gemm_us:.2f} us",
                        flush=True,
                    )

            best_backend = None
            best_time = float("inf")
            row_times: Dict[str, Optional[float]] = {}
            for name, backend in backends.items():
                barrier()
                reason = backend.check(tensor)
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
                            graph_calls,
                            use_median,
                            before=before,
                        )
                    except Exception as e:
                        t = None
                        reason = str(e).splitlines()[0] if str(e) else type(e).__name__
                barrier()
                if t is not None and gemm_us is not None:
                    t -= gemm_us
                if verbose:
                    shown = f"{t:.2f} us" if t is not None else f"skipped: {reason}"
                    print(f"{tag}: {name} -> {shown}", flush=True)
                row_times[name] = t
                if t is not None and t < best_time:
                    best_time, best_backend = t, name

            row: Dict[str, Any] = {"size_bytes": size_bytes, "times_us": row_times}
            if gemm_us is not None:
                row["gemm_us"] = gemm_us
                row["gemm"] = shape
            measurements.append(row)
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
            del tensor, before
            torch.cuda.empty_cache()
    finally:
        # Close together: a rank that frees its IPC buffers early can leave a
        # peer still using them.
        barrier()
        for backend in backends.values():
            backend.close()

    if verbose:
        print(
            f"\n  All-reduce tuning ({mode}) finished in "
            f"{time.perf_counter() - total_start:.2f}s, "
            f"{len(results)}/{len(sizes)} sizes",
            flush=True,
        )
    return results, measurements


def run_allreduce_tuning(
    model_config: Dict[str, Any],
    tp_size: int,
    ep_size: int = 1,
    mode: str = "eager",
    group=None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    qr_quants: Optional[List[str]] = None,
    num_iters: int = 100,
    graph_calls: int = 10,
    sizes: Optional[List[int]] = None,
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    use_median: bool = False,
    preceding_gemm: bool = False,
) -> Dict[str, Any]:
    """Tune, then write ``allreduce_configs_<mode>.json`` into ``output_dir``."""
    if verbose:
        print(f"\n{'=' * 60}\nAll-reduce tuning, {mode} mode\n{'=' * 60}")
        print(f"  TP: {tp_size}, EP: {ep_size}")

    qr_quants = _normalize_qr_quants(qr_quants)
    results, measurements = tune_allreduce(
        model_config,
        tp_size,
        mode=mode,
        group=group,
        num_iters=num_iters,
        graph_calls=graph_calls,
        verbose=verbose,
        qr_quants=qr_quants,
        sizes=sizes,
        ignore_acr_max_size=ignore_acr_max_size,
        ignore_scr_max_size=ignore_scr_max_size,
        qr_cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
        use_median=use_median,
        preceding_gemm=preceding_gemm,
    )
    if not results:
        if verbose:
            print("  No results.")
        return {}

    output = {
        "mode": mode,
        "env": "SGLANG_TUNED_GRAPH_CONFIG" if mode == "graph" else "SGLANG_TUNED_EAGER_CONFIG",
        "tp_size": tp_size,
        "ep_size": ep_size,
        "qr_quants": qr_quants,
        "qr_cast_bf16_to_fp16": int(qr_cast_bf16_to_fp16),
        "ignore_acr_max_size": bool(ignore_acr_max_size),
        "ignore_scr_max_size": bool(ignore_scr_max_size),
        "num_iters": int(num_iters),
        "graph_calls": int(graph_calls) if mode == "graph" else None,
        "timing": ("median" if use_median else "mean")
        + ("_after_gemm" if preceding_gemm else ""),
        "preceding_gemm": bool(preceding_gemm),
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
    show_gemm = any(row.get("gemm_us") is not None for row in measurements)
    print(
        f"\nAll-reduce comparison, microseconds per call ({timing}). "
        "skip means the kernel declined the tensor."
        + (" GEMM-only time already subtracted." if show_gemm else "")
    )
    header = f"{'size':>10}  " + "  ".join(f"{name:<{width}}" for name in names)
    if show_gemm:
        header += f"  {'gemm':<{width}}"
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
        line = f"{row['size_bytes']:>10}  " + "  ".join(cells)
        if show_gemm:
            line += f"  {row.get('gemm_us', 0.0):<{width}.2f}"
        print(line)


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
