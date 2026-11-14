"""Allreduce algorithm auto-tuner for sglang auto_tune.

Benchmarks and selects the optimal allreduce backend for a given
model and TP configuration.

Supported backends:
- custom_allreduce: sglang's own all-reduce, or AITER's when --use-aiter-allreduce is set. The ROCm server dispatches AITER.
- custom_allreduce_v2: SGLang CustomAllReduceV2, with --use-car-v2. ROCm configs cover MI355 and Helios.
- nccl: standard NCCL allreduce
- torch_symm_mem: torch symmetric memory all-gather + reduce
- quick_allreduce_<codec>: AMD QuickReduce. Default codec is INT4. With the default cast, the timed call converts bf16 to fp16 inside that kernel.

Part of #13363 item 4.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Codecs QuickAllReduce reads from ROCM_QUICK_REDUCE_QUANTIZATION. NONE leaves
# the communicator disabled, so it is not a benchmarkable backend.
QR_QUANT_REGIMES = ("FP", "INT8", "INT6", "INT4")
_QR_QUANT_ENVS = (
    "ROCM_QUICK_REDUCE_QUANTIZATION",
    "AITER_QUICK_REDUCE_QUANTIZATION",
)


@dataclasses.dataclass
class AllreduceConfig:
    """Best allreduce configuration for a given message size."""

    size_bytes: int
    backend: str
    time_us: float
    tp_size: int
    dtype: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _get_tensor_sizes(model_config: Dict[str, Any]) -> List[int]:
    """Derive key tensor sizes (in bytes) for allreduce operations.

    In transformer models, allreduce happens after:
    - QKV projection (3 * hidden_size)
    - Attention output (hidden_size)
    - MLP gate + up (intermediate_size)
    - MoE fused (shared_expert hidden_size)
    """
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config.get("shard_intermediate_size", 4 * hidden_size)

    import torch

    dtype = model_config.get("dtype", torch.bfloat16)
    elem_size = 2  # default bf16/fp16
    if dtype == torch.float32:
        elem_size = 4
    elif dtype == torch.float8_e4m3fn:
        elem_size = 1

    sizes = [
        hidden_size * elem_size,           # attention output
        3 * hidden_size * elem_size,       # QKV
        intermediate_size * elem_size,     # MLP up
        hidden_size * elem_size,           # MLP down (same as attn)
    ]

    sizes.extend([
        4 * hidden_size * elem_size,       # typical fused
        8 * hidden_size * elem_size,       # large fused
        16 * hidden_size * elem_size,      # very large
    ])

    return sorted(set(sizes))


def _time_gpu_calls(
    invoke, num_iters: int, num_warmup: int, use_median: bool = False
) -> Optional[float]:
    """Time ``invoke`` on the GPU, in microseconds. ``None`` means not eligible.

    The default is the mean of one timed loop. With ``use_median``, each call
    gets its own events and the result is the median of those calls.
    """
    import torch

    for _ in range(num_warmup):
        if invoke() is None:
            return None
    torch.cuda.synchronize()
    if not use_median:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            if invoke() is None:
                return None
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / num_iters

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        starts[i].record()
        if invoke() is None:
            return None
        ends[i].record()
    torch.cuda.synchronize()
    samples_us = [
        starts[i].elapsed_time(ends[i]) * 1000 for i in range(num_iters)
    ]
    return statistics.median(samples_us)


def _agree(ok: bool, cpu_group: "torch.distributed.ProcessGroup") -> bool:
    """True only when every rank passes ``ok``.

    A rank that skips a backend while the others enter its collectives leaves
    them waiting forever, so every rank must take the same branch.
    """
    import torch
    import torch.distributed as dist

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(flag.item())


class _Rejected(Exception):
    """The kernel will not run this tensor. The message is the log line."""


def _shape_reason(tensor, supported_dtypes) -> Optional[str]:
    """Why this tensor's shape or dtype is not one the kernel can run.

    Size caps are not a reason. A tensor that passes here is handed to the
    kernel, which rejects it itself when the workspace cannot hold it.
    """
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


def _custom_topology_ok(car) -> bool:
    from sglang.srt.utils import is_hip

    if not is_hip():
        return car.world_size == 2 or car.full_nvlink
    if car.use_amd_deterministic_impl:
        return True
    return bool(car.full_nvlink)


def _class_qualname(cls) -> str:
    func = getattr(cls, "func", cls)
    return (
        f"{getattr(func, '__module__', '')}.{getattr(func, '__name__', type(func).__name__)}"
    )


def _relax_aiter_size_cap(car) -> str:
    """Ignore AITER_CUSTOM_AR_MAX_SIZE. The registered pool still bounds the call.

    The cutoff is read once, into ``_car_max_size``, when the communicator is
    built. Raising that attribute is enough for a build that has it. An older
    build bakes 64 MiB into ``should_custom_ar``; the wrapper below keeps the
    other gates and compares against the pool instead.
    """
    pool = int(getattr(car, "max_size", 0) or 0)
    if hasattr(car, "_car_max_size") and pool > 0:
        car._car_max_size = pool
        return f"AITER_CUSTOM_AR_MAX_SIZE ignored; cutoff is the {pool}B pool"
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_weak_contiguous,
    )

    original = getattr(car, "should_custom_ar", None)
    if original is None or pool <= 0:
        return "AITER size cap left unchanged; this build has no cutoff to lift"

    def should(inp, prefill_support=False):
        if original(inp, prefill_support=prefill_support):
            return True
        if getattr(car, "disabled", True):
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes % 16 != 0 or nbytes > pool or not is_weak_contiguous(inp):
            return False
        if not (
            getattr(car, "world_size", 0) == 2 or getattr(car, "fully_connected", False)
        ):
            return False
        return True

    car.should_custom_ar = should
    return f"AITER policy cutoff bypassed; cutoff is the {pool}B pool"


def _open_custom_allreduce(
    cpu_group, device, ignore_max_size=False, use_aiter=False, ignore_acr_max_size=False
):
    """Build one custom all-reduce and time it under the name ``custom_allreduce``.

    ``use_aiter`` asks for the class ``dispatch_custom_allreduce`` returns.
    On ROCm that is AITER's communicator, which is the one the server runs
    for a ``custom_allreduce`` row. The sglang class is a different kernel.
    """
    car = None
    ok = False
    try:
        if use_aiter:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            opened = _class_qualname(AiterCustomAllreduce)
            car = AiterCustomAllreduce(group=cpu_group, device=device)
            ok = not getattr(car, "disabled", True)
        else:
            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            opened = _class_qualname(CustomAllreduce)
            car = CustomAllreduce(cpu_group, device)
            ok = not car.disabled
    except Exception as e:
        opened = ""
        logger.warning(
            "custom_allreduce (%s) failed: %s",
            "aiter" if use_aiter else "sglang",
            e,
        )
    if not _agree(ok, cpu_group):
        if car is not None and hasattr(car, "close"):
            car.close()
        return None

    cap_note = ""
    if use_aiter and ignore_acr_max_size:
        cap_note = _relax_aiter_size_cap(car)

    supported = getattr(car, "_SUPPORTED_DTYPES", None) or torch_float_dtypes()

    def run(tensor, num_iters, num_warmup=20, use_median=False):
        reason = _shape_reason(tensor, supported)
        if reason:
            raise _Rejected(reason)
        # The sglang class has a topology gate the server also applies to it.
        # AITER decides for itself inside should_custom_ar.
        if not use_aiter and not _custom_topology_ok(car):
            raise _Rejected("not supported on this link topology")
        if ignore_max_size and hasattr(car, "_all_reduce_impl"):
            # Past the published cap, the kernel rejects a tensor that does
            # not fit the workspace. That rejection is the skip reason.
            invoke = lambda: car._all_reduce_impl(tensor, registered=False)
        else:
            if hasattr(car, "should_custom_ar") and not car.should_custom_ar(tensor):
                nbytes = tensor.numel() * tensor.element_size()
                limit = getattr(car, "max_size", None)
                extra = f" (workspace {limit}B)" if limit is not None else ""
                raise _Rejected(f"{nbytes}B was declined by should_custom_ar{extra}")
            invoke = lambda: car.custom_all_reduce(tensor)

        def timed():
            out = invoke()
            if out is None:
                raise _Rejected("custom all-reduce declined the tensor")
            return out

        return _time_gpu_calls(timed, num_iters, num_warmup, use_median=use_median)

    run.impl = opened
    run.note = cap_note
    close = car.close if hasattr(car, "close") else (lambda: None)
    return run, close


def _open_car_v2(cpu_group, device):
    """Time SGLang's CustomAllReduceV2.

    On ROCm the server uses it only when SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1.
    The class itself already carries the MI355 (gfx950) and Helios (gfx1250)
    configs. A size its heuristic declines is skipped, the same way the server
    would leave that size to the next backend.
    """
    car = None
    ok = False
    opened = ""
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        opened = _class_qualname(CustomAllReduceV2)
        car = CustomAllReduceV2(group=cpu_group, device=device)
        ok = not getattr(car, "disabled", True)
    except Exception as e:
        logger.warning("custom_allreduce_v2 failed: %s", e)
    if not _agree(ok, cpu_group):
        if car is not None and hasattr(car, "close"):
            car.close()
        return None

    def run(tensor, num_iters, num_warmup=20, use_median=False):
        if not car.should_custom_ar(tensor):
            nbytes = tensor.numel() * tensor.element_size()
            limit = getattr(car, "max_size", None)
            extra = f" (workspace {limit}B)" if limit is not None else ""
            raise _Rejected(f"{nbytes}B was declined by custom all-reduce v2{extra}")

        def timed():
            out = car.custom_all_reduce(tensor)
            if out is None:
                raise _Rejected("custom all-reduce v2 declined the tensor")
            return out

        return _time_gpu_calls(timed, num_iters, num_warmup, use_median=use_median)

    run.impl = opened
    return run, car.close if hasattr(car, "close") else (lambda: None)


def torch_float_dtypes():
    import torch

    return (torch.float32, torch.float16, torch.bfloat16)


def _open_nccl_allreduce(group, cpu_group, device):
    import torch.distributed as dist

    def run(tensor, num_iters, num_warmup=20, use_median=False):
        # dist.all_reduce returns None, which _time_gpu_calls would read as
        # "not eligible", so report success explicitly.
        def invoke():
            dist.all_reduce(tensor, group=group)
            return True

        return _time_gpu_calls(invoke, num_iters, num_warmup, use_median=use_median)

    return run, None


def _open_torch_symm_mem(group, cpu_group, device):
    comm = None
    ok = False
    try:
        from sglang.srt.distributed.device_communicators.torch_symm_mem import (
            TorchSymmMemCommunicator,
        )

        comm = TorchSymmMemCommunicator(group, device)
        ok = not comm.disabled
    except Exception as e:
        logger.warning("torch_symm_mem failed: %s", e)
    if not _agree(ok, cpu_group):
        return None

    def run(tensor, num_iters, num_warmup=20, use_median=False):
        if not comm.should_torch_symm_mem_allreduce(tensor):
            raise _Rejected(_shape_reason(tensor, (comm.dtype,)) or "tensor was declined")
        return _time_gpu_calls(
            lambda: comm.all_reduce(tensor),
            num_iters,
            num_warmup,
            use_median=use_median,
        )

    return run, None


def _normalize_qr_quants(qr_quants: Optional[List[str]]) -> List[str]:
    if qr_quants is None:
        qr_quants = ["INT4"]
    selected: List[str] = []
    for quant in qr_quants:
        if quant in QR_QUANT_REGIMES and quant not in selected:
            selected.append(quant)
    return selected


def _set_qr_quant(quant: str) -> Dict[str, Optional[str]]:
    previous = {name: os.environ.get(name) for name in _QR_QUANT_ENVS}
    for name in _QR_QUANT_ENVS:
        os.environ[name] = quant
    return previous


def _restore_qr_quant(previous: Dict[str, Optional[str]]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _open_quick_allreduce(
    quant,
    cpu_group,
    device,
    ignore_min_size=False,
    ignore_max_size=False,
    cast_bf16_to_fp16=True,
):
    """Build QuickReduce once at one codec. The codec is read at construction.

    ``cast_bf16_to_fp16`` is ``ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16``. The
    communicator reads it once and passes it into every ``qr_all_reduce``.
    The cast is on the kernel's load and store, so the timed call includes it.
    """
    os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = (
        "1" if cast_bf16_to_fp16 else "0"
    )
    previous = _set_qr_quant(quant)
    qr = None
    ok = False
    try:
        from sglang.srt.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qr = QuickAllReduce(cpu_group, device)
        ok = not qr.disabled
        if ok and bool(qr.use_fp16_kernels) != bool(cast_bf16_to_fp16):
            logger.warning(
                "quick_allreduce %s cast flag is %s, requested %s",
                quant,
                qr.use_fp16_kernels,
                int(cast_bf16_to_fp16),
            )
            ok = False
    except Exception as e:
        logger.warning("quick_allreduce %s failed: %s", quant, e)
    finally:
        _restore_qr_quant(previous)
    if not _agree(ok, cpu_group):
        if qr is not None:
            qr.close()
        return None

    def run(tensor, num_iters, num_warmup=20, use_median=False):
        reason = _shape_reason(tensor, qr._SUPPORTED_DTYPES)
        if reason:
            raise _Rejected(reason)
        import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops

        if not ops.qr_supports(qr._ptr, tensor.dtype):
            raise _Rejected(f"this build does not support {tensor.dtype}")
        if not ignore_min_size or not ignore_max_size:
            if not qr.should_quick_allreduce(
                tensor,
                ignore_min_size=ignore_min_size,
                ignore_max_size=ignore_max_size,
            ):
                nbytes = tensor.numel() * tensor.element_size()
                raise _Rejected(f"{nbytes}B is outside the published size window")

        def timed():
            # use_fp16_kernels was fixed at init from
            # ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16. quick_all_reduce passes it
            # into the kernel, which casts bf16 to fp16 on load and back on
            # store. That conversion is inside this timed call.
            out = qr.quick_all_reduce(tensor)
            if out is None:
                raise _Rejected("QuickReduce declined the tensor")
            return out

        return _time_gpu_calls(timed, num_iters, num_warmup, use_median=use_median)

    return run, qr.close


def _cpu_group(group: "torch.distributed.ProcessGroup"):
    """Custom all-reduce and QuickReduce reject an NCCL group.

    The server attaches both to a gloo group. torchrun's default group is NCCL.
    """
    import torch.distributed as dist

    if dist.get_backend(group) != dist.Backend.NCCL:
        return group
    return dist.new_group(backend="gloo")


def _open_backends(
    qr_quants: List[str],
    group,
    cpu_group,
    device,
    verbose: bool,
    qr_ignore_min_size: bool = False,
    qr_ignore_max_size: bool = False,
    use_aiter_allreduce: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    use_car_v2: bool = False,
    ignore_acr_max_size: bool = False,
) -> Dict[str, Any]:
    """Open every backend once, in the same order on every rank.

    Returns name -> (run, close). Backends that cannot start are left out.
    """
    # All three custom all-reduces are separate entries. A flag does not
    # drop the others: the run is a competition, and one winner is stored
    # per size under that kernel's own name.
    openers = [
        (
            "sglang_custom_allreduce",
            lambda: _open_custom_allreduce(
                cpu_group,
                device,
                ignore_max_size=qr_ignore_max_size,
                use_aiter=False,
            ),
        ),
        (
            "aiter_custom_allreduce",
            lambda: _open_custom_allreduce(
                cpu_group,
                device,
                ignore_max_size=False,
                use_aiter=True,
                ignore_acr_max_size=ignore_acr_max_size,
            ),
        ),
        ("custom_allreduce_v2", lambda: _open_car_v2(cpu_group, device)),
        ("nccl", lambda: _open_nccl_allreduce(group, cpu_group, device)),
        ("torch_symm_mem", lambda: _open_torch_symm_mem(group, cpu_group, device)),
    ]
    for quant in qr_quants:
        openers.append(
            (
                f"quick_allreduce_{quant}",
                lambda quant=quant: _open_quick_allreduce(
                    quant,
                    cpu_group,
                    device,
                    ignore_min_size=qr_ignore_min_size,
                    ignore_max_size=qr_ignore_max_size,
                    cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
                ),
            )
        )

    backends: Dict[str, Any] = {}
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
            impl = getattr(opened[0], "impl", "")
            note = getattr(opened[0], "note", "")
            suffix = f" ({impl})" if impl else ""
            print(f"  {name}: ready{suffix}", flush=True)
            if note:
                print(f"  {note}", flush=True)
    return backends


def tune_allreduce(
    model_config: Dict[str, Any],
    tp_size: int,
    group: Optional["torch.distributed.ProcessGroup"] = None,
    num_iters: int = 100,
    verbose: bool = True,
    qr_quants: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    qr_ignore_min_size: bool = False,
    qr_ignore_max_size: bool = False,
    use_aiter_allreduce: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    use_car_v2: bool = False,
    ignore_acr_max_size: bool = False,
    use_median: bool = False,
) -> Tuple[List[AllreduceConfig], List[Dict[str, Any]]]:
    """Tune allreduce algorithm for the given model and TP size.

    Benchmarks each backend for key tensor sizes and returns the
    best configuration for each size.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        logger.warning(
            "torch.distributed is not initialized, so allreduce tuning was skipped. "
            "Launch one process per GPU with torchrun, for example: "
            "torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce "
            "--model-path <model> --tp 8 --qr-quant INT4"
        )
        return []

    if group is None:
        group = dist.group.WORLD

    world_size = dist.get_world_size(group)
    if world_size != tp_size:
        logger.warning(
            "World size %d != TP size %d; using world size %d",
            world_size, tp_size, world_size,
        )
        tp_size = world_size

    qr_quants = _normalize_qr_quants(qr_quants)
    cpu_group = _cpu_group(group)
    device = torch.device("cuda", torch.cuda.current_device())
    if sizes is None:
        sizes = _get_tensor_sizes(model_config)
    else:
        sizes = sorted({int(size) for size in sizes})
    backends = _open_backends(
        qr_quants,
        group,
        cpu_group,
        device,
        verbose,
        qr_ignore_min_size,
        qr_ignore_max_size=qr_ignore_max_size,
        use_aiter_allreduce=use_aiter_allreduce,
        qr_cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
        use_car_v2=use_car_v2,
        ignore_acr_max_size=ignore_acr_max_size,
    )
    if verbose:
        print(f"  Backends: {', '.join(backends)}", flush=True)
        if qr_ignore_min_size or qr_ignore_max_size:
            print(
                "  Published size caps: not applied. "
                "A skip is the kernel rejecting the tensor.",
                flush=True,
            )
        cast = int(qr_cast_bf16_to_fp16)
        print(
            f"  ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16={cast}. "
            "The bf16 to fp16 cast runs inside the timed QuickReduce kernel.",
            flush=True,
        )
        print(
            "  Kernels in this run: sglang custom all-reduce, AITER custom "
            "all-reduce, and custom all-reduce v2. Each keeps its own size "
            "cutoff. --ignore-acr-max-size lifts only AITER's.",
            flush=True,
        )
        stat = "median of each call" if use_median else "mean of one timed loop"
        print(f"  Timing: {stat}, {num_iters} calls after 20 warmup.", flush=True)
    dtype = model_config.get("dtype", torch.bfloat16)
    if verbose:
        print(f"  Sizes: {sizes}", flush=True)

    results = []
    measurements: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    try:
        for idx, size_bytes in enumerate(sizes):
            num_elements = size_bytes // 2  # bf16/fp16
            if dtype == torch.float32:
                num_elements = size_bytes // 4

            if verbose:
                print(
                    f"  [{idx + 1}/{len(sizes)}] {size_bytes}B: allocating "
                    f"{num_elements} x {dtype}",
                    flush=True,
                )
            tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

            best_backend = None
            best_time = float("inf")
            row_times: Dict[str, Optional[float]] = {}

            for backend_name, (run, _close) in backends.items():
                if verbose:
                    print(
                        f"  [{idx + 1}/{len(sizes)}] {size_bytes}B: "
                        f"timing {backend_name}",
                        flush=True,
                    )
                # Every rank starts each backend together and finishes it
                # before any rank moves on.
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)
                reason = None
                try:
                    t = run(tensor, num_iters, use_median=use_median)
                except Exception as e:
                    t = None
                    reason = str(e).splitlines()[0]
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)
                if verbose:
                    shown = f"{t:.2f} us" if t is not None else f"skipped: {reason}"
                    print(
                        f"  [{idx + 1}/{len(sizes)}] {size_bytes}B: "
                        f"{backend_name} -> {shown}",
                        flush=True,
                    )

                row_times[backend_name] = t
                if t is not None and t < best_time:
                    best_time = t
                    best_backend = backend_name

            measurements.append(
                {"size_bytes": size_bytes, "times_us": row_times}
            )
            if best_backend is not None:
                config = AllreduceConfig(
                    size_bytes=size_bytes,
                    backend=best_backend,
                    time_us=best_time,
                    tp_size=tp_size,
                    dtype=str(dtype),
                )
                results.append(config)
                if verbose:
                    print(
                        f"  [{idx + 1}/{len(sizes)}] {size_bytes}B: "
                        f"best {best_backend:20s} {best_time:8.2f} us",
                        flush=True,
                    )
    finally:
        # Close together: a rank that frees its IPC buffers early can leave a
        # peer still using them.
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        for _run, close in backends.values():
            if close is not None:
                close()

    total_end = time.perf_counter()
    if verbose:
        print(f"\n  Allreduce tuning completed in {total_end - total_start:.2f}s")
        print(f"  Sizes profiled: {len(results)}/{len(sizes)}")

    return results, measurements


def run_allreduce_tuning(
    model_config: Dict[str, Any],
    tp_size: int,
    ep_size: int = 1,
    group: Optional["torch.distributed.ProcessGroup"] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    qr_quants: Optional[List[str]] = None,
    num_iters: int = 100,
    sizes: Optional[List[int]] = None,
    qr_ignore_min_size: bool = False,
    qr_ignore_max_size: bool = False,
    use_aiter_allreduce: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    use_car_v2: bool = False,
    ignore_acr_max_size: bool = False,
    use_median: bool = False,
) -> Dict[str, Any]:
    """Entry point for allreduce auto-tuning.

    Returns a dict with the best backend for each tensor size,
    plus a summary of which backend dominates.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("Allreduce Algorithm Tuning")
        print(f"{'=' * 60}")
        print(f"  TP: {tp_size}, EP: {ep_size}")

    qr_quants = _normalize_qr_quants(qr_quants)
    results, measurements = tune_allreduce(
        model_config,
        tp_size,
        group=group,
        verbose=verbose,
        qr_quants=qr_quants,
        num_iters=num_iters,
        sizes=sizes,
        qr_ignore_min_size=qr_ignore_min_size,
        qr_ignore_max_size=qr_ignore_max_size,
        use_aiter_allreduce=use_aiter_allreduce,
        qr_cast_bf16_to_fp16=qr_cast_bf16_to_fp16,
        use_car_v2=use_car_v2,
        ignore_acr_max_size=ignore_acr_max_size,
        use_median=use_median,
    )

    if not results:
        if verbose:
            print("  No results. Skipping.")
        return {}

    output = {
        "tp_size": tp_size,
        "ep_size": ep_size,
        "qr_quants": qr_quants,
        "qr_cast_bf16_to_fp16": int(qr_cast_bf16_to_fp16),
        "custom_allreduce_impl": "sglang+aiter+v2",
        "use_car_v2": True,
        "ignore_acr_max_size": bool(ignore_acr_max_size),
        "num_iters": int(num_iters),
        "timing": "median" if use_median else "mean",
        "configs": [r.to_dict() for r in results],
        "measurements": measurements,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "allreduce_configs.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"  Saved to: {path}")

    return output


def _print_comparison(
    measurements: List[Dict[str, Any]], timing: str = "mean"
) -> None:
    """One row per message size, one column per kernel that was opened."""
    if not measurements:
        return
    names: List[str] = []
    for row in measurements:
        for name in row.get("times_us", {}):
            if name not in names:
                names.append(name)
    if not names:
        return
    width = max(12, *(len(name) for name in names))
    print(
        f"\nAllreduce comparison, microseconds ({timing}). "
        "A star marks the fastest kernel at that size. "
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
            text = f"{value:.2f}"
            if best is not None and value == best:
                text += "*"
            cells.append(f"{text:<{width}}")
        print(f"{row['size_bytes']:>10}  " + "  ".join(cells))


def print_allreduce_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of allreduce tuning results."""
    configs = results.get("configs", [])
    if not configs:
        print("  No allreduce configs to summarize.")
        return

    timing = results.get("timing", "mean")
    n = results.get("num_iters")
    count = f", {n} calls" if n else ""
    print(f"\n{'=' * 60}")
    print(f"Allreduce Tuning Summary ({timing}{count})")
    print(f"{'=' * 60}")
    for c in configs:
        print(f"  {c['size_bytes']:>10}B  {c['backend']:20s}  {c['time_us']:8.2f} us")
    print(f"{'=' * 60}")
    _print_comparison(results.get("measurements") or [], timing=timing)