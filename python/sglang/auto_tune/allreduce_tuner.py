"""Allreduce algorithm auto-tuner for sglang auto_tune.

Benchmarks and selects the optimal allreduce backend for a given
model and TP configuration.

Supported backends:
- custom_allreduce: sglang's NVLink-based allreduce
- nccl: standard NCCL allreduce
- torch_symm_mem: torch symmetric memory all-gather + reduce
- quick_allreduce_<codec>: AMD QuickReduce. Default codec is INT4.

Part of #13363 item 4.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
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


def _time_gpu_calls(invoke, num_iters: int, num_warmup: int) -> Optional[float]:
    """Time ``invoke`` on the GPU. ``None`` means this size is not eligible."""
    import torch

    for _ in range(num_warmup):
        if invoke() is None:
            return None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        if invoke() is None:
            return None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / num_iters


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


def _open_custom_allreduce(cpu_group, device, ignore_max_size=False):
    """Build custom all-reduce once. The server also builds it once."""
    car = None
    ok = False
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        car = CustomAllreduce(cpu_group, device)
        ok = not car.disabled
    except Exception as e:
        logger.warning("custom_allreduce failed: %s", e)
    if not _agree(ok, cpu_group):
        if car is not None:
            car.close()
        return None

    def run(tensor, num_iters, num_warmup=20):
        reason = _shape_reason(tensor, car._SUPPORTED_DTYPES)
        if reason:
            raise _Rejected(reason)
        if not _custom_topology_ok(car):
            raise _Rejected("not supported on this link topology")
        if ignore_max_size:
            # Past the published cap, the kernel rejects a tensor that does
            # not fit the workspace. That rejection is the skip reason.
            invoke = lambda: car._all_reduce_impl(tensor, registered=False)
        else:
            if not car.should_custom_ar(tensor):
                nbytes = tensor.numel() * tensor.element_size()
                raise _Rejected(f"{nbytes}B exceeds the {car.max_size}B workspace")
            invoke = lambda: car.custom_all_reduce(tensor)

        def timed():
            out = invoke()
            if out is None:
                raise _Rejected("custom all-reduce declined the tensor")
            return out

        return _time_gpu_calls(timed, num_iters, num_warmup)

    return run, car.close


def _open_nccl_allreduce(group, cpu_group, device):
    import torch.distributed as dist

    def run(tensor, num_iters, num_warmup=20):
        # dist.all_reduce returns None, which _time_gpu_calls would read as
        # "not eligible", so report success explicitly.
        def invoke():
            dist.all_reduce(tensor, group=group)
            return True

        return _time_gpu_calls(invoke, num_iters, num_warmup)

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

    def run(tensor, num_iters, num_warmup=20):
        if not comm.should_torch_symm_mem_allreduce(tensor):
            raise _Rejected(_shape_reason(tensor, (comm.dtype,)) or "tensor was declined")
        return _time_gpu_calls(lambda: comm.all_reduce(tensor), num_iters, num_warmup)

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
    quant, cpu_group, device, ignore_min_size=False, ignore_max_size=False
):
    """Build QuickReduce once at one codec. The codec is read at construction."""
    previous = _set_qr_quant(quant)
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
        _restore_qr_quant(previous)
    if not _agree(ok, cpu_group):
        if qr is not None:
            qr.close()
        return None

    def run(tensor, num_iters, num_warmup=20):
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
            out = qr.quick_all_reduce(tensor)
            if out is None:
                raise _Rejected("QuickReduce declined the tensor")
            return out

        return _time_gpu_calls(timed, num_iters, num_warmup)

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
) -> Dict[str, Any]:
    """Open every backend once, in the same order on every rank.

    Returns name -> (run, close). Backends that cannot start are left out.
    """
    openers = [
        (
            "custom_allreduce",
            lambda: _open_custom_allreduce(
                cpu_group, device, ignore_max_size=qr_ignore_max_size
            ),
        ),
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
            print(f"  {name}: ready", flush=True)
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
) -> List[AllreduceConfig]:
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
    )
    if verbose:
        print(f"  Backends: {', '.join(backends)}", flush=True)
        if qr_ignore_min_size or qr_ignore_max_size:
            print(
                "  Published size caps: not applied. "
                "A skip is the kernel rejecting the tensor.",
                flush=True,
            )
    dtype = model_config.get("dtype", torch.bfloat16)
    if verbose:
        print(f"  Sizes: {sizes}", flush=True)

    results = []
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
                    t = run(tensor, num_iters)
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

                if t is not None and t < best_time:
                    best_time = t
                    best_backend = backend_name

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

    return results


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
    results = tune_allreduce(
        model_config,
        tp_size,
        group=group,
        verbose=verbose,
        qr_quants=qr_quants,
        num_iters=num_iters,
        sizes=sizes,
        qr_ignore_min_size=qr_ignore_min_size,
        qr_ignore_max_size=qr_ignore_max_size,
    )

    if not results:
        if verbose:
            print("  No results. Skipping.")
        return {}

    output = {
        "tp_size": tp_size,
        "ep_size": ep_size,
        "qr_quants": qr_quants,
        "configs": [r.to_dict() for r in results],
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "allreduce_configs.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"  Saved to: {path}")

    return output


def print_allreduce_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of allreduce tuning results."""
    configs = results.get("configs", [])
    if not configs:
        print("  No allreduce configs to summarize.")
        return

    print(f"\n{'=' * 60}")
    print("Allreduce Tuning Summary")
    print(f"{'=' * 60}")
    for c in configs:
        print(f"  {c['size_bytes']:>10}B  {c['backend']:20s}  {c['time_us']:8.2f} us")
    print(f"{'=' * 60}")