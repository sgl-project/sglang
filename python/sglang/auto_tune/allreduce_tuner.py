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


def _benchmark_custom_allreduce(
    tensor: "torch.Tensor",
    group: "torch.distributed.ProcessGroup",
    num_iters: int = 100,
    num_warmup: int = 20,
) -> Optional[float]:
    """Benchmark sglang's custom allreduce."""
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        car = CustomAllreduce(group, tensor.device)
        try:
            if car.disabled or not car.should_custom_ar(tensor):
                return None
            return _time_gpu_calls(
                lambda: car.custom_all_reduce(tensor), num_iters, num_warmup
            )
        finally:
            car.close()
    except Exception as e:
        logger.warning("custom_allreduce failed: %s", e)
        return None


def _benchmark_nccl_allreduce(
    tensor: "torch.Tensor",
    group: "torch.distributed.ProcessGroup",
    num_iters: int = 100,
    num_warmup: int = 20,
) -> Optional[float]:
    """Benchmark standard NCCL allreduce."""
    import torch
    import torch.distributed as dist

    try:
        return _time_gpu_calls(
            lambda: dist.all_reduce(tensor, group=group), num_iters, num_warmup
        )
    except Exception as e:
        logger.warning("nccl allreduce failed: %s", e)
        return None


def _benchmark_torch_symm_mem(
    tensor: "torch.Tensor",
    group: "torch.distributed.ProcessGroup",
    num_iters: int = 100,
    num_warmup: int = 20,
) -> Optional[float]:
    """Benchmark torch symmetric memory allreduce."""
    try:
        from sglang.srt.distributed.device_communicators.torch_symm_mem import (
            TorchSymmMemCommunicator,
        )

        comm = TorchSymmMemCommunicator(group, tensor.device)
        if comm.disabled or not comm.should_torch_symm_mem_allreduce(tensor):
            return None
        return _time_gpu_calls(
            lambda: comm.all_reduce(tensor), num_iters, num_warmup
        )
    except Exception as e:
        logger.warning("torch_symm_mem failed: %s", e)
        return None


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


def _benchmark_quick_allreduce(
    tensor: "torch.Tensor",
    group: "torch.distributed.ProcessGroup",
    num_iters: int = 100,
    num_warmup: int = 20,
    quant: str = "INT4",
) -> Optional[float]:
    """Benchmark QuickReduce at one codec. INT4 is the usual compressed regime."""
    if quant not in QR_QUANT_REGIMES:
        logger.warning("quick_allreduce quant %s is not one of %s", quant, QR_QUANT_REGIMES)
        return None
    previous = _set_qr_quant(quant)
    try:
        from sglang.srt.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qr = QuickAllReduce(group, tensor.device)
        try:
            if qr.disabled or not qr.should_quick_allreduce(tensor):
                return None
            return _time_gpu_calls(
                lambda: qr.quick_all_reduce(tensor), num_iters, num_warmup
            )
        finally:
            qr.close()
    except Exception as e:
        logger.warning("quick_allreduce %s failed: %s", quant, e)
        return None
    finally:
        _restore_qr_quant(previous)


def _quickreduce_group(group: "torch.distributed.ProcessGroup"):
    """QuickAllReduce rejects an NCCL group. The server attaches it to gloo."""
    import torch.distributed as dist

    if dist.get_backend(group) != dist.Backend.NCCL:
        return group
    return dist.new_group(backend="gloo")


def _bind_quick_allreduce(quant: str, qr_group: "torch.distributed.ProcessGroup"):
    def _run(tensor, group, num_iters=100, num_warmup=20):
        del group
        return _benchmark_quick_allreduce(
            tensor, qr_group, num_iters=num_iters, num_warmup=num_warmup, quant=quant
        )

    return _run


def _allreduce_backends(
    qr_quants: List[str],
    qr_group: Optional["torch.distributed.ProcessGroup"],
) -> Dict[str, Any]:
    backends: Dict[str, Any] = {
        "custom_allreduce": _benchmark_custom_allreduce,
        "nccl": _benchmark_nccl_allreduce,
        "torch_symm_mem": _benchmark_torch_symm_mem,
    }
    if qr_group is None:
        return backends
    for quant in qr_quants:
        backends[f"quick_allreduce_{quant}"] = _bind_quick_allreduce(quant, qr_group)
    return backends


def tune_allreduce(
    model_config: Dict[str, Any],
    tp_size: int,
    group: Optional["torch.distributed.ProcessGroup"] = None,
    num_iters: int = 100,
    verbose: bool = True,
    qr_quants: Optional[List[str]] = None,
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
    qr_group = None
    if qr_quants:
        try:
            qr_group = _quickreduce_group(group)
        except Exception as e:
            logger.warning("QuickReduce group setup failed: %s", e)
    backends = _allreduce_backends(qr_quants, qr_group)
    if verbose and qr_quants and qr_group is not None:
        print(f"  QuickReduce codecs: {', '.join(qr_quants)}")

    sizes = _get_tensor_sizes(model_config)
    dtype = model_config.get("dtype", torch.bfloat16)

    results = []
    total_start = time.perf_counter()

    for idx, size_bytes in enumerate(sizes):
        num_elements = size_bytes // 2  # bf16/fp16
        if dtype == torch.float32:
            num_elements = size_bytes // 4

        tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

        best_backend = None
        best_time = float("inf")

        for backend_name, bench_fn in backends.items():
            try:
                t = bench_fn(tensor, group, num_iters=num_iters)
            except Exception as e:
                logger.debug("%s: %s", backend_name, e)
                t = None

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
                    f"{best_backend:20s} {best_time:8.2f} us"
                )

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