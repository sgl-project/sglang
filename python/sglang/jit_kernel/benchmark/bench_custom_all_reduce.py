"""
Benchmark JIT custom all-reduce (v2) vs NCCL vs AOT custom all-reduce (v1).

Usage (torchrun required for multi-GPU):
    torchrun --nproc_per_node=2 bench_custom_all_reduce.py
    torchrun --nproc_per_node=4 bench_custom_all_reduce.py --dtype float16
    torchrun --nproc_per_node=8 bench_custom_all_reduce.py --warmup 10 --iters 100

The script initializes all three backends, then benchmarks each over a sweep
of message sizes.  Results are printed as a comparison table on rank 0.
"""

import argparse
import contextlib
import gc
import logging
import os
from math import isnan
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.jit_kernel.benchmark.utils import is_in_ci

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

MESSAGE_SIZES_BYTES = [
    4 * 1024,  # 4K
    16 * 1024,  # 16K
    64 * 1024,  # 64K
    128 * 1024,  # 128K
    3 * 64 * 1024,  # 192K
    4 * 64 * 1024,  # 256K
    3 * 128 * 1024,  # 384K
    4 * 128 * 1024,  # 512K
    5 * 128 * 1024,  # 640K
    6 * 128 * 1024,  # 768K
    7 * 128 * 1024,  # 896K
    1 * 1024 * 1024,  # 1M
    2 * 1024 * 1024,  # 2M
    3 * 1024 * 1024,  # 2M
    4 * 1024 * 1024,  # 4M
    8 * 1024 * 1024,  # 8M
    16 * 1024 * 1024,  # 16M
    32 * 1024 * 1024,  # 32M
]


# ---------------------------------------------------------------------------
# Backend wrappers - each exposes a uniform interface:
#   .name          - display name
#   .capture()     - context manager for CUDA-graph recording
#   .all_reduce()  - perform an all-reduce and return the result tensor
# ---------------------------------------------------------------------------


class NCCLAllReduceBackend:
    name = "NCCL"

    def __init__(self, group: dist.ProcessGroup):
        self.group = group

    def capture(self, register_input: bool):
        return contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self.group)
        return tensor


class AOTAllReduceBackend:
    name = "AOT"

    def __init__(self, group: dist.ProcessGroup, device: torch.device):
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        max_size = max(MESSAGE_SIZES_BYTES)
        self.comm = CustomAllreduce(group, device, max_size=max_size)
        if self.comm.disabled:
            raise RuntimeError("AOT CustomAllreduce is disabled on this system")

    def capture(self, register_input: bool):
        return self.comm.capture()  # ignore register_input since v1 always requires it

    def all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


class JITAllReduceBackend:
    name = "JIT"

    def __init__(self, group: dist.ProcessGroup, device: torch.device):
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        max_size = max(MESSAGE_SIZES_BYTES)
        self.comm = CustomAllReduceV2(group, device, max_pull_size=max_size)
        if self.comm.disabled:
            raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")

    def capture(self, register_input: bool):
        return self.comm.capture() if register_input else contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


class FlashInferAllReduceBackend:
    name = "FI"

    def __init__(self, group: dist.ProcessGroup, dtype: torch.dtype):
        import flashinfer.comm as comm

        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
        max_size = max(MESSAGE_SIZES_BYTES)
        hidden_dim = min(MESSAGE_SIZES_BYTES) // 2
        num_tokens = max_size // hidden_dim
        self.comm = comm
        self.hidden_dim = hidden_dim
        self.workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=num_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )

    def capture(self, *_):
        return contextlib.nullcontext()

    def all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        return self.comm.allreduce_fusion(
            input=tensor.view(-1, self.hidden_dim),
            workspace=self.workspace,
            pattern=self.comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            fp32_acc=True,
        )


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="bfloat16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--no-inplace", dest="register_input", action="store_false")
    return p.parse_args()


@torch.inference_mode()
def bench_one(
    backend,
    inp: torch.Tensor,
    warmup: int,
    iters: int,
    group: dist.ProcessGroup,
    register_input: bool,
) -> float:
    """
    Run *warmup* iterations of all-reduce first.
    Return the average time for *iters* iterations of all-reduce.
    """
    dist.barrier(group=group)
    for _ in range(warmup):
        backend.all_reduce(inp)
    torch.cuda.synchronize()

    # Capture a CUDA graph with *iters* all-reduce calls.
    inp_batch = torch.stack([inp] * 4)
    graph = torch.cuda.CUDAGraph()
    with backend.capture(register_input):
        with torch.cuda.graph(graph):
            for i in range(iters):
                backend.all_reduce(inp_batch[i % 4])

    torch.cuda.synchronize()
    # Warm up the graph once.
    graph.replay()

    # Timed replay.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph.replay()  # make the stream busy
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_sweep(
    backend,
    sizes_bytes: List[int],
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    group: dist.ProcessGroup,
    register_input: bool,
) -> Dict[int, float]:
    """Benchmark one backend over all message sizes."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    results: Dict[int, float] = {}
    for sz in sizes_bytes:
        numel = sz // elem_size
        inp = torch.zeros(numel, dtype=dtype, device=device)
        try:
            elapsed_ms = bench_one(backend, inp, warmup, iters, group, register_input)
            results[sz] = elapsed_ms * 1000  # convert to us per iter
        except AssertionError:
            results[sz] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------


def print_results(
    backends: list,
    all_results: Dict[str, Dict[int, float]],
    sizes_bytes: List[int],
) -> None:
    """Print a comparison table on rank 0."""

    def human_bytes(n: int) -> str:
        for suffix, unit in [("M", 1 << 20), ("K", 1 << 10)]:
            if n >= unit and n % unit == 0:
                return f"{n // unit}{suffix}"
        return f"{n}B"

    def fmt_us(v: float) -> str:
        return f"{v:13.1f}" if not isnan(v) else "     n/a"

    names = [b.name for b in backends]
    nccl_name = "NCCL"

    # Header
    header_cols = [f"{n:>13}" for n in names]
    speedup_cols = [f"{n:>13}/NCCL" for n in names if n != nccl_name]
    header = f"{'Size':>8}  " + "  ".join(header_cols)
    for sc in speedup_cols:
        header += f"  {sc}"
    header += "  "
    print()
    print(header)
    print("-" * len(header))

    # Rows
    for sz in sizes_bytes:
        row = f"{human_bytes(sz):>8}"
        nccl_lat = all_results[nccl_name][sz]
        for n in names:
            row += f"  {fmt_us(all_results[n][sz])}"
        for n in names:
            if n == nccl_name:
                continue
            lat = all_results[n][sz]
            if not isnan(lat):
                row += f"  {nccl_lat / lat:17.2f}x"
            else:
                row += f"  {'n/a':>17}"
        print(row)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def init_distributed():
    """Initialize distributed groups using torchrun env vars.

    Returns (rank, world_size, device, cpu_group, nccl_group).
    """
    import sglang.srt.distributed.parallel_state as ps

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.set_stream(torch.cuda.Stream())  # use a non-default stream

    torch.distributed.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )

    cpu_group = coord.cpu_group
    nccl_group = coord.device_group
    assert nccl_group is not None
    return rank, world_size, device, cpu_group, nccl_group


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    rank, world_size, device, cpu_group, nccl_group = init_distributed()

    # Instantiate backends.
    backends = [
        NCCLAllReduceBackend(nccl_group),
        JITAllReduceBackend(cpu_group, device),
    ]
    if world_size in [2, 4, 6, 8]:
        backends.insert(1, AOTAllReduceBackend(cpu_group, device))
    if world_size in [2, 4, 8]:
        backends.append(FlashInferAllReduceBackend(cpu_group, dtype))

    # Run benchmarks.
    all_results: Dict[str, Dict[int, float]] = {}
    torch.cuda.synchronize()
    for backend in backends:
        if rank == 0:
            print(f"Benchmarking {backend.name} ...")
        all_results[backend.name] = bench_sweep(
            backend,
            MESSAGE_SIZES_BYTES,
            dtype,
            device,
            args.warmup,
            args.iters,
            cpu_group,
            args.register_input,
        )

    # Aggregate across ranks (use max to reflect the slowest rank).
    for name in list(all_results):
        for sz in MESSAGE_SIZES_BYTES:
            val = all_results[name].get(sz)
            if val is None:
                continue
            t = torch.tensor([val], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=nccl_group)
            all_results[name][sz] = t.item()

    # Print results on rank 0.
    if rank == 0:
        print_results(backends, all_results, MESSAGE_SIZES_BYTES)

    del backends, all_results
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__" and not is_in_ci():
    main()
