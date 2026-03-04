"""
Benchmark JIT custom all-reduce (v2) vs PyNccl vs AOT custom all-reduce (v1).

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

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

MESSAGE_SIZES_BYTES = [
    4 * 1024,  # 4K
    16 * 1024,  # 16K
    64 * 1024,  # 64K
    128 * 1024,  # 128K  (1-shot / 2-shot boundary)
    256 * 1024,  # 256K
    512 * 1024,  # 512K
    1 * 1024 * 1024,  # 1M
    2 * 1024 * 1024,  # 2M
    4 * 1024 * 1024,  # 4M
    8 * 1024 * 1024,  # 8M
    16 * 1024 * 1024,  # 16M
    32 * 1024 * 1024,  # 32M
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="bfloat16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args()


def human_bytes(n: int) -> str:
    for suffix, unit in [("M", 1 << 20), ("K", 1 << 10)]:
        if n >= unit and n % unit == 0:
            return f"{n // unit}{suffix}"
    return f"{n}B"


def fmt_us(v: float) -> str:
    return f"{v:8.1f}" if not isnan(v) else "     n/a"


class NCCLBackend:
    """torch.distributed NCCL all-reduce (in-place, used as baseline)."""

    name = "NCCL"

    def __init__(self, group):
        self.group = group

    def capture(self):
        return contextlib.nullcontext()

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        dist.all_reduce(out, group=self.group)
        return out


class AOTCustomARBackend:
    """AOT sgl-kernel custom all-reduce (v1)."""

    name = "AOT-CAR"

    def __init__(self, group, device):
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.comm = CustomAllreduce(group=group, device=device)
        if self.comm.disabled:
            raise RuntimeError("AOT CustomAllreduce is disabled on this system")

    def capture(self):
        return self.comm.capture()

    def run(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


class JITCustomARBackend:
    """JIT custom all-reduce (v2)."""

    name = "JIT-CAR"

    def __init__(self, group, device):
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        self.comm = CustomAllReduceV2(group=group, device=device)

        if self.comm.disabled:
            raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")

    def capture(self):
        self.comm.obj.reset_graph()  # reset graph to clear any previous inputs/state
        return self.comm.capture()

    def run(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        assert self.comm.should_custom_ar(tensor), str(tensor.shape)
        return self.comm.custom_all_reduce(tensor)


@torch.inference_mode()
def bench_one(
    backend,
    inp: torch.Tensor,
    warmup: int,
    iters: int,
    pg: dist.ProcessGroup,
) -> float:
    """Run *warmup* + *iters* iterations, return median latency in us."""

    # warmup
    dist.barrier(group=pg)
    for _ in range(warmup):
        backend.run(inp)
    torch.cuda.synchronize()

    # timed
    inp_4 = torch.stack([inp] * 4)
    # use cuda graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with backend.capture():
            for i in range(iters):
                backend.run(inp_4[i % 4])

    graph.replay()

    # replay and time
    torch.cuda.synchronize()
    dist.barrier(group=pg)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(10_000_000)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def bench_sweep(
    backend,
    sizes_bytes: List[int],
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    pg: dist.ProcessGroup,
) -> Dict[int, float]:
    """Benchmark one backend over all message sizes."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    results: Dict[int, float] = {}
    for sz in sizes_bytes:
        numel = sz // elem_size
        inp = torch.randn(numel, dtype=dtype, device=device)
        try:
            lat = bench_one(backend, inp, warmup, iters, pg)
            results[sz] = lat * 1000 / iters
        except AssertionError as e:
            results[sz] = float("nan")
    return results


def main():
    import sglang.srt.distributed.parallel_state as ps

    # disable info
    logging.basicConfig(level=logging.WARNING)

    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_stream(torch.cuda.Stream())  # use a non-default stream
    torch.cuda.set_device(device)
    # -- distributed init via torchrun env vars --
    torch.distributed.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )

    # We need a non-NCCL group for custom AR control path
    cpu_group = coord.cpu_group
    nccl_group = coord.device_group
    assert nccl_group is not None

    # -- instantiate backends --
    backends = []

    # 1) NCCL baseline
    backends.append(NCCLBackend(nccl_group))
    # 2) AOT custom AR
    backends.append(AOTCustomARBackend(cpu_group, device))
    # 3) JIT custom AR
    backends.append(JITCustomARBackend(cpu_group, device))

    # -- run benchmarks --
    all_results: Dict[str, Dict[int, float]] = {}
    for be in backends:
        if rank == 0:
            print(f"Benchmarking {be.name} ...")
        r = bench_sweep(
            be, MESSAGE_SIZES_BYTES, dtype, device, args.warmup, args.iters, nccl_group
        )
        all_results[be.name] = r

    # -- aggregate across ranks (use max to reflect the slowest rank) --
    for name in list(all_results):
        for sz in MESSAGE_SIZES_BYTES:
            local_val = all_results[name].get(sz)
            if local_val is None:
                continue
            t = torch.tensor([local_val], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=nccl_group)
            all_results[name][sz] = t.item()

    # -- print results on rank 0 --
    if rank == 0:
        names = [be.name for be in backends]
        header_cols = [f"{n:>10}" for n in names]
        # Add speedup columns
        nccl_name = "NCCL"
        speedup_cols = []
        for n in names:
            if n != nccl_name:
                speedup_cols.append(f"{n}/NCCL")
        header = f"{'Size':>8}  " + "  ".join(header_cols)
        for sc in speedup_cols:
            header += f"  {sc:>10}"
        header += "  (us, median, lower is better)"
        print()
        print(header)
        print("-" * len(header))
        for sz in MESSAGE_SIZES_BYTES:
            row = f"{human_bytes(sz):>8}"
            nccl_lat = all_results[nccl_name][sz]
            for n in names:
                row += f"  {fmt_us(all_results[n][sz])}"
            for n in names:
                if n == nccl_name:
                    continue
                lat = all_results[n][sz]
                if not isnan(lat):
                    speedup = nccl_lat / lat
                    row += f"  {speedup:10.2f}x"
                else:
                    row += f"  {'n/a':>10}"
            print(row)

    del backends, all_results  # cleanup before dist.destroy_process_group()
    gc.collect()

    # -- cleanup --
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
