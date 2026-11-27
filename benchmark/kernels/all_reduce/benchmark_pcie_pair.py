"""PCIe pair all-reduce against NCCL, per call inside a CUDA graph of back-to-back calls.

For 4 GPUs where ranks 0-1 and 2-3 each share a PCIe switch:

    torchrun --nproc-per-node 4 benchmark/kernels/all_reduce/benchmark_pcie_pair.py \
        [--rows 1,2,4,6,8,12,16,24,32,48,64] [--hidden 5120]

Prints microseconds per call (median of 5 graph replays) for NCCL and for the pair
kernel with fp32 and bf16 pair sums, and checks the pair results against a fixed-order
reference on every rank.
"""

import argparse
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.pcie_pair_ar import (
    PciePairCommunicator,
)
from sglang.srt.environ import envs


def time_graph(fn, x, iters, reps=5):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [fn(x) for _ in range(iters)]
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iters)
    out = outs[-1].clone()
    del graph
    return sorted(times)[len(times) // 2], out


def reference(x, wide):
    parts = [torch.empty_like(x) for _ in range(4)]
    dist.all_gather(parts, x)
    f = [p.float() for p in parts]
    if wide:
        return ((f[0] + f[1]) + (f[2] + f[3])).bfloat16()
    return (
        (f[0] + f[1]).bfloat16().float() + (f[2] + f[3]).bfloat16().float()
    ).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    cpu = dist.new_group(backend="gloo")
    rank = dist.get_rank()

    comms = {}
    for wide in (True, False):
        with (
            envs.SGLANG_PCIE_PAIR_AR_FP32.override(wide),
            envs.SGLANG_PCIE_PAIR_AR_MAX_BYTES.override(1 << 30),
        ):
            comm = PciePairCommunicator(cpu, device)
        if comm.disabled:
            raise SystemExit("ranks 0-1 and 2-3 are not two PCIe switch pairs")
        comms["fp32" if wide else "bf16"] = comm

    def nccl(x):
        y = x.clone()
        dist.all_reduce(y)
        return y

    if rank == 0:
        print("| rows | KB | NCCL us | pair fp32 us | pair bf16 us | pair exact |")
        print("|---|---|---|---|---|---|")
    for rows in [int(r) for r in args.rows.split(",")]:
        g = torch.Generator(device="cpu").manual_seed(1234 + rank)
        x = (torch.randn(rows, args.hidden, generator=g) * 0.5).bfloat16().to(device)
        t_nccl, _ = time_graph(nccl, x, args.iters)
        t, exact = {}, True
        for name, comm in comms.items():
            t[name], out = time_graph(comm.pcie_pair_all_reduce, x, args.iters)
            exact &= torch.equal(out, reference(x, name == "fp32"))
        ok = torch.tensor([int(exact)], device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(
                f"| {rows} | {rows * args.hidden * 2 / 1024:.0f} | {t_nccl:.1f} "
                f"| {t['fp32']:.1f} | {t['bf16']:.1f} | {'yes' if ok.item() else 'NO'} |",
                flush=True,
            )
    for comm in comms.values():
        comm.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
