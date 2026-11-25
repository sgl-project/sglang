"""Benchmark the KV-shard prefix gather used where PyNccl is unavailable (XPU).

Under --enable-kv-cache-sharding each rank gathers the next layer's prefix pages
in place on a side stream while the current layer computes
(PageInterleaveKVPoolMixin._prefetch_layer). This times the pool's own
_TorchDistributedAllGather, which owns a process group, against the same gather
on the shared world group, and reports how much of the gather hides behind a
concurrent compute load.

Run with one process per device, e.g. on 2 XPUs:
    torchrun --nproc-per-node 2 benchmark/kernels/all_gather/bench_kv_shard_gather.py --device xpu
CPU smoke run (gloo, no overlap numbers worth reading):
    torchrun --nproc-per-node 2 benchmark/kernels/all_gather/bench_kv_shard_gather.py --device cpu

Each call gathers into the next of several buffer copies whose total is at least
--rotate-mb, so small blocks are not timed L2-hot. The matmul is only the
concurrent load the gather has to hide behind; its cache behavior is not measured.
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.page_interleave_pool import _TorchDistributedAllGather

_BACKEND = {"xpu": "xccl", "cuda": "nccl", "cpu": "gloo"}


class _Timer:
    """Wall-clock spans that include every queued device stream."""

    def __init__(self, device: torch.device):
        self.module = torch.get_device_module(device) if device.type != "cpu" else None

    def sync(self):
        if self.module is not None:
            self.module.synchronize()

    def run(self, fn, repeat: int) -> float:
        self.sync()
        start = time.perf_counter()
        for _ in range(repeat):
            fn()
        self.sync()
        return (time.perf_counter() - start) * 1e6 / repeat


def _max_across_ranks(value: float, device: torch.device) -> float:
    # On the device: an xccl/nccl world group has no CPU backend.
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _rotating_buffers(args, device, block_rows):
    """(output, send) pairs, send being this rank's block of its output."""
    world, rank = dist.get_world_size(), dist.get_rank()
    output_bytes = world * block_rows * args.row_bytes
    copies = max(1, -(-args.rotate_mb * 2**20 // output_bytes))
    pairs = []
    for _ in range(copies):
        output = torch.zeros(
            world * block_rows, args.row_bytes, dtype=torch.uint8, device=device
        )
        send = output[rank * block_rows : (rank + 1) * block_rows]
        send.fill_(rank + 1)
        pairs.append((output, send))
    return pairs


def _bench_size(args, device, timer, gathers, block_rows):
    world = dist.get_world_size()
    buffers = _rotating_buffers(args, device, block_rows)
    cursor = [0]

    def next_buffers():
        cursor[0] = (cursor[0] + 1) % len(buffers)
        return buffers[cursor[0]]

    a = torch.randn(
        args.matmul_dim, args.matmul_dim, device=device, dtype=torch.float32
    )
    side = timer.module.Stream() if timer.module is not None else None

    def compute():
        for _ in range(args.matmuls):
            a @ a

    def gather_on_side(gather):
        output, send = next_buffers()
        if side is None:
            return gather(output, send)
        side.wait_stream(timer.module.current_stream())
        with timer.module.stream(side):
            gather(output, send)
        timer.module.current_stream().wait_stream(side)

    def both(gather):
        output, send = next_buffers()
        if side is None:
            gather(output, send)
            return compute()
        side.wait_stream(timer.module.current_stream())
        with timer.module.stream(side):
            gather(output, send)
        compute()
        timer.module.current_stream().wait_stream(side)

    for fn in [compute] + [lambda g=g: both(g) for g in gathers.values()]:
        fn()  # warm up kernels and communicators
    t_compute = _max_across_ranks(timer.run(compute, args.repeat), device)
    row = {"block_mb": block_rows * args.row_bytes / 2**20, "compute_us": t_compute}
    for name, gather in gathers.items():
        t_gather = _max_across_ranks(
            timer.run(lambda: gather_on_side(gather), args.repeat), device
        )
        t_both = _max_across_ranks(timer.run(lambda: both(gather), args.repeat), device)
        hidden = max(0.0, t_compute + t_gather - t_both)
        output_bytes = buffers[0][0].numel()
        row[name] = {
            "gather_us": t_gather,
            # Output bytes per second; each rank moves (world - 1) / world of them.
            "alg_gb_s": output_bytes / t_gather / 1e3,
            "bus_gb_s": output_bytes * (world - 1) / world / t_gather / 1e3,
            "both_us": t_both,
            # Share of the gather (or the compute, if shorter) that overlapped.
            # Capped: run-to-run noise can make both faster than the parts.
            "overlap": min(1.0, hidden / min(t_compute, t_gather)),
        }
    # Correctness pass: every copy, every group, after timing.
    for gather in gathers.values():
        for output, send in buffers:
            output.zero_()
            send.fill_(dist.get_rank() + 1)
            gather(output, send)
            timer.sync()
            expected = torch.arange(1, world + 1, dtype=output.dtype, device=device)
            assert torch.equal(
                output.view(world, -1),
                expected[:, None].expand(world, block_rows * args.row_bytes),
            ), "gather produced wrong rank blocks"
    row["buffer_copies"] = len(buffers)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=sorted(_BACKEND), default="xpu")
    # Default row: one token's K+V for 8 bf16 heads of dim 128.
    parser.add_argument("--row-bytes", type=int, default=2 * 8 * 128 * 2)
    parser.add_argument(
        "--block-tokens", type=int, nargs="+", default=[1024, 4096, 16384]
    )
    parser.add_argument("--matmul-dim", type=int, default=4096)
    parser.add_argument("--matmuls", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--rotate-mb",
        type=int,
        default=256,
        help="Minimum total size of the rotated gather buffers (defeats L2 reuse)",
    )
    parser.add_argument("--json", help="Write the per-size results to this file")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        torch.get_device_module(args.device).set_device(local_rank)
        device = torch.device(args.device, local_rank)
    dist.init_process_group(_BACKEND[args.device])
    world_group = dist.group.WORLD
    shard_group = SimpleNamespace(
        ranks=list(range(dist.get_world_size())), device_group=world_group
    )
    dedicated = _TorchDistributedAllGather(shard_group)
    gathers = {
        "dedicated_group": dedicated.all_gather,
        "shared_group": lambda out, send: dist.all_gather_into_tensor(
            out, send, group=world_group
        ),
    }

    timer = _Timer(device)
    results = [_bench_size(args, device, timer, gathers, n) for n in args.block_tokens]
    if dist.get_rank() == 0:
        print(
            f"world={dist.get_world_size()} device={args.device} row_bytes={args.row_bytes}"
        )
        print(
            f"{'block MB':>9} {'compute us':>11} | {'group':>15} {'gather us':>10} "
            f"{'algGB/s':>8} {'busGB/s':>8} {'both us':>9} {'overlap':>8}"
        )
        for row in results:
            for name in gathers:
                r = row[name]
                print(
                    f"{row['block_mb']:9.2f} {row['compute_us']:11.1f} | {name:>15} "
                    f"{r['gather_us']:10.1f} {r['alg_gb_s']:8.2f} {r['bus_gb_s']:8.2f} "
                    f"{r['both_us']:9.1f} {r['overlap']:8.0%}"
                )
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"args": vars(args), "results": results}, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
