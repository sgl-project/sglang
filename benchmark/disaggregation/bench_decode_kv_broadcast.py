# pyright: basic
"""Benchmark the MLA decode-side KV relay (`DecodeKVBroadcaster`).

How long the relay adds to a decode scheduler tick, and how much of that is the
collective vs. the gather/scatter kernel.

    python benchmark/disaggregation/bench_decode_kv_broadcast.py --world-size=8

    # DeepSeek-V3.2 shape, with the DSA indexer K cache relayed alongside the KV
    python benchmark/disaggregation/bench_decode_kv_broadcast.py --world-size=8 --dsa

With `--world-size=1` the collective degrades to a no-op, so the numbers isolate
the kernel.

Reported per token count, as the max over ranks (the scheduler waits for the
slowest):

  relay(ms)   full `DecodeKVBroadcaster.broadcast()` -- what the scheduler pays
  kernel(ms)  the same call with the collective stubbed out -- gather/scatter only
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.decode_kv_broadcast import (
    DecodeKVBroadcaster,
    default_chunk_bytes,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64


class _DistComm:
    """The slice of the relay communicator the broadcaster uses, over WORLD."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.available = True
        # Cleared by the broadcaster, like a real PyNcclCommunicator's.
        self.disabled = True

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        if self.world_size > 1:
            dist.broadcast(tensor, src=src)


class _NoopComm(_DistComm):
    """Keeps the gather/scatter kernels but drops the collective."""

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        pass


def _make_pool(args, device: str):
    if args.dsa:
        return DSATokenToKVPool(
            size=args.pool_size,
            page_size=PAGE_SIZE,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=args.layer_num,
            device=device,
            index_head_dim=INDEX_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        )
    return MLATokenToKVPool(
        size=args.pool_size,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=args.layer_num,
        device=device,
        enable_memory_saver=False,
    )


def _payload_stats(
    broadcaster: DecodeKVBroadcaster, num_tokens: int, num_pages: int
) -> tuple[int, int]:
    """Bytes relayed, and the number of chunks `_relay` splits them into."""
    total_bytes = 0
    num_chunks = 0
    for groups, num_rows in (
        (broadcaster.kv_groups, num_tokens),
        (broadcaster.state_groups, num_pages),
    ):
        for group in groups:
            total_bytes += group.relay_elems(num_rows) * 4
            num_chunks += -(-num_rows // group.rows_per_chunk)
    return total_bytes, num_chunks


def _median_ms(fn: Callable[[], None], iters: int, warmup: int) -> float:
    """Median wall time of `fn`, with every rank stepping in lockstep."""

    def barrier() -> None:
        if dist.is_initialized():
            dist.barrier()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        barrier()
    return statistics.median(samples)


def _max_over_ranks(value: float) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _bench_one(args, pool, num_tokens: int, chunk_bytes: int, device: str, log) -> None:
    # Rows are scattered across the pool: a decode pool that has been running for
    # a while hands out free slots, not a contiguous span, and the gather kernel's
    # read pattern follows that. Seeded, so every rank relays the same row count --
    # the broadcaster is a collective and mismatched counts deadlock.
    generator = torch.Generator(device=device).manual_seed(num_tokens)
    kv_indices = torch.randperm(args.pool_size, device=device, generator=generator)[
        :num_tokens
    ]
    state_indices = [torch.unique(kv_indices // PAGE_SIZE)] if args.dsa else []

    def make(comm_cls) -> DecodeKVBroadcaster:
        return DecodeKVBroadcaster(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=None,
            relay_comm=comm_cls(args.world_size, args.rank),
            attn_tp_rank=args.rank,
            attn_tp_size=args.world_size,
            # No forward here, so the relay's fence against it is a no-op and
            # the timed stream still sees the whole relay.
            forward_stream=torch.cuda.current_stream(),
            chunk_bytes=chunk_bytes,
        )

    relay, kernel_only = make(_DistComm), make(_NoopComm)
    total_bytes, num_chunks = _payload_stats(
        relay, num_tokens, sum(pages.numel() for pages in state_indices)
    )
    timings = [
        _max_over_ranks(
            _median_ms(
                lambda: broadcaster.broadcast([kv_indices], state_indices),
                args.iters,
                args.warmup,
            )
        )
        for broadcaster in (relay, kernel_only)
    ]

    mb = total_bytes / 1e6
    log(
        f"{num_tokens:>8}{mb:>10.1f}{relay.relay_buf.numel() * 4 / 2**20:>10.1f}"
        f"{num_chunks:>8}{timings[0]:>12.2f}{timings[1]:>12.2f}{mb / timings[0]:>10.1f}"
    )
    del relay, kernel_only
    torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=str,
        default="256,1024,4096,16384,65536,131072",
        help="Comma-separated transferred-token counts per relay call.",
    )
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--layer-num", type=int, default=61)
    parser.add_argument(
        "--dsa",
        action="store_true",
        help="Use DSATokenToKVPool, so the indexer K cache is relayed too.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=str,
        default=str(default_chunk_bytes()),
        help="Comma-separated relay buffer sizes, in bytes. Sets rows/chunk, "
        "and is also the persistent HBM the relay holds.",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    args.token_counts = [int(t) for t in args.tokens.split(",")]
    args.chunk_byte_sizes = [int(c) for c in args.chunk_bytes.split(",")]
    pool_size = 2 * max(args.token_counts)
    args.pool_size = (pool_size + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    return args


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run_rank(rank: int, args: argparse.Namespace, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    device = f"cuda:{rank}"
    if args.world_size > 1:
        dist.init_process_group(backend="nccl", device_id=torch.device(device))

    def log(line: str) -> None:
        if rank == 0:
            print(line, flush=True)

    args.rank = rank
    pool = _make_pool(args, device)
    log(
        f"world_size={args.world_size} pool={type(pool).__name__} "
        f"layers={args.layer_num}"
    )
    log(
        f"{'tokens':>8}{'MB':>10}{'stageMB':>10}{'chunks':>8}"
        f"{'relay(ms)':>12}{'kernel(ms)':>12}{'GB/s':>10}"
    )
    for num_tokens in args.token_counts:
        for chunk_bytes in args.chunk_byte_sizes:
            _bench_one(args, pool, num_tokens, chunk_bytes, device, log)

    if args.world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    args = _parse_args()
    mp.spawn(_run_rank, args=(args, _free_port()), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
