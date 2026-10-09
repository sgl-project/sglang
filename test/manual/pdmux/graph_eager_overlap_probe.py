"""Multi-GPU graph/eager ordering probe, without loading a model.

Run with torchrun and an external timeout. Passing this probe does not verify
model state, HiCache, output parity, or the original longcodebench regression.
"""

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--layers-per-slice", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=12)
    parser.add_argument("--delay-ms", type=float, default=3)
    parser.add_argument("--pdmux-config", default="")
    parser.add_argument("--stream-idx", type=int, default=1)
    parser.add_argument("--dynamic-alloc", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--use-existing-ordering",
        action="store_true",
        help="Diagnostic A/B only: bypass configuration and keep the current NCCL env",
    )
    args = parser.parse_args()
    for name in (
        "chunks",
        "layers",
        "layers_per_slice",
        "tokens",
        "hidden",
        "decode_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.multiplex.launch_order import configure_pdmux_nccl_launch_order

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not args.use_existing_ordering:
        configure_pdmux_nccl_launch_order()
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        raise ValueError("This probe needs at least two GPUs/processes")
    prefill_group = dist.new_group(backend="gloo")
    device = torch.device("cuda", local_rank)
    decode_comm = PyNcclCommunicator(dist.group.WORLD, device)
    prefill_comm = PyNcclCommunicator(prefill_group, device)
    decode_comm.disabled = prefill_comm.disabled = False

    if args.pdmux_config:
        from sglang.srt.multiplex.pdmux_context import (
            get_stream_groups,
            initialize_stream_groups,
            load_pdmux_config,
        )

        initialize_stream_groups(local_rank, load_pdmux_config(args.pdmux_config))
        prefill_stream, decode_stream = get_stream_groups()[args.stream_idx]
    else:
        prefill_stream = torch.cuda.Stream()
        decode_stream = torch.cuda.Stream()
    weight = torch.eye(args.hidden, dtype=torch.bfloat16, device=device)
    decode_x = torch.ones(
        args.decode_tokens, args.hidden, dtype=torch.bfloat16, device=device
    )
    decode_y = torch.empty_like(decode_x)
    prefill_x = torch.ones(
        args.tokens, args.hidden, dtype=torch.bfloat16, device=device
    )
    prefill_y = torch.empty_like(prefill_x)
    # Publish allocations before warmup/capture on the two lane streams.
    torch.cuda.synchronize()

    def decode_step():
        for _ in range(args.layers):
            torch.mm(decode_x, weight.t(), out=decode_y)
            decode_comm.all_reduce(decode_y)
            decode_x.copy_(decode_y).mul_(1 / world)

    with torch.cuda.stream(decode_stream):
        for _ in range(3):
            decode_step()
    decode_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=decode_stream):
        decode_step()
    decode_stream.synchronize()
    tick = 0

    def trace(phase, chunk, start):
        if args.trace:
            print(
                f"rank={rank} tick={tick} chunk={chunk} layer={start} phase={phase}",
                flush=True,
            )

    began = time.monotonic()
    for chunk in range(args.chunks):
        # Rotate the slow, active prefill rank; other ranks still submit every
        # prefill collective. Decode has one IDLE rank and a common graph.
        busy = (5 + chunk) % world
        live = 1 + (chunk * 137) % args.tokens if args.dynamic_alloc else args.tokens
        start = 0
        ready = False
        done = torch.cuda.Event()
        while not ready:
            counts = torch.tensor([int(rank != busy), live if rank == busy else 0])
            global_counts = [torch.empty_like(counts) for _ in range(world)]
            dist.all_gather(global_counts, counts)
            assert sum(int(c[1]) for c in global_counts) == live
            trace("decode_begin", chunk, start)
            if rank == (busy + 1) % world:
                time.sleep(args.delay_ms / 1000)
            with torch.cuda.stream(decode_stream):
                decode_x.fill_(float(rank != busy))
                graph.replay()
            trace("decode_returned", chunk, start)
            if start < args.layers:
                trace("prefill_begin", chunk, start)
                if rank == busy:
                    time.sleep(args.delay_ms / 1000)
                end = min(start + args.layers_per_slice, args.layers)
                with torch.cuda.stream(prefill_stream):
                    for _ in range(start, end):
                        prefill_y.zero_()
                        if rank == busy:
                            if args.dynamic_alloc:
                                prefill_y[:live].copy_(
                                    F.linear(prefill_x[:live], weight)
                                )
                            else:
                                torch.mm(prefill_x, weight.t(), out=prefill_y)
                        prefill_comm.all_reduce(prefill_y)
                    if end == args.layers:
                        done.record()
                trace("prefill_returned", chunk, start)
                start = end
            trace("decode_wait_begin", chunk, start)
            decode_stream.synchronize()
            trace("decode_wait_end", chunk, start)
            if start == args.layers:
                vote = torch.tensor([int(done.query())])
                dist.all_reduce(vote)
                ready = int(vote[0]) == world
            tick += 1
        # Both lane fences are complete now; value checks cannot hold up a peer
        # that still needs to submit the other lane's collective.
        torch.testing.assert_close(
            decode_x.cpu().float(),
            torch.full(decode_x.shape, (world - 1) / world),
            rtol=0,
            atol=0.01,
        )
        expected = torch.zeros(prefill_y.shape)
        expected[:live] = 1
        torch.testing.assert_close(prefill_y.cpu().float(), expected, rtol=0, atol=0)
        if rank == 0:
            print(
                f"completed_chunks={chunk + 1} ticks={tick} elapsed={time.monotonic() - began:.2f}s",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
