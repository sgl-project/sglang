#!/usr/bin/env python3
"""Verify the single-image broadcast fast path vs the pad-to-max all_gather:
bitwise equivalence + timing, on real NCCL over 8 ranks. Mirrors what
run_dp_sharded_mrope_vision_model does for a single image (one owner rank
holds the embedding, the rest are empty). torchrun --nproc_per_node=8."""

import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    owner = 3  # arbitrary non-zero owner, as LB would pick
    n_tok, hidden = 5476, 4096  # ~2048^2 image, typical tower output (~44MB bf16)

    # deterministic ground-truth owner embedding, known to every rank
    gen = torch.Generator(device=dev).manual_seed(12345)
    owner_truth = torch.randn(
        n_tok, hidden, dtype=torch.bfloat16, device=dev, generator=gen
    )
    emb = (
        owner_truth.clone()
        if rank == owner
        else torch.empty(0, hidden, dtype=torch.bfloat16, device=dev)
    )
    max_len = n_tok  # single image: max over ranks == owner's length

    def path_a():  # current: pad-to-max all_gather + reconstruct owner rows
        padded = torch.empty(max_len, hidden, dtype=torch.bfloat16, device=dev)
        if emb.shape[0] > 0:
            padded[: emb.shape[0]].copy_(emb)
        gathered = [
            torch.empty(max_len, hidden, dtype=torch.bfloat16, device=dev)
            for _ in range(world)
        ]
        dist.all_gather(gathered, padded)
        return gathered[owner][:n_tok]

    def path_b():  # fast path: broadcast from owner
        buf = (
            emb.contiguous()
            if rank == owner
            else torch.empty(n_tok, hidden, dtype=torch.bfloat16, device=dev)
        )
        dist.broadcast(buf, src=owner)
        return buf

    out_a, out_b = path_a(), path_b()
    eq_truth = torch.equal(out_a, owner_truth)
    eq_ab = torch.equal(out_a, out_b)
    eq_b_truth = torch.equal(out_b, owner_truth)

    def timeit(fn, n=100):
        for _ in range(15):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        return (time.perf_counter() - t0) / n * 1000

    ta, tb = timeit(path_a), timeit(path_b)
    # gather correctness flags from all ranks
    flags = torch.tensor(
        [eq_ab and eq_truth and eq_b_truth], device=dev, dtype=torch.int32
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"world={world} owner={owner} shape=[{n_tok},{hidden}] "
            f"(~{n_tok * hidden * 2 / 1e6:.0f}MB) | all_ranks_bitwise_ok={bool(flags.item())} "
            f"(A==truth={eq_truth} A==B={eq_ab}) | all_gather {ta:.3f}ms "
            f"broadcast {tb:.3f}ms speedup {ta / tb:.2f}x",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
