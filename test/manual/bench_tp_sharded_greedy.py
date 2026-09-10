"""Compare full-vocab all-gather+argmax with TP candidate all-gather.

Example:
  torchrun --standalone --nproc-per-node=2 test/manual/bench_tp_sharded_greedy.py
"""

import argparse
import os

import torch
import torch.distributed as dist

from sglang.srt.layers.tp_sharded_greedy import tp_sharded_greedy_argmax


def _time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    local_width = (args.vocab_size + world_size - 1) // world_size
    padded_vocab = local_width * world_size
    vocab_start = rank * local_width
    vocab_end = min(vocab_start + local_width, args.vocab_size)

    if rank == 0:
        print("batch,full_allgather_argmax_us,candidate_allgather_us,speedup")
    for batch in args.batches:
        gen = torch.Generator(device=device).manual_seed(1234 + rank)
        local_logits = torch.randn(
            batch,
            local_width,
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        ).float()
        if vocab_end < vocab_start + local_width:
            local_logits[:, vocab_end - vocab_start :] = -torch.inf

        full_buffer = torch.empty(
            padded_vocab * batch, dtype=local_logits.dtype, device=device
        )

        def full_path():
            dist.all_gather_into_tensor(full_buffer, local_logits.contiguous().view(-1))
            return (
                full_buffer.view(world_size, batch, local_width)
                .permute(1, 0, 2)
                .reshape(batch, padded_vocab)[:, : args.vocab_size]
                .argmax(dim=-1)
            )

        def candidate_path():
            return tp_sharded_greedy_argmax(
                local_logits,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
                process_group=dist.group.WORLD,
                world_size=world_size,
            )

        torch.testing.assert_close(candidate_path(), full_path(), rtol=0, atol=0)
        full_ms = _time_ms(full_path, args.warmup, args.iterations)
        candidate_ms = _time_ms(candidate_path, args.warmup, args.iterations)
        if rank == 0:
            print(
                f"{batch},{full_ms * 1000:.2f},{candidate_ms * 1000:.2f},"
                f"{full_ms / candidate_ms:.3f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
