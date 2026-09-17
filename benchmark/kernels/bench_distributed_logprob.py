"""TP benchmark for exact input logprobs without full-vocab AllGather.

Example:

  torchrun --standalone --nproc-per-node=4 \
    benchmark/kernels/bench_distributed_logprob.py --rows 2048 8192
"""

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from sglang.srt.layers.logprob_processor import (
    compute_distributed_row_log_normalizer,
    compute_row_log_normalizer,
    get_distributed_token_scores,
)


class TorchTpGroup:
    def __init__(self):
        self.device_group = dist.group.WORLD

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.device_group)
        return tensor


@dataclass
class BenchResult:
    rows: int
    vocab_size: int
    tp_size: int
    baseline_ms: float
    distributed_ms: float
    speedup: float
    baseline_peak_mib: float
    distributed_peak_mib: float
    peak_reduction: float
    max_abs_error: float
    baseline_allgather_receive_mib_per_rank: float
    distributed_scalar_receive_mib_per_rank: float


def shard_bounds(vocab_size: int, rank: int, world_size: int) -> tuple[int, int, int]:
    padded_vocab = ((vocab_size + 63) // 64) * 64
    assert padded_vocab % world_size == 0
    width = padded_vocab // world_size
    start = rank * width
    end = min(start + width, vocab_size)
    return start, end, width


def make_local_logits(
    rows: int,
    vocab_size: int,
    rank: int,
    world_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int]:
    start, end, width = shard_bounds(vocab_size, rank, world_size)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260809 + rank)
    logits = torch.randn((rows, width), device="cuda", dtype=dtype, generator=generator)
    valid_width = end - start
    if valid_width < width:
        logits[:, valid_width:] = 1e4
    return logits, start, end


def max_across_ranks(value: float) -> float:
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        output = fn()
        del output
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = fn()
        del output
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def peak_memory_mib(fn) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return (peak - base) / (1024**2)


def run_correctness(group: TorchTpGroup, rank: int, world_size: int) -> dict:
    rows, vocab_size = 17, 154883
    start, end, width = shard_bounds(vocab_size, rank, world_size)

    # Every rank constructs the same small reference, then takes its shard.
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234)
    full = torch.randn((rows, vocab_size), device="cuda", generator=generator)
    full = full + 1e4
    local = torch.empty((rows, width), device="cuda", dtype=torch.float32)
    local.fill_(1e6)
    if end > start:
        local[:, : end - start] = full[:, start:end]

    row_max, row_log_sum = compute_distributed_row_log_normalizer(
        local, end - start, group
    )
    row_indices = torch.arange(rows, device="cuda").repeat_interleave(world_size)
    owner_tokens = torch.tensor(
        [min((owner * width) + 3, vocab_size - 1) for owner in range(world_size)],
        device="cuda",
    )
    token_ids = owner_tokens.repeat(rows)
    raw_scores = get_distributed_token_scores(
        local, row_indices, token_ids, start, end, group
    )
    got = (raw_scores - row_max[row_indices]) - row_log_sum[row_indices]
    reference = torch.log_softmax(full, dim=-1)[row_indices, token_ids]
    max_abs_error = max_across_ranks((got - reference).abs().max().item())
    return {
        "rows": rows,
        "vocab_size": vocab_size,
        "max_abs_error": max_abs_error,
        "padding_excluded": True,
        "owners_covered": world_size,
    }


def run_benchmark(
    group: TorchTpGroup,
    rank: int,
    world_size: int,
    rows: int,
    vocab_size: int,
    warmup: int,
    iterations: int,
) -> BenchResult:
    local_logits, vocab_start, vocab_end = make_local_logits(
        rows, vocab_size, rank, world_size, torch.bfloat16
    )
    local_width = local_logits.shape[1]
    valid_width = vocab_end - vocab_start
    target_ids = torch.arange(rows, device="cuda", dtype=torch.long) % vocab_size
    row_indices = torch.arange(rows, device="cuda", dtype=torch.long)

    def baseline():
        gathered = torch.empty(
            (world_size * rows, local_width),
            device="cuda",
            dtype=local_logits.dtype,
        )
        dist.all_gather_into_tensor(gathered, local_logits)
        full_logits = (
            gathered.view(world_size, rows, local_width)
            .permute(1, 0, 2)
            .reshape(rows, world_size * local_width)[:, :vocab_size]
            .float()
        )
        row_max, row_log_sum = compute_row_log_normalizer(full_logits)
        raw_scores = full_logits[row_indices, target_ids]
        return (raw_scores - row_max) - row_log_sum

    def distributed():
        local_fp32 = local_logits.float()
        row_max, row_log_sum = compute_distributed_row_log_normalizer(
            local_fp32, valid_width, group
        )
        raw_scores = get_distributed_token_scores(
            local_fp32,
            row_indices,
            target_ids,
            vocab_start,
            vocab_end,
            group,
        )
        return (raw_scores - row_max) - row_log_sum

    reference = baseline()
    candidate = distributed()
    torch.cuda.synchronize()
    max_abs_error = max_across_ranks((candidate - reference).abs().max().item())
    del reference, candidate

    baseline_ms = max_across_ranks(time_cuda(baseline, warmup, iterations))
    distributed_ms = max_across_ranks(time_cuda(distributed, warmup, iterations))
    baseline_peak_mib = max_across_ranks(peak_memory_mib(baseline))
    distributed_peak_mib = max_across_ranks(peak_memory_mib(distributed))

    full_bf16_bytes = rows * vocab_size * 2
    baseline_receive = full_bf16_bytes * (world_size - 1) / world_size
    # MAX, SUM, and selected-score SUM. Ring all-reduce receives
    # 2*(p-1)/p times the tensor size per rank.
    scalar_receive = 3 * (2 * (world_size - 1) / world_size) * rows * 4

    return BenchResult(
        rows=rows,
        vocab_size=vocab_size,
        tp_size=world_size,
        baseline_ms=baseline_ms,
        distributed_ms=distributed_ms,
        speedup=baseline_ms / distributed_ms,
        baseline_peak_mib=baseline_peak_mib,
        distributed_peak_mib=distributed_peak_mib,
        peak_reduction=baseline_peak_mib / distributed_peak_mib,
        max_abs_error=max_abs_error,
        baseline_allgather_receive_mib_per_rank=baseline_receive / (1024**2),
        distributed_scalar_receive_mib_per_rank=scalar_receive / (1024**2),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = TorchTpGroup()
    try:
        correctness = run_correctness(group, dist.get_rank(), dist.get_world_size())
        results = [
            asdict(
                run_benchmark(
                    group,
                    dist.get_rank(),
                    dist.get_world_size(),
                    rows,
                    args.vocab_size,
                    args.warmup,
                    args.iterations,
                )
            )
            for rows in args.rows
        ]
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {"correctness": correctness, "benchmarks": results}, indent=2
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
