"""Benchmark script for Triton sampling kernels.

Usage:
    python benchmark_triton_sampling.py             # full benchmark with individual latencies
    python benchmark_triton_sampling.py --metric-only  # single number for autoresearch
"""

import argparse

import torch
import torch.nn.functional as F

from sglang.srt.speculative.triton_sampling_kernels import (
    top_k_renorm_prob,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
)


def bench_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep * 1000  # µs


def bench_top_k(bs=1, vocab_size=131072):
    probs = torch.rand(bs, vocab_size, device="cuda", dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    k = torch.full((bs,), 50, device="cuda", dtype=torch.int64)
    return bench_fn(lambda: top_k_renorm_prob(probs, k))


def bench_top_p(bs=1, vocab_size=131072):
    probs = torch.rand(bs, vocab_size, device="cuda", dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    p = torch.full((bs,), 0.9, device="cuda", dtype=torch.float32)
    return bench_fn(lambda: top_p_renorm_prob(probs, p))


def bench_tree_spec(bs=1, num_draft_tokens=6, vocab_size=20):
    device = "cuda"
    candidates = torch.randint(0, vocab_size, (bs, num_draft_tokens), device=device)
    retrive_index = torch.arange(
        bs * num_draft_tokens, device=device, dtype=torch.int64
    ).reshape(bs, num_draft_tokens)
    retrive_next_token = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=device
    )
    for b in range(bs):
        for i in range(num_draft_tokens - 1):
            retrive_next_token[b, i] = i + 1

    num_spec = 4
    target_logits = torch.randn(bs, num_draft_tokens, vocab_size, device=device)
    target_probs = F.softmax(target_logits, dim=-1)
    draft_probs = torch.zeros_like(target_probs)
    predicts = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full((bs, num_spec), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_final = torch.rand(bs, device=device, dtype=torch.float32)

    def run():
        predicts.fill_(-1)
        accept_index.fill_(-1)
        accept_token_num.fill_(0)
        tree_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            coins,
            coins_final,
            target_probs,
            draft_probs,
        )

    return bench_fn(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-only", action="store_true")
    args = parser.parse_args()

    top_k_us = bench_top_k()
    top_p_us = bench_top_p()
    tree_us = bench_tree_spec()

    total = top_k_us + top_p_us + tree_us

    if args.metric_only:
        print(f"{total:.1f}")
    else:
        print(f"top_k_renorm_prob:  {top_k_us:8.1f} µs")
        print(f"top_p_renorm_prob:  {top_p_us:8.1f} µs")
        print(f"tree_spec_sampling: {tree_us:8.1f} µs")
        print(f"TOTAL:              {total:8.1f} µs")


if __name__ == "__main__":
    main()
