"""Benchmark seeded versus unseeded EAGLE draft Gumbel sampling.

Example:
    python benchmark/kernels/sampling/bench_deterministic_fast_sample.py \
        --batch-sizes 1 8 32 128 --vocab-sizes 32000 128256 163840
"""

import argparse

import torch
import triton

from sglang.srt.speculative.spec_utils import fast_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[32000, 128256, 163840]
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument(
        "--modes", nargs="+", choices=("eager", "graph"), default=["graph"]
    )
    return parser.parse_args()


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def main():
    args = parse_args()
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(device)}")
    print("mode\tbatch\tvocab\tunseeded_us\tseeded_us\tslowdown")

    for vocab_size in args.vocab_sizes:
        for batch_size in args.batch_sizes:
            logits = torch.randn(
                (batch_size, vocab_size), device=device, dtype=torch.float32
            )
            probs = torch.softmax(logits, dim=-1)
            seeds = torch.arange(batch_size, device=device, dtype=torch.int64) + 12345
            positions = (
                torch.arange(batch_size, device=device, dtype=torch.int64) + 4096
            )

            def unseeded():
                return fast_sample(probs)

            def seeded():
                return fast_sample(
                    probs,
                    sampling_seed=seeds,
                    positions=positions,
                    draft_step=1,
                )

            # Compile/JIT and populate allocator caches before timing.
            unseeded()
            seeded()
            torch.cuda.synchronize()

            for mode in args.modes:
                unseeded_fn = capture(unseeded) if mode == "graph" else unseeded
                seeded_fn = capture(seeded) if mode == "graph" else seeded
                unseeded_ms = triton.testing.do_bench(
                    unseeded_fn, warmup=args.warmup_ms, rep=args.rep_ms
                )
                seeded_ms = triton.testing.do_bench(
                    seeded_fn, warmup=args.warmup_ms, rep=args.rep_ms
                )
                print(
                    f"{mode}\t{batch_size}\t{vocab_size}\t"
                    f"{unseeded_ms * 1000:.1f}\t{seeded_ms * 1000:.1f}\t"
                    f"{seeded_ms / unseeded_ms:.2f}x"
                )


if __name__ == "__main__":
    main()
