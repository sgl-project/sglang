"""Compare token and block samplers on precomputed FP32 probability tensors."""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)


def benchmark(args: argparse.Namespace) -> None:
    samplers = [("current", chain_speculative_sampling_triton)]
    if args.baseline_file is not None:
        spec = importlib.util.spec_from_file_location(
            "baseline_reject_sampling", args.baseline_file
        )
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        samplers.insert(0, ("baseline", baseline.chain_speculative_sampling_triton))

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "seed": args.seed,
                "steps": args.steps,
                "vocab": args.vocab,
                "warmup_ms": args.warmup,
                "measurement_ms": args.measurement,
            }
        ),
        flush=True,
    )
    torch.manual_seed(args.seed)
    for batch in args.batches:
        slots = args.steps + 1
        target = torch.randn(batch, slots, args.vocab, device="cuda").softmax(-1)
        draft = (
            0.7 * target[:, : args.steps]
            + 0.3
            * torch.randn(batch, args.steps, args.vocab, device="cuda").softmax(-1)
        ).contiguous()
        candidates = torch.zeros(batch, slots, dtype=torch.long, device="cuda")
        if args.steps:
            candidates[:, 1:] = torch.multinomial(draft.flatten(0, 1), 1).view(
                batch, args.steps
            )
        indices = torch.arange(batch * slots, device="cuda").view(batch, slots)
        predicts = torch.empty(batch * slots, dtype=torch.int32, device="cuda")
        accepted = torch.empty_like(indices, dtype=torch.int32)
        lengths = torch.empty(batch, dtype=torch.int32, device="cuda")
        coins = torch.rand(batch, slots, device="cuda")
        final_coins = torch.rand(batch, device="cuda")

        measurements = {
            (name, block): [] for name, _ in samplers for block in (False, True)
        }
        for repeat in range(args.repeats):
            order = samplers if repeat % 2 == 0 else samplers[::-1]
            for name, sampler in order:
                for block in (False, True):

                    def run():
                        sampler(
                            predicts,
                            accepted,
                            lengths,
                            candidates,
                            indices,
                            None,
                            None,
                            coins,
                            final_coins,
                            target,
                            draft,
                            1.0,
                            1.0,
                            True,
                            block_verification=block,
                        )

                    run()
                    milliseconds = triton.testing.do_bench(
                        run, warmup=args.warmup, rep=args.measurement
                    )
                    measurements[name, block].append(milliseconds * 1000)
        for (name, block), microseconds in measurements.items():
            print(
                json.dumps(
                    {
                        "batch": batch,
                        "implementation": name,
                        "block": block,
                        "microseconds": microseconds,
                        "median_us": statistics.median(microseconds),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--vocab", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=100, help="Warmup milliseconds")
    parser.add_argument(
        "--measurement", type=int, default=500, help="Measurement milliseconds"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-file", type=Path)
    benchmark(parser.parse_args())
