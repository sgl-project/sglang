# ViT attention with FA4 on SM103: row max from tcgen05.ld.red
# (SGLANG_FA4_TMEM_LOAD_RED_MAX=1) vs FMNMX (=0). Shapes and call follow
# VisionFlash4Attention (varlen, non-causal, no LSE). Each round CUDA-graph times both
# back to back, in alternating order; the per-round ratio is the headline number.
# Outputs must be bit-identical.
#
# Run on B300 / GB300 (elsewhere the flag is ignored):
#   python benchmark/kernels/attention/bench_flash_attention_tmem_red_max.py

from __future__ import annotations

import argparse
import os
from typing import Iterable

import torch

from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    flash_attn_varlen_func,
)

RED_MAX_ENV = "SGLANG_FA4_TMEM_LOAD_RED_MAX"

# images, tokens per image, heads, head dim
SHAPES = [
    (8, 4096, 16, 80),  # Qwen2.5-VL
    (8, 4096, 16, 72),  # Qwen3-VL
    (8, 4096, 16, 64),  # InternViT, GLM / DeepSeek-V4 ViT
    (64, 1024, 16, 80),
    (64, 1024, 16, 64),
    (1, 32768, 16, 80),  # one large image or a video
    (1, 32768, 16, 72),
    (1, 32768, 16, 64),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    return t.contiguous().view(int_dtype)


def time_graph(fn, iters: int, calls_per_graph: int = 5) -> float:
    """Mean seconds per call, replaying a CUDA graph of `calls_per_graph` calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / (iters * calls_per_graph)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Rounds per shape; each round times both variants back to back.",
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Graph replays per round."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"This benchmark is for SM10x. Got {major}.{minor}.")
    if minor < 3:
        print(
            f"Note: SM{major}{minor} has no tcgen05.ld.red; {RED_MAX_ENV} is ignored "
            "and both variants run the same kernel."
        )

    torch.manual_seed(args.seed)
    device = "cuda"
    mismatches = []
    print(
        f"{'shape':28s} {'FMNMX us':>9s} {'ld.red us':>9s} {'ld.red TF':>9s} "
        f"{'best':>7s} {'paired':>7s} {'wins':>5s}  bitwise"
    )
    for images, tokens, heads, head_dim in SHAPES:
        torch.cuda.empty_cache()
        q, k, v = [
            torch.randn(
                images * tokens, heads, head_dim, device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        ]
        cu_seqlens = torch.arange(images + 1, dtype=torch.int32, device=device) * tokens
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=tokens,
            max_seqlen_k=tokens,
        )

        outputs, times = {}, {False: [], True: []}
        for red_max in (False, True):
            os.environ[RED_MAX_ENV] = "1" if red_max else "0"
            outputs[red_max] = flash_attn_varlen_func(**kwargs)[0]
        # Swap the order every round so clock and thermal drift hit both equally.
        for r in range(args.rounds):
            for red_max in (False, True) if r % 2 == 0 else (True, False):
                os.environ[RED_MAX_ENV] = "1" if red_max else "0"
                times[red_max].append(
                    time_graph(lambda: flash_attn_varlen_func(**kwargs), args.iters)
                )
        same = torch.equal(_bits(outputs[True]), _bits(outputs[False]))
        name = f"{images} x {tokens} tok, h{heads} d{head_dim}"
        if not same:
            mismatches.append(name)
        t0, t1 = min(times[False]), min(times[True])
        # Ratio of the two variants within each round, then the median over rounds.
        paired = sorted(a / c for a, c in zip(times[False], times[True]))
        wins = f"{sum(p > 1 for p in paired)}/{len(paired)}"
        flops = 4.0 * images * heads * head_dim * tokens * tokens
        print(
            f"{name:28s} {t0 * 1e6:9.1f} {t1 * 1e6:9.1f} {flops / t1 / 1e12:9.1f} "
            f"{t0 / t1:6.3f}x {paired[len(paired) // 2]:6.3f}x {wins:>5s}  "
            f"{'identical' if same else 'DIFFERENT'}",
            flush=True,
        )
        del q, k, v, outputs

    os.environ.pop(RED_MAX_ENV, None)
    if mismatches:
        print(f"\nOutputs differ between the variants for: {', '.join(mismatches)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
