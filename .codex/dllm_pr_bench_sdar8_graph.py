#!/usr/bin/env python3
"""Run the upstream-matching SDAR-8B matrix with decode CUDA Graph enabled."""

from pathlib import Path

import dllm_pr_bench as bench


bench.ROOT = Path("/results/upstream-sync-sdar8-cudagraph")
bench.MODELS = (
    bench.ModelCase(
        key="sdar-8b",
        served_name="JetLM/SDAR-8B-Chat",
        path="/models/JetLM/SDAR-8B-Chat",
        concurrencies=(4, 8, 16, 32),
        repeats=1,
        disable_cuda_graph=False,
    ),
)


if __name__ == "__main__":
    raise SystemExit(bench.main())
