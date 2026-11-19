"""Benchmark settings, read once from the environment at import."""

import os

from sglang.srt.environ import envs

DISABLE_LOG_BANDWIDTH = envs.SGLANG_JIT_BENCHMARK_DISABLE_LOG_BANDWIDTH.get()
DISABLE_LOG_FLOPS = envs.SGLANG_JIT_BENCHMARK_DISABLE_LOG_FLOPS.get()
OUTPUT_DIR = envs.SGLANG_JIT_BENCHMARK_OUTPUT.get() or None
CASE_FILTER = frozenset(envs.SGLANG_JIT_BENCHMARK_CASE_FILTER.get())
CI_MODE = envs.SGLANG_IS_IN_CI.get() and not envs.SGLANG_JIT_BENCHMARK_FORCE_FULL.get()
# torchrun's process layout; a process outside torchrun is a world of one.
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
