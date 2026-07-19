from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.jit_kernel.kv_canary.scatter_req_token_ids import (
    launch_scatter_req_token_ids_kernel,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)
# AMD mirrors the CUDA nightly registration (nightly-only, no per-PR suite).
# Note: amd_ci_exec.sh sets SGLANG_IS_IN_CI, so this runs the CI-reduced range
# (_BS_AXIS_CI/_SEQ_LEN_AXIS_CI via get_benchmark_range), same as CUDA nightly.
register_amd_ci(est_time=180, suite="nightly-amd-kernel-1-gpu", nightly=True)


_BS_AXIS_FULL: list[int] = [1, 8, 64, 256]
_SEQ_LEN_AXIS_FULL: list[int] = [128, 512, 2048, 8192]
_BS_AXIS_CI: list[int] = [1, 64]
_SEQ_LEN_AXIS_CI: list[int] = [512, 2048]


@dataclass(frozen=True, slots=True, kw_only=True)
class _BenchCase:
    bs: int
    seq_len: int


def _build_cases() -> list[_BenchCase]:
    bs_axis = get_benchmark_range(full_range=_BS_AXIS_FULL, ci_range=_BS_AXIS_CI)
    seq_axis = get_benchmark_range(
        full_range=_SEQ_LEN_AXIS_FULL, ci_range=_SEQ_LEN_AXIS_CI
    )
    return [
        _BenchCase(bs=bs, seq_len=seq_len) for bs in bs_axis for seq_len in seq_axis
    ]


_X_NAMES = ["bs", "seq_len"]
_X_VALS = [(c.bs, c.seq_len) for c in _build_cases()]


def _inputs_from_lens(lens: torch.Tensor, device: torch.device) -> dict:
    """Build kernel inputs from a per-req length vector (uniform or skewed)."""
    bs = int(lens.shape[0])
    max_reqs = max(bs + 1, 4)
    max_context_len = max(int(lens.max().item()) + 1, 1) if bs else 1
    total_tokens = int(lens.sum().item())

    flat = torch.randint(
        low=0,
        high=1 << 30,
        size=(max(total_tokens, 1),),
        dtype=torch.int64,
        device=device,
    )[:total_tokens]
    offsets = torch.zeros(bs + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lens, dim=0)

    req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int64, device=device)
    pool = torch.zeros((max_reqs, max_context_len), dtype=torch.int32, device=device)
    return dict(
        flat_in=flat,
        offsets=offsets,
        req_pool_indices=req_pool_indices,
        pool_out=pool,
    )


def _build_inputs(*, bs: int, seq_len: int, device: torch.device) -> dict:
    lens = torch.full((bs,), seq_len, dtype=torch.int64, device=device)
    return _inputs_from_lens(lens, device)


# Skewed workloads: a few long requests among many short/empty ones. These are the
# shapes the (request, column-block) grid + per-request early-return are designed for,
# and that the uniform bs x seq_len grid above does not cover.
_SKEWED_FULL: list[str] = [
    "one_long_among_short_bs256",
    "one_long_among_short_bs64",
    "few_long_among_short_bs256",
    "many_medium_bs256",
]
_SKEWED_CI: list[str] = ["one_long_among_short_bs256", "many_medium_bs256"]


def _build_skewed_lens(name: str, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(0)
    if name == "one_long_among_short_bs256":
        lens = torch.randint(0, 4, (256,), generator=g, device=device)
        lens[128] = 32768
    elif name == "one_long_among_short_bs64":
        lens = torch.randint(0, 4, (64,), generator=g, device=device)
        lens[32] = 32768
    elif name == "few_long_among_short_bs256":
        lens = torch.randint(0, 4, (256,), generator=g, device=device)
        lens[::64] = 16384
    elif name == "many_medium_bs256":
        lens = torch.full((256,), 64, dtype=torch.int64, device=device)
    else:
        raise ValueError(f"unknown skewed workload {name}")
    return lens.to(torch.int64)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_X_NAMES,
        x_vals=_X_VALS,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("blue", "-")],
        ylabel="time (us)",
        plot_name="kv-canary-scatter-req-token-ids",
        args={},
    )
)
def benchmark(bs: int, seq_len: int, provider: str) -> tuple[float, float, float]:
    inputs = _build_inputs(bs=bs, seq_len=seq_len, device=torch.device(DEFAULT_DEVICE))
    return run_benchmark_no_cudagraph(
        lambda: launch_scatter_req_token_ids_kernel(**inputs)
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["workload"],
        x_vals=get_benchmark_range(full_range=_SKEWED_FULL, ci_range=_SKEWED_CI),
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("green", "-")],
        ylabel="time (us)",
        plot_name="kv-canary-scatter-req-token-ids-skewed",
        args={},
    )
)
def benchmark_skewed(workload: str, provider: str) -> tuple[float, float, float]:
    device = torch.device(DEFAULT_DEVICE)
    lens = _build_skewed_lens(workload, device)
    inputs = _inputs_from_lens(lens, device)
    return run_benchmark_no_cudagraph(
        lambda: launch_scatter_req_token_ids_kernel(**inputs)
    )


if __name__ == "__main__":
    benchmark.run(print_data=True)
    benchmark_skewed.run(print_data=True)
