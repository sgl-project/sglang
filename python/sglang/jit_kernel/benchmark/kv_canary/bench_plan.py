from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.kv_canary.bench_helpers import (
    POOL_AXIS,
    SWA_WINDOW,
    BenchCase,
    build_fast_matrix_cases,
    build_full_matrix_cases,
    naive_cumsum_fn,
)
from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="base-b-kernel-benchmark-1-gpu-large")
register_cuda_ci(est_time=900, suite="nightly-kernel-1-gpu", nightly=True)


_TOTAL_TOKENS_AXIS: list[int] = [256, 4096, 65536, 262144]
_TOTAL_TOKENS_BS_AXIS: list[int] = [1, 32, 256]


@dataclass(frozen=True, slots=True, kw_only=True)
class _TotalTokensBenchCase:
    bs: int
    total_tokens: int
    pool_kind: str


def _build_total_tokens_cases() -> list[_TotalTokensBenchCase]:
    cases: list[_TotalTokensBenchCase] = []
    for bs in _TOTAL_TOKENS_BS_AXIS:
        for total_tokens in _TOTAL_TOKENS_AXIS:
            if total_tokens < bs:
                continue
            for pool_kind in POOL_AXIS:
                cases.append(
                    _TotalTokensBenchCase(
                        bs=bs, total_tokens=total_tokens, pool_kind=pool_kind
                    )
                )
    return cases


_X_NAMES_MATRIX = ["scenario", "bs", "prefix_len", "mode", "extend_len", "pool_kind"]


def _cases_to_matrix_x_vals(
    cases: list[BenchCase],
) -> list[tuple[str, int, int, str, int, str]]:
    return [
        (c.scenario, c.bs, c.prefix_len, c.mode, c.extend_len, c.pool_kind)
        for c in cases
    ]


_X_VALS_MATRIX = _cases_to_matrix_x_vals(
    get_benchmark_range(
        full_range=build_full_matrix_cases(),
        ci_range=build_fast_matrix_cases(),
    )
)

_X_NAMES_TT = ["bs", "total_tokens", "pool_kind"]
_X_VALS_TT = [(c.bs, c.total_tokens, c.pool_kind) for c in _build_total_tokens_cases()]


def _build_plan_inputs(
    *,
    bs: int,
    prefix_len: int,
    extend_len: int,
    pool_kind: str,
    device: torch.device,
) -> dict:
    swa_window_size = SWA_WINDOW if pool_kind == "swa_window_128" else 0
    verify_per_req = min(prefix_len, SWA_WINDOW) if swa_window_size > 0 else prefix_len
    verify_capacity = max(1, bs * verify_per_req)
    write_req_capacity = max(1, bs)

    verify_plan = VerifyPlan.allocate(verify_capacity=verify_capacity, device=device)
    write_plan = WritePlan.allocate(
        write_req_capacity=write_req_capacity, device=device
    )

    fb_req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int32, device=device)
    fb_prefix_lens = torch.full((bs,), prefix_len, dtype=torch.int32, device=device)
    fb_extend_seq_lens = torch.full((bs,), extend_len, dtype=torch.int32, device=device)

    max_seq_len = max(prefix_len + extend_len, 1)
    req_to_token_rows = bs + 1
    req_to_token = torch.zeros(
        req_to_token_rows,
        max_seq_len,
        dtype=torch.int32,
        device=device,
    )
    if bs > 0:
        row_idx = torch.arange(1, bs + 1, dtype=torch.int32, device=device).unsqueeze(1)
        col_idx = torch.arange(max_seq_len, dtype=torch.int32, device=device).unsqueeze(
            0
        )
        req_to_token[1:] = (row_idx - 1) * max_seq_len + col_idx

    if swa_window_size > 0:
        full_pool_size = bs * max_seq_len + 1
        full_to_swa: Optional[torch.Tensor] = torch.arange(
            full_pool_size + 1, dtype=torch.int64, device=device
        )
        full_to_swa[-1] = -1
    else:
        full_to_swa = None

    return dict(
        verify_plan_out=verify_plan,
        write_plan_out=write_plan,
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa,
        verify_capacity=int(verify_plan.verify_slot_indices.shape[0]),
    )


def _make_plan_callable(inputs: dict):
    def fn() -> None:
        canary_plan_step(
            verify_plan_out=inputs["verify_plan_out"],
            write_plan_out=inputs["write_plan_out"],
            fb_req_pool_indices=inputs["fb_req_pool_indices"],
            fb_prefix_lens=inputs["fb_prefix_lens"],
            fb_extend_seq_lens=inputs["fb_extend_seq_lens"],
            req_to_token=inputs["req_to_token"],
            swa_window_size=inputs["swa_window_size"],
            full_to_swa_index_mapping=inputs["full_to_swa_index_mapping"],
            verify_capacity=inputs["verify_capacity"],
        )

    return fn


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_X_NAMES_MATRIX,
        x_vals=_X_VALS_MATRIX,
        line_arg="provider",
        line_vals=["canary", "naive"],
        line_names=["canary_plan_step", "naive torch.cumsum"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="kv-canary-plan-matrix-perf",
        args={},
    )
)
def benchmark_matrix(
    scenario: str,
    bs: int,
    prefix_len: int,
    mode: str,
    extend_len: int,
    pool_kind: str,
    provider: str,
) -> Tuple[float, float, float]:
    del scenario
    del mode

    device = torch.device(DEFAULT_DEVICE)
    if provider == "canary":
        inputs = _build_plan_inputs(
            bs=bs,
            prefix_len=prefix_len,
            extend_len=extend_len,
            pool_kind=pool_kind,
            device=device,
        )
        fn = _make_plan_callable(inputs)
    else:
        fn = naive_cumsum_fn(bs=bs, device=device)
    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_X_NAMES_TT,
        x_vals=_X_VALS_TT,
        line_arg="provider",
        line_vals=["canary", "naive"],
        line_names=["canary_plan_step", "naive torch.cumsum"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="kv-canary-plan-total-tokens-perf",
        args={},
    )
)
def benchmark_total_tokens(
    bs: int,
    total_tokens: int,
    pool_kind: str,
    provider: str,
) -> Tuple[float, float, float]:
    device = torch.device(DEFAULT_DEVICE)
    per_req_prefix = max(1, total_tokens // max(bs, 1))
    if provider == "canary":
        inputs = _build_plan_inputs(
            bs=bs,
            prefix_len=per_req_prefix,
            extend_len=1,
            pool_kind=pool_kind,
            device=device,
        )
        fn = _make_plan_callable(inputs)
    else:
        fn = naive_cumsum_fn(bs=bs, device=device)
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark_matrix.run(print_data=True)
    benchmark_total_tokens.run(print_data=True)
