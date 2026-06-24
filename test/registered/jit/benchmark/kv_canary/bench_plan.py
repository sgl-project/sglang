from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.kv_canary.utils import (
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
from sglang.jit_kernel.kv_canary.plan import launch_canary_plan_kernels
from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=900, suite="nightly-kernel-1-gpu", nightly=True)
# AMD mirrors the CUDA nightly registration (nightly-only, no per-PR suite).
register_amd_ci(est_time=900, suite="nightly-amd-kernel-1-gpu", nightly=True)


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


_POOL_CAPACITY_VERIFY_CAP_AXIS: list[int] = [16384, 262144, 1398028]
_POOL_CAPACITY_BS_AXIS: list[int] = [1, 4, 32]
_POOL_CAPACITY_PREFIX_LEN: int = 512
_POOL_CAPACITY_BS_PADDED_AXIS: list[Optional[int]] = [None, 4096]


@dataclass(frozen=True, slots=True, kw_only=True)
class _PoolCapacityBenchCase:
    """One pool-capacity bench point.

    Attributes:
        bs: Number of active (non-padding) requests in the launch.
        bs_padded: Total request-axis size of the input tensors. ``None`` means
            no padding (``bs_padded == bs``); a concrete value pads ``req_pool_indices``
            with ``REQ_POOL_IDX_PADDING`` sentinels in rows ``[bs, bs_padded)``.
        prefix_len: Per-active-request prefix length.
        verify_capacity: Plan tensor row capacity.
        pool_kind: "full".
    """

    bs: int
    bs_padded: Optional[int]
    prefix_len: int
    verify_capacity: int
    pool_kind: str


def _build_pool_capacity_cases() -> list[_PoolCapacityBenchCase]:
    cases: list[_PoolCapacityBenchCase] = []
    for bs in _POOL_CAPACITY_BS_AXIS:
        for verify_capacity in _POOL_CAPACITY_VERIFY_CAP_AXIS:
            for bs_padded in _POOL_CAPACITY_BS_PADDED_AXIS:
                if bs_padded is not None and bs_padded < bs:
                    continue
                cases.append(
                    _PoolCapacityBenchCase(
                        bs=bs,
                        bs_padded=bs_padded,
                        prefix_len=_POOL_CAPACITY_PREFIX_LEN,
                        verify_capacity=verify_capacity,
                        pool_kind="full",
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

_X_NAMES_PC = ["bs", "bs_padded", "prefix_len", "verify_capacity", "pool_kind"]
_X_VALS_PC = [
    (
        c.bs,
        c.bs_padded if c.bs_padded is not None else c.bs,
        c.prefix_len,
        c.verify_capacity,
        c.pool_kind,
    )
    for c in _build_pool_capacity_cases()
]


def _build_plan_inputs(
    *,
    bs: int,
    prefix_len: int,
    extend_len: int,
    pool_kind: str,
    device: torch.device,
    verify_capacity_override: Optional[int] = None,
    bs_padded: Optional[int] = None,
) -> dict:
    swa_window_size = SWA_WINDOW if pool_kind == "swa_window_128" else 0
    verify_per_req = min(prefix_len, SWA_WINDOW) if swa_window_size > 0 else prefix_len
    if verify_capacity_override is not None:
        verify_capacity = max(1, verify_capacity_override)
    else:
        verify_capacity = max(1, bs * verify_per_req)

    effective_bs = bs_padded if bs_padded is not None else bs
    if effective_bs < bs:
        raise ValueError(f"kv-canary bench: bs_padded={bs_padded} must be >= bs={bs}")
    write_req_capacity = max(1, effective_bs)

    verify_plan = VerifyPlan.allocate(verify_capacity=verify_capacity, device=device)
    write_plan = WritePlan.allocate(
        write_req_capacity=write_req_capacity, device=device
    )

    req_pool_indices = torch.zeros(effective_bs, dtype=torch.int64, device=device)
    req_pool_indices[:bs] = torch.arange(1, bs + 1, dtype=torch.int64, device=device)
    prefix_lens = torch.zeros(effective_bs, dtype=torch.int64, device=device)
    prefix_lens[:bs] = prefix_len
    extend_seq_lens = torch.zeros(effective_bs, dtype=torch.int64, device=device)
    extend_seq_lens[:bs] = extend_len

    max_seq_len = max(prefix_len + extend_len, 1)
    req_to_token_rows = effective_bs + 1
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
        req_to_token[1 : bs + 1] = (row_idx - 1) * max_seq_len + col_idx

    if swa_window_size > 0:
        full_pool_size = effective_bs * max_seq_len + 1
        full_to_swa: Optional[torch.Tensor] = torch.arange(
            full_pool_size + 1, dtype=torch.int64, device=device
        )
        full_to_swa[-1] = -1
    else:
        full_to_swa = None

    return dict(
        verify_plan_out=verify_plan,
        write_plan_out=write_plan,
        req_pool_indices=req_pool_indices,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa,
        verify_capacity=int(verify_plan.verify_slot_indices.shape[0]),
    )


def _make_plan_callable(inputs: dict):
    def fn() -> None:
        launch_canary_plan_kernels(
            verify_plan_out=inputs["verify_plan_out"],
            write_plan_out=inputs["write_plan_out"],
            req_pool_indices=inputs["req_pool_indices"],
            prefix_lens=inputs["prefix_lens"],
            extend_seq_lens=inputs["extend_seq_lens"],
            req_to_token=inputs["req_to_token"],
            swa_window_size=inputs["swa_window_size"],
            full_to_swa_index_mapping=inputs["full_to_swa_index_mapping"],
            verify_capacity=inputs["verify_capacity"],
            req_to_verify_expected_tokens=None,
            req_to_verify_expected_tokens_valid_lens=None,
            kv_token_id_vs_position_offset=0,
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=_X_NAMES_PC,
        x_vals=_X_VALS_PC,
        line_arg="provider",
        line_vals=["canary", "naive"],
        line_names=["canary_plan_step", "naive torch.cumsum"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="kv-canary-plan-pool-capacity-perf",
        args={},
    )
)
def benchmark_pool_capacity(
    bs: int,
    bs_padded: int,
    prefix_len: int,
    verify_capacity: int,
    pool_kind: str,
    provider: str,
) -> Tuple[float, float, float]:
    device = torch.device(DEFAULT_DEVICE)
    if provider == "canary":
        inputs = _build_plan_inputs(
            bs=bs,
            prefix_len=prefix_len,
            extend_len=1,
            pool_kind=pool_kind,
            device=device,
            verify_capacity_override=verify_capacity,
            bs_padded=bs_padded,
        )
        fn = _make_plan_callable(inputs)
    else:
        fn = naive_cumsum_fn(bs=bs, device=device)
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark_matrix.run(print_data=True)
    benchmark_total_tokens.run(print_data=True)
    benchmark_pool_capacity.run(print_data=True)
