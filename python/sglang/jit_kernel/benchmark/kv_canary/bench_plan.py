"""Sweep-matrix benchmark for canary_plan_step.

Cartesian product over (bs, prefix_len, mode, pool_kind), plus a separate ``total_tokens`` axis to
expose how the plan kernel scales with (bs, total_tokens) rather than chunk_size. Fast subset runs by
default; full slow subset gated behind ``--runslow`` / ``--bench-full``.

Per case the bench reports: name, microseconds per call, nanoseconds per verify entry, and the ratio
against a naive ``torch.cumsum`` baseline of the same per-req count array (cumsum is the closest "naive
plan baseline" because exclusive prefix-sum dominates the plan kernel's second pass).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from sglang.jit_kernel.benchmark.kv_canary.bench_helpers import (
    POOL_AXIS,
    SWA_WINDOW,
    BenchCase,
    baseline_us_cumsum,
    do_bench,
    select_matrix_cases,
)
from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE
from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.verify import VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")
register_cuda_ci(est_time=900, suite="nightly-kernel-1-gpu", nightly=True)


_TOTAL_TOKENS_AXIS = [256, 4096, 65536, 262144]
_TOTAL_TOKENS_BS_AXIS = [1, 32, 256]


@dataclass(frozen=True, slots=True, kw_only=True)
class _TotalTokensBenchCase:
    bs: int
    total_tokens: int
    pool_kind: str

    @property
    def case_id(self) -> str:
        return f"tt_bs{self.bs}_total{self.total_tokens}_{self.pool_kind}"


def _build_total_tokens_params() -> list[_TotalTokensBenchCase]:
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


def _total_tokens_params() -> list:
    return [pytest.param(c, id=c.case_id) for c in _build_total_tokens_params()]


def _verify_entry_count(*, bs: int, prefix_len: int, pool_kind: str) -> int:
    if pool_kind == "swa_window_128":
        per_req = min(prefix_len, SWA_WINDOW)
    else:
        per_req = prefix_len
    return bs * per_req


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

    extras_cap = 1
    extra_verify_slot_indices = torch.zeros(
        extras_cap, dtype=torch.int32, device=device
    )
    extra_verify_positions = torch.zeros(extras_cap, dtype=torch.int32, device=device)
    extra_verify_prev_slot_indices = torch.zeros(
        extras_cap, dtype=torch.int32, device=device
    )
    extra_verify_num_valid = torch.zeros(1, dtype=torch.int32, device=device)

    if swa_window_size > 0:
        full_pool_size = bs * max_seq_len + 1
        full_to_swa: Optional[torch.Tensor] = torch.arange(
            full_pool_size + 1, dtype=torch.int32, device=device
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
        extra_verify_slot_indices=extra_verify_slot_indices,
        extra_verify_positions=extra_verify_positions,
        extra_verify_prev_slot_indices=extra_verify_prev_slot_indices,
        extra_verify_num_valid=extra_verify_num_valid,
        swa_window_size=swa_window_size,
        full_to_swa_index_mapping=full_to_swa,
    )


def _run_plan(inputs: dict) -> None:
    canary_plan_step(
        verify_plan_out=inputs["verify_plan_out"],
        write_plan_out=inputs["write_plan_out"],
        fb_req_pool_indices=inputs["fb_req_pool_indices"],
        fb_prefix_lens=inputs["fb_prefix_lens"],
        fb_extend_seq_lens=inputs["fb_extend_seq_lens"],
        req_to_token=inputs["req_to_token"],
        extra_verify_slot_indices=inputs["extra_verify_slot_indices"],
        extra_verify_positions=inputs["extra_verify_positions"],
        extra_verify_prev_slot_indices=inputs["extra_verify_prev_slot_indices"],
        extra_verify_num_valid=inputs["extra_verify_num_valid"],
        swa_window_size=inputs["swa_window_size"],
        full_to_swa_index_mapping=inputs["full_to_swa_index_mapping"],
    )


def _run_matrix_case(case: BenchCase) -> dict:
    device = torch.device(DEFAULT_DEVICE)
    inputs = _build_plan_inputs(
        bs=case.bs,
        prefix_len=case.prefix_len,
        extend_len=case.extend_len,
        pool_kind=case.pool_kind,
        device=device,
    )

    def fn() -> None:
        _run_plan(inputs)

    fn()
    torch.cuda.synchronize()

    plan_us = do_bench(fn)
    baseline_us = baseline_us_cumsum(bs=case.bs, device=device)
    total_entries = _verify_entry_count(
        bs=case.bs, prefix_len=case.prefix_len, pool_kind=case.pool_kind
    )
    per_slot_ns = (plan_us * 1000.0 / total_entries) if total_entries > 0 else 0.0
    ratio = plan_us / baseline_us if baseline_us > 0 else float("inf")

    return {
        "name": case.case_id,
        "us_per_call": plan_us,
        "per_slot_ns": per_slot_ns,
        "ratio": ratio,
    }


def _run_total_tokens_case(case: _TotalTokensBenchCase) -> dict:
    device = torch.device(DEFAULT_DEVICE)
    per_req_prefix = max(1, case.total_tokens // max(case.bs, 1))
    inputs = _build_plan_inputs(
        bs=case.bs,
        prefix_len=per_req_prefix,
        extend_len=1,
        pool_kind=case.pool_kind,
        device=device,
    )

    def fn() -> None:
        _run_plan(inputs)

    fn()
    torch.cuda.synchronize()

    plan_us = do_bench(fn)
    baseline_us = baseline_us_cumsum(bs=case.bs, device=device)
    total_entries = _verify_entry_count(
        bs=case.bs, prefix_len=per_req_prefix, pool_kind=case.pool_kind
    )
    per_slot_ns = (plan_us * 1000.0 / total_entries) if total_entries > 0 else 0.0
    ratio = plan_us / baseline_us if baseline_us > 0 else float("inf")

    return {
        "name": case.case_id,
        "us_per_call": plan_us,
        "per_slot_ns": per_slot_ns,
        "ratio": ratio,
    }


@pytest.fixture(scope="module", autouse=True)
def _print_header():
    print(
        "\n[bench_kv_canary_plan] name | us/call | per_slot_ns | ratio_vs_cumsum",
        flush=True,
    )
    yield


@pytest.mark.parametrize("case", select_matrix_cases())
def test_bench_plan_matrix(case: BenchCase) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    result = _run_matrix_case(case)
    print(
        f"  {result['name']} | {result['us_per_call']:.3f} | "
        f"{result['per_slot_ns']:.3f} | {result['ratio']:.3f}",
        flush=True,
    )


@pytest.mark.parametrize("case", _total_tokens_params())
def test_bench_plan_total_tokens(case: _TotalTokensBenchCase) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    result = _run_total_tokens_case(case)
    print(
        f"  {result['name']} | {result['us_per_call']:.3f} | "
        f"{result['per_slot_ns']:.3f} | {result['ratio']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
