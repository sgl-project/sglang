from __future__ import annotations

from sglang.jit_kernel.benchmark.kv_canary.utils import (
    MAX_EXTEND_TOKENS_PER_FORWARD,
    build_fast_matrix_cases,
    build_full_matrix_cases,
    cases_to_x_vals,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def test_fast_matrix_cases_include_e2e_decode_and_chunked_prefill_scenarios() -> None:
    cases = build_fast_matrix_cases()
    scenarios = {case.scenario for case in cases}

    assert {
        "e2e_decode_steady",
        "e2e_decode_tail",
        "e2e_prefill_chunk_first",
        "e2e_prefill_chunk_second",
        "e2e_prefill_chunk_mid",
        "e2e_prefill_chunk_last",
    } <= scenarios


def test_extend_cases_are_bounded_to_scheduler_chunk_size() -> None:
    cases = build_full_matrix_cases()
    bad_cases = [
        case
        for case in cases
        if case.mode == "extend"
        and case.bs * case.extend_len > MAX_EXTEND_TOKENS_PER_FORWARD
    ]

    assert bad_cases == []


def test_cases_to_x_vals_includes_scenario_axis() -> None:
    case = build_fast_matrix_cases()[0]

    x_vals = cases_to_x_vals([case])

    assert x_vals == [
        (
            case.scenario,
            case.bs,
            case.prefix_len,
            case.mode,
            case.extend_len,
            case.pool_kind,
            case.real_kv_kind,
            case.hash_mode,
        )
    ]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
