import json

import pytest
from test_simulation_sglang_serving import (
    SIM_CONFIGS,
    SGLangServingRunner,
    assert_decode_metrics,
)

REQUEST_RATE = 1
SEED = 123
RELATIVE_TOLERANCES = {
    "duration": 0.01,
    "request_throughput": 0.01,
    "input_throughput": 0.01,
    "output_throughput": 0.01,
    "mean_e2e_latency_ms": 0.10,
    "mean_ttft_ms": 0.10,
    "mean_tpot_ms": 0.10,
    "mean_itl_ms": 0.10,
}


def _relative_error(actual, expected):
    return abs(actual - expected) / abs(expected)


def _run_mode(mode, tmp_path):
    case_dir = tmp_path / mode
    case_dir.mkdir()
    runner = SGLangServingRunner(SIM_CONFIGS["aic_sol"], case_dir, mode=mode)
    try:
        metrics = runner.benchmark(
            case_dir / "benchmark.json", request_rate=REQUEST_RATE, seed=SEED
        )
    finally:
        runner.shutdown()

    requests = [
        json.loads(line)
        for line in (runner.output_dir / "request.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    requests.sort(key=lambda request: request["created_time"])
    return metrics, requests


def test_request_rate_offline_matches_blocking(tmp_path):
    offline_metrics, offline_requests = _run_mode("offline", tmp_path)
    blocking_metrics, blocking_requests = _run_mode("blocking", tmp_path)

    for metrics in (offline_metrics, blocking_metrics):
        assert_decode_metrics(metrics)

    assert len(offline_requests) == len(blocking_requests) == 3

    offline_arrivals = [request["created_time"] for request in offline_requests]
    blocking_arrivals = [request["created_time"] for request in blocking_requests]
    assert offline_arrivals[1] > 0.5
    assert blocking_arrivals[1] > 0.5
    assert offline_arrivals == pytest.approx(blocking_arrivals, abs=0.02)
    assert (
        offline_metrics["max_concurrent_requests"]
        == blocking_metrics["max_concurrent_requests"]
        == 1
    )

    for key in ("completed", "total_input", "total_output"):
        assert offline_metrics[key] == blocking_metrics[key]

    for key, tolerance in RELATIVE_TOLERANCES.items():
        error = _relative_error(offline_metrics[key], blocking_metrics[key])
        assert error <= tolerance, (
            key,
            offline_metrics[key],
            blocking_metrics[key],
            error,
            tolerance,
        )
