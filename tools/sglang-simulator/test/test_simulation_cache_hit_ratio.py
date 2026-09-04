import json

import pytest
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from test_simulation_sglang_runner import make_fixed_dataset, make_sglang_runner
from test_simulation_sglang_serving import (
    SIM_CONFIGS,
    SGLangServingRunner,
    assert_decode_metrics,
)


def test_in_process_runner_reports_each_cache_tier(tmp_path):
    runner = make_sglang_runner(tmp_path)
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=False)
    cached_ds = make_fixed_dataset(1000, 8)
    evict_l1_ds = make_fixed_dataset(2000, 10)
    evict_l2_ds = make_fixed_dataset(3000, 20)

    try:
        metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
        assert metrics["completed"] == len(cached_ds)
        assert metrics["prefix_cache_reused_ratio"] == 0

        metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
        assert metrics["kv_cache_device_hit_ratio"] > 0.95

        runner.benchmark(benchmark_config, dataset=evict_l1_ds)
        metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
        assert metrics["kv_cache_host_hit_ratio"] > 0.95

        runner.benchmark(benchmark_config, dataset=evict_l2_ds)
        metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
        assert metrics["kv_cache_storage_hit_ratio"] > 0.95
    finally:
        runner.shutdown()


def test_second_replay_benchmark_hits_all_reusable_prefix_tokens(tmp_path, monkeypatch):
    # This test validates cache reuse across consecutive benchmark runs.
    monkeypatch.setenv("SGLANG_IS_IN_CI", "false")

    runner = SGLangServingRunner(SIM_CONFIGS["replay"], tmp_path)
    try:
        first_metrics = runner.benchmark(tmp_path / "benchmark-first.json")
        second_metrics = runner.benchmark(tmp_path / "benchmark-second.json")
    finally:
        runner.shutdown()

    assert_decode_metrics(first_metrics)
    assert_decode_metrics(second_metrics)

    assert second_metrics["total_input"] == 24
    assert second_metrics["total_new_input"] == 3
    assert second_metrics["prefix_cache_reused_ratio"] == pytest.approx(0.875)
    assert second_metrics["kv_cache_device_hit_ratio"] == pytest.approx(0.875)
    assert second_metrics["kv_cache_host_hit_ratio"] == 0
    assert second_metrics["kv_cache_storage_hit_ratio"] == 0

    requests = [
        json.loads(line)
        for line in (runner.output_dir / "request.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(requests) == 3
    assert all(request["input_length"] == 8 for request in requests)
    assert all(request["final_device_hit_len"] == 7 for request in requests)
