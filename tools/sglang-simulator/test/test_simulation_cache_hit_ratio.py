import json

import pytest
from test_simulation_sglang_serving import (
    SIM_CONFIGS,
    SGLangServingRunner,
    assert_decode_metrics,
)


def test_second_replay_benchmark_hits_all_reusable_prefix_tokens(tmp_path):
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
