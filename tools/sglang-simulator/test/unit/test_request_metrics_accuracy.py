import pytest
from sglang_simulator.simulation.types import RequestStats
from sglang_simulator.simulation.utils import calc_metrics


def test_request_metrics_preserve_prefix_and_tier_hit_ratios():
    requests = [
        RequestStats(
            rid="r1",
            input_length=100,
            output_length=2,
            final_device_hit_len=60,
            final_host_hit_len=20,
            final_storage_hit_len=5,
            created_time=-0.01,
            queue_start=0.00,
            queue_end=0.01,
            last_event_time=0.20,
            gen_token_latencies=[0.10, 0.02],
        ),
        RequestStats(
            rid="r2",
            input_length=100,
            output_length=1,
            final_device_hit_len=40,
            final_host_hit_len=10,
            final_storage_hit_len=0,
            created_time=0.03,
            queue_start=0.05,
            queue_end=0.07,
            last_event_time=0.50,
            gen_token_latencies=[0.20],
        ),
    ]

    metrics = calc_metrics(requests)

    assert metrics["completed"] == 2
    assert metrics["total_input"] == 200
    assert metrics["total_output"] == 3
    assert metrics["duration"] == pytest.approx(0.50)
    assert metrics["prefix_cache_reused_ratio"] == pytest.approx(0.50)
    assert metrics["kv_cache_device_hit_ratio"] == pytest.approx(0.35)
    assert metrics["kv_cache_host_hit_ratio"] == pytest.approx(0.125)
    assert metrics["kv_cache_storage_hit_ratio"] == pytest.approx(0.025)
    assert metrics["mean_ttft_ms"] == pytest.approx(150)
    assert metrics["mean_queue_ms"] == pytest.approx(15)
    assert metrics["mean_dispatch_wait_ms"] == pytest.approx(15)
    assert metrics["mean_arrival_to_prefill_ms"] == pytest.approx(30)
    assert metrics["p95_ttft_ms"] >= metrics["p90_ttft_ms"]
    assert metrics["p99_ttft_ms"] >= metrics["p95_ttft_ms"]
    assert metrics["p95_e2e_latency_ms"] >= metrics["p90_e2e_latency_ms"]
    assert metrics["p99_e2e_latency_ms"] >= metrics["p95_e2e_latency_ms"]
    assert metrics["concurrency"] == pytest.approx((0.12 + 0.20) / 0.50)
    assert metrics["max_output_tokens_per_s"] == 3
    assert metrics["max_concurrent_requests"] == 2
