"""Regular router performance benchmark test."""

import pytest


@pytest.mark.e2e
@pytest.mark.workers(count=4)
@pytest.mark.gateway(policy="cache_aware")
@pytest.mark.parametrize("setup_backend", ["http", "grpc"], indirect=True)
class TestRegularPerf:
    """Performance benchmark for regular (non-PD) router."""

    def test_regular_perf(self, setup_backend, genai_bench_runner):
        """Run genai-bench against regular router and validate metrics."""
        backend, model_path, client, gateway = setup_backend
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder=f"benchmark_cache_aware_regular_{backend}",
            thresholds={
                "ttft_mean_max": 6,
                "e2e_latency_mean_max": 14,
                "input_throughput_mean_min": 800,
                "output_throughput_mean_min": 12,
                "gpu_util_p50_min": 99,
            },
        )
