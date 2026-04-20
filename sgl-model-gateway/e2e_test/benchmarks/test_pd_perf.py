"""PD (prefill/decode disaggregation) router performance benchmark test."""

import pytest


@pytest.mark.e2e
@pytest.mark.workers(prefill=2, decode=2)
@pytest.mark.parametrize("setup_backend", ["pd"], indirect=True)
class TestPDPerf:
    """Performance benchmark for PD disaggregation router."""

    def test_pd_perf(self, setup_backend, genai_bench_runner):
        """Run genai-bench against PD router and validate metrics."""
        backend, model_path, client, gateway = setup_backend
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder="benchmark_round_robin_pd",
            thresholds={
                "ttft_mean_max": 13,
                "e2e_latency_mean_max": 16,
                "input_throughput_mean_min": 350,
                "output_throughput_mean_min": 18,
                "gpu_util_p50_min": 99,
            },
        )
