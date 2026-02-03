"""EPD router performance benchmark tests."""

import pytest


@pytest.mark.e2e
@pytest.mark.model("qwen-vl-7b")
@pytest.mark.workers(encode=2, prefill=2, decode=2)
@pytest.mark.epd_backend("mooncake")
@pytest.mark.parametrize("setup_backend", ["epd"], indirect=True)
class TestEPDPerf:
    """Performance benchmark for EPD disaggregation router."""

    def test_epd_perf(self, setup_backend, genai_bench_runner):
        """Run genai-bench against EPD router and validate metrics."""
        _, model_path, _, gateway = setup_backend
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder="benchmark_round_robin_epd",
            thresholds={
                "ttft_mean_max": 15,
                "e2e_latency_mean_max": 20,
                "input_throughput_mean_min": 200,
                "output_throughput_mean_min": 10,
                "gpu_util_p50_min": 90,
            },
        )
