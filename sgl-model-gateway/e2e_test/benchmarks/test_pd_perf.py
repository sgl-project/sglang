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
                # Lowered from 99 — the new 4-gpu-h100 runner produces
                # only ~12 GPU-util samples per run, so p50 catches idle
                # moments between PD requests (observed 3-50%). Keep a
                # weak floor so a fully stuck worker still trips the
                # gate; recalibrate once the bench window is longer.
                "gpu_util_p50_min": 1,
            },
        )
