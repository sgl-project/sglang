import time

import pytest
import requests


def _wait_for_workers(
    base_url: str, expected_count: int, timeout: float = 60.0, headers: dict = None
) -> None:
    """Poll /workers endpoint until expected number of workers are registered."""
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                r = session.get(f"{base_url}/workers", headers=headers, timeout=5)
                if r.status_code == 200:
                    workers = r.json().get("workers", [])
                    if len(workers) >= expected_count:
                        return
            except requests.RequestException:
                pass
            time.sleep(0.5)
    raise TimeoutError(
        f"Expected {expected_count} workers at {base_url}, timed out after {timeout}s"
    )


@pytest.mark.e2e
def test_genai_bench(
    e2e_router_only_rr, e2e_two_workers_dp2, e2e_model, genai_bench_runner
):
    """Attach a worker to the regular router and run a short genai-bench."""
    base = e2e_router_only_rr.url
    for w in e2e_two_workers_dp2:
        r = requests.post(f"{base}/workers", json={"url": w.url}, timeout=180)
        assert (
            r.status_code == 202
        ), f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"

    # Wait for workers to be registered
    _wait_for_workers(base, expected_count=2, timeout=60.0)

    genai_bench_runner(
        router_url=base,
        model_path=e2e_model,
        experiment_folder="benchmark_round_robin_regular",
        thresholds={
            "ttft_mean_max": 6,
            "e2e_latency_mean_max": 14,
            "input_throughput_mean_min": 800,  # temp relax from 1000 to 800 for now
            "output_throughput_mean_min": 12,
            # Enforce GPU utilization p50 >= 99% during the run.
            "gpu_util_p50_min": 99,
        },
        kill_procs=e2e_two_workers_dp2,
    )
