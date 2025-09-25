from __future__ import annotations

import logging

import pytest
from py_test.perf.conftest import PolicyPerfRecord, build_perf_record

logger = logging.getLogger(__name__)

POLICIES = ["random", "round_robin", "cache_aware", "power_of_two"]
THRESHOLDS = {
    "ttft_mean_max": 4.7,
    "e2e_latency_mean_max": 35.0,
    "input_throughput_mean_min": 10_000,
    "output_throughput_mean_min": 68.0,
}


@pytest.mark.perf
@pytest.mark.parametrize("policy", POLICIES)
def test_pd_router_policy_perf(
    policy: str,
    request: pytest.FixtureRequest,
    perf_model_path: str,
    pd_router_factory,
    router_smoke_checker,
    genai_bench_runner,
):
    router = pd_router_factory(policy)
    try:
        router_smoke_checker(router)

        experiment_folder = f"pd_perf_benchmark_{policy}"
        result = genai_bench_runner(
            router_url=router.url,
            model_path=perf_model_path,
            experiment_folder=experiment_folder,
            thresholds=THRESHOLDS,
            num_concurrency=64,
            traffic_scenario="D(8000,2000)",
            clean_experiment=True,
        )

        record = build_perf_record(
            policy=policy,
            stats=result.stats,
            experiment_dir=str(result.experiment_folder),
        )
        _attach_perf_record(request, record)
        logger.info(
            "Policy %s metrics: ttft=%.3fs e2e=%.3fs input_tp=%.0f output_tp=%.0f",
            policy,
            record.ttft_mean,
            record.e2e_latency_mean,
            record.input_throughput_mean,
            record.output_throughput_mean,
        )
    finally:
        router.stop()


def _attach_perf_record(
    request: pytest.FixtureRequest, record: PolicyPerfRecord
) -> None:
    existing = getattr(request.node, "_pd_perf_records", [])
    existing.append(record)
    request.node._pd_perf_records = existing
