"""Timing driver correctness on real CUDA/fake EP; no communication speed claim."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
from nccl_ep_test.benchmark import METRICS, benchmark_steps, summarize_pair
from nccl_ep_test.oracle import RoutingBatch


def local_fixture(bucket, *, case, step, change):
    assert case == "balanced" and change == "all"
    x = torch.full((bucket, 2048), 2 ** (step % 3), dtype=torch.bfloat16)
    ids = torch.tensor([[1, 0] if step % 2 else [0, 1]] * bucket)
    weights = torch.tensor([[0.75, 0.25] if step % 2 else [0.25, 0.75]] * bucket)
    return RoutingBatch((x,), (ids,), (weights,), 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA required")
def test_matched_driver_changes_inputs_and_preserves_both_layer_outputs():
    from nccl_ep_test.fake_ep import dispatcher_environment

    with dispatcher_environment(capacity=16) as ep:
        result = benchmark_steps(
            0,
            ep.coordinator,
            [ep.dispatcher(layer_id=i) for i in range(2)],
            buckets=(8, 16),
            cases=("balanced",),
            samples=4,
            warmups=2,
            rounds=2,
            fixture=local_fixture,
        )
        assert result["checked"] == 48
        assert len(result["records"]) == 8
        assert [r["mode"] for r in result["records"]] == [
            "eager",
            "graph",
            "graph",
            "eager",
        ] * 2
        assert all(
            r["statistics_ms"][m]["count"] == 4
            for r in result["records"]
            for m in METRICS
        )
        # One dedicated Graph group and one separate eager group. The Graph
        # group keeps a single handle while eager creates per-step handles.
        assert len(ep.groups) == 2
        graph = next(g for g in ep.groups if g.config.rdma_buffer_size == 0)
        assert len(graph.handles) == 1 and graph.destroyed
    assert "native_ep_tested" not in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA required")
def test_benchmark_does_not_publish_timings_after_oracle_failure(monkeypatch, tmp_path):
    import nccl_ep_test.benchmark as benchmark
    from nccl_ep_test.fake_ep import dispatcher_environment

    monkeypatch.setenv("NCCL_EP_REPORT_DIR", str(tmp_path))
    original = benchmark.forward_layer

    def corrupt(*args, **kwargs):
        received, counters, combined = original(*args, **kwargs)
        return received, counters, combined * 0

    monkeypatch.setattr(benchmark, "forward_layer", corrupt)
    with dispatcher_environment(capacity=8) as ep:
        with pytest.raises(AssertionError):
            benchmark_steps(
                0,
                ep.coordinator,
                [ep.dispatcher(layer_id=i) for i in range(2)],
                buckets=(8,),
                cases=("balanced",),
                samples=2,
                warmups=2,
                rounds=2,
                fixture=local_fixture,
            )
    assert list(tmp_path.glob("mismatch-rank0-*.pt"))


def synthetic_pair():
    reports = []
    for rank in (0, 1):
        records = []
        for rnd in range(2):
            for mode in ("eager", "graph"):
                factor = (2 if mode == "eager" else 1) * (rank + 1)
                records.append(
                    {
                        "bucket": 8,
                        "case": "balanced",
                        "round": rnd,
                        "mode": mode,
                        "samples": {m: [factor, 2 * factor] for m in METRICS},
                    }
                )
        reports.append(
            {
                "rank": rank,
                "status": "PASS",
                "synthetic_fixture": True,
                "result": {
                    "native_ep_tested": True,
                    "resource_cleanup_returned": True,
                    "tolerance": {"rtol": 0, "atol": 0},
                    "config": {
                        "buckets": [8],
                        "cases": ["balanced"],
                        "rounds": 2,
                        "samples_per_round": 2,
                    },
                    "records": records,
                },
            }
        )
    return reports


def test_summary_aligns_ranks_and_reports_raw_sample_count():
    summary = summarize_pair(synthetic_pair())
    row = summary["rows"][0]
    eager = row["modes"]["eager"]["cuda_step_ms"]
    assert eager["rank0"]["median"] == 3
    assert eager["rank1"]["median"] == 6
    assert eager["max_rank_per_sample"] == {"count": 4, "median": 6, "p95": 8}
    assert row["median_speedup"] == {"cuda_step_ms": 2, "host_step_ms": 2}


@pytest.mark.parametrize(
    "case",
    [
        "missing_rank",
        "skip",
        "local_fake",
        "config",
        "block",
        "duplicate",
        "samples",
        "nan",
    ],
)
def test_incomplete_or_invalid_benchmark_cannot_produce_speedup(case):
    reports = copy.deepcopy(synthetic_pair())
    result = reports[1]["result"]
    if case == "missing_rank":
        reports.pop()
    elif case == "skip":
        reports[1]["status"] = "SKIP"
    elif case == "local_fake":
        result["native_ep_tested"] = False
    elif case == "config":
        result["config"]["rounds"] = 4
    elif case == "block":
        result["records"].pop()
    elif case == "duplicate":
        result["records"].append(result["records"][0])
    elif case == "samples":
        result["records"][0]["samples"]["cuda_step_ms"].pop()
    else:
        result["records"][0]["samples"]["cuda_step_ms"][0] = float("nan")
    with pytest.raises(ValueError):
        summarize_pair(reports)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
