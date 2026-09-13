"""Real MoE timing harness correctness; local EP is a one-rank test double."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test import overlap_benchmark
from nccl_ep_test.fake_ep import dispatcher_environment
from registered.unit.layers.moe.test_nccl_ep_benchmark import (
    local_fixture,
    synthetic_pair,
)

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA required")
@pytest.mark.parametrize(
    "sbo,tbo", [(False, False), (True, False), (False, True), (True, True)]
)
def test_overlap_timing_checks_real_expert_outputs_and_keeps_ep_unwrapped(
    monkeypatch, sbo, tbo
):
    construct = overlap_benchmark.make_moe

    def make_model(*args, **kwargs):
        model = construct(*args, **kwargs)
        # Only the transport double is one rank; exercise the real EP stages.
        model.ep_size = 2
        return model

    monkeypatch.setattr(overlap_benchmark, "make_moe", make_model)
    with dispatcher_environment(capacity=8) as ep:
        handle_dispatch = ep.ep.Handle.dispatch
        result = overlap_benchmark.run(
            ep.coordinator,
            sbo=sbo,
            tbo=tbo,
            fixture_fn=local_fixture,
            buckets=(8,),
            cases=("balanced",),
            samples=2,
            warmups=2,
            rounds=2,
        )
        assert result["checked"] == 24
        assert result["config"]["sbo"] == sbo and result["config"]["tbo"] == tbo
        assert len(result["records"]) == 4
        assert ep.ep.Handle.dispatch is handle_dispatch
        assert len(ep.groups) == (4 if tbo else 2)
        assert all(group.destroyed for group in ep.groups)


def test_summary_aligns_ranks_and_rejects_incomplete_or_stale_evidence():
    reports = synthetic_pair()
    for report in reports:
        report.update(passed=True, native_ep_tested=True, source_head="same")
        report["result"]["implementation"] = "nccl_ep_shared_triton_overlap"
    summary = overlap_benchmark.summarize(reports)
    eager = next(row for row in summary["rows"] if row["mode"] == "eager")
    assert eager["max_rank_per_sample_ms"]["cuda_step_ms"]["median"] == 6
    for mutation in ("native", "sha", "sample", "block"):
        changed = copy.deepcopy(reports)
        if mutation == "native":
            changed[1]["native_ep_tested"] = False
        elif mutation == "sha":
            changed[1]["source_head"] = "old"
        elif mutation == "sample":
            changed[1]["result"]["records"][0]["samples"]["cuda_step_ms"].pop()
        else:
            changed[1]["result"]["records"].pop()
        with pytest.raises(ValueError):
            overlap_benchmark.summarize(changed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA required")
def test_bad_expert_oracle_prevents_timing_result(monkeypatch):
    monkeypatch.setattr(
        overlap_benchmark,
        "expected_output",
        lambda batch, rank, *a, **kw: torch.full_like(
            batch.tokens[rank], float("nan"), dtype=torch.float32
        ),
    )
    with dispatcher_environment(capacity=8) as ep:
        with pytest.raises(AssertionError):
            overlap_benchmark.run(
                ep.coordinator,
                sbo=True,
                fixture_fn=local_fixture,
                buckets=(8,),
                cases=("balanced",),
                samples=2,
                warmups=2,
                rounds=2,
            )
        assert all(group.destroyed for group in ep.groups)
