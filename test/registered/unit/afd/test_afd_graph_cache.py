"""Bucket lifecycle contracts migrated from per-operation to whole-role graphs."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.contracts import AFDError, AFDExecutionKind, AFDReason
from sglang.srt.afd.role_graph import copy_tensor_prefix, cuda_memory_diagnostic
from sglang.test.afd.graph_fixtures import (
    RecordingDriver,
    make_shape,
    role_service,
    role_step,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_capture_replay_values_and_program_ownership_stay_stable():
    cfg, service = role_service(num_layers=78)
    first, _ = role_step(service, cfg, step_id=0)
    program = service._driver.programs[0]
    # Arming must use exactly the peer-coordinated replay, never a private
    # extra exchange. Later steps replay the same program without rearming.
    assert program.armed == program.replays == 1
    armed, _ = role_step(service, cfg, step_id=1)
    replay, _ = role_step(service, cfg, step_id=2, compute=lambda staged: ())
    assert first.reason == AFDReason.ARMING
    assert armed.kind == AFDExecutionKind.REPLAY
    assert replay.kind == AFDExecutionKind.REPLAY
    for eager, captured, actual in zip(first.value, armed.value, replay.value):
        torch.testing.assert_close(eager[0], captured[0])
        torch.testing.assert_close(actual[0], eager[0])
    assert service._driver.programs == [program]
    assert service._driver.replay_ids == [id(program)] * 3
    assert program.armed == 1
    assert program.replays == 3
    assert service._driver.captures == 1
    usage = service.usage()
    bucket_usage = next(iter(usage["buckets"].values()))["usage"]
    assert service._eligible_steps == 3
    assert bucket_usage["installs"] == 1
    assert bucket_usage["replays"] == 2

    service.close()
    service.close()
    assert service._driver.closed == 1
    assert program.closed == 1


def test_unexecuted_arming_step_rolls_back_without_installing():
    cfg, service = role_service()
    shape = make_shape(
        lane=0, lane_rows=((3, 3),), hidden_size=16, dtype="bfloat16", config=cfg
    )
    service.begin_step(step_id=0, shape=shape, eligible=True, capture=True)
    assert service.end_step() == AFDReason.PARTIAL_CAPTURE
    bucket = service._cache.buckets[0]
    assert bucket.program is None


def test_capture_failure_keeps_runtime_pool_and_cannot_repeat_exchange():
    cfg, service = role_service(driver=RecordingDriver(fail_at=0))
    for step in range(2):
        with pytest.raises(AFDError, match="STEP_NOT_GRAPHABLE"):
            role_step(service, cfg, step_id=step)
    assert service._driver.programs == []
    assert service._driver.captures == 1
    assert service._driver.closed == 0
    assert service.usage()["retained_hbm_bytes"] == 0


def test_eight_buckets_never_evict_or_allocate_a_ninth_pool():
    cfg, service = role_service()
    for step in range(8):
        role_step(service, cfg, step_id=step, hidden=16 + step)
    programs = tuple(service._driver.programs)
    result, end = role_step(service, cfg, step_id=8, hidden=24)
    assert result.reason == end == AFDReason.BUCKET_LIMIT
    assert service._driver.captures == 8
    assert tuple(service._driver.programs) == programs
    assert not service._driver.closed


def test_hbm_gate_rejects_before_capture_allocation():
    cfg, service = role_service(max_hbm_bytes=1)
    with pytest.raises(AFDError, match="STEP_NOT_GRAPHABLE"):
        role_step(service, cfg, step_id=0)
    assert service._driver.captures == 0


def test_measured_pool_growth_exceeding_budget_releases_the_program():
    class Driver(RecordingDriver):
        def memory_usage(self, *, device):
            return (10000, 10000) if self.programs else (0, 0)

    cfg, service = role_service(driver=Driver(), max_hbm_bytes=5000)
    with pytest.raises(AFDError, match="STEP_NOT_GRAPHABLE"):
        role_step(service, cfg, step_id=0)
    assert service._driver.programs[0].closed == 1
    assert service._driver.closed == 0
    assert service.usage()["retained_hbm_bytes"] == 10000
    assert service.usage()["capture_hbm_bytes"] == 10000


def test_real_row_changes_within_one_planned_bucket_reuse_program_and_pool():
    cfg, service = role_service()
    role_step(service, cfg, step_id=0, rows=(3, 3))
    first = service._driver.programs[0]
    result, _ = role_step(service, cfg, step_id=1, rows=(5, 1))
    assert result.kind == AFDExecutionKind.REPLAY
    assert service._driver.programs == [first]
    assert service._driver.captures == 1
    assert [values[0].shape[0] for values in result.value] == [5, 1]


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_memory_diagnostic_absorbs_only_ordinary_errors(monkeypatch, error_type):
    def fail(*args):
        raise error_type("memory probe")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(memory_allocated=fail)),
    )
    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            cuda_memory_diagnostic("cuda:0")
    else:
        assert "RuntimeError" in cuda_memory_diagnostic("cuda:0")["unavailable"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_prefix_copy_large_small_empty_large_and_alias(noncontiguous):
    target = torch.ones(3, 8).T if noncontiguous else torch.ones(8, 3)
    for rows in (8, 3, 0, 8):
        source = (
            torch.arange(rows * 3).reshape(3, rows).T.float()
            if rows
            else torch.empty(0, 3)
        )
        expected = source.clone()
        copy_tensor_prefix(target=target, source=source, real_rows=rows)
        torch.testing.assert_close(target[:rows], expected)
        assert torch.count_nonzero(target[rows:]) == 0
    expected = target[:3].clone()
    copy_tensor_prefix(target=target, source=target[:3], real_rows=3)
    torch.testing.assert_close(target[:3], expected)
    assert torch.count_nonzero(target[3:]) == 0


def test_snapshot_skips_usage_construction_when_info_is_disabled(monkeypatch, caplog):
    cfg, service = role_service()
    with caplog.at_level("WARNING"):
        monkeypatch.setattr(
            service._cache, "usage", lambda **kw: pytest.fail("snapshot materialized")
        )
        service._log_step_snapshot(step_id=0, eligible=True)


def test_sampled_snapshots_keep_cumulative_replay_and_eager_evidence(caplog):
    import json

    cfg, service = role_service()
    service._log_interval = 4
    with caplog.at_level("INFO"):
        for step in range(9):
            role_step(service, cfg, step_id=step)
    snapshots = [
        json.loads(record.getMessage().split("AFD_GRAPH_USAGE_SNAPSHOT ", 1)[1])
        for record in caplog.records
        if "AFD_GRAPH_USAGE_SNAPSHOT " in record.getMessage()
    ]
    assert len(snapshots) == 3
    before, after = snapshots[-2:]
    assert after["completed_steps"] - before["completed_steps"] == 4
    assert after["replays"] - before["replays"] == 4
    assert before["non_replay_steps"] == after["non_replay_steps"] == 1
