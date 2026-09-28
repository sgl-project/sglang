"""Exception-safe bucket release with real cache/accounting and CPU storage."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.cache import AFDBucketPhase, AFDShapeCache
from sglang.srt.afd.config import AFDConfig
from sglang.srt.afd.contracts import AFDReason, AFDRole
from sglang.srt.afd.role_graph import AFDRoleGraphService
from sglang.test.afd.graph_fixtures import make_shape
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CAPTURE_SIZES = tuple(range(1, 9))


class _ProgramBoundary:
    def __init__(self, events, name, failure=None):
        self.events = events
        self.name = name
        self.failure = failure
        self.storage = torch.ones(2, 3)
        self.storage_ref = weakref.ref(self.storage)

    def close(self):
        self.events.append(self.name)
        self.storage = None
        self.guard = None
        if self.failure is not None:
            raise self.failure


class _GuardOwner:
    pass


def _config():
    return AFDConfig(stages=2, max_hbm_bytes=4096)


def _populate(cache, config, events, *, rows, installed, program_error=None):
    shape = make_shape(
        capture_sizes=CAPTURE_SIZES,
        lane=0,
        lane_rows=((rows, rows),),
        hidden_size=8,
        dtype="bfloat16",
        config=config,
    )
    selection = cache.select(shape=shape, estimated_hbm_bytes=128, capture=True)
    bucket = selection.bucket
    program = _ProgramBoundary(events, f"bucket{rows}.program0", program_error)
    bucket.program = program
    programs = [program]
    guard = _GuardOwner()
    guard_ref = weakref.ref(guard)
    program.guard = guard
    bucket.retained_backing_hbm_bytes = 64
    if installed:
        cache.install(bucket=bucket)
    return selection, programs, guard_ref


def _assert_released(bucket, programs, guard_ref):
    assert bucket.program is None
    assert bucket.retained_backing_hbm_bytes == 0
    assert bucket.usage.retained_hbm_bytes == 0
    assert guard_ref() is None
    assert all(program.storage_ref() is None for program in programs)


@pytest.mark.parametrize(
    "action,installed",
    (("rollback", False), ("invalidate", False), ("invalidate", True)),
)
@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_terminal_transition_releases_owners_despite_cleanup_error(
    action, installed, error_type
):
    config, events = _config(), []
    cache = AFDShapeCache(capture_sizes=CAPTURE_SIZES, config=config, base_hbm_bytes=16)
    failure = error_type("first release failed")
    selection, programs, guard_ref = _populate(
        cache,
        config,
        events,
        rows=1,
        installed=installed,
        program_error=failure,
    )
    other, other_programs, other_guard = _populate(
        cache, config, events, rows=2, installed=True
    )
    method = getattr(cache, action)
    with pytest.raises(error_type) as caught:
        method(bucket=selection.bucket, reason=AFDReason.CAPTURE_FAILED)
    assert caught.value is failure
    assert events == ["bucket1.program0"]
    _assert_released(selection.bucket, programs, guard_ref)
    assert selection.bucket.phase is AFDBucketPhase.TERMINAL_EAGER
    assert selection.bucket.terminal_reason is AFDReason.CAPTURE_FAILED
    assert selection.bucket.usage.rollbacks == 1
    assert selection.bucket.usage.eager == {AFDReason.CAPTURE_FAILED.value: 1}
    assert cache.retained_hbm_bytes == 16 + 128
    after = list(events)
    method(bucket=selection.bucket, reason=AFDReason.CAPTURE_FAILED)
    assert events == after
    assert all(program.storage is not None for program in other_programs)
    cache.close()
    _assert_released(other.bucket, other_programs, other_guard)
    assert cache.retained_hbm_bytes == 0


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_cache_close_releases_later_buckets_and_reports_first_original_error(
    error_type,
):
    config, events = _config(), []
    cache = AFDShapeCache(capture_sizes=CAPTURE_SIZES, config=config, base_hbm_bytes=16)
    failure = error_type("first program failed")
    first, programs1, guard1 = _populate(
        cache,
        config,
        events,
        rows=1,
        installed=True,
        program_error=failure,
    )
    second, programs2, guard2 = _populate(
        cache,
        config,
        events,
        rows=2,
        installed=False,
        program_error=RuntimeError("later bucket program failed"),
    )
    with pytest.raises(error_type) as caught:
        cache.close()
    assert caught.value is failure
    assert events == [
        "bucket1.program0",
        "bucket2.program0",
    ]
    for selection, programs, guard in (
        (first, programs1, guard1),
        (second, programs2, guard2),
    ):
        _assert_released(selection.bucket, programs, guard)
        assert selection.bucket.phase is AFDBucketPhase.CLOSED
    assert cache.retained_hbm_bytes == 0
    after = list(events)
    usage = cache.close()
    assert events == after
    assert usage["status"] == "CLOSED"
    assert all(
        item["usage"]["retained_hbm_bytes"] == 0 for item in usage["buckets"].values()
    )
    closed_selection = cache.select(shape=first.bucket.shape, estimated_hbm_bytes=128)
    assert closed_selection.reason is AFDReason.CLOSED


def test_successful_cache_close_clears_per_bucket_accounting():
    config, events = _config(), []
    cache = AFDShapeCache(capture_sizes=CAPTURE_SIZES, config=config, base_hbm_bytes=16)
    selection, programs, guard = _populate(
        cache, config, events, rows=1, installed=True
    )
    usage = cache.close()
    _assert_released(selection.bucket, programs, guard)
    assert usage["retained_hbm_bytes"] == 0


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_service_close_finishes_cache_after_active_rollback_failure(error_type):
    config, events = _config(), []
    service = AFDRoleGraphService(
        capture_sizes=CAPTURE_SIZES,
        role=AFDRole.ATTENTION,
        config=config,
        num_layers=2,
        driver=SimpleNamespace(close=lambda: events.append("driver.close")),
        base_hbm_bytes=16,
        device="cuda:0",
    )
    earlier, earlier_programs, earlier_guard = _populate(
        service._cache,
        config,
        events,
        rows=1,
        installed=True,
        program_error=RuntimeError("older installed program failed"),
    )
    failure = error_type("active rollback failed")
    active, active_programs, active_guard = _populate(
        service._cache,
        config,
        events,
        rows=2,
        installed=False,
        program_error=failure,
    )
    service._selection = active
    service._retained_backing_hbm_bytes = 64
    with pytest.raises(error_type) as caught:
        service.close()
    assert caught.value is failure
    assert events == [
        "bucket2.program0",
        "bucket1.program0",
        "driver.close",
    ]
    for selection, programs, guard in (
        (active, active_programs, active_guard),
        (earlier, earlier_programs, earlier_guard),
    ):
        _assert_released(selection.bucket, programs, guard)
        assert selection.bucket.phase is AFDBucketPhase.CLOSED
    assert active.bucket.terminal_reason is AFDReason.PARTIAL_CAPTURE
    assert active.bucket.usage.rollbacks == 1
    assert service._selection is None
    assert service._retained_backing_hbm_bytes == 0
    after = list(events)
    assert service.close()["status"] == "CLOSED"
    assert service._cache.retained_hbm_bytes == 0
    assert events == after


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
