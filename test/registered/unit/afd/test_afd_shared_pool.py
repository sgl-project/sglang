"""Shape rollback preserves the role's shared pool and other programs."""

import pytest

from sglang.srt.afd.contracts import AFDError
from sglang.test.afd.graph_fixtures import RecordingDriver, role_service, role_step
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_failed_second_bucket_keeps_first_program_and_shared_pool():
    driver = RecordingDriver(fail_at=1)
    cfg, service = role_service(driver=driver)
    role_step(service, cfg, step_id=0)
    first = service._cache.buckets[0]
    with pytest.raises(AFDError, match="STEP_NOT_GRAPHABLE"):
        role_step(service, cfg, step_id=1, rows=(17, 17))
    assert first.program is not None
    assert driver.captures == 2
    assert driver.closed == 0
    role_step(service, cfg, step_id=2)
    service.close()
    service.close()
    assert driver.closed == 1
    assert first.program is None


def test_capture_growth_is_counted_once_and_not_refunded_by_single_bucket_release():
    class Driver(RecordingDriver):
        def memory_usage(self, *, device):
            # The second shape reuses allocator space; no additional growth.
            return (1024, 2048) if self.programs else (0, 0)

    cfg, service = role_service(driver=Driver())
    role_step(service, cfg, step_id=0)
    role_step(service, cfg, step_id=1, rows=(17, 17))
    assert service.usage()["capture_hbm_bytes"] == 2048
    assert service.usage()["retained_hbm_bytes"] == 2048
    from sglang.srt.afd.contracts import AFDReason

    service._cache.invalidate(
        bucket=service._cache.buckets[0], reason=AFDReason.REPLAY_FAILED
    )
    assert service.usage()["retained_hbm_bytes"] == 2048
    assert service._cache.buckets[1].program is not None
    service.close()
    assert service.usage()["retained_hbm_bytes"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
