"""Delta registration leaves ordinary dispatch unchanged and owns its pause fence."""

import time
from types import SimpleNamespace

from sglang.srt.weight_sync.gpu_delta import io as io
from sglang.srt.weight_sync.gpu_delta import session as delta_runtime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.utils import TypeBasedDispatcher

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_update_owns_pause_fence_retract_and_resume_order(monkeypatch):
    events = []
    scheduler = SimpleNamespace(_engine_paused=False)

    def fence():
        assert scheduler._engine_paused
        events.append("fence")

    def pause(obj):
        assert scheduler._engine_paused and obj.mode == "retract"
        events.append("retract")

    def flush(empty_cache):
        assert not empty_cache
        events.append("flush")
        return True

    def apply():
        events.append("apply")
        return {"applied": True}

    def resume(obj):
        assert not obj.torch_empty_cache
        events.append("resume")
        scheduler._engine_paused = False

    scheduler.device_module = SimpleNamespace(synchronize=fence)
    scheduler.pause_generation = pause
    scheduler.flush_cache = flush
    scheduler.continue_generation = resume
    scheduler.record_weight_version_change = lambda version: events.append(
        ("version", version)
    )
    control = delta_runtime.GpuDeltaSchedulerControl(scheduler)
    monkeypatch.setattr(delta_runtime, "GpuDeltaSchedulerControl", lambda _: control)
    wrapped = delta_runtime.with_gpu_delta_controls(scheduler, TypeBasedDispatcher([]))
    who = {"engine_id": "e0", "rank_id": "original"}
    session = control.session = delta_runtime.DeltaSession(
        who,
        SimpleNamespace(
            prepare=lambda *args: SimpleNamespace(
                apply=apply,
                close=lambda: None,
                release_and_close=lambda: events.append("release"),
            ),
            host_arena=SimpleNamespace(directory=None),
            close=lambda: events.append("close"),
        ),
    )
    try:
        session.prepare(
            dict(
                session_id="p",
                manifest_path="/manifest",
                manifest_sha256="a" * 64,
                stream_id="run",
                base_version=0,
                target_version=1,
                plan_digest="b" * 64,
                participants=[who],
            )
        )
        deadline = time.monotonic() + 2
        while session.status()["state"] == "PREPARING" and time.monotonic() < deadline:
            time.sleep(0.001)
        result = wrapped(
            io.UpdateWeightsFromDeltaReqInput(session_id="p", rid="apply-rid")
        )
        assert result.rid == "apply-rid" and scheduler._engine_paused
        assert result.success and events == ["fence", "retract", "flush", "apply"]
        result = wrapped(io.ResumeWeightsFromDeltaReqInput(session_id="p"))
        assert result.success and result.participant["state"] == "RESUMED"
        assert events[4:6] == [("version", "1"), "resume"]
        assert not scheduler._engine_paused
        cleared = wrapped(io.ClearWeightsDeltaStateReqInput())
        assert cleared.success and cleared.participant["version"] == 1
        assert events[6:] == ["release", "close"]
        released = wrapped(
            io.ReleaseWeightsDeltaCacheReqInput(owner_rank_ids=["original"])
        )
        assert released.success and released.participant["state"] == "CLEARED"
        assert not scheduler._engine_paused
    finally:
        session._executor.shutdown(wait=True)
