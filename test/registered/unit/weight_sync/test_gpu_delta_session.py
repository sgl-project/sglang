"""CPU tests for distributed delta transaction failure boundaries."""

import asyncio
import copy
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sglang.srt.weight_sync import gpu_delta_session as delta_runtime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.weight_sync.gpu_delta_session import (
    DeltaSession,
    GpuDeltaSchedulerControl,
    guard_tokenizer_dispatch,
)


class Payload:
    def __init__(self, fail=False):
        self.fail = fail
        self.applications = 0
        self.closed = threading.Event()

    def apply(self):
        self.applications += 1
        if self.fail:
            raise RuntimeError("device verdict failed after a possible write")
        return {"applied": True, "verification": "artifact-sha256-and-decoder-status"}

    def close(self):
        self.closed.set()


class Backend:
    def __init__(self, payload=None):
        self.payload = payload or Payload()
        self.started = threading.Event()
        self.ready = threading.Event()
        self.error = None

    def prepare(self, *args):
        self.started.set()
        assert self.ready.wait(5)
        if self.error:
            raise self.error
        return self.payload


def identity(engine="a", rank=0):
    return dict(
        engine_id=engine,
        rank_id=f"{engine}-{rank}-original",
        pid=10 + rank,
        start_ticks=42,
        tp_rank=rank,
    )


def request(participants, cohort=None):
    cohort = cohort or participants
    return dict(
        session_id="publication-1",
        manifest_path="/immutable/manifest.json",
        manifest_sha256="a" * 64,
        stream_id="run-1",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        participants=participants,
        cohort=cohort,
        expected_engines=sorted({item["engine_id"] for item in cohort}),
    )


def wait_state(session, state):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        receipt = session.status("publication-1")
        if receipt["state"] == state:
            return receipt
        time.sleep(0.001)
    pytest.fail(f"state did not reach {state}: {receipt}")


@pytest.fixture
def make_session():
    instances = []

    def make(who=None, backend=None):
        backend = backend or Backend()
        session = DeltaSession(who or identity(), backend)
        instances.append(session)
        return session, backend

    yield make
    for session in instances:
        session.backend.ready.set()
        session._executor.shutdown(wait=True)


def prepare_ready(session, backend, req=None):
    session.prepare(req or request([session.identity]))
    backend.ready.set()
    return wait_state(session, "PREPARED")


def applied(session, backend, req=None):
    prepare_ready(session, backend, req)
    session.quiesce(lambda: None)
    session.test_quiescence = [session.status("publication-1")]
    return session.apply(
        "publication-1",
        (req or request([session.identity]))["participants"],
        session.test_quiescence,
        lambda: True,
    )


def test_prepare_and_status_do_not_wait_for_file_io(make_session):
    session, backend = make_session()
    returned = threading.Event()

    def submit():
        assert session.prepare(request([session.identity]))["state"] == "PREPARING"
        returned.set()

    caller = threading.Thread(target=submit)
    caller.start()
    assert backend.started.wait(2)
    assert returned.wait(2), "submission waited for background I/O"
    caller.join()
    assert session.status("publication-1")["state"] == "PREPARING"
    assert backend.payload.applications == 0
    session.abort("publication-1")
    backend.ready.set()
    assert backend.payload.closed.wait(2)
    assert session.status("publication-1")["state"] == "ABORTED"


def test_bad_publication_never_quiesces_or_mutates(make_session):
    session, backend = make_session()
    backend.error = ValueError("encoded checksum mismatch")
    session.prepare(request([session.identity]))
    backend.ready.set()
    assert "checksum" in wait_state(session, "FAILED")["message"]
    with pytest.raises(ValueError, match="cannot pause"):
        session.quiesce(lambda: pytest.fail("must not fence old-model generation"))
    assert backend.payload.applications == 0
    session.abort("publication-1")
    assert not session.leased


def test_update_lost_ack_does_not_apply_xor_twice(make_session):
    session, backend = make_session()
    receipt = applied(session, backend)
    again = session.apply(
        "publication-1",
        [session.identity],
        session.test_quiescence,
        lambda: pytest.fail("duplicate apply flushed"),
    )
    assert receipt == again
    assert backend.payload.applications == 1
    with pytest.raises(ValueError, match="different publication"):
        session.prepare(request([session.identity]) | {"manifest_sha256": "c" * 64})
    with pytest.raises(ValueError, match="already leased"):
        session.prepare(
            request([session.identity]) | {"session_id": "retry-under-new-id"}
        )


def test_scheduler_blocked_timing_excludes_prepare_and_preserves_retry_boundaries(
    make_session, monkeypatch
):
    now = [1_000_000_000]
    monkeypatch.setattr(
        delta_runtime, "time", SimpleNamespace(monotonic_ns=lambda: now[0])
    )
    session, backend = make_session()
    prepare_ready(session, backend)
    assert (
        session.status("publication-1")["scheduler_timing"]["pause_started_ns"] is None
    )
    now[0] = 10_000_000_000

    def fence():
        timing = session.status("publication-1")["scheduler_timing"]
        assert timing["pause_started_ns"] == 10_000_000_000
        assert timing["reader_fence_completed_ns"] is None
        now[0] = 20_000_000_000

    session.quiesce(fence)
    quiesced = session.status("publication-1")
    assert quiesced["scheduler_timing"]["blocked_s"] is None
    now[0] = 30_000_000_000
    session.quiesce(lambda: None)  # Repeated pause must not shorten the span.
    assert session.status("publication-1") == quiesced
    receipt = session.apply(
        "publication-1", [session.identity], [quiesced], lambda: True
    )
    committed = session.commit("publication-1", [receipt])
    assert committed["scheduler_timing"]["resumed_ns"] is None
    assert committed["scheduler_timing"]["blocked_s"] is None
    session.authorize_resume("publication-1", [committed])
    now[0] = 60_000_000_000
    resumed = session.resumed("publication-1")
    assert resumed["scheduler_timing"] == {
        "clock": "monotonic_ns",
        "pause_started_ns": 10_000_000_000,
        "reader_fence_completed_ns": 20_000_000_000,
        "resumed_ns": 60_000_000_000,
        "blocked_s": 50.0,
    }
    now[0] = 90_000_000_000
    assert session.resumed("publication-1") == resumed


def test_exact_rank_and_all_engine_commit_before_resume(make_session):
    ids = [identity("a"), identity("b")]
    sessions = [make_session(who) for who in ids]
    for who, (session, backend) in zip(ids, sessions):
        prepare_ready(session, backend, request([who], ids))
        session.quiesce(lambda: None)
    quiesced = [session.status("publication-1") for session, _ in sessions]
    receipts = [
        session.apply("publication-1", [who], quiesced, lambda: True)
        for who, (session, _) in zip(ids, sessions)
    ]
    first = sessions[0][0]
    with pytest.raises(ValueError, match="every original rank"):
        first.commit("publication-1", receipts[:1])
    recycled = copy.deepcopy(receipts)
    recycled[1]["identity"]["start_ticks"] += 1
    with pytest.raises(ValueError, match="every original rank"):
        first.commit("publication-1", recycled)
    committed_a = first.commit("publication-1", receipts)
    with pytest.raises(ValueError, match="requires COMMITTED"):
        first.authorize_resume("publication-1", [committed_a, receipts[1]])
    committed_b = sessions[1][0].commit("publication-1", receipts)
    first.authorize_resume("publication-1", [committed_a, committed_b])
    assert first.resumed("publication-1")["state"] == "RESUMED"
    assert not first.leased
    assert first.version == 1


def test_partial_write_poison_cannot_abort_commit_or_retry(make_session):
    session, backend = make_session(backend=Backend(Payload(fail=True)))
    prepare_ready(session, backend)
    session.quiesce(lambda: None)
    quiesced = [session.status("publication-1")]
    with pytest.raises(RuntimeError, match="possible write"):
        session.apply("publication-1", [session.identity], quiesced, lambda: True)
    assert session.status("publication-1")["state"] == "POISONED"
    timing = session.status("publication-1")["scheduler_timing"]
    assert timing["pause_started_ns"] is not None
    assert timing["resumed_ns"] is None
    assert timing["blocked_s"] is None
    with pytest.raises(ValueError, match="cannot abort POISONED"):
        session.abort("publication-1")
    with pytest.raises(ValueError, match="requires retract pause"):
        session.apply("publication-1", [session.identity], quiesced, lambda: True)
    assert session.leased
    assert backend.payload.applications == 1


def test_reader_fence_and_cache_flush_precede_first_write(make_session):
    session, backend = make_session()
    prepare_ready(session, backend)
    with pytest.raises(ValueError, match="retract pause"):
        session.apply(
            "publication-1",
            [session.identity],
            [session.status("publication-1")],
            lambda: True,
        )
    with pytest.raises(RuntimeError, match="reader"):
        session.quiesce(
            lambda: (_ for _ in ()).throw(RuntimeError("reader fence failed"))
        )
    assert session.status("publication-1")["state"] == "PREPARED"
    session.quiesce(lambda: None)
    with pytest.raises(ValueError, match="cache flush"):
        session.apply(
            "publication-1",
            [session.identity],
            [session.status("publication-1")],
            lambda: False,
        )
    assert backend.payload.applications == 0
    assert (
        session.apply(
            "publication-1",
            [session.identity],
            [session.status("publication-1")],
            lambda: True,
        )["state"]
        == "APPLIED"
    )


def test_lease_blocks_competing_disk_update_but_not_generation_or_status():
    manager = SimpleNamespace(_gpu_delta_session_id="leased")
    for name in (
        "UpdateWeightFromDiskReqInput",
        "BeginWeightUpdateReqInput",
        "ReleaseMemoryOccupationReqInput",
        "PdRoleSwitchReqInput",
    ):
        with pytest.raises(ValueError, match="competing mutation"):
            guard_tokenizer_dispatch(manager, type(name, (), {})())
    for name in ("TokenizedGenerateReqInput", "GetWeightsDeltaStatusReqInput"):
        guard_tokenizer_dispatch(manager, type(name, (), {})())
    manager._gpu_delta_session_id = None
    guard_tokenizer_dispatch(manager, type("UpdateWeightFromDiskReqInput", (), {})())


def test_other_update_path_invalidates_startup_baseline_before_first_prepare():
    control = GpuDeltaSchedulerControl(SimpleNamespace())
    disk_update = type("UpdateWeightFromDiskReqInput", (), {})()
    assert control.reject_conflicting(disk_update) is None  # old API still runs
    assert control.legacy_mutated
    with pytest.raises(ValueError, match="fresh engine"):
        control._describe("engine-0")


@pytest.mark.parametrize("cache", ["shared IPC", "derived HPC"])
def test_unsafe_weight_caches_reject_before_plan_or_session_creation(
    cache, monkeypatch
):
    from sglang.srt import runtime_context

    calls = []

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1, attn_cp_size=1, pp_size=1),
    )

    def check_shared(op):
        assert op == "update_weights_from_delta"
        calls.append("shared")
        if cache == "shared IPC":
            raise RuntimeError(cache)

    model = object()

    def check_derived(candidate):
        assert candidate is model
        calls.append("derived")
        return cache

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner_components.weight_updater",
        SimpleNamespace(_unsupported_derived_weight_cache_error=check_derived),
    )
    runner = SimpleNamespace(
        weight_updater=SimpleNamespace(_assert_weight_cache_inactive=check_shared),
        model=model,
    )
    scheduler = SimpleNamespace(
        enable_lora=False,
        disaggregation_mode=SimpleNamespace(value="null"),
        rust_server=None,
        tp_worker=SimpleNamespace(model_runner=runner),
    )
    control = GpuDeltaSchedulerControl(scheduler)
    with pytest.raises((RuntimeError, ValueError), match=cache):
        control._describe("engine-0")
    assert calls == (["shared"] if cache == "shared IPC" else ["shared", "derived"])
    assert (
        control.identity is None and control.backend is None and control.session is None
    )


def test_apply_requires_whole_cohort_quiescence(make_session):
    local = identity("a")
    remote = identity("b")
    session, backend = make_session(local)
    prepare_ready(session, backend, request([local], [local, remote]))
    session.quiesce(lambda: None)
    local_receipt = session.status("publication-1")
    with pytest.raises(ValueError, match="every original rank"):
        session.apply("publication-1", [local], [local_receipt], lambda: True)
    remote_receipt = copy.deepcopy(local_receipt)
    remote_receipt["identity"] = remote
    remote_receipt["state"] = "PREPARED"
    with pytest.raises(ValueError, match="requires QUIESCED"):
        session.apply(
            "publication-1", [local], [local_receipt, remote_receipt], lambda: True
        )
    assert backend.payload.applications == 0


def test_late_reply_after_cancellation_cannot_acknowledge_resume():
    from sglang.srt.managers.communicator import FanOutCommunicator

    async def scenario():
        sent = []
        comm = FanOutCommunicator(sent.append, 1, correlate_rid=True)
        old = asyncio.create_task(comm(SimpleNamespace(rid=None)))
        await asyncio.sleep(0)
        old_rid = sent[-1].rid
        old.cancel()
        with pytest.raises(asyncio.CancelledError):
            await old
        resume = asyncio.create_task(comm(SimpleNamespace(rid=None)))
        await asyncio.sleep(0)
        comm.handle_recv(SimpleNamespace(rid=old_rid, state="COMMITTED"))
        await asyncio.sleep(0)
        assert not resume.done()
        comm.handle_recv(SimpleNamespace(rid=sent[-1].rid, state="RESUMED"))
        assert (await resume)[0].state == "RESUMED"

    asyncio.run(scenario())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
