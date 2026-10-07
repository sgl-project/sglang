"""CPU tests for distributed delta transaction failure boundaries."""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sglang.srt.weight_sync.gpu_delta import session as delta_runtime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.weight_sync.gpu_delta.session import (
    DeltaSession,
    GpuDeltaCommunicator,
    GpuDeltaSchedulerControl,
)


class Payload:
    def __init__(self, fail=False):
        self.fail = fail
        self.applications = 0
        self.closed = threading.Event()
        self.released = threading.Event()

    def apply(self):
        self.applications += 1
        if self.fail:
            raise RuntimeError("device verdict failed after a possible write")
        return {"applied": True, "verification": "manifest-sha256-and-decoder-status"}

    def close(self):
        self.closed.set()

    def release_and_close(self):
        self.released.set()
        self.close()


class Backend:
    def __init__(self, payload=None):
        self.payload = payload or Payload()
        self.started = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.host_arena = SimpleNamespace(directory=None)
        self.closed = False

    def close(self):
        self.closed = True

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


def request(participants):
    return dict(
        session_id="publication-1",
        manifest_path="/immutable/manifest.json",
        manifest_sha256="a" * 64,
        stream_id="run-1",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        participants=participants,
    )


def wait_state(session, state):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        receipt = session.status()
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
        instances.append((session, backend))
        return session, backend

    yield make
    for session, backend in instances:
        backend.ready.set()
        session._executor.shutdown(wait=True)


def prepare_ready(session, backend, req=None):
    session.prepare(req or request([session.identity]))
    backend.ready.set()
    return wait_state(session, "PREPARED")


def applied(session, backend, req=None):
    prepare_ready(session, backend, req)
    return session.apply(lambda: None, lambda: None, lambda: True)


def test_prepare_and_status_do_not_wait_for_file_io(make_session):
    session, backend = make_session()
    req = request([session.identity])
    preparing = session.prepare(req)
    assert preparing["state"] == "PREPARING"
    assert backend.started.wait(2)
    status = session.status()
    assert status["state"] == "PREPARING"
    assert backend.payload.applications == 0
    session.abort("publication-1")
    backend.ready.set()
    assert backend.payload.closed.wait(2)
    assert session.status()["state"] == "ABORTED"
    assert not backend.payload.released.is_set()


def test_engine_release_runs_off_scheduler_and_before_next_prepare(make_session):
    session, backend = make_session()
    other, other_backend = make_session(identity("b"))
    other.prepare(request([other.identity]))
    assert other_backend.started.wait(2)
    applied(session, backend)
    entered, release = threading.Event(), threading.Event()
    caller = threading.get_ident()

    def cleanup():
        assert threading.get_ident() != caller
        entered.set()
        assert release.wait(5)
        backend.payload.released.set()
        backend.payload.close()

    backend.payload.release_and_close = cleanup
    try:
        assert session.resume(lambda _: None)["state"] == "RESUMED"
        assert entered.wait(2)
        assert other.status()["state"] == "PREPARING"
        assert not other_backend.payload.released.is_set()
        backend.started.clear()
        req = request([session.identity]) | dict(
            session_id="publication-2", base_version=1, target_version=2
        )
        assert session.prepare(req)["state"] == "PREPARING"
        assert not backend.started.is_set()
    finally:
        release.set()
    assert backend.started.wait(2)
    assert backend.payload.released.is_set()


def test_bad_publication_never_mutates(make_session):
    session, backend = make_session()
    backend.error = ValueError("encoded checksum mismatch")
    session.prepare(request([session.identity]))
    backend.ready.set()
    assert "checksum" in wait_state(session, "FAILED")["message"]
    assert backend.payload.applications == 0
    session.abort("publication-1")
    assert session.status()["state"] == "ABORTED"


@pytest.mark.parametrize("failure", ["fence", "retract", "flush", "apply"])
def test_update_failure_is_terminal_and_never_reclaims_after_failed_fence(
    make_session, failure
):
    session, backend = make_session(backend=Backend(Payload(fail=failure == "apply")))
    prepare_ready(session, backend)
    events = []

    def phase(name):
        assert session.status()["state"] == "APPLYING"
        events.append(name)
        if name == failure:
            raise RuntimeError(name + " failed")
        return True

    with pytest.raises(RuntimeError):
        session.apply(
            lambda: phase("fence"),
            lambda: phase("retract"),
            lambda: phase("flush"),
        )
    expected = ["fence", "retract", "flush"]
    assert (
        events == expected[: expected.index(failure) + 1]
        if failure in expected
        else events == expected
    )
    assert backend.payload.applications == (1 if failure == "apply" else 0)
    status = session.status()
    assert (
        status["state"] == "POISONED"
        and status["scheduler_timing"]["blocked_s"] is None
    )
    assert not backend.payload.released.is_set()


def test_resume_failure_retains_ownership(make_session):
    session, backend = make_session()
    applied(session, backend)
    with pytest.raises(RuntimeError, match="resume failed"):
        session.resume(
            lambda _: (_ for _ in ()).throw(RuntimeError("resume failed")),
        )
    assert session.status()["state"] == "RESUMING"
    assert not backend.payload.released.is_set()
    assert session.status()["scheduler_timing"]["blocked_s"] is None


def test_scheduler_blocked_timing_excludes_background_prepare(
    make_session, monkeypatch
):
    now = [1_000_000_000]
    monkeypatch.setattr(
        delta_runtime, "time", SimpleNamespace(monotonic_ns=lambda: now[0])
    )
    session, backend = make_session()
    prepare_ready(session, backend)
    assert session.status()["scheduler_timing"]["pause_started_ns"] is None
    now[0] = 10_000_000_000

    def fence():
        assert session.status()["scheduler_timing"]["pause_started_ns"] == now[0]
        now[0] = 20_000_000_000

    receipt = session.apply(fence, lambda: None, lambda: True)
    now[0] = 60_000_000_000
    resumed = session.resume(lambda _: None)
    assert resumed["scheduler_timing"] == {
        "clock": "monotonic_ns",
        "pause_started_ns": 10_000_000_000,
        "reader_fence_completed_ns": 20_000_000_000,
        "resumed_ns": 60_000_000_000,
        "blocked_s": 50.0,
    }


@pytest.mark.parametrize("local_control,ep_joiner", [(True, False), (False, True)])
def test_describe_rejects_unsynchronized_control_topologies(
    monkeypatch, local_control, ep_joiner
):
    from sglang.srt import runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(
            enable_dp_attention_local_control_broadcast=local_control
        ),
    )
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=ep_joiner)),
    )
    control = GpuDeltaSchedulerControl(SimpleNamespace())
    with pytest.raises(ValueError, match="global control broadcast"):
        control._describe("engine-0")
    assert control.session is None


@pytest.mark.parametrize("cache", ["shared IPC", "derived HPC"])
def test_unsafe_weight_caches_reject_before_plan_or_session_creation(
    cache, monkeypatch
):
    from sglang.srt import runtime_context

    calls = []
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=False)),
    )

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(
            attn_tp_size=1,
            attn_cp_size=1,
            pp_size=1,
            enable_dp_attention_local_control_broadcast=False,
        ),
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
    assert control.session is None


def test_late_reply_after_cancellation_cannot_acknowledge_resume():
    async def scenario():
        sent = []
        comm = GpuDeltaCommunicator(sent.append, 1)
        old = asyncio.create_task(comm(SimpleNamespace(rid=None)))
        await asyncio.sleep(0)
        old_rid = sent[-1].rid
        old.cancel()
        with pytest.raises(asyncio.CancelledError):
            await old
        resume = asyncio.create_task(comm(SimpleNamespace(rid=None)))
        await asyncio.sleep(0)
        comm.handle_recv(SimpleNamespace(rid=old_rid, state="APPLIED"))
        await asyncio.sleep(0)
        assert not resume.done()
        comm.handle_recv(SimpleNamespace(rid=sent[-1].rid, state="RESUMED"))
        assert (await resume)[0].state == "RESUMED"

    asyncio.run(scenario())


def test_clear_drains_release_before_private_close_and_shared_unlink(
    make_session, tmp_path
):
    session, backend = make_session()
    directory = tmp_path / "own-stream-cache"
    directory.mkdir()
    (directory / "state.json").write_text("READY")
    source = tmp_path / "immutable-publication"
    source.write_bytes(b"retained")
    backend.host_arena.directory = directory
    req = request([session.identity]) | dict(target_version=7)
    applied(session, backend, req)
    entered, release = threading.Event(), threading.Event()

    def cleanup():
        entered.set()
        assert release.wait(5)
        assert not backend.closed
        assert (directory / "state.json").read_text() == "READY"
        backend.payload.release_and_close_original()

    backend.payload.release_and_close_original = backend.payload.release_and_close
    backend.payload.release_and_close = cleanup
    session.resume(lambda version: None)
    assert entered.wait(2)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as pool:
        clearing = pool.submit(session.clear)
        try:
            assert not clearing.done()
            assert not backend.closed and directory.exists()
        finally:
            release.set()
        receipt = clearing.result(timeout=2)
    assert receipt["state"] == "CLEARED" and receipt["version"] == 7
    assert receipt["target_version"] == 7 and receipt["identity"] == session.identity
    assert backend.closed and backend.payload.closed.is_set()
    assert session.backend is None and session._session is None
    # Closing all rank-local resources does not itself unlink the shared cache.
    assert directory.exists()
    session.release_cache(owner=True)
    assert not directory.exists() and source.read_bytes() == b"retained"
    assert session.clear()["version"] == 7
    session.release_cache(owner=True)
    with pytest.raises(ValueError, match="base version"):
        session.prepare(request([session.identity]))


@pytest.mark.parametrize(
    "state", ["PREPARING", "PREPARED", "APPLYING", "APPLIED", "POISONED", "RESUMING"]
)
def test_clear_cannot_discard_an_unsettled_update(make_session, state):
    session, backend = make_session()
    prepare_ready(session, backend)
    session._session.state = state
    with pytest.raises(ValueError, match="idle or resumed"):
        session.clear()
    assert not backend.closed and session.backend is backend
    assert not backend.payload.closed.is_set()


def test_clear_surfaces_failed_async_release_before_backend_close(make_session):
    session, backend = make_session()
    applied(session, backend)

    def fail_release():
        raise OSError("state release failed")

    backend.payload.release_and_close = fail_release
    session.resume(lambda _: None)
    with pytest.raises(OSError, match="state release failed"):
        session.clear()
    assert session.version == 1 and not backend.closed
    assert session.backend is backend
