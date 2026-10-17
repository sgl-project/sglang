"""CPU tests for distributed delta transaction failure boundaries."""

import asyncio
import io
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sglang.srt.weight_sync.gpu_delta import io as delta_io
from sglang.srt.weight_sync.gpu_delta import session as delta_runtime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

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
        assert op == "apply_gpu_delta"
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
    scheduler = SimpleNamespace(_engine_paused=False)
    control = GpuDeltaSchedulerControl(scheduler)
    control.session = session
    rejected = control.handle(
        delta_io.PrepareGpuDeltaReqInput(**request([session.identity]))
    )
    assert not rejected.success and "base version" in rejected.message
    assert rejected.participant["state"] == "CLEARED"
    assert rejected.participant["version"] == 7
    assert rejected.participant["identity"] == session.identity
    assert not rejected.participant["generation_paused"]


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


@pytest.mark.parametrize("flush_cache", [True, False])
@pytest.mark.parametrize("keep_pause", [True, False])
@pytest.mark.parametrize("weight_version", [None, "0", "actor-42"])
def test_update_owns_pause_fence_retract_and_resume_order(
    monkeypatch, flush_cache, keep_pause, weight_version
):
    from sglang.utils import TypeBasedDispatcher

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
            delta_io.ApplyGpuDeltaReqInput(
                session_id="p", rid="apply-rid", flush_cache=flush_cache
            )
        )
        assert result.rid == "apply-rid" and scheduler._engine_paused
        expected = ["fence", "retract"] + (["flush"] if flush_cache else []) + ["apply"]
        assert result.success and events == expected
        result = wrapped(
            delta_io.ResumeGpuDeltaReqInput(
                session_id="p", keep_pause=keep_pause, weight_version=weight_version
            )
        )
        assert result.success and result.participant["state"] == "RESUMED"
        expected += [("version", "1" if weight_version is None else weight_version)] + (
            [] if keep_pause else ["resume"]
        )
        assert events[: len(expected)] == expected
        assert scheduler._engine_paused is keep_pause
        assert result.participant["generation_paused"] is keep_pause
        timing = result.participant["scheduler_timing"]
        assert (timing["resumed_ns"] is None) is keep_pause
        assert (timing["blocked_s"] is None) is keep_pause
        repeated = wrapped(
            delta_io.ResumeGpuDeltaReqInput(
                session_id="p", weight_version="conflicting-version"
            )
        )
        assert (
            not repeated.success and "requires an applied session" in repeated.message
        )
        assert events[: len(expected)] == expected
        assert not any(event == ("version", "conflicting-version") for event in events)
        assert scheduler._engine_paused is keep_pause
        cleared = wrapped(delta_io.ClearGpuDeltaStateReqInput())
        assert cleared.success and cleared.participant["version"] == 1
        assert events[len(expected) :] == ["release", "close"]
        released = wrapped(
            delta_io.ReleaseGpuDeltaCacheReqInput(owner_rank_ids=["original"])
        )
        assert released.success and released.participant["state"] == "CLEARED"
        assert scheduler._engine_paused is keep_pause
        assert released.participant["generation_paused"] is keep_pause
    finally:
        session._executor.shutdown(wait=True)


def test_describe_binds_published_scheduler_ranks(monkeypatch):
    rank = 7
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    try:
        # Derive the serving identity from the published launch-time ranks.
        publish(
            ServerArgs(
                model_path="dummy",
                tp_size=8,
                dp_size=8,
                ep_size=8,
                enable_dp_attention=True,
            ),
            role="test",
            ranks=SpawnRanks(world_rank=rank, dp_rank=rank),
        )
        runner = SimpleNamespace(
            model=object(),
            weight_updater=SimpleNamespace(
                _assert_weight_cache_inactive=lambda _: None
            ),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_lora = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.rust_server = None
        scheduler.tp_worker = SimpleNamespace(model_runner=runner)

        plan = {"tensors": []}
        backends = []

        def make_backend(model_runner, identity):
            assert model_runner is runner
            backends.append(identity.copy())
            return SimpleNamespace(
                describe=lambda: plan,
                host_arena=SimpleNamespace(directory=None),
                close=lambda: None,
            )

        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.weight_sync.gpu_delta.layout",
            SimpleNamespace(GpuDeltaBackend=make_backend),
        )
        monkeypatch.setattr(
            "sglang.srt.model_executor.model_runner_components.weight_updater."
            "_unsupported_derived_weight_cache_error",
            lambda _: None,
        )
        # Linux starttime field; comm deliberately contains spaces and ')'.
        stat = "123 (scheduler worker)) " + " ".join(["0"] * 19 + ["456"])
        monkeypatch.setattr(
            "sglang.srt.weight_sync.gpu_delta.session.open",
            lambda path: io.StringIO(stat),
            raising=False,
        )
        cache_engines = []

        def cache_id(engine_id):
            cache_engines.append(engine_id)
            return "shared-host-test"

        monkeypatch.setattr(
            "sglang.srt.weight_sync.gpu_delta.host.host_cache_id",
            cache_id,
        )
        control = GpuDeltaSchedulerControl(scheduler)
        receipt = control._describe("engine-0")
        assert receipt["identity"] | {
            "rank_id": "ignored",
            "hostname": "ignored",
            "pid": 0,
        } == {
            "engine_id": "engine-0",
            "rank_id": "ignored",
            "hostname": "ignored",
            "host_cache_id": "shared-host-test",
            "pid": 0,
            "start_ticks": 456,
            "tp_rank": rank,
            "dp_rank": rank,
            "pp_rank": 0,
        }
        assert receipt["state"] == "IDLE" and receipt["version"] == 0
        assert receipt["plan"] == plan
        assert control._describe("engine-0") == receipt
        assert len(backends) == 1
        assert cache_engines == ["engine-0"]
        control.session.version = 7
        assert control.session.clear()["version"] == 7
        control.session.release_cache(owner=True)
        fresh = control._describe("unused-new-engine-name")
        assert fresh["identity"] == receipt["identity"] and fresh["version"] == 7
        assert len(backends) == 2 and cache_engines == ["engine-0"]
        control.session._executor.shutdown(wait=True)
    finally:
        reset_context()


def test_nvtx_disabled_identity_and_worker_exception_balance(monkeypatch):
    profiling = delta_runtime.profiling
    events = []
    nvtx = SimpleNamespace(
        range_push=lambda name: events.append((threading.get_ident(), "push", name)),
        range_pop=lambda: events.append((threading.get_ident(), "pop")),
    )
    monkeypatch.setattr(profiling, "nvtx", nvtx, raising=False)

    def work(fail=False):
        if fail:
            raise RuntimeError("worker failure")
        return threading.get_ident()

    monkeypatch.setattr(profiling, "ENABLED", False)
    assert profiling.annotate("unused")(work) is work
    monkeypatch.setattr(profiling, "ENABLED", True)
    wrapped = profiling.annotate("gpu_delta.worker")(work)
    with delta_runtime.ThreadPoolExecutor(max_workers=1) as pool:
        worker_id = pool.submit(wrapped).result()
        with pytest.raises(RuntimeError, match="worker failure"):
            pool.submit(wrapped, True).result()
    assert worker_id != threading.get_ident()
    assert events == [
        (worker_id, "push", "gpu_delta.worker"),
        (worker_id, "pop"),
        (worker_id, "push", "gpu_delta.worker"),
        (worker_id, "pop"),
    ]


@pytest.mark.parametrize(
    "outcome", ["resumed", "retained", "apply_failed", "resume_failed"]
)
def test_nvtx_pause_ends_only_after_successful_unpause(
    make_session, monkeypatch, outcome
):
    events = []
    profiling = delta_runtime.profiling
    monkeypatch.setattr(profiling, "ENABLED", True)

    def start(name):
        events.append(("start", name))
        return 17

    monkeypatch.setattr(
        profiling,
        "nvtx",
        SimpleNamespace(
            range_push=lambda name: None,
            range_pop=lambda: None,
            range_start=start,
            range_end=lambda handle: events.append(("end", handle)),
            mark=lambda name: events.append(("mark", name)),
        ),
        raising=False,
    )
    session, backend = make_session()
    prepare_ready(session, backend)

    def fail():
        raise RuntimeError("control failure")

    if outcome == "apply_failed":
        with pytest.raises(RuntimeError, match="control failure"):
            session.apply(fail, lambda: None, None)
        assert session.status()["state"] == "POISONED"
        assert events[-1] == ("mark", "gpu_delta.pause.poisoned")
    else:
        session.apply(lambda: None, lambda: None, None)
        assert events == [("start", "gpu_delta.pause")]

        def resume(version):
            assert version == 1
            events.append(("resume", version))
            if outcome == "resume_failed":
                fail()

        if outcome == "resume_failed":
            with pytest.raises(RuntimeError, match="control failure"):
                session.resume(resume)
        else:
            session.resume(resume, keep_pause=outcome == "retained")
    if outcome == "resumed":
        assert events[-2:] == [("resume", 1), ("end", 17)]
        assert session._session.nvtx_pause_id is None
        assert session.status()["scheduler_timing"]["blocked_s"] is not None
    else:
        assert not any(event[0] == "end" for event in events)
        assert session._session.nvtx_pause_id == 17
        assert session.status()["scheduler_timing"]["blocked_s"] is None
