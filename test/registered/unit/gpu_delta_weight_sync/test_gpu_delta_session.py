"""CPU tests for distributed delta transaction failure boundaries."""

import hashlib
import io
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sglang.srt.weight_sync.gpu_delta import io as delta_io
from sglang.srt.weight_sync.gpu_delta import session as delta_runtime
from sglang.srt.weight_sync.gpu_delta.receiver import GpuDeltaBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

from sglang.srt.weight_sync.gpu_delta.session import (
    DeltaSession,
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
        return {
            "applied": True,
            "target_version": self.target_version,
            "verification": "manifest-sha256-and-decoder-status",
            "timings": {"host_prepare_s": 0.25},
        }

    def close(self):
        self.closed.set()

    def release_and_close(self):
        self.released.set()
        self.close()


class Backend:
    discard = GpuDeltaBackend.discard

    def __init__(self, payload=None):
        self.payload = payload or Payload()
        self.started = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.host_arena = SimpleNamespace(directory=None)
        self.closed = False

    def close(self):
        self.closed = True

    def prepare(self, request, cancelled):
        self.started.set()
        assert self.ready.wait(5)
        if self.error:
            raise self.error
        self.payload.target_version = request["target_version"]
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


@pytest.mark.parametrize(
    "abort_state,drain_failure",
    [("PREPARING", False), ("PREPARED", False), ("PREPARING", True)],
)
def test_prepare_and_status_do_not_wait_for_file_io(
    make_session, abort_state, drain_failure
):
    session, backend = make_session()
    req = request([session.identity])
    control = GpuDeltaSchedulerControl(SimpleNamespace(_engine_paused=False))
    control.session = session
    accepted = control.handle(
        delta_io.PrepareGpuDeltaReqInput(**req, validate_payload_hash=False)
    )
    assert accepted.success
    preparing = accepted.participants[0]
    assert session._session.request["validate_payload_hash"] is False
    assert preparing["state"] == "PREPARING"
    assert backend.started.wait(2)
    status = session.status()
    assert status["state"] == "PREPARING"
    assert backend.payload.applications == 0
    matched = control.handle(
        delta_io.GetGpuDeltaStatusReqInput(session_id="publication-1")
    )
    assert matched.success and matched.participants[0]["session_id"] == "publication-1"
    mismatched = control.handle(
        delta_io.GetGpuDeltaStatusReqInput(session_id="another-publication")
    )
    assert not mismatched.success and "different session" in mismatched.message
    for receipt in (
        preparing,
        status,
        matched.participants[0],
        mismatched.participants[0],
    ):
        for key in req.keys() - {"manifest_path", "participants"}:
            assert receipt[key] == req[key]
    assert session.status()["state"] == "PREPARING"
    if abort_state == "PREPARED":
        backend.ready.set()
        wait_state(session, "PREPARED")
    cleanup = []
    close_payload, close_backend = backend.payload.close, backend.close

    def drain():
        assert not backend.closed
        cleanup.append("drain")
        if drain_failure:
            raise RuntimeError("GPU drain failed")
        close_payload()

    def close():
        assert backend.payload.closed.is_set()
        cleanup.append("close")
        close_backend()

    backend.payload.close, backend.close = drain, close
    session.abort("publication-1")
    backend.ready.set()
    session._executor.submit(lambda: None).result(timeout=2)
    if drain_failure:
        with pytest.raises(RuntimeError, match="GPU drain failed"):
            session._prepare_future.result()
        assert cleanup == ["drain"]
        assert not backend.closed and not backend.payload.closed.is_set()
    else:
        assert cleanup == ["drain", "close"]
        assert backend.closed and backend.payload.closed.is_set()
    assert session.status()["state"] == "ABORTED"
    assert not backend.payload.released.is_set()
    backend.started.clear()
    with pytest.raises(ValueError, match="requires restart"):
        session.prepare(req)
    assert not backend.started.is_set()


def test_abort_unblocks_rank_waiting_for_shared_preparation_owner(
    make_session, monkeypatch, tmp_path
):
    from sglang.srt.weight_sync.gpu_delta import host

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(host, "_cache_base", lambda: cache)
    manifest = {"codec": "bitcomp-zstd", "files": [], "tensors": []}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    participants = [
        identity(rank=rank)
        | dict(host_cache_id="shared-host", dp_rank=0, pid=os.getpid())
        for rank in range(2)
    ]
    arena = host.HostArena("a", 0, participants[1])

    class WaitingBackend(Backend):
        def prepare(self, request, cancelled):
            self.started.set()
            return arena.prepare_publication(request, {}, cancelled)

    backend = WaitingBackend()
    backend.host_arena = arena
    session, _ = make_session(participants[1], backend)
    req = request(participants) | dict(
        manifest_path=str(path),
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validate_payload_hash=True,
    )
    try:
        session.prepare(req)
        assert backend.started.wait(2)
        assert session.status()["state"] == "PREPARING"
        assert arena._worker is None
    finally:
        session.abort(req["session_id"])
        # A queued callback proves the blocked prepare actually exited.
        session._executor.submit(lambda: None).result(timeout=2)
        arena.close()
    assert session.status()["state"] == "ABORTED"
    assert arena.allocation is None and not backend.payload.released.is_set()


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
            session_id="publication-2",
            manifest_sha256="c" * 64,
            base_version=1,
            target_version=2,
        )
        receipt = session.prepare(req)
        assert receipt["state"] == "PREPARING"
        for key in req.keys() - {"manifest_path", "participants"}:
            assert receipt[key] == req[key]
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
    assert receipt["state"] == "APPLIED"
    assert receipt["identity"] == session.identity
    assert receipt["result"] == {"applied": True, "target_version": 1}
    full_result = session.status()["result"]
    assert full_result["timings"] == {"host_prepare_s": 0.25}
    now[0] = 60_000_000_000
    resumed = session.resume(lambda _: None)
    assert resumed["result"] == full_result
    assert {
        k: v
        for k, v in receipt.items()
        if k not in {"state", "result", "scheduler_timing"}
    } == {
        k: v
        for k, v in resumed.items()
        if k not in {"state", "result", "scheduler_timing"}
    }
    assert resumed["scheduler_timing"] == {
        "clock": "monotonic_ns",
        "pause_started_ns": 10_000_000_000,
        "reader_fence_completed_ns": 20_000_000_000,
        "resumed_ns": 60_000_000_000,
        "blocked_s": 50.0,
    }


@pytest.mark.parametrize(
    "override,ep_joiner,message",
    [
        (
            {"enable_dp_attention_local_control_broadcast": True},
            False,
            "global control",
        ),
        ({}, True, "global control"),
        (
            {"attn_cp_size": 2, "enable_cp_decode_attn_tp": True},
            False,
            "CP decode attention TP caches",
        ),
        ({"attn_dcp_size": 2}, False, "DCP groups inside attention TP"),
        ({"pp_size": 2}, False, "PP1"),
    ],
)
def test_describe_rejects_unsupported_control_topologies(
    monkeypatch, override, ep_joiner, message
):
    from sglang.srt import runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(
            **(
                dict(
                    enable_dp_attention_local_control_broadcast=False,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    attn_dcp_size=1,
                    enable_cp_decode_attn_tp=False,
                    pp_size=1,
                )
                | override
            )
        ),
    )
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=ep_joiner)),
    )
    control = GpuDeltaSchedulerControl(SimpleNamespace())
    with pytest.raises(ValueError, match=message):
        control._describe("engine-0")
    assert control.session is None


@pytest.mark.parametrize(
    "cp_size,tp_size,rank",
    [(1, 1, 0), (1, 2, 0), (1, 2, 1), (2, 1, 0), (2, 1, 1)]
    + [(2, 2, rank) for rank in range(4)],
)
def test_cp_tp_replies_preserve_follower_failure_and_original_identity(
    monkeypatch, cp_size, tp_size, rank
):
    # Use a nonzero global root, as in another attention-DP shard.
    cp_rank, tp_rank = divmod(rank, tp_size)
    cp_ranks = [4 + cp * tp_size + tp_rank for cp in range(cp_size)]
    tp_ranks = [4 + cp_rank * tp_size + tp for tp in range(tp_size)]
    groups = tuple(
        SimpleNamespace(
            rank_in_group=local_rank,
            world_size=len(ranks),
            ranks=ranks,
            cpu_group=object(),
        )
        for local_rank, ranks in ((cp_rank, cp_ranks), (tp_rank, tp_ranks))
        if len(ranks) > 1
    )
    count = cp_size * tp_size
    receipts = [
        dict(
            identity=identity(rank=4 + index),
            session_id="publication-1",
            state="POISONED" if count > 1 and index == count - 1 else "APPLIED",
            message=(
                "failed follower decode" if count > 1 and index == count - 1 else ""
            ),
            generation_paused=True,
        )
        for index in range(count)
    ]
    values = [(r["state"] == "APPLIED", r["message"], r) for r in receipts]
    control = GpuDeltaSchedulerControl(SimpleNamespace(_engine_paused=True), groups)
    control.session = SimpleNamespace(status=lambda: receipts[rank].copy())
    calls = []

    def gather_object(local, output, dst, group):
        expected = groups[len(calls)]
        assert dst == expected.ranks[0] and group is expected.cpu_group
        if cp_size > 1 and not calls:
            assert local == [values[rank]]
            peer_values = [[values[peer - 4]] for peer in cp_ranks]
        else:
            assert cp_rank == 0
            assert local == [values[cp * tp_size + tp_rank] for cp in range(cp_size)]
            peer_values = [
                [values[cp * tp_size + tp] for cp in range(cp_size)]
                for tp in range(tp_size)
            ]
        calls.append(dst)
        if expected.rank_in_group == 0:
            output[:] = peer_values
        else:
            assert output is None

    monkeypatch.setattr("torch.distributed.gather_object", gather_object)
    result = control.handle(
        delta_io.GetGpuDeltaStatusReqInput(session_id="publication-1", rid="status")
    )
    if rank == 0:
        assert len(calls) == len(groups)
        assert result.success is (count == 1)
        assert result.rid == "status" and result.message == receipts[-1]["message"]
        assert (
            sorted(result.participants, key=lambda r: r["identity"]["pid"]) == receipts
        )
    else:
        assert len(calls) == (1 if cp_rank else len(groups)) and result is None


@pytest.mark.parametrize("cache", ["shared IPC", "derived HPC"])
@pytest.mark.parametrize("attn_cp_size", [1, 2])
def test_unsafe_weight_caches_reject_before_plan_or_session_creation(
    cache, attn_cp_size, monkeypatch
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
            attn_cp_size=attn_cp_size,
            attn_dcp_size=1,
            enable_cp_decode_attn_tp=False,
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
    assert rejected.participants[0]["state"] == "CLEARED"
    assert rejected.participants[0]["version"] == 7
    assert rejected.participants[0]["identity"] == session.identity
    assert not rejected.participants[0]["generation_paused"]
    for receipt in (receipt, session.status(), rejected.participants[0]):
        for key in req.keys() - {"manifest_path", "participants"}:
            assert receipt[key] == req[key]


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


@pytest.mark.parametrize("next_operation", ["clear", "prepare"])
def test_failed_async_release_prevents_resource_reuse(make_session, next_operation):
    session, backend = make_session()
    applied(session, backend)

    def fail_release():
        raise OSError("state release failed")

    backend.payload.release_and_close = fail_release
    session.resume(lambda _: None)
    if next_operation == "clear":
        with pytest.raises(OSError, match="state release failed"):
            session.clear()
    else:
        backend.started.clear()
        session.prepare(
            request([session.identity])
            | dict(session_id="publication-2", base_version=1, target_version=2)
        )
        receipt = wait_state(session, "FAILED")
        assert "state release failed" in receipt["message"]
        assert not backend.started.is_set()
        assert backend.payload.applications == 1
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
        return {
            "applied": True,
            "target_version": 1,
            "timings": {"host_prepare_s": 0.25},
        }

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
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(attn_cp_size=1, attn_tp_size=1),
    )

    def make_control(_, groups):
        assert groups == ()
        return control

    monkeypatch.setattr(delta_runtime, "GpuDeltaSchedulerControl", make_control)
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
        assert result.participants[0]["generation_paused"]
        assert result.participants[0]["result"] == {
            "applied": True,
            "target_version": 1,
        }
        result = wrapped(
            delta_io.ResumeGpuDeltaReqInput(
                session_id="p", keep_pause=keep_pause, weight_version=weight_version
            )
        )
        assert result.success and result.participants[0]["state"] == "RESUMED"
        assert result.participants[0]["result"]["timings"] == {"host_prepare_s": 0.25}
        expected += [("version", "1" if weight_version is None else weight_version)] + (
            [] if keep_pause else ["resume"]
        )
        assert events[: len(expected)] == expected
        assert scheduler._engine_paused is keep_pause
        assert result.participants[0]["generation_paused"] is keep_pause
        timing = result.participants[0]["scheduler_timing"]
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
        assert cleared.success and cleared.participants[0]["version"] == 1
        assert events[len(expected) :] == ["release", "close"]
        released = wrapped(
            delta_io.ReleaseGpuDeltaCacheReqInput(owner_rank_ids=["original"])
        )
        assert released.success and released.participants[0]["state"] == "CLEARED"
        assert scheduler._engine_paused is keep_pause
        assert released.participants[0]["generation_paused"] is keep_pause
    finally:
        session._executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "tp_size,dp_size,attn_dp_size,dcp_size,world_rank,dp_rank",
    [
        (8, 1, 8, 1, 7, 7),
        (4, 1, 2, 1, 3, 1),
        (2, 1, 1, 1, 1, None),
        (4, 1, 2, 2, 3, 1),
        (2, 1, 1, 2, 1, None),
        (1, 2, 1, 1, 0, 1),
        (1, 1, 1, 1, 0, None),
    ],
    ids=[
        "attention-dp",
        "attention-dp-tp",
        "attention-tp",
        "attention-dp-tp-dcp",
        "attention-tp-dcp",
        "replica-dp",
        "singleton",
    ],
)
def test_describe_binds_published_scheduler_ranks(
    monkeypatch, tp_size, dp_size, attn_dp_size, dcp_size, world_rank, dp_rank
):
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
                tp_size=tp_size,
                dp_size=dp_size,
                attn_dp_size=attn_dp_size,
                dcp_size=dcp_size,
                ep_size=tp_size,
            ),
            role="test",
            ranks=SpawnRanks(world_rank=world_rank, dp_rank=dp_rank),
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
            "sglang.srt.weight_sync.gpu_delta.receiver",
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
            "tp_rank": world_rank,
            "dp_rank": 0 if dp_rank is None else dp_rank,
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
