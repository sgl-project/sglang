"""GPU-delta HTTP contract and tokenizer transaction coordination, without engines."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
from sglang.srt.utils.auth import AuthLevel, add_api_key_middleware
from sglang.srt.weight_sync.gpu_delta import io as io
from sglang.srt.weight_sync.gpu_delta import tokenizer as tokenizer
from sglang.srt.weight_sync.gpu_delta.http import register_gpu_delta_routes
from sglang.srt.weight_sync.gpu_delta.io import (
    ClearGpuDeltaStateReqInput,
    GetGpuDeltaInfoReqInput,
    GetGpuDeltaStatusReqInput,
    GpuDeltaReqOutput,
    UpdateWeightsFromGpuDeltaReqInput,
)
from sglang.srt.weight_sync.gpu_delta.tokenizer import GpuDeltaCommunicator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def http_delta():
    class CustomRoute(APIRoute):
        pass

    app = FastAPI()
    app.router.route_class = CustomRoute
    state = SimpleNamespace(tokenizer_manager=None)
    register_gpu_delta_routes(app, lambda: state)
    add_api_key_middleware(app, api_key="ordinary-key", admin_api_key="admin-key")
    calls = []
    reply = {"success": True, "message": "", "participants": []}

    async def dispatch(obj):
        calls.append(obj)
        return reply

    # Global state is populated at startup, after the routes are registered.
    state.tokenizer_manager = SimpleNamespace(
        gpu_delta=SimpleNamespace(request=dispatch)
    )
    with TestClient(app, headers={"Authorization": "Bearer admin-key"}) as client:
        yield client, app, calls, reply, CustomRoute


def test_delta_routes_preserve_typed_requests_auth_and_app_route_class(http_delta):
    client, app, calls, reply, route_class = http_delta
    session = {"session_id": "publication-1"}
    receipts = [
        {
            "identity": {"engine_id": "e0", "rank_id": "original"},
            "state": "APPLIED",
            "session_id": "publication-1",
            "target_version": 1,
        }
    ]
    publication = session | {
        "manifest_path": "/immutable/manifest.json",
        "manifest_sha256": "a" * 64,
        "stream_id": "run-1",
        "base_version": 0,
        "target_version": 1,
        "plan_digest": "b" * 64,
        "participants": [],
    }
    cases = [
        (
            "update_weights_from_gpu_delta",
            UpdateWeightsFromGpuDeltaReqInput,
            publication,
        ),
        (
            "update_weights_from_gpu_delta",
            UpdateWeightsFromGpuDeltaReqInput,
            publication
            | {
                "release_state": True,
                "flush_cache": False,
                "abort_all_requests": True,
                "weight_version": "actor-42",
                "validate_payload_hash": False,
            },
        ),
        ("clear_gpu_delta_state", ClearGpuDeltaStateReqInput, {}),
        ("get_gpu_delta_info", GetGpuDeltaInfoReqInput, {"engine_id": "e0"}),
        ("get_gpu_delta_status", GetGpuDeltaStatusReqInput, session),
    ]
    assert {route.path for route in app.routes if "gpu_delta" in route.path} == {
        "/update_weights_from_gpu_delta",
        "/clear_gpu_delta_state",
        "/get_gpu_delta_info",
        "/get_gpu_delta_status",
    }
    for name, request_type, payload in cases:
        route = next(route for route in app.routes if route.path == f"/{name}")
        assert isinstance(route, route_class)
        assert route.methods == {"POST"}
        assert route.endpoint._auth_level is AuthLevel.ADMIN_OPTIONAL
        rejected = client.post(
            route.path,
            json=payload,
            headers={"Authorization": "Bearer ordinary-key"},
        )
        assert rejected.status_code == 401
        result = client.post(route.path, json=payload)
        assert result.status_code == 200 and result.json() == reply
        assert isinstance(calls[-1], request_type)
        received = msgpack_decode(msgpack_encode(calls[-1]))
        assert type(received) is request_type
        assert received == calls[-1]
        if request_type is UpdateWeightsFromGpuDeltaReqInput:
            assert received.validate_payload_hash is payload.get(
                "validate_payload_hash", True
            )
    response = GpuDeltaReqOutput(
        success=True, message="", participant=receipts[0], rid="control-reply"
    )
    received = msgpack_decode(msgpack_encode(response))
    assert type(received) is GpuDeltaReqOutput and received == response
    assert len(calls) == len(cases)
    assert calls[0].release_state is False
    rejected = client.post(
        "/update_weights_from_gpu_delta", json={"manifest_path": "/manifest.json"}
    )
    assert rejected.status_code == 422 and len(calls) == len(cases)

    # A real scheduler failure retains the same typed HTTP route and response.
    reply.update(success=False, message="delta payload checksum failed")
    result = client.post("/get_gpu_delta_status", json={"session_id": "p"})
    assert result.status_code == 409 and result.json() == reply
    assert len(calls) == len(cases) + 1


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setattr(
        tokenizer, "get_serving", lambda: SimpleNamespace(tokenizer_worker_num=1)
    )
    sent, versions = [], []
    manager = SimpleNamespace(
        is_pause=False,
        mm_processor=None,
        is_pause_cond=asyncio.Condition(),
        _dispatch_to_scheduler=sent.append,
        auto_create_handle_loop=lambda: None,
        _update_weight_version_if_provided=versions.append,
    )
    control = tokenizer.GpuDeltaTokenizerControl.__new__(
        tokenizer.GpuDeltaTokenizerControl
    )
    control.manager = manager
    control.communicator = GpuDeltaCommunicator(manager._dispatch_to_scheduler, 1)
    return control, sent, versions


@pytest.fixture
def publication():
    return dict(
        session_id="publication-1",
        manifest_path="/immutable/manifest.json",
        manifest_sha256="a" * 64,
        stream_id="run-1",
        base_version=2,
        target_version=7,
        plan_digest="b" * 64,
        participants=[
            dict(
                engine_id="e0",
                rank_id=f"r{i}",
                hostname=f"host-{i // 2}",
                host_cache_id=f"cache-{i // 2}",
            )
            for i in range(3)
        ],
    )


def reply(control, obj, state, rank=0):
    control.communicator.handle_recv(
        io.GpuDeltaReqOutput(
            rid=obj.rid,
            success=True,
            message="",
            participant={
                "identity": {"engine_id": "e0", "rank_id": f"r{rank}"},
                "session_id": obj.session_id,
                "state": state,
                "target_version": 7,
            },
        )
    )


def test_update_waits_for_every_rank_before_apply_and_resume(
    control, publication, monkeypatch
):
    async def run():
        facade, _, versions = control
        setup_calls = []
        facade.manager.auto_create_handle_loop = lambda: setup_calls.append("loop")

        def serving():
            setup_calls.append("workers")
            return SimpleNamespace(tokenizer_worker_num=1)

        monkeypatch.setattr(tokenizer, "get_serving", serving)
        sent = asyncio.Queue()
        facade.communicator._send = sent.put_nowait
        facade.communicator.set_fan_out(2)
        task = asyncio.create_task(
            facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
        )
        prepare = await sent.get()
        assert type(prepare) is io.PrepareGpuDeltaReqInput
        assert prepare.validate_payload_hash is True
        assert not facade.manager.is_pause
        reply(facade, prepare, "PREPARED")
        await asyncio.sleep(0)
        assert sent.empty() and not task.done()
        reply(facade, prepare, "PREPARED", rank=1)
        apply = await sent.get()
        assert type(apply) is io.ApplyGpuDeltaReqInput
        assert facade.manager.is_pause
        reply(facade, apply, "APPLIED")
        await asyncio.sleep(0)
        assert sent.empty() and facade.manager.is_pause
        reply(facade, apply, "APPLIED", rank=1)
        resume = await sent.get()
        assert type(resume) is io.ResumeGpuDeltaReqInput
        assert facade.manager.is_pause and versions == []
        reply(facade, resume, "RESUMED")
        await asyncio.sleep(0)
        assert not task.done() and facade.manager.is_pause and versions == []
        reply(facade, resume, "RESUMED", rank=1)
        result = await task
        assert result["success"] and not facade.manager.is_pause
        assert versions == ["7"] and sent.empty()
        assert [r["state"] for r in result["applied_participants"]] == ["APPLIED"] * 2
        assert [r["state"] for r in result["participants"]] == ["RESUMED"] * 2
        assert setup_calls == ["loop", "workers"]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize(
    "failure,release_state,flush_cache,abort_all_requests,weight_version,pause_at",
    [
        (None, False, True, False, None, None),
        (None, True, False, True, "0", None),
        (None, False, False, True, "actor-42", None),
        (None, True, True, False, None, "entry"),
        (None, False, True, False, None, "prepare"),
        ("prepare", True, True, False, None, None),
        ("apply", True, True, False, None, None),
        ("resume", True, True, False, None, None),
        ("uncertain_apply", True, True, False, None, None),
        ("clear", True, True, False, None, None),
        ("release", True, True, False, None, None),
    ],
)
def test_update_preserves_receipts_options_and_failure_boundaries(
    control,
    publication,
    monkeypatch,
    failure,
    release_state,
    flush_cache,
    abort_all_requests,
    weight_version,
    pause_at,
):
    # The same negotiated input contract supports fresh-base and rolling updates.
    publication["validate_payload_hash"] = False
    if release_state:
        publication["base_version"] = 0
    now = [0.0]
    monkeypatch.setattr(tokenizer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    serving_version = "7" if weight_version is None else weight_version
    calls = []

    async def run():
        facade, _, versions = control
        facade.manager.is_pause = pause_at == "entry"
        cache_clears, aborted = [], []
        facade.manager.mm_processor = SimpleNamespace(
            clear_preprocess_cache=lambda: cache_clears.append("clear")
        )

        def abort(abort_all):
            assert abort_all and facade.manager.is_pause
            aborted.append("abort")

        facade.manager.abort_request = abort

        async def request(obj):
            calls.append(obj)
            if type(obj) is io.PrepareGpuDeltaReqInput:
                assert not facade.manager.is_pause_cond.locked()
                assert facade.manager.is_pause is (pause_at == "entry")
                assert obj == io.PrepareGpuDeltaReqInput(**publication)
                state, phase, duration = "PREPARING", "prepare", 2
            elif isinstance(obj, io.GetGpuDeltaStatusReqInput):
                if pause_at == "prepare":
                    facade.manager.is_pause = True
                state, phase, duration = "PREPARED", "status", 3
            elif isinstance(obj, io.ApplyGpuDeltaReqInput):
                assert facade.manager.is_pause
                assert obj.flush_cache is flush_cache
                assert obj.abort_all_requests is abort_all_requests
                assert aborted == (["abort"] if abort_all_requests else [])
                if failure == "uncertain_apply":
                    raise OSError("lost apply acknowledgment")
                state, phase, duration = "APPLIED", "apply", 7
            elif isinstance(obj, io.ResumeGpuDeltaReqInput):
                assert facade.manager.is_pause
                assert obj.keep_pause is (pause_at is not None)
                assert obj.weight_version == weight_version
                state, phase, duration = "RESUMED", "resume", 11
            elif isinstance(obj, io.AbortGpuDeltaReqInput):
                state, phase, duration = "ABORTED", "abort", 0
            else:
                assert facade.manager.is_pause is (pause_at is not None)
                assert versions == [serving_version]
                state, phase, duration = "CLEARED", "clear", 13
                if isinstance(obj, io.ReleaseGpuDeltaCacheReqInput):
                    phase, duration = "release", 17
                    assert obj.owner_rank_ids == ["r0", "r2"]
                    assert isinstance(calls[-2], io.ClearGpuDeltaStateReqInput)
                else:
                    assert isinstance(obj, io.ClearGpuDeltaStateReqInput)
            now[0] += duration
            return {
                "success": failure != phase,
                "message": "" if failure != phase else "injected failure",
                "participants": [
                    dict(
                        identity=identity,
                        session_id=publication["session_id"],
                        state=state,
                        version=7,
                        target_version=7,
                    )
                    for identity in publication["participants"]
                ],
            }

        facade._request = request
        result = await facade.request(
            io.UpdateWeightsFromGpuDeltaReqInput(
                **publication,
                release_state=release_state,
                flush_cache=flush_cache,
                abort_all_requests=abort_all_requests,
                weight_version=weight_version,
            )
        )
        assert result["success"] is (failure is None)
        assert cache_clears == (
            ["clear"]
            if flush_cache and failure in {None, "resume", "clear", "release"}
            else []
        )
        assert aborted == (
            ["abort"] if abort_all_requests and failure != "prepare" else []
        )
        if failure is None:
            assert versions == [serving_version]
            expected = [
                io.PrepareGpuDeltaReqInput,
                io.GetGpuDeltaStatusReqInput,
                io.ApplyGpuDeltaReqInput,
                io.ResumeGpuDeltaReqInput,
            ]
            if release_state:
                expected += [
                    io.ClearGpuDeltaStateReqInput,
                    io.ReleaseGpuDeltaCacheReqInput,
                ]
            assert [type(obj) for obj in calls] == expected
            assert facade.manager.is_pause is (pause_at is not None)
            assert all(rank["state"] == "RESUMED" for rank in result["participants"])
            assert all(
                rank["state"] == "APPLIED" for rank in result["applied_participants"]
            )
            assert result["timings"] == {
                "prepare_s": 5.0,
                "apply_s": 7.0,
                "resume_s": 11.0,
                "activation_s": 23.0,
            }
            assert now[0] == (53.0 if release_state else 23.0)
        else:
            assert any(isinstance(obj, io.AbortGpuDeltaReqInput) for obj in calls) is (
                failure == "prepare"
            )
            assert facade.manager.is_pause is (
                failure in {"apply", "resume", "uncertain_apply"}
            )
            if failure == "clear":
                assert not any(
                    isinstance(obj, io.ReleaseGpuDeltaCacheReqInput) for obj in calls
                )
            elif failure != "release":
                assert not any(
                    isinstance(obj, io.ClearGpuDeltaStateReqInput) for obj in calls
                )
            assert versions == (
                [serving_version] if failure in {"clear", "release"} else []
            )

    asyncio.run(run())


def test_public_request_rejects_multiple_tokenizer_workers_before_dispatch(
    control, publication, monkeypatch
):
    facade, sent, versions = control
    monkeypatch.setattr(
        tokenizer, "get_serving", lambda: SimpleNamespace(tokenizer_worker_num=2)
    )
    result = asyncio.run(
        facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
    )
    assert not result["success"] and "one Python tokenizer worker" in result["message"]
    assert sent == versions == [] and not facade.manager.is_pause


@pytest.mark.parametrize("phase", ["prepare", "apply"])
def test_cancel_only_aborts_before_apply_and_never_reopens_admission(
    control, publication, phase
):
    async def run():
        facade, _, versions = control
        reached = asyncio.Event()
        calls = []

        async def request(obj):
            calls.append(obj)
            if isinstance(obj, io.PrepareGpuDeltaReqInput):
                state, current = "PREPARED", "prepare"
            elif isinstance(obj, io.ApplyGpuDeltaReqInput):
                state, current = "APPLIED", "apply"
            else:
                assert isinstance(obj, io.AbortGpuDeltaReqInput)
                state, current = "ABORTED", "abort"
            if current == phase:
                reached.set()
                await asyncio.Future()
            return {"success": True, "participants": [{"state": state}]}

        facade._request = request
        task = asyncio.create_task(
            facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
        )
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert facade.manager.is_pause is (phase == "apply")
        assert versions == []
        assert any(isinstance(obj, io.AbortGpuDeltaReqInput) for obj in calls) is (
            phase == "prepare"
        )
        assert not any(isinstance(obj, io.ResumeGpuDeltaReqInput) for obj in calls)

    asyncio.run(asyncio.wait_for(run(), timeout=2))


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
