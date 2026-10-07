"""Admission remains paused until the dedicated delta acknowledgment succeeds."""

import asyncio
from types import SimpleNamespace

import pytest

from sglang.srt.weight_sync.gpu_delta import io as io
from sglang.srt.weight_sync.gpu_delta import tokenizer as tokenizer
from sglang.srt.weight_sync.gpu_delta.session import GpuDeltaCommunicator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setattr(
        tokenizer, "get_serving", lambda: SimpleNamespace(tokenizer_worker_num=1)
    )
    sent, versions = [], []
    manager = SimpleNamespace(
        is_pause=False,
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


def reply(control, obj, state, success=True):
    control.communicator.handle_recv(
        io.DeltaWeightsReqOutput(
            rid=obj.rid,
            success=success,
            message="" if success else "rejected",
            participant={
                "identity": {"engine_id": "engine-0", "rank_id": "rank-0"},
                "session_id": obj.session_id,
                "state": state,
                "target_version": 1,
            },
        )
    )


@pytest.mark.parametrize("resume_success", [True, False])
def test_only_update_gates_admission_and_successful_resume_releases_it(
    control, resume_success
):
    async def run():
        facade, sent, versions = control
        status = io.GetWeightsDeltaStatusReqInput(session_id="publication-1")
        task = asyncio.create_task(facade.request(status))
        await asyncio.sleep(0)
        assert not facade.manager.is_pause
        reply(facade, status, "PREPARED")
        assert (await task)["success"]
        update = io.UpdateWeightsFromDeltaReqInput(session_id="publication-1")
        task = asyncio.create_task(facade.request(update))
        await asyncio.sleep(0)
        assert sent[-1] is update and facade.manager.is_pause
        reply(facade, update, "APPLIED")
        assert (await task)["success"] and facade.manager.is_pause
        resume = io.ResumeWeightsFromDeltaReqInput(session_id="publication-1")
        task = asyncio.create_task(facade.request(resume))
        await asyncio.sleep(0)
        reply(
            facade,
            resume,
            "RESUMED" if resume_success else "RESUMING",
            success=resume_success,
        )
        assert (await task)["success"] is resume_success
        assert facade.manager.is_pause is not resume_success
        assert versions == (["1"] if resume_success else [])

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["send", "cancel"])
def test_uncertain_update_keeps_admission_paused(control, failure):
    async def run():
        facade, _, _ = control
        if failure == "send":

            def fail(_):
                raise OSError("transport unavailable")

            facade.communicator._send = fail
        update = io.UpdateWeightsFromDeltaReqInput(session_id="publication-1")
        task = asyncio.create_task(facade.request(update))
        await asyncio.sleep(0)
        if failure == "cancel":
            task.cancel()
        with pytest.raises(OSError if failure == "send" else asyncio.CancelledError):
            await task
        assert facade.manager.is_pause

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", [None, "prepare", "apply", "resume", "uncertain_apply"]
)
@pytest.mark.parametrize("release_state", [True, False])
def test_standalone_load_uses_fresh_participants_and_clears_only_after_resume(
    control, tmp_path, failure, release_state
):
    import hashlib

    import orjson

    manifest = {
        "base_version": 0,
        "target_version": 7,
        "stream_id": "saved-training-run",
        "plan_digest": "b" * 64,
        "codec": "lz4",
    }
    path = tmp_path / "manifest.json"
    content = orjson.dumps(manifest)
    path.write_bytes(content)
    identities = [
        dict(
            engine_id="e0",
            rank_id=f"r{i}",
            hostname=f"host-{i // 2}",
            host_cache_id=f"cache-{i // 2}",
        )
        for i in range(3)
    ]
    calls = []

    async def run():
        facade, _, versions = control

        async def request(obj):
            calls.append(obj)
            if isinstance(obj, io.GetWeightsDeltaInfoReqInput):
                state, phase = "IDLE", "describe"
            elif isinstance(obj, io.PrepareWeightsFromDeltaReqInput):
                state, phase = "PREPARING", "prepare"
                assert obj.participants == identities
                assert obj.manifest_sha256 == hashlib.sha256(content).hexdigest()
                assert obj.base_version == 0 and obj.target_version == 7
                assert obj.stream_id == manifest["stream_id"]
                assert obj.plan_digest == manifest["plan_digest"]
            elif isinstance(obj, io.GetWeightsDeltaStatusReqInput):
                state, phase = "PREPARED", "status"
            elif isinstance(obj, io.UpdateWeightsFromDeltaReqInput):
                assert facade.manager.is_pause
                if failure == "uncertain_apply":
                    raise OSError("lost apply acknowledgment")
                state, phase = "APPLIED", "apply"
            elif isinstance(obj, io.ResumeWeightsFromDeltaReqInput):
                assert facade.manager.is_pause
                state, phase = "RESUMED", "resume"
            elif isinstance(obj, io.AbortWeightsFromDeltaReqInput):
                state, phase = "ABORTED", "abort"
            else:
                assert not facade.manager.is_pause
                assert versions == ["7"]
                state, phase = "CLEARED", "clear"
                if isinstance(obj, io.ReleaseWeightsDeltaCacheReqInput):
                    assert obj.owner_rank_ids == ["r0", "r2"]
                    assert isinstance(calls[-2], io.ClearWeightsDeltaStateReqInput)
            return {
                "success": failure != phase,
                "message": "" if failure != phase else "injected failure",
                "participants": [
                    dict(
                        identity=identity,
                        state=state,
                        version=0 if phase == "describe" else 7,
                        target_version=7,
                    )
                    for identity in identities
                ],
            }

        facade._request = request
        # Omitting the option preserves the standalone path-only API.
        options = {} if release_state else {"release_state": False}
        result = await facade.request(
            io.LoadWeightsFromDeltaReqInput(manifest_path=str(path), **options)
        )
        assert result["success"] is (failure is None)
        assert path.read_bytes() == content
        if failure is None:
            expected = [
                io.GetWeightsDeltaInfoReqInput,
                io.PrepareWeightsFromDeltaReqInput,
                io.GetWeightsDeltaStatusReqInput,
                io.UpdateWeightsFromDeltaReqInput,
                io.ResumeWeightsFromDeltaReqInput,
            ]
            if release_state:
                expected += [
                    io.ClearWeightsDeltaStateReqInput,
                    io.ReleaseWeightsDeltaCacheReqInput,
                ]
            assert [type(obj) for obj in calls] == expected
            assert not facade.manager.is_pause
            assert all(rank["version"] == 7 for rank in result["participants"])
            assert all(
                rank["state"] == ("CLEARED" if release_state else "RESUMED")
                for rank in result["participants"]
            )
        else:
            assert not any(
                isinstance(obj, io.ClearWeightsDeltaStateReqInput) for obj in calls
            )
            assert facade.manager.is_pause is (failure != "prepare")
            assert any(
                isinstance(obj, io.AbortWeightsFromDeltaReqInput) for obj in calls
            ) is (failure == "prepare")

    asyncio.run(run())


@pytest.mark.parametrize("state,version", [("RESUMED", 7), ("POISONED", 0)])
def test_standalone_load_never_replays_applied_or_ambiguous_base(
    control, tmp_path, state, version
):
    import orjson

    path = tmp_path / "manifest.json"
    path.write_bytes(orjson.dumps({"base_version": 0}))

    async def run():
        facade, _, _ = control
        calls = []

        async def request(obj):
            calls.append(obj)
            return {
                "success": True,
                "participants": [{"state": state, "version": version}],
            }

        facade._request = request
        result = await facade.request(
            io.LoadWeightsFromDeltaReqInput(manifest_path=str(path))
        )
        assert not result["success"] and "freshly loaded base" in result["message"]
        assert len(calls) == 1

    asyncio.run(run())


def test_clear_does_not_remove_any_cache_until_all_ranks_closed(control):
    async def run():
        facade, _, _ = control
        calls = []

        async def request(obj):
            calls.append(obj)
            return {
                "success": False,
                "message": "rank release failed",
                "participants": [],
            }

        facade._request = request
        result = await facade.request(io.ClearWeightsDeltaStateReqInput())
        assert not result["success"]
        assert [type(obj) for obj in calls] == [io.ClearWeightsDeltaStateReqInput]

    asyncio.run(run())
