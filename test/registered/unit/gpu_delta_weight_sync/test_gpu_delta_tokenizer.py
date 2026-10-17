"""Unified updates retain preparation, mutation and resume failure boundaries."""

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


def test_update_waits_for_every_rank_before_apply_and_resume(control, publication):
    async def run():
        facade, _, versions = control
        sent = asyncio.Queue()
        facade.communicator._send = sent.put_nowait
        facade.communicator.set_fan_out(2)
        task = asyncio.create_task(
            facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
        )
        prepare = await sent.get()
        assert type(prepare) is io.PrepareGpuDeltaReqInput
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
