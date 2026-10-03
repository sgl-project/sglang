"""Admission remains paused until the dedicated delta acknowledgment succeeds."""

import asyncio
from types import SimpleNamespace

import pytest
from sglang.srt.managers import io_struct as io
from sglang.srt.weight_sync import gpu_delta_tokenizer as tokenizer
from sglang.srt.weight_sync.gpu_delta_session import GpuDeltaCommunicator
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
    control.session_id = "publication-1"
    control.participants = [{"engine_id": "engine-0", "rank_id": "rank-0"}]
    control.communicator = GpuDeltaCommunicator(control._send, 1)
    return control, sent, versions


def reply(control, obj, state, success=True):
    control.communicator.handle_recv(
        io.DeltaWeightsReqOutput(
            rid=obj.rid,
            success=success,
            message="" if success else "rejected",
            participant={
                "identity": control.participants[0],
                "session_id": obj.session_id,
                "state": state,
                "target_version": 1,
            },
        )
    )


def test_only_update_gates_admission_and_only_successful_resume_releases_it(control):
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

        resume = io.ResumeWeightsFromDeltaReqInput(
            session_id="publication-1", receipts=[]
        )
        task = asyncio.create_task(facade.request(resume))
        await asyncio.sleep(0)
        reply(facade, resume, "APPLIED", success=False)
        assert not (await task)["success"]
        assert facade.manager.is_pause and facade.session_id == "publication-1"
        assert versions == []

        resume = io.ResumeWeightsFromDeltaReqInput(
            session_id="publication-1", receipts=[{"state": "APPLIED"}]
        )
        task = asyncio.create_task(facade.request(resume))
        await asyncio.sleep(0)
        assert facade.manager.is_pause and not task.done()
        reply(facade, resume, "RESUMED")
        assert (await task)["success"]
        assert not facade.manager.is_pause and facade.session_id is None
        assert versions == ["1"]

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["send", "cancel"])
def test_uncertain_update_keeps_admission_paused(control, failure):
    async def run():
        facade, _, _ = control
        if failure == "send":

            def fail(_):
                raise OSError("transport unavailable")

            facade.manager._dispatch_to_scheduler = fail
        update = io.UpdateWeightsFromDeltaReqInput(session_id="publication-1")
        task = asyncio.create_task(facade.request(update))
        await asyncio.sleep(0)
        if failure == "cancel":
            task.cancel()
        with pytest.raises(OSError if failure == "send" else asyncio.CancelledError):
            await task
        assert facade.manager.is_pause and facade.session_id == "publication-1"

    asyncio.run(run())


def test_prepare_ownership_survives_failed_send(control):
    async def run():
        facade, _, _ = control
        facade.session_id = None

        def fail(_):
            raise OSError("transport unavailable")

        facade.manager._dispatch_to_scheduler = fail
        prepare = io.PrepareWeightsFromDeltaReqInput(
            session_id="publication-1",
            engine_id="engine-0",
            manifest_path="/manifest",
            manifest_sha256="a" * 64,
            stream_id="run",
            base_version=0,
            target_version=1,
            plan_digest="b" * 64,
            participants=facade.participants,
            host_tensor_names={"host": []},
        )
        with pytest.raises(OSError):
            await facade.request(prepare)
        assert facade.session_id == "publication-1" and not facade.manager.is_pause
        with pytest.raises(tokenizer.GpuDeltaConflict, match="another delta session"):
            await facade.request(prepare)

    asyncio.run(run())
