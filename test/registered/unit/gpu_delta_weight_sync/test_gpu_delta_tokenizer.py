"""Admission remains paused until the dedicated delta acknowledgment succeeds."""

import asyncio
from types import SimpleNamespace

import pytest

from sglang.srt.weight_sync import gpu_delta_io as io
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
