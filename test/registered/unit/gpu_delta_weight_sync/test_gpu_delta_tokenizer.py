"""Resume admission and pause-state regressions at the real tokenizer boundary."""

import asyncio
from types import SimpleNamespace

import pytest

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.weight_sync.gpu_delta_tokenizer import (
    GpuDeltaConflict,
    GpuDeltaTokenizerControl,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def make_manager():
    manager = SimpleNamespace(is_pause=True, is_pause_cond=asyncio.Condition())
    manager.gpu_delta = GpuDeltaTokenizerControl.__new__(GpuDeltaTokenizerControl)
    manager.gpu_delta.manager = manager
    manager.gpu_delta.session_id = None
    return manager


def ordinary_continue():
    return SimpleNamespace(delta_session_id=None, delta_commit_receipts=None)


def test_queued_ordinary_resume_rechecks_delta_lease_under_pause_condition():
    async def run():
        manager = make_manager()
        dispatched = []

        async def dispatch(request):
            dispatched.append(request)

        manager._async_dispatch_to_scheduler = dispatch
        async with manager.is_pause_cond:
            task = asyncio.create_task(
                TokenizerManager.continue_generation(manager, ordinary_continue())
            )
            await asyncio.sleep(0)
            assert not task.done()
            # Prepare acquires ownership while the ordinary resume is queued.
            prepare = type(
                "PrepareWeightsFromDeltaReqInput",
                (),
                {"session_id": "publication-1"},
            )()
            manager.gpu_delta.guard_dispatch(prepare)

        with pytest.raises(GpuDeltaConflict, match="global commit certificate"):
            await task
        assert manager.is_pause
        assert manager.gpu_delta.session_id == "publication-1"
        assert dispatched == []

    asyncio.run(run())


def test_resume_unpauses_only_after_successful_dispatch_or_delta_ack():
    async def run():
        manager = make_manager()
        dispatched = []
        fail_dispatch = True

        async def dispatch(request):
            assert manager.is_pause
            dispatched.append(request)
            if fail_dispatch:
                raise OSError("transport unavailable")

        manager._async_dispatch_to_scheduler = dispatch
        with pytest.raises(OSError, match="transport unavailable"):
            await TokenizerManager.continue_generation(manager, ordinary_continue())
        assert manager.is_pause

        fail_dispatch = False
        assert (
            await TokenizerManager.continue_generation(manager, ordinary_continue())
            is None
        )
        assert not manager.is_pause
        assert len(dispatched) == 2

        manager.is_pause = True
        manager.gpu_delta.session_id = "publication-1"
        requested = asyncio.Event()
        acknowledgement = asyncio.get_running_loop().create_future()
        receipts = [{"session_id": "publication-1", "state": "COMMITTED"}]
        versions = []

        async def delta_request(request):
            assert request.session_id == "publication-1"
            assert request.receipts == receipts
            requested.set()
            return await acknowledgement

        def update_version(version):
            assert manager.is_pause
            versions.append(version)

        manager.gpu_delta.request = delta_request
        manager._update_weight_version_if_provided = update_version
        task = asyncio.create_task(
            TokenizerManager.continue_generation(
                manager,
                SimpleNamespace(
                    delta_session_id="publication-1", delta_commit_receipts=receipts
                ),
            )
        )
        await asyncio.wait_for(requested.wait(), timeout=1)
        assert manager.is_pause
        assert manager.gpu_delta.session_id == "publication-1"
        assert not task.done()
        assert versions == []
        assert len(dispatched) == 2

        participants = [
            {
                "identity": {"engine_id": "engine-0", "rank_id": "rank-0"},
                "session_id": "publication-1",
                "state": "RESUMED",
                "target_version": 7,
            }
        ]
        acknowledgement.set_result(
            {"success": True, "message": "", "participants": participants}
        )
        assert await task == {"success": True, "participants": participants}
        assert not manager.is_pause
        assert manager.gpu_delta.session_id is None
        assert versions == ["7"]
        assert len(dispatched) == 2

    asyncio.run(run())
