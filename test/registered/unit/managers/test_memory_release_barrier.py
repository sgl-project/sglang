import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from sglang.srt.managers.io_struct import (
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
)
from sglang.srt.managers.multi_tokenizer_mixin import TokenizerWorker
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bare_tokenizer_manager():
    tm = object.__new__(TokenizerManager)
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = RWLock()
    tm._async_dispatch_to_scheduler = AsyncMock()
    return tm


@pytest.mark.asyncio
async def test_in_place_pause_refuses_active_request_without_pausing():
    tm = _bare_tokenizer_manager()
    await tm.model_update_lock.acquire_reader()

    with pytest.raises(RuntimeError, match="requests are active"):
        await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))

    assert tm.is_pause is False
    tm._async_dispatch_to_scheduler.assert_not_awaited()
    await tm.model_update_lock.release_reader()


@pytest.mark.asyncio
async def test_in_place_refusal_preserves_existing_pause():
    tm = _bare_tokenizer_manager()
    tm.is_pause = True
    await tm.model_update_lock.acquire_reader()

    with pytest.raises(RuntimeError, match="requests are active"):
        await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))

    assert tm.is_pause is True
    await tm.model_update_lock.release_reader()


@pytest.mark.asyncio
async def test_request_admission_cannot_be_liminal_during_pause():
    tm = _bare_tokenizer_manager()
    original_acquire_reader = tm.model_update_lock.acquire_reader
    admission_entered = asyncio.Event()
    admission_release = asyncio.Event()

    async def delayed_acquire_reader():
        admission_entered.set()
        await admission_release.wait()
        await original_acquire_reader()

    tm.model_update_lock.acquire_reader = delayed_acquire_reader
    admission_task = asyncio.create_task(tm._acquire_generation_reader())
    await admission_entered.wait()
    pause_task = asyncio.create_task(
        tm.pause_generation(PauseGenerationReqInput(mode="in_place"))
    )
    await asyncio.wait_for(pause_task, timeout=1)
    tm._async_dispatch_to_scheduler.assert_awaited_once()

    admission_release.set()
    await asyncio.sleep(0)
    assert not admission_task.done(), "request crossed the closed pause gate"

    async with tm.is_pause_cond:
        tm.is_pause = False
        tm.is_pause_cond.notify_all()
    reader_lock = await admission_task
    await reader_lock.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_waiting_writer_does_not_block_abort_from_closing_gate():
    tm = _bare_tokenizer_manager()
    abort_called = asyncio.Event()
    tm.abort_request = Mock(side_effect=lambda **_: abort_called.set())
    await tm.model_update_lock.acquire_reader()

    writer_release = asyncio.Event()

    async def hold_writer():
        async with tm.model_update_lock.writer_lock:
            await writer_release.wait()

    writer_task = asyncio.create_task(hold_writer())
    while tm.model_update_lock._waiting_writers == 0:
        await asyncio.sleep(0)

    admission_task = asyncio.create_task(tm._acquire_generation_reader())
    pause_task = asyncio.create_task(
        tm.pause_generation(PauseGenerationReqInput(mode="abort"))
    )
    await asyncio.wait_for(abort_called.wait(), timeout=1)
    tm.abort_request.assert_called_with(abort_all=True)

    pause_task.cancel()
    admission_task.cancel()
    await asyncio.gather(pause_task, admission_task, return_exceptions=True)
    await tm.model_update_lock.release_reader()
    writer_release.set()
    await writer_task


@pytest.mark.asyncio
async def test_cancelled_admission_releases_acquired_reader():
    tm = _bare_tokenizer_manager()
    acquired = asyncio.Event()
    proceed = asyncio.Event()
    original_acquire_reader = tm.model_update_lock.acquire_reader

    async def observed_acquire_reader():
        await original_acquire_reader()
        acquired.set()
        await proceed.wait()

    tm.model_update_lock.acquire_reader = observed_acquire_reader
    admission_task = asyncio.create_task(tm._acquire_generation_reader())
    await acquired.wait()
    async with tm.is_pause_cond:
        proceed.set()
        await asyncio.sleep(0)
        admission_task.cancel()
        await asyncio.gather(admission_task, return_exceptions=True)
        assert await tm.model_update_lock.is_locked() is False


@pytest.mark.asyncio
async def test_failed_in_place_pause_reopens_admission():
    tm = _bare_tokenizer_manager()
    tm._async_dispatch_to_scheduler.side_effect = RuntimeError("dispatch failed")

    with pytest.raises(RuntimeError, match="dispatch failed"):
        await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))

    assert tm.is_pause is False
    reader_lock = await asyncio.wait_for(tm._acquire_generation_reader(), timeout=1)
    await reader_lock.__aexit__(None, None, None)

    tm.is_pause = True
    with pytest.raises(RuntimeError, match="dispatch failed"):
        await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))
    assert tm.is_pause is True


@pytest.mark.asyncio
async def test_multi_tokenizer_rejects_in_place_pause_before_dispatch():
    worker = object.__new__(TokenizerWorker)
    worker._dispatch_to_scheduler = Mock()

    with pytest.raises(RuntimeError, match="multiple tokenizer workers"):
        await asyncio.wait_for(
            worker.pause_generation(PauseGenerationReqInput(mode="in_place")),
            timeout=1,
        )

    worker._dispatch_to_scheduler.assert_not_called()


@pytest.mark.asyncio
async def test_release_surfaces_fanout_failure():
    manager = object.__new__(TokenizerManager)
    manager.auto_create_handle_loop = Mock()
    manager.release_memory_occupation_communicator = AsyncMock(
        return_value=[
            ReleaseMemoryOccupationReqOutput(),
            ReleaseMemoryOccupationReqOutput(success=False, message="rank busy"),
        ]
    )

    with pytest.raises(RuntimeError, match="rank busy"):
        await TokenizerControlMixin.release_memory_occupation(
            manager, ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        )


def test_busy_release_returns_failure_instead_of_asserting():
    updater = object.__new__(SchedulerWeightUpdaterManager)
    updater.is_fully_idle = Mock(return_value=False)

    result = updater.release_memory_occupation(
        ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
    )

    assert result.success is False
    assert "idle scheduler" in result.message
