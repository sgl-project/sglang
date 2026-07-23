import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from sglang.srt.managers.io_struct import (
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock


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
    await asyncio.sleep(0)
    assert not pause_task.done(), "pause crossed a request still being admitted"

    admission_release.set()
    reader_lock = await admission_task

    with pytest.raises(RuntimeError, match="requests are active"):
        await pause_task
    assert tm.is_pause is False
    tm._async_dispatch_to_scheduler.assert_not_awaited()
    await reader_lock.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_failed_in_place_pause_reopens_admission():
    tm = _bare_tokenizer_manager()
    tm._async_dispatch_to_scheduler.side_effect = RuntimeError("dispatch failed")

    with pytest.raises(RuntimeError, match="dispatch failed"):
        await tm.pause_generation(PauseGenerationReqInput(mode="in_place"))

    assert tm.is_pause is False
    reader_lock = await asyncio.wait_for(tm._acquire_generation_reader(), timeout=1)
    await reader_lock.__aexit__(None, None, None)


def test_busy_release_returns_failure_instead_of_asserting():
    updater = object.__new__(SchedulerWeightUpdaterManager)
    updater.is_fully_idle = Mock(return_value=False)

    result = updater.release_memory_occupation(
        ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
    )

    assert result.success is False
    assert "idle scheduler" in result.message
