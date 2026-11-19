"""MQA logits chunking must distinguish cached memory from live allocations."""

from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.layers.attention import mqa_logits_utils as mqa
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GIB = 1 << 30


@pytest.fixture
def memory(monkeypatch):
    state = SimpleNamespace(free=0, reserved=0, allocated=0)
    monkeypatch.setattr(mqa, "mqa_logits_static_budget_bytes", lambda **_: 2 * GIB)
    monkeypatch.setattr(mqa, "mqa_logits_free_mem_fraction", lambda: 0.2)
    monkeypatch.setattr(mqa, "is_hip", lambda: False)
    monkeypatch.setattr(mqa, "is_xpu", lambda: False)
    with (
        mock.patch.object(
            mqa.torch.cuda, "mem_get_info", side_effect=lambda _: (state.free, 64 * GIB)
        ) as free,
        mock.patch.object(
            mqa.torch.cuda, "memory_reserved", side_effect=lambda _: state.reserved
        ) as reserved,
        mock.patch.object(
            mqa.torch.cuda, "memory_allocated", side_effect=lambda _: state.allocated
        ) as allocated,
        mock.patch.object(mqa.torch.cuda, "empty_cache") as empty_cache,
    ):
        state.queries = (free, reserved, allocated)
        yield state
        empty_cache.assert_not_called()


def budget():
    return mqa.mqa_logits_budget_bytes(device_index=1, allow_sync=True)


def test_allocator_cache_does_not_force_one_row_chunks(memory):
    # The driver has no free memory, but 8 GiB of cached allocator blocks are
    # not occupied by tensors. A CP8 prefill's 7392 x 16384 fp32 slab fits.
    memory.reserved = 60 * GIB
    memory.allocated = 52 * GIB
    value = budget()
    assert value == int(8 * GIB * 0.2)
    assert (
        mqa.mqa_logits_rows_per_chunk(
            num_rows=7392,
            row_bytes=mqa.mqa_logits_row_bytes(16384),
            budget_bytes=value,
        )
        is None
    )
    for query in memory.queries:
        query.assert_called_once_with(1)


def test_moving_free_memory_into_the_allocator_cache_preserves_budget(memory):
    memory.free = 8 * GIB
    memory.reserved = memory.allocated = 52 * GIB
    before = budget()
    memory.free = 0
    memory.reserved += 8 * GIB
    assert budget() == before


def test_live_allocations_still_reduce_the_budget(memory):
    memory.reserved = 60 * GIB
    memory.allocated = 52 * GIB
    before = budget()
    memory.allocated += 4 * GIB
    assert budget() == before // 2
    memory.allocated = memory.reserved
    assert budget() == 1
    assert (
        mqa.mqa_logits_rows_per_chunk(
            num_rows=7392, row_bytes=65536, budget_bytes=budget()
        )
        == 1
    )


def test_driver_free_and_cached_memory_share_the_safety_fraction(memory):
    memory.free = GIB
    memory.reserved = 8 * GIB
    memory.allocated = 4 * GIB
    assert budget() == GIB


def test_static_cap_still_limits_an_ample_allocator_cache(memory):
    memory.reserved = 32 * GIB
    assert budget() == 2 * GIB


def test_inconsistent_allocator_snapshot_does_not_subtract_driver_free(memory):
    memory.free = GIB
    memory.reserved = 4 * GIB
    memory.allocated = 5 * GIB
    assert budget() == int(GIB * 0.2)


def test_capture_budget_does_not_query_cuda_memory(memory):
    assert mqa.mqa_logits_budget_bytes(device_index=1, allow_sync=False) == 2 * GIB
    for query in memory.queries:
        query.assert_not_called()


def test_xpu_budget_does_not_query_cuda_memory(memory, monkeypatch):
    monkeypatch.setattr(mqa, "is_xpu", lambda: True)
    assert budget() == 2 * GIB
    for query in memory.queries:
        query.assert_not_called()


def test_rocm_logits_ceiling_still_applies(memory, monkeypatch):
    monkeypatch.setattr(mqa, "mqa_logits_static_budget_bytes", lambda **_: 4 * GIB)
    monkeypatch.setattr(mqa, "is_hip", lambda: True)
    memory.reserved = 32 * GIB
    assert budget() == mqa.MQA_LOGITS_MAX_BYTES_ROCM
