from __future__ import annotations

import pytest
import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_allocator(kind, need_sort, device, *, enable_tracking=True):
    unit = 4 if kind == "paged" else 1
    if kind == "mamba":
        allocator = MambaSlotAllocator(8, device)
    elif kind == "paged":
        allocator = PagedTokenToKVPoolAllocator(
            8 * unit, unit, torch.float32, device, None, need_sort
        )
    else:
        allocator = TokenToKVPoolAllocator(8, torch.float32, device, None, need_sort)
    if enable_tracking:
        allocator.reset_reuse_tracking()
    return allocator, unit


def free_slots(allocator, slots):
    if isinstance(allocator, PagedTokenToKVPoolAllocator):
        allocator.free_segment(slots, start_pos=0)
    else:
        allocator.free(slots)


@pytest.mark.parametrize("kind", ["token", "paged"])
@pytest.mark.parametrize("need_sort", [False, True])
@pytest.mark.parametrize("with_pending", [False, True])
def test_refill_respects_sort_setting_and_pending_boundary(
    kind, need_sort, with_pending
):
    allocator, unit = make_allocator(kind, need_sort, "cpu")
    slots = allocator.alloc(8 * unit).reshape(8, unit)
    ready = slots[[5, 2, 7]].flatten()
    free_slots(allocator, ready)
    allocator.reset_reuse_tracking()
    pending = slots[[1, 0]].flatten()
    if with_pending:
        free_slots(allocator, pending)

    restored = allocator.alloc(3 * unit)

    expected = slots[[2, 5, 7]].flatten() if need_sort else ready
    assert torch.equal(restored, expected)
    assert not allocator.needs_restore_fence
    if with_pending:
        assert torch.equal(allocator.alloc(2 * unit), pending)
        assert allocator.needs_restore_fence
    assert allocator.available_size() == 0


@pytest.mark.parametrize("kind", ["token", "paged", "mamba"])
@pytest.mark.parametrize("need_sort", [False, True])
@pytest.mark.parametrize("next_forward", [False, True])
def test_launch_reset_makes_older_frees_ready(kind, need_sort, next_forward):
    allocator, unit = make_allocator(kind, need_sort, "cpu")
    first = allocator.alloc(2 * unit)
    second = allocator.alloc(2 * unit)
    free_slots(allocator, first)
    if next_forward:
        allocator.reset_reuse_tracking()
    free_slots(allocator, second)
    fresh = allocator.alloc(4 * unit)
    assert set(fresh.tolist()).isdisjoint(torch.cat((first, second)).tolist())
    assert not allocator.needs_restore_fence
    assert allocator.available_size() == 4 * unit
    assert torch.equal(allocator.alloc(2 * unit), first)
    assert allocator.needs_restore_fence == (not next_forward)
    assert torch.equal(allocator.alloc(2 * unit), second)
    assert allocator.needs_restore_fence
    assert allocator.available_size() == 0


@pytest.mark.parametrize("kind", ["token", "paged", "mamba"])
@pytest.mark.parametrize("need_sort", [False, True])
def test_pending_reuse_stays_flagged_across_later_allocations(kind, need_sort):
    allocator, unit = make_allocator(kind, need_sort, "cpu")
    assert allocator.alloc(9 * unit) is None
    assert not allocator.needs_restore_fence
    first, second = allocator.alloc(8 * unit).split(4 * unit)
    assert not allocator.needs_restore_fence
    free_slots(allocator, first)
    free_slots(allocator, second)
    assert torch.equal(allocator.alloc(5 * unit), torch.cat((first, second[:unit])))
    assert allocator.needs_restore_fence
    assert allocator.alloc(4 * unit) is None
    assert allocator.needs_restore_fence
    assert torch.equal(allocator.alloc(3 * unit), second[unit:])
    # Consuming the entire free list must not lose the fence requirement.
    assert allocator.alloc(0).numel() == 0
    assert allocator.needs_restore_fence
    allocator.reset_reuse_tracking()
    assert not allocator.needs_restore_fence


@pytest.mark.parametrize("kind", ["token", "paged", "mamba"])
@pytest.mark.parametrize("enabled", [False, True])
def test_clear_resets_reuse_tracking_and_preserves_enablement(kind, enabled):
    allocator, unit = make_allocator(kind, False, "cpu", enable_tracking=enabled)
    free_slots(allocator, allocator.alloc(8 * unit))
    allocator.alloc(unit)
    assert allocator.needs_restore_fence == enabled
    allocator.clear()
    assert not allocator.needs_restore_fence
    assert allocator.available_size() == 8 * unit
    fresh = allocator.alloc(8 * unit)
    assert not allocator.needs_restore_fence
    free_slots(allocator, fresh)
    allocator.alloc(unit)
    assert allocator.needs_restore_fence == enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("kind", ["token", "paged", "mamba"])
@pytest.mark.parametrize("need_sort", [False, True])
def test_fresh_dma_overlaps_forward_and_recycled_pages_require_fence(kind, need_sort):
    allocator, unit = make_allocator(kind, need_sort, "cuda")
    forward = torch.cuda.Stream()
    schedule = torch.cuda.Stream()
    dma = torch.cuda.Stream()
    # Warm lazy kernels before testing stream independence.
    with torch.cuda.stream(schedule):
        slots = allocator.alloc(2 * unit)
        free_slots(allocator, slots)
        allocator.alloc(8 * unit)
    schedule.synchronize()
    allocator.clear()
    old = allocator.alloc(2 * unit)
    old_cpu = old.cpu()
    values = torch.zeros(9 * unit, device="cuda")
    values.index_fill_(0, old, 0)
    payload = torch.full((2 * unit,), 33.0, pin_memory=True)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    with torch.cuda.stream(forward):
        torch.cuda._sleep(400_000_000)
        values.index_fill_(0, old, 7)
        previous_done = forward.record_event()

    with torch.cuda.stream(schedule):
        free_slots(allocator, old)
        fresh = allocator.alloc(2 * unit)
        fresh_cpu = fresh.cpu()
        assert set(old_cpu.tolist()).isdisjoint(fresh_cpu.tolist())
        assert not allocator.needs_restore_fence
        ready = schedule.record_event()
        with torch.cuda.stream(dma):
            dma.wait_event(ready)
            start = int(fresh_cpu[0])
            values[start : start + len(payload)].copy_(payload, non_blocking=True)
        dma.synchronize()
        assert not previous_done.query(), "DMA waited for the previous forward"

        assert allocator.alloc(4 * unit) is not None
        assert not allocator.needs_restore_fence
        reused = allocator.alloc(2 * unit)
        assert torch.equal(reused.cpu().sort().values, old_cpu.sort().values)
        assert allocator.needs_restore_fence
        schedule.synchronize()
        assert not previous_done.query(), "allocation waited for the previous forward"

    # The controller must fence before submitting DMA to the flagged allocation.
    previous_done.synchronize()
    assert torch.all(values[reused].cpu() == 7)
    assert torch.all(values[fresh].cpu() == 33)


def test_grouped_frees_preserve_pending_suffix_and_own_index_storage():
    allocator, unit = make_allocator("paged", True, "cpu")
    ids = allocator.alloc(2 * unit)
    allocator.free_group_begin()
    allocator.free_segment(ids, start_pos=0)
    ids.fill_(0)
    allocator.free_group_end()
    assert allocator.available_size() == 8 * unit
    assert allocator.get_all_free_pages().sort().values.tolist() == list(range(1, 9))
    assert allocator.alloc(6 * unit) is not None
    assert not allocator.needs_restore_fence
    assert allocator.alloc(2 * unit).tolist() == list(range(4, 12))
    assert allocator.needs_restore_fence


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("next_forward", [False, True])
def test_paged_extend_and_decode_account_for_pending_reuse(decode, next_forward):
    allocator, unit = make_allocator("paged", True, "cuda")
    old = allocator.alloc(8 * unit)
    allocator.free_segment(old, start_pos=0)
    if next_forward:
        allocator.reset_reuse_tracking()
    prefix_cpu = torch.tensor([0], dtype=torch.int64)
    seq_cpu = torch.tensor([1 if decode else 4], dtype=torch.int64)
    prefix, seq = prefix_cpu.cuda(), seq_cpu.cuda()
    last = torch.tensor([-1], device="cuda", dtype=torch.int64)
    if decode:
        out = allocator.alloc_decode(seq, seq_cpu, last)
    else:
        out = allocator.alloc_extend(prefix, prefix_cpu, seq, seq_cpu, last, 4)
    assert out.cpu().tolist() == ([4] if decode else [4, 5, 6, 7])
    assert allocator.needs_restore_fence == (not next_forward)
    assert allocator.alloc(7 * unit) is not None
    assert allocator.needs_restore_fence == (not next_forward)


def test_mamba_group_flags_pending_reuse_at_reservation_time():
    allocator, _ = make_allocator("mamba", False, "cpu")
    pending = allocator.alloc(4)
    allocator.free(pending)
    allocator.alloc_group_begin(8)
    assert allocator.needs_restore_fence
    assert allocator.alloc(1).tolist() == [5]
    assert allocator.alloc(1).tolist() == [6]
    allocator.alloc_group_end()
    assert allocator.available_size() == 6
    assert allocator.alloc(6).tolist() == [7, 8, 1, 2, 3, 4]
    assert allocator.needs_restore_fence


@pytest.mark.parametrize("kind", ["token", "paged", "mamba"])
@pytest.mark.parametrize("need_sort", [False, True])
def test_allocation_rollback_conservatively_requires_fence(kind, need_sort):
    allocator, unit = make_allocator(kind, need_sort, "cpu")
    old = allocator.alloc(2 * unit)
    free_slots(allocator, old)
    destination = allocator.alloc(6 * unit)
    assert not allocator.needs_restore_fence
    free_slots(allocator, destination)
    retry = allocator.alloc(8 * unit)
    assert set(retry.tolist()) == set(torch.cat((old, destination)).tolist())
    assert allocator.needs_restore_fence


def test_unused_mamba_reservation_is_conservatively_pending():
    allocator, _ = make_allocator("mamba", False, "cpu")
    pending = allocator.alloc(2)
    allocator.free(pending)
    allocator.alloc_group_begin(6)
    assert not allocator.needs_restore_fence
    assert allocator.alloc(1).tolist() == [3]
    assert not allocator.needs_restore_fence
    allocator.alloc_group_end()
    assert allocator.available_size() == 7
    assert not allocator.needs_restore_fence
    assert allocator.alloc(7).tolist() == [1, 2, 4, 5, 6, 7, 8]
    assert allocator.needs_restore_fence
