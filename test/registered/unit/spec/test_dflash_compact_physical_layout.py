from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner_components import spec_aux_hidden_state
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _compact_dflash_fixed_bytes,
    _compact_dflash_linear_budget,
)
from sglang.srt.speculative.dflash_compact_physical_layout import (
    CompactDFlashPhysicalLayout,
)
from sglang.srt.speculative.dflash_draft_content_allocator import (
    DFlashDraftContentAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _layout(*, owners=2, window=2048, block=8, page=1, content=0):
    return CompactDFlashPhysicalLayout.build(
        owner_count=owners,
        window_size=window,
        block_size=block,
        page_size=page,
        content_tokens=content,
    )


def _enable_compact_budget(monkeypatch):
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_spec",
        lambda: SimpleNamespace(speculative_dflash_compact_cache=True),
    )


def test_tp2_geometry_and_fixed_bytes(monkeypatch):
    value = _layout()
    assert value.guard_rows == 16
    assert value.scratch_rows == 16
    assert value.owner_span == 2080
    assert value.physical_tokens == 4160

    _enable_compact_budget(monkeypatch)
    assert _compact_dflash_linear_budget(10_240) == 0
    assert (
        _compact_dflash_fixed_bytes(
            10_240,
            owner_count=2,
            window_size=2048,
            block_size=8,
            page_size=1,
        )
        == 42_608_640
    )


def test_fixed_bytes_include_one_sentinel_page(monkeypatch):
    _enable_compact_budget(monkeypatch)
    layout = _layout(owners=3, window=2048, block=8, page=1)
    assert (
        _compact_dflash_fixed_bytes(
            10_240,
            owner_count=3,
            window_size=2048,
            block_size=8,
            page_size=1,
        )
        == (layout.physical_tokens + 1) * 10_240
    )


def test_content_region_is_page_aligned_disjoint_and_budgeted(monkeypatch):
    layout = _layout(owners=2, window=32, block=8, page=4, content=13)
    assert layout.content_start == layout.request_tokens
    assert layout.content_tokens == 16
    assert layout.content_end == layout.physical_tokens

    request_rows = torch.cat(
        (
            layout.committed_locs(torch.ones(32), torch.arange(32)),
            layout.scratch_locs(torch.tensor([1]), 8).reshape(-1),
        )
    )
    content_rows = torch.arange(layout.content_start, layout.content_end)
    assert not bool(torch.isin(request_rows, content_rows).any())
    layout.assert_content_locs(content_rows)
    with pytest.raises(RuntimeError, match="content loc OOB"):
        layout.assert_content_locs(request_rows[:1])

    _enable_compact_budget(monkeypatch)
    assert (
        _compact_dflash_fixed_bytes(
            32,
            owner_count=2,
            window_size=32,
            block_size=8,
            page_size=1,
            content_tokens=13,
        )
        == (_layout(owners=2, window=32, block=8, content=13).physical_tokens + 1) * 32
    )


def test_content_allocator_exhaustion_free_double_free_and_clear():
    allocator = DFlashDraftContentAllocator(start=100, size=5, device="cpu")
    first = allocator.alloc(3)
    second = allocator.alloc(2)
    assert first.tolist() == [100, 101, 102]
    assert second.tolist() == [103, 104]
    assert allocator.available_size() == 0
    assert allocator.alloc(1) is None
    allocator.assert_allocated(torch.tensor([100, 104]))

    allocator.free(torch.tensor([101, 103]))
    assert allocator.available_size() == 2
    with pytest.raises(RuntimeError, match="freed twice"):
        allocator.free(torch.tensor([101]))
    with pytest.raises(RuntimeError, match="outside allocator range"):
        allocator.free(torch.tensor([99]))

    reused = allocator.alloc(2)
    assert reused.tolist() == [101, 103]
    allocator.clear()
    assert allocator.available_size() == 5
    with pytest.raises(RuntimeError, match="unallocated row"):
        allocator.assert_allocated(torch.tensor([100]))


def test_fixed_budget_rejects_paged_modulo_aliasing(monkeypatch):
    _enable_compact_budget(monkeypatch)
    with pytest.raises(RuntimeError, match="requires page_size=1"):
        _compact_dflash_fixed_bytes(
            10_240,
            owner_count=3,
            window_size=2048,
            block_size=8,
            page_size=16,
        )


def test_owner_budget_uses_local_attention_dp_and_decode_extra_slots(monkeypatch):
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_schedule",
        lambda: SimpleNamespace(max_running_requests=8),
    )
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_disagg",
        lambda: SimpleNamespace(
            disaggregation_mode="decode", disaggregation_decode_extra_slots=3
        ),
    )
    assert spec_aux_hidden_state._compact_dflash_owner_count(attn_dp_size=2) == 7


@pytest.mark.parametrize("max_running_requests", [None, 7])
def test_owner_budget_rejects_unresolved_or_uneven_geometry(
    monkeypatch, max_running_requests
):
    monkeypatch.setattr(
        spec_aux_hidden_state,
        "get_schedule",
        lambda: SimpleNamespace(max_running_requests=max_running_requests),
    )
    with pytest.raises(RuntimeError, match="max_running_requests"):
        spec_aux_hidden_state._compact_dflash_owner_count(attn_dp_size=2)


@pytest.mark.parametrize("page_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("owner_count", [1, 2, 7])
def test_regions_are_bounded_disjoint_and_periodic(page_size, owner_count):
    value = _layout(owners=owner_count, window=32, block=8, page=page_size)
    all_regions = []
    for owner in range(1, owner_count + 1):
        positions = torch.arange(32)
        owners = torch.full_like(positions, owner)
        committed = value.committed_locs(owners, positions)
        assert torch.equal(committed, value.committed_locs(owners, positions + 352))
        scratch = value.scratch_locs(torch.tensor([owner]), 8).reshape(-1)
        region = torch.cat((committed, scratch))
        assert torch.unique(region).numel() == region.numel()
        assert 0 <= int(region.min()) <= int(region.max()) < value.physical_tokens
        all_regions.append(region)
    combined = torch.cat(all_regions)
    assert torch.unique(combined).numel() == combined.numel()


def test_generation_reuse_fails_closed_and_explicit_rebind_is_counted():
    value = _layout()
    owner_generation = torch.zeros(3, dtype=torch.int64)
    current = torch.tensor([0, 3, 9], dtype=torch.int64)
    assert (
        value.bind_first_use_or_assert_generation(
            torch.tensor([1, 2]),
            owner_generation,
            current,
            torch.tensor([3, 9]),
            torch.tensor([True, True]),
        )
        == 0
    )
    current[1] += 1
    with pytest.raises(RuntimeError, match="generation mismatch"):
        value.bind_first_use_or_assert_generation(
            torch.tensor([1]),
            owner_generation,
            current,
            torch.tensor([3]),
            torch.tensor([False]),
        )
    assert (
        value.bind_first_use_or_assert_generation(
            torch.tensor([1]),
            owner_generation,
            current,
            torch.tensor([4]),
            torch.tensor([True]),
        )
        == 1
    )


def test_request_generation_is_monotonic_across_clear():
    pool = ReqToTokenPool(
        size=1, max_context_len=4, device="cpu", enable_memory_saver=False
    )
    first = SimpleNamespace(req_pool_idx=None)
    assert pool.alloc([first]) == [1]
    first_generation = int(pool.req_generation[1])
    pool.clear()
    second = SimpleNamespace(req_pool_idx=None)
    assert pool.alloc([second]) == [1]
    assert int(pool.req_generation[1]) == first_generation + 1


def test_decode_request_generation_is_monotonic_across_clear():
    pool = DecodeReqToTokenPool(
        size=1,
        max_context_len=4,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=0,
    )
    first = SimpleNamespace(req_pool_idx=None)
    assert pool.alloc([first]) == [1]
    first_generation = int(pool.req_generation[1])
    pool.clear()
    second = SimpleNamespace(req_pool_idx=None)
    assert pool.alloc([second]) == [1]
    assert int(pool.req_generation[1]) == first_generation + 1


def test_invalid_geometry_fails_closed(monkeypatch):
    _enable_compact_budget(monkeypatch)
    with pytest.raises(RuntimeError, match="resolved positive geometry"):
        _compact_dflash_fixed_bytes(
            10_240,
            owner_count=None,
            window_size=2048,
            block_size=8,
            page_size=1,
        )
    with pytest.raises(ValueError):
        _layout(owners=1, window=2049, block=8, page=16)
    with pytest.raises(ValueError, match="cover one complete DFlash commit block"):
        _layout(owners=1, window=4, block=8, page=1)


def test_window_equal_to_commit_block_has_unique_rows():
    layout = _layout(owners=1, window=8, block=8, page=1)
    locs = layout.committed_locs(torch.ones(8), torch.arange(8))
    assert torch.unique(locs).numel() == 8
