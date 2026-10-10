from types import SimpleNamespace

import pytest
import torch

from sglang.srt.state_capturer.base import (
    BaseHostCache,
    BaseTopkCapturer,
    TopkCaptureOutput,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def _cache(num_tokens=16, num_layers=2, topk=3):
    # These tests exercise CPU-side metadata only; avoid leaking repeated
    # cudaHostRegister allocations across short-lived cache instances.
    return BaseHostCache(num_tokens, num_layers, topk, name="test", device="npu")


def test_hicache_restore_remaps_experts_to_new_kv_slots():
    cache = _cache()
    old_slots = torch.tensor([1, 2])
    host_slots = torch.tensor([10, 11])
    new_slots = torch.tensor([7, 8])
    expected = torch.arange(12, dtype=torch.int32).reshape(2, 2, 3)

    cache.store(old_slots, expected)
    cache.backup_to_hicache(old_slots, host_slots)

    # Simulate a previous owner leaving unrelated data in the slots selected
    # by HiCache load-back.
    dirty = torch.full_like(expected, 999)
    cache.store(new_slots, dirty)

    assert cache.restore_from_hicache(host_slots, new_slots)
    torch.testing.assert_close(cache.buffer[new_slots], expected)
    assert cache.valid[new_slots].all()


def test_missing_hicache_sidecar_invalidates_dirty_destination():
    cache = _cache()
    new_slot = torch.tensor([7])
    cache.store(new_slot, torch.full((1, 2, 3), 999, dtype=torch.int32))

    assert not cache.restore_from_hicache(torch.tensor([123]), new_slot)
    assert not cache.valid[new_slot].any()

    capturer = BaseTopkCapturer.__new__(BaseTopkCapturer)
    capturer.host_cache = cache
    req_pool = SimpleNamespace(
        req_to_token=torch.tensor([[0, 7, 0]], dtype=torch.int64)
    )
    with pytest.raises(RuntimeError, match="refusing to return stale"):
        capturer.get_topk(
            req_pool_idx=0,
            seqlen=3,
            req_to_token_pool=req_pool,
            start_len=1,
        )


def test_capture_finalize_marks_slots_valid():
    cache = _cache()
    slots = torch.tensor([3, 4])
    values = torch.arange(12, dtype=torch.int32).reshape(2, 2, 3)

    TopkCaptureOutput(slots, values, cache).finalize()

    torch.testing.assert_close(cache.buffer[slots], values)
    assert cache.valid[slots].all()


def test_backup_preserves_uncaptured_padding_slots():
    cache = _cache()

    cache.backup_to_hicache(torch.tensor([5]), torch.tensor([9]))
    torch.testing.assert_close(cache._hicache_rows[9], cache.buffer[5])


def test_clear_hicache_removes_l1_validity_and_l2_rows():
    cache = _cache()
    cache.store(torch.tensor([1]), torch.ones((1, 2, 3), dtype=torch.int32))
    cache.backup_to_hicache(torch.tensor([1]), torch.tensor([9]))

    cache.clear_hicache()

    assert not cache.valid.any()
    assert not cache.restore_from_hicache(torch.tensor([9]), torch.tensor([2]))


def test_clear_removes_host_cache_and_hicache_rows():
    cache = _cache()
    cache.store(torch.tensor([1]), torch.ones((1, 2, 3), dtype=torch.int32))
    cache.backup_to_hicache(torch.tensor([1]), torch.tensor([9]))

    cache.clear()

    assert not cache.buffer.any()
    assert not cache.valid.any()
    assert cache._hicache_rows == {}
    assert not cache.restore_from_hicache(torch.tensor([9]), torch.tensor([2]))


def test_storage_owner_invalidates_reused_hicache_host_slots():
    cache = _cache()
    old_slots = torch.tensor([1, 2])
    host_slots = torch.tensor([9, 10])
    destination = torch.tensor([6, 7])
    values = torch.arange(12, dtype=torch.int32).reshape(2, 2, 3)

    cache.store(old_slots, values)
    cache.backup_to_hicache(old_slots, host_slots)
    cache.invalidate_hicache(host_slots)

    assert not cache.restore_from_hicache(host_slots, destination)
    assert not cache.valid[destination].any()


def test_unified_cache_resolves_global_routed_experts_host_cache(monkeypatch):
    from sglang.srt.state_capturer import routed_experts

    host_cache = object()
    capturer = SimpleNamespace(host_cache=host_cache)
    monkeypatch.setattr(
        routed_experts,
        "get_global_experts_capturer",
        lambda: capturer,
    )

    assert UnifiedRadixCache._get_routed_experts_host_cache() is host_cache
