"""Public ownership contract shared by LMCache and HiCache recovery adapters."""

import pytest
import torch
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_load_recovery import prepare_partial_kv_load


@pytest.mark.parametrize("hole", [(0, 4), (4, 8), (8, 12)])
@pytest.mark.parametrize("policy", ["recompute", "repair"])
def test_partial_load_preserves_prefix_and_releases_only_owned_slots(hole, policy):
    allocator = TokenToKVPoolAllocator(32, torch.float32, "cpu", None, False)
    indices = allocator.alloc(12)
    prefix, recovery = prepare_partial_kv_load(
        loaded_start=0,
        loaded_indices=indices,
        missing_span=hole,
        page_size=1,
        allocator=allocator,
        policy=policy,
    )
    assert torch.equal(prefix, indices[: hole[0]])
    assert allocator.available_size() == 20 + (
        hole[1] - hole[0] if policy == "repair" else 12 - hole[0]
    )
    if policy == "repair":
        assert recovery.span_start == hole[0] and recovery.span_end == hole[1]
        assert recovery.restored_extent == 12
        assert torch.equal(recovery.suffix_indices, indices[hole[1] :])
        recovery.abort(allocator)
        recovery.abort(allocator)
    else:
        assert recovery is None
    assert allocator.available_size() == 32 - hole[0]
    allocator.free(prefix)
    assert allocator.available_size() == 32


@pytest.mark.parametrize(
    "span,page", [((-1, 4), 1), ((4, 4), 1), ((4, 13), 1), ((3, 8), 4)]
)
def test_invalid_descriptor_does_not_release_any_allocation(span, page):
    allocator = TokenToKVPoolAllocator(32, torch.float32, "cpu", None, False)
    indices = allocator.alloc(12)
    with pytest.raises(ValueError):
        prepare_partial_kv_load(
            loaded_start=0,
            loaded_indices=indices,
            missing_span=span,
            page_size=page,
            allocator=allocator,
            policy="repair",
        )
    assert allocator.available_size() == 20


def test_commit_restores_the_private_suffix_only_after_repair_forward():
    from array import array

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    allocator = TokenToKVPoolAllocator(32, torch.float32, "cpu", None, False)
    pool = ReqToTokenPool(
        size=1,
        row_width=16,
        device="cpu",
        enable_memory_saver=False,
    )
    cache = RadixCache(CacheInitParams(False, pool, allocator, 1))
    req = Req("repair", "", array("q", range(16)), SamplingParams())
    pool.alloc([req])
    indices = allocator.alloc(12)
    prefix, recovery = prepare_partial_kv_load(
        loaded_start=0,
        loaded_indices=indices,
        missing_span=(4, 8),
        page_size=1,
        allocator=allocator,
        policy="repair",
    )
    req.kv_load_recovery = recovery
    fresh = allocator.alloc(4)
    pool.write((req.kv.req_pool_idx, slice(0, 8)), torch.cat([prefix, fresh]))
    req.extend_end = 7
    req.kv.set_row_len(7)
    with pytest.raises(RuntimeError):
        recovery.commit(req, cache)
    assert not pool.req_to_token[req.kv.req_pool_idx, 8:12].any()
    req.extend_end = 8
    req.kv.set_row_len(8)
    recovery.commit(req, cache)
    assert req.extend_end == 12 and req.kv.kv_committed_len == 12
    assert torch.equal(
        pool.req_to_token[req.kv.req_pool_idx, 8:12].long(), indices[8:12]
    )
    assert req.kv_load_recovery is None
    before = allocator.available_size()
    recovery.abort(allocator)
    assert allocator.available_size() == before
