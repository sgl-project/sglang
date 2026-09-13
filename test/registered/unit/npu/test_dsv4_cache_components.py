"""Host-safe regression tests for DSV4 Unified Radix Cache extensions."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    DSV4SWAComponent,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="base-a-test-1-npu-a2")


def test_dsv4_swa_branch_keeps_the_c128_cache_boundary_authoritative():
    component = DSV4SWAComponent.__new__(DSV4SWAComponent)
    component.tree_core = SimpleNamespace(is_eagle=False)
    req = SimpleNamespace(
        kv=SimpleNamespace(cache_protected_len=96256, swa_evicted_seqlen=117376),
        swa_branching_seqlen=117632,
    )
    params = InsertParams()

    cache_len = component.prepare_for_caching_req(
        req, params, token_ids_len=117633, is_finished=False
    )

    assert cache_len is None
    assert params.swa_evicted_seqlen == 117376
    assert params.swa_branching_seqlen == 117632


def test_dsv4_full_free_coalesces_only_adjacent_segments_in_one_page():
    allocator = DSV4NPUTokenToKVPoolAllocator.__new__(DSV4NPUTokenToKVPoolAllocator)
    allocator.page_size = 128
    segments = [
        (torch.tensor([100], dtype=torch.int64), 129024),
        (torch.tensor([101, 102, 103], dtype=torch.int64), 129025),
    ]

    with patch.object(SWATokenToKVPoolAllocator, "free_full_segments") as free:
        allocator.free_full_segments(segments)

    normalized = free.call_args.args[1]
    assert len(normalized) == 1
    assert normalized[0][1] == 129024
    assert normalized[0][0].tolist() == [100, 101, 102, 103]
