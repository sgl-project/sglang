"""Current KVDemand restore eviction names FULL and each SWA tail separately."""

import pytest
import torch
from test_flexkv_hybrid_radix_cache import (
    FlexKVHybridRadixCache,
    _make_layerwise_restore,
)

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("page,hit", [(1, 8), (4, 128), (256, 512)])
@pytest.mark.parametrize("retry", [False, True])
def test_restore_evicts_full_hit_and_only_live_window_tail(page, hit, retry):
    cache, req, _, _ = _make_layerwise_restore()
    cache.page_size = page
    cache._inner_cache.path_device_indices.return_value = torch.empty(
        0, dtype=torch.int64
    )
    cache._inner_cache.window_sub_pools.return_value = (("swa", 128),)
    cache.supports_swa = lambda: True
    req.prefix_len = 0
    allocator = cache.token_to_kv_pool_allocator
    slots = torch.arange(hit)
    allocate = allocator.alloc if page == 1 else allocator.alloc_extend_swa_tail
    allocate.side_effect = [None, slots] if retry else [slots]
    assert FlexKVHybridRadixCache._alloc_restore_slots(cache, req, hit) is slots
    if retry:
        cache._inner_cache.evict_for_alloc.assert_called_once_with(
            KVDemand(
                full_tokens=hit,
                window_tokens={"swa": hit if page == 1 else min(page, hit)},
            )
        )
    else:
        cache._inner_cache.evict_for_alloc.assert_not_called()
