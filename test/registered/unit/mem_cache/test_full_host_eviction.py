"""CPU coverage for Full host eviction after duplicate reclamation."""

import sys
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import slot_of, tree_slots_of

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _key(tokens):
    return RadixKey(array("q", tokens))


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_host_eviction_skips_heap_only_when_target_is_satisfied(num_tokens):
    ct = ComponentType.FULL
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_slots=tree_slots_of((ct,)),
    )
    component = FullComponent(
        SimpleNamespace(enable_session_radix_cache=False),
        params,
        slot_of(params, ct),
    )
    core = UnifiedTreeCore(params, {ct: component})
    core.set_hicache_enabled()
    core.is_write_back = True
    step = core.begin_insert(
        InsertParams(key=_key([1]), value=torch.tensor([10], dtype=torch.int64))
    )
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    duplicate = step.result.last_device_node
    core.commit_backup(duplicate, torch.tensor([100], dtype=torch.int64), {})
    core.mark_write_through_pending([duplicate], duplicate)
    core.finish_write_through([duplicate], duplicate)
    core.insert_host(
        core.root_node_handle(), _key([2]), torch.tensor([200]), ["02" * 32]
    )

    with patch.object(
        core.eviction_strategy,
        "get_priority",
        wraps=core.eviction_strategy.get_priority,
    ) as priority:
        result = core.drive_host_eviction(ct, num_tokens)
    try:
        expected = [100] if num_tokens == 1 else [100, 200]
        freed = [
            slot for tensor in result.host_frees.get(ct, []) for slot in tensor.tolist()
        ]
        assert freed == expected
        assert result.tracker[ct] == num_tokens
        assert not result.device_frees
        assert not result.cache_actions
        needs_heap = num_tokens == 2
        assert priority.call_count == int(needs_heap)
        duplicate_data = core.node_by_id(duplicate).component_data[ct]
        assert duplicate_data.value.tolist() == [10]
        assert duplicate_data.host_value is None
        match = core.match_prefix(MatchPrefixParams(key=_key([2])))
        assert match.host_hit_length == int(not needs_heap)
        core.sanity_check([], [])
    finally:
        result.device_frees.clear()
        result.host_frees.clear()
        result.cache_actions.clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
