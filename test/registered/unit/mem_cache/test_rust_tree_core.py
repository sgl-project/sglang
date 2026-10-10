"""Smoke tests for the in-tree Rust TreeCore backend (``rust``).

Requires a Rust toolchain: the extension builds with cargo on first use.
"""

from array import array

import pytest
import torch
from unified_tree_core_inspection_interface import UnifiedTreeCoreInspectionInterface

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import create_tree_core
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=17, suite="base-a-test-cpu")


def _tree_core():
    return create_tree_core(
        "rust",
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        ),
        components={},
    )


def _key(token_ids, extra_key=None):
    return RadixKey(array("q", token_ids), extra_key=extra_key)


def _pump_insert(core, params):
    step = core.begin_insert(params)
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    return step.result


def test_registry_resolves_the_rust_backend_lazily():
    core = _tree_core()
    assert type(core).__name__ == "RustUnifiedTreeCore"
    assert not isinstance(core, UnifiedTreeCoreInspectionInterface)
    assert not any(name.startswith("inspect_") for name in dir(core._binding))


def test_backfill_hashes_existing_nodes_in_parent_order():
    expected = _tree_core()
    expected.enable_storage = True
    _pump_insert(
        expected,
        InsertParams(key=_key([1, 2]), value=torch.tensor([10, 11], dtype=torch.int64)),
    )
    _pump_insert(
        expected,
        InsertParams(
            key=_key([1, 2, 3, 4]),
            value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        ),
    )

    late = _tree_core()
    _pump_insert(
        late,
        InsertParams(key=_key([1, 2]), value=torch.tensor([10, 11], dtype=torch.int64)),
    )
    _pump_insert(
        late,
        InsertParams(
            key=_key([1, 2, 3, 4]),
            value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        ),
    )

    parent = late.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    child = late.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 4]))).best_match_node
    expected_parent = expected.match_prefix(
        MatchPrefixParams(key=_key([1, 2]))
    ).best_match_node
    expected_child = expected.match_prefix(
        MatchPrefixParams(key=_key([1, 2, 3, 4]))
    ).best_match_node

    assert late.get_hash_values(parent) == []
    assert late.get_hash_values(child) == []
    assert late.backfill_missing_hash_values() == 2
    assert late.get_hash_values(parent) == expected.get_hash_values(expected_parent)
    assert late.get_hash_values(child) == expected.get_hash_values(expected_child)
    assert late.backfill_missing_hash_values() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
