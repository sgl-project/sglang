"""Smoke tests for the in-tree Rust TreeCore backend (``rust``).

Requires a Rust toolchain: the extension builds with cargo on first use.
"""

import shutil
from array import array

import pytest
import torch

if shutil.which("cargo") is None:
    pytest.skip("the rust backend builds with cargo", allow_module_level=True)

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import create_tree_core


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


def test_insert_then_match_round_trips():
    core = _tree_core()
    _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2, 3]), value=torch.tensor([10, 11, 12], dtype=torch.int64)
        ),
    )
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert matched.device_indices.tolist() == [10, 11, 12]


def test_lock_moves_tokens_between_evictable_and_protected():
    core = _tree_core()
    _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2]), value=torch.tensor([10, 11], dtype=torch.int64)
        ),
    )
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2])))
    core.inc_lock_ref(matched.best_match_node)
    assert core.protected_size() == 2
    assert core.evictable_size() == 0
    core.dec_lock_ref(matched.best_match_node)
    assert core.evictable_size() == 2


def test_namespaces_isolate_the_same_tokens():
    core = _tree_core()
    _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2], extra_key="chat"),
            value=torch.tensor([20, 21], dtype=torch.int64),
        ),
    )
    salted = core.match_prefix(MatchPrefixParams(key=_key([1, 2], extra_key="chat")))
    assert salted.device_indices.tolist() == [20, 21]
    unsalted = core.match_prefix(MatchPrefixParams(key=_key([1, 2])))
    assert unsalted.device_indices.numel() == 0
