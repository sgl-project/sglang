"""Integration tests driving the real compiled Rust mem_cache extension."""

import hashlib
import math
import sys
from array import array
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=17, suite="base-a-test-cpu")

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.evict_policy import TLRUStrategy
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_tree_core.adapter import (
    RustUnifiedTreeCore,
    _tlru_float_config,
)
from sglang.srt.mem_cache.rust_tree_core.extension import bindings as mem_cache
from sglang.srt.mem_cache.rust_tree_core.extension import load_tree_core_extension
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    FreeComponentHostSlot,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    RecoverSWAWithLockedFull,
    ReplaceWriteThroughOnNodeSplit,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.utils import get_storage_hash_str, hash_str_to_int64
from sglang.srt.runtime_context import get_context


def _tree_core(**params_overrides) -> RustUnifiedTreeCore:
    params = dict(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_components=(ComponentType.FULL,),
    )
    params.update(params_overrides)
    return RustUnifiedTreeCore(CacheInitParams(**params))


def _key(token_ids: list[int]) -> RadixKey:
    return RadixKey(array("q", token_ids))


def _insert(core: RustUnifiedTreeCore, token_ids: list[int], indices: list[int]):
    return _pump_insert(
        core,
        InsertParams(
            key=_key(token_ids),
            value=torch.tensor(indices, dtype=torch.int64),
        ),
    )


def _binding(**init_overrides):
    return mem_cache.RustUnifiedTreeCoreBinding(
        mem_cache.TreeCoreInitParamsBinding(**init_overrides),
        [int(ComponentType.FULL)],
    )


def _pump_insert(core: RustUnifiedTreeCore, params: InsertParams) -> InsertResult:
    """Drive the resumable-insert protocol, folding step actions into the result."""
    step = core.begin_insert(params)
    actions = list(step.actions)
    while step.result is None:
        step = core.resume_insert()
        actions.extend(step.actions)
    core.end_insert()
    return InsertResult(
        prefix_len=step.result.prefix_len,
        last_device_node=step.result.last_device_node,
        mamba_exist=step.result.mamba_exist,
        swa_branch_inserted=step.result.swa_branch_inserted,
        rotation_tail_declined=step.result.rotation_tail_declined,
        adopted_ranges=step.result.adopted_ranges,
        cache_actions=actions,
    )


def _accumulate_step(result, tracker, device_frees, host_frees):
    """Fold an eviction step into running accumulators (the Controller
    consumption contract: deltas add, freed tensors append), draining it."""
    for component, delta in result.tracker.items():
        tracker[component] = tracker.get(component, 0) + delta
    for component, tensors in result.device_frees.items():
        device_frees.setdefault(component, []).extend(tensors)
    for component, tensors in result.host_frees.items():
        host_frees.setdefault(component, []).extend(tensors)
    result.device_frees.clear()
    result.host_frees.clear()


def test_match_on_the_empty_tree_returns_no_indices():
    core = _tree_core()
    result = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert result.device_indices.numel() == 0


def test_default_backend_constructs_real_rust_cpu_cache(monkeypatch):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    monkeypatch.delenv("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", raising=False)
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    assert cache._tree_core_backend == "rust"
    assert isinstance(cache.tree_core, RustUnifiedTreeCore)
    cache.insert(InsertParams(key=_key([1, 2, 3]), value=torch.tensor([11, 12, 13])))
    matched = cache.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert matched.device_indices.tolist() == [11, 12, 13]
    cache.tree_core.sanity_check([], [])


def test_insert_then_match_back_returns_the_exact_indices():
    core = _tree_core()
    result = _insert(core, [1, 2, 3], [10, 11, 12])
    assert result.prefix_len == 0
    assert result.cache_actions == []
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert result.last_device_node == matched.last_device_node
    assert matched.device_indices.tolist() == [10, 11, 12]


def test_root_node_handle_is_namespace_independent():
    core = _tree_core()
    root = core.root_node_handle()
    # The single root serves every namespace, seen or not.
    assert core.root_node_handle("ghost") == root
    _pump_insert(
        core,
        InsertParams(
            key=RadixKey(array("q", [1, 2]), extra_key="chat"),
            value=torch.tensor([10, 11], dtype=torch.int64),
        ),
    )
    assert core.root_node_handle("chat") == root
    # A full miss in the namespace anchors its match at the root.
    missed = core.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [9]), extra_key="chat"))
    )
    assert missed.best_match_node == root


@pytest.mark.parametrize("is_eagle", [False, True])
def test_rotation_decline_precedes_host_restore_and_respects_namespaces(is_eagle):
    core = _tree_core(page_size=2, is_eagle=is_eagle, enable_kv_cache_events=True)
    core.set_hicache_enabled()
    tokens = array("q", range(9 if is_eagle else 8))
    key = RadixKey(tokens, extra_key="adapter", cache_salt="tenant-a")
    inserted = _pump_insert(
        core,
        InsertParams(
            key=key,
            value=torch.arange(10, 18),
            rotation_base=1,
        ),
    )
    leaf = inserted.last_device_node
    core.commit_backup(leaf, torch.arange(100, 108), {})
    _accumulate_step(core.demote(leaf), {}, {}, {})
    core.take_events()

    step = core.begin_insert(
        InsertParams(
            key=key,
            value=torch.arange(20, 28),
            rotation_base=3,
            track_adopted_ranges=True,
        )
    )
    assert step.result is not None
    assert step.result.rotation_tail_declined
    assert step.result.prefix_len == 8
    assert step.result.last_device_node == leaf
    assert step.result.adopted_ranges == {}
    assert step.actions == []
    assert not core.has_ongoing_insert()
    assert core.end_insert() == []
    assert core.is_full_device_evicted(leaf)
    assert core.rotation_base_of(leaf) == 1
    assert core.take_events() == []

    other = _pump_insert(
        core,
        InsertParams(
            key=RadixKey(tokens, extra_key="adapter", cache_salt="tenant-b"),
            value=torch.arange(20, 28),
            rotation_base=3,
        ),
    )
    assert not other.rotation_tail_declined
    assert core.rotation_base_of(other.last_device_node) == 3
    assert core.rotation_base_of(core.root_node_handle()) is None
    core.sanity_check([], [])


@pytest.mark.parametrize(
    "config,evict_recent",
    [
        (None, True),
        ({"protected_threshold": 4}, False),
        ({"protected_threshold": 0}, False),
    ],
)
def test_slru_config_changes_which_leaf_is_evicted(config, evict_recent):
    core = _tree_core(eviction_policy="SLRU", eviction_policy_config=config)
    for _ in range(3):
        old = _insert(core, [1, 2], [10, 11]).last_device_node
    recent = _insert(core, [3, 4], [12, 13]).last_device_node

    core.evict_device_start(ComponentType.FULL, 2)
    try:
        step = core.evict_device_next_node(ComponentType.FULL, {})
        assert step.node_id == (recent if evict_recent else old)
    finally:
        core.evict_device_end(ComponentType.FULL)


def _tlru_tree_core(backend, config=None, page_size=1, is_eagle=False):
    from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=page_size,
        is_eagle=is_eagle,
        tree_components=(ComponentType.FULL,),
        eviction_policy="tlru",
        eviction_policy_config=config,
    )
    if backend == "rust":
        return RustUnifiedTreeCore(params)
    cache = SimpleNamespace(enable_session_radix_cache=False)
    return UnifiedTreeCore(params, {ComponentType.FULL: FullComponent(cache, params)})


def _tlru_insert(core, token_ids, indices, tier):
    key = RadixKey(array("q", token_ids), is_bigram=core.is_eagle)
    if tier == "device":
        return _pump_insert(
            core, InsertParams(key=key, value=torch.tensor(indices, dtype=torch.int64))
        ).last_device_node
    core.set_hicache_enabled()
    return core.insert_host(
        core.root_node_handle(),
        key,
        torch.tensor(indices, dtype=torch.int64),
        [f"{index:064x}" for index in range(len(indices) // core.page_size)],
    ).inserted_host_node


def _tlru_evict_one_leaf(core, tier):
    tracker, device_frees, host_frees = {}, {}, {}
    if tier == "host":
        _accumulate_step(
            core.drive_host_eviction(ComponentType.FULL, 1),
            tracker,
            device_frees,
            host_frees,
        )
        return torch.cat(host_frees[ComponentType.FULL]).tolist()

    core.evict_device_start(ComponentType.FULL, 1)
    try:
        step = core.evict_device_next_node(ComponentType.FULL, tracker)
        node = step.node_id
        _accumulate_step(step, tracker, device_frees, host_frees)
        assert node is not None
        _accumulate_step(
            core.evict_device_leaf(node, is_write_back=False),
            tracker,
            device_frees,
            host_frees,
        )
        assert core.evict_device_next_node(ComponentType.FULL, tracker).node_id is None
    finally:
        core.evict_device_end(ComponentType.FULL)
    return torch.cat(device_frees[ComponentType.FULL]).tolist()


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize(
    "config,evict_recent",
    [
        (None, False),
        ({"threshold": 4, "next_prompt_estimate": 0}, True),
        ({"threshold": 4, "next_prompt_estimate": 2}, True),
        ({"threshold": 4, "next_prompt_estimate": 4}, False),
        ({"threshold": 4, "next_prompt_estimate": 6}, False),
        ({"threshold": -2, "next_prompt_estimate": 0}, False),
        ({"threshold": 2, "next_prompt_estimate": -2}, True),
        ({"threshold": 2**100, "next_prompt_estimate": 2**100 - 2}, True),
        ({"threshold": 2**100, "next_prompt_estimate": 2**100 + 2}, False),
        ({"threshold": 2**100, "next_prompt_estimate": -(2**100)}, False),
        ({"threshold": -(2**100), "next_prompt_estimate": 2**100}, False),
    ],
)
def test_tlru_config_prioritizes_safe_tails_then_recency(backend, config, evict_recent):
    core = _tlru_tree_core(backend, config)
    # The older conversation occupies one oversized node, so trimming it would
    # exceed the tail budget. The recent short conversation can fit entirely.
    _insert(core, list(range(100, 108)), list(range(200, 208)))
    _insert(core, [1, 2], [10, 11])

    assert _tlru_evict_one_leaf(core, "device") == (
        [10, 11] if evict_recent else list(range(200, 208))
    )
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("tier", ["device", "host"])
@pytest.mark.parametrize(
    "threshold,next_prompt_estimate,evict_recent",
    [
        pytest.param(4096.0, 4095.0, True, id="integral-floats"),
        pytest.param(1.5, 0.25, True, id="fractional-safe"),
        pytest.param(1.5, 0.75, False, id="fractional-protected"),
        pytest.param(1.5, 0, True, id="float-threshold"),
        pytest.param(2, 0.5, True, id="float-estimate"),
        pytest.param(1.2, 0.2, False, id="addition-before-subtraction"),
        pytest.param(1.2000000000000002, 0.2, True, id="adjacent-float-boundary"),
        pytest.param(
            float(2**53 + 2),
            2**53 + 1,
            False,
            id="integer-addition-before-float-conversion",
        ),
        pytest.param(2**53 + 1, float(2**53), False, id="large-integer-threshold"),
        pytest.param(
            -float(2**53),
            -(2**53 + 1),
            True,
            id="negative-integer-addition-before-float-conversion",
        ),
        pytest.param(2**100 + 1, 2**100, True, id="arbitrary-integer-safe"),
        pytest.param(2**100, 2**100, False, id="arbitrary-integer-protected"),
        pytest.param(-0.5, -2, True, id="negative-threshold"),
        pytest.param(0, -1.5, True, id="negative-estimate"),
        pytest.param(True, False, True, id="boolean-counts"),
        pytest.param(float("nan"), 0, False, id="nan-threshold"),
        pytest.param(0, float("nan"), False, id="nan-estimate"),
        pytest.param(float("inf"), 0, False, id="positive-infinite-threshold"),
        pytest.param(float("-inf"), 0, False, id="negative-infinite-threshold"),
        pytest.param(0, float("inf"), False, id="positive-infinite-estimate"),
        pytest.param(0, float("-inf"), False, id="negative-infinite-estimate"),
        pytest.param(float("inf"), float("inf"), False, id="infinite-cancellation"),
    ],
)
def test_tlru_numeric_configuration_matches_python_eviction(
    backend, tier, threshold, next_prompt_estimate, evict_recent
):
    core = _tlru_tree_core(
        backend,
        {"threshold": threshold, "next_prompt_estimate": next_prompt_estimate},
    )
    _tlru_insert(core, list(range(100, 108)), list(range(200, 208)), tier)
    _tlru_insert(core, [0], [10], tier)
    _tlru_insert(core, [0, 1], [10, 11], tier)

    # Removing the recent leaf leaves one token cached from a two-token history.
    # In particular, 2 + 0.2 - 1.2 exceeds 1, while rearranging the expression to
    # 1.2 - 0.2 would incorrectly mark that leaf safe. Non-finite configurations
    # put both leaves in the same priority class and retain recency ordering.
    assert _tlru_evict_one_leaf(core, tier) == (
        [11] if evict_recent else list(range(200, 208))
    )
    core.sanity_check([], [])


def _assert_tlru_float_priorities_match_python(
    bindings, threshold, next_prompt_estimate, histories, cached_counts
):
    native = _tlru_float_config(bindings, threshold, next_prompt_estimate)
    strategy = TLRUStrategy(threshold, next_prompt_estimate)
    for history in histories:
        for cached in cached_counts:
            if cached > history:
                continue
            node = SimpleNamespace(
                _tlru_history_len=history,
                _tlru_cached_prefix_len=cached,
                key=(),
                last_access_time=0.0,
            )
            expected = strategy.get_priority(node)[0] == -1
            assert native.inspect_is_tel_safe(history, cached) == expected, (
                threshold,
                next_prompt_estimate,
                history,
                cached,
            )


def test_tlru_float_priorities_match_python_at_integer_comparison_boundaries():
    bindings = load_tree_core_extension(inspection=True)
    max_history = 2 * sys.maxsize + 1
    counts = [
        0,
        1,
        2,
        3,
        8,
        2**53 - 1,
        2**53,
        2**53 + 1,
        2**53 + 2,
        2**53 + 3,
        2**53 + 4,
        max_history - 1,
        max_history,
    ]
    for threshold, estimate in (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.2, 0.2),
        (1.2000000000000002, 0.2),
        (-0.5, -2.0),
        (4096.0, 1024.25),
        (float(2**53), 0.0),
        (2**53 + 1, float(2**53)),
        (0.0, math.nextafter(float(max_history + 1), 0.0)),
        (0.0, float(max_history + 1)),
        (-sys.float_info.max, sys.float_info.max),
        (sys.float_info.max, -sys.float_info.max),
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (0.0, float("inf")),
        (0.0, float("-inf")),
        (float("inf"), float("inf")),
        (float("-inf"), float("-inf")),
    ):
        _assert_tlru_float_priorities_match_python(
            bindings, threshold, estimate, counts, counts
        )


def test_tlru_integer_estimate_priorities_match_python_at_i128_endpoints():
    bindings = load_tree_core_extension(inspection=True)
    max_history = 2 * sys.maxsize + 1
    max_estimate = 2**127 - 1
    max_safe_estimate = max_estimate - max_history
    history_candidates = [0, 1, 2, 2**53 + 1, max_history - 1, max_history]
    for estimate in (
        -(2**127),
        -(2**127) + 1,
        max_safe_estimate - 1,
        max_safe_estimate,
        max_safe_estimate + 1,
        max_estimate - 2,
        max_estimate,
    ):
        # Only histories whose actual addition fits i128 belong in the parity
        # oracle; native arithmetic overflow is covered by the Rust panic test.
        histories = [
            history
            for history in history_candidates
            if history <= max_estimate - estimate
        ]
        for threshold in (float(estimate), math.nextafter(float(estimate), math.inf)):
            _assert_tlru_float_priorities_match_python(
                bindings, threshold, estimate, histories, histories
            )


@pytest.mark.parametrize("exponent", [53, 100, 126])
def test_tlru_integer_estimate_priorities_match_python_at_rounding_ties(exponent):
    bindings = load_tree_core_extension(inspection=True)
    max_history = 2 * sys.maxsize + 1
    for sign in (-1, 1):
        lower = sign * math.ldexp(1.0, exponent)
        # Consecutive float pairs exercise both directions of ties-to-even.
        for _ in range(2):
            upper = math.nextafter(lower, math.inf)
            midpoint = (int(lower) + int(upper)) // 2
            for tie_history in (0, 1, 2, 2**53 + 1, sys.maxsize, max_history):
                estimate = midpoint - tie_history
                histories = sorted(
                    {
                        0,
                        1,
                        2,
                        2**53 + 1,
                        max_history - 1,
                        max_history,
                        max(tie_history - 1, 0),
                        tie_history,
                        min(tie_history + 1, max_history),
                    }
                )
                _assert_tlru_float_priorities_match_python(
                    bindings, lower, estimate, histories, [0, 1, max_history]
                )
            lower = upper


@pytest.mark.parametrize("estimate", [-(2**127) - 1, 2**127])
def test_tlru_mixed_integer_estimate_outside_i128_is_rejected(estimate):
    with pytest.raises(OverflowError):
        _tlru_tree_core("rust", {"threshold": 0.0, "next_prompt_estimate": estimate})


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_tlru_near_i128_max_estimate_accepts_small_history(backend):
    core = _tlru_tree_core(
        backend,
        {"threshold": float(2**127 - 1), "next_prompt_estimate": 2**127 - 3},
    )
    _insert(core, [0, 1], [10, 11])
    assert _tlru_evict_one_leaf(core, "device") == [10, 11]
    core.sanity_check([], [])


@pytest.mark.parametrize(
    "threshold,next_prompt_estimate",
    [
        pytest.param("1.0", 0, id="string-threshold"),
        pytest.param(0, "1.0", id="string-estimate"),
        pytest.param(None, 0, id="null-threshold"),
        pytest.param(0, None, id="null-estimate"),
        pytest.param([1], 0, id="list-threshold"),
        pytest.param(0, {}, id="object-estimate"),
    ],
)
def test_tlru_non_numeric_configuration_raises_type_error(
    threshold, next_prompt_estimate
):
    config = {"threshold": threshold, "next_prompt_estimate": next_prompt_estimate}
    with pytest.raises(TypeError):
        _tlru_tree_core("rust", config)
    node = SimpleNamespace(
        _tlru_history_len=2,
        _tlru_cached_prefix_len=2,
        key=(0,),
        last_access_time=0.0,
    )
    with pytest.raises(TypeError):
        TLRUStrategy(**config).get_priority(node)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("tier", ["device", "host"])
@pytest.mark.parametrize("page_size", [1, 2])
@pytest.mark.parametrize("is_eagle", [False, True])
def test_tlru_repeated_tail_eviction_preserves_history(
    backend, tier, page_size, is_eagle
):
    core = _tlru_tree_core(
        backend, {"threshold": 6, "next_prompt_estimate": 2}, page_size, is_eagle
    )
    _tlru_insert(
        core,
        list(range(100, 108 + is_eagle)),
        list(range(200, 208)),
        tier,
    )
    for length in (2, 4, 6, 8):
        _tlru_insert(
            core, list(range(length + is_eagle)), list(range(10, 10 + length)), tier
        )

    # Four tokens are safe. A fresh eviction walk must retain the original
    # eight-token high-water mark and protect the shortened conversation.
    assert _tlru_evict_one_leaf(core, tier) == [16, 17]
    assert _tlru_evict_one_leaf(core, tier) == [14, 15]
    assert _tlru_evict_one_leaf(core, tier) == list(range(200, 208))
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("tier", ["device", "host"])
@pytest.mark.parametrize("page_size", [1, 2])
def test_tlru_split_and_compacted_branch_preserve_history(backend, tier, page_size):
    core = _tlru_tree_core(
        backend, {"threshold": 6, "next_prompt_estimate": 2}, page_size
    )
    _tlru_insert(core, list(range(100, 108)), list(range(200, 208)), tier)
    _tlru_insert(core, list(range(8)), list(range(10, 18)), tier)
    # Matching splits the long turn at depth four; inserting a compacted turn
    # then splits that prefix again at depth two. Neither split shortens history.
    core.match_prefix(MatchPrefixParams(key=_key(list(range(4)))))
    _tlru_insert(core, [0, 1, 90, 91], [10, 11, 30, 31], tier)

    assert _tlru_evict_one_leaf(core, tier) == [14, 15, 16, 17]
    assert _tlru_evict_one_leaf(core, tier) == [30, 31]
    assert _tlru_evict_one_leaf(core, tier) == list(range(200, 208))
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_tlru_host_refill_raises_device_ancestor_history(backend):
    core = _tlru_tree_core(backend, {"threshold": 6, "next_prompt_estimate": 2})
    core.set_hicache_enabled()
    old = _insert(core, list(range(100, 108)), list(range(200, 208))).last_device_node
    prefix = _insert(core, [0, 1], [10, 11]).last_device_node
    _complete_backup(core, prefix)
    inserted = core.insert_host(
        prefix,
        _key(list(range(2, 8))),
        torch.arange(1012, 1018),
        [f"{index:064x}" for index in range(6)],
    )
    assert inserted.inserted_host_node is not None

    # Prefetch creates no new device KV, but its depth must protect the short
    # device ancestor from T-LRU's first phase.
    core.evict_device_start(ComponentType.FULL, 1)
    try:
        assert core.evict_device_next_node(ComponentType.FULL, {}).node_id == old
    finally:
        core.evict_device_end(ComponentType.FULL)
    core.sanity_check([], [])


@pytest.mark.parametrize(
    "policy,config",
    [("lru", {"protected_threshold": 4}), ("slru", {"unknown_option": 4})],
)
def test_eviction_config_rejects_unknown_constructor_options(policy, config):
    with pytest.raises(TypeError):
        _tree_core(eviction_policy=policy, eviction_policy_config=config)


def test_stale_handle_reads_raise_key_error_without_poisoning_the_core():
    core = _tree_core()
    stale_root = core.root_node_handle()
    core.reset()
    live_root = core.root_node_handle()

    accessors = (
        core.is_backuped,
        core.is_root,
        core.get_last_hash_value,
        core.get_prefix_hash_values,
        core.prefetch_anchor_info,
    )
    for accessor in accessors:
        with pytest.raises(Exception) as exc_info:
            accessor(stale_root)
        assert isinstance(exc_info.value, KeyError)
        assert exc_info.value.args == (stale_root,)
        assert core.is_root(live_root)


def test_stale_match_finalizer_handles_raise_key_error_without_poisoning_the_core():
    from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

    core = RustUnifiedTreeCoreInspector(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    stale_root = core.root_node_handle()
    core.reset()
    live_root = core.root_node_handle()
    result = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=live_root,
        last_host_node=live_root,
        best_match_node=live_root,
    )
    params = MatchPrefixParams(key=_key([]))

    for field in ("last_device_node", "last_host_node", "best_match_node"):
        with pytest.raises(KeyError) as exc_info:
            core.finalize_component_match_result(
                ComponentType.FULL,
                result._replace(**{field: stale_root}),
                params,
                value_chunks=[],
                best_value_len=0,
            )
        assert exc_info.value.args == (stale_root,), field
        assert core.is_root(live_root), field

    finalized = core.finalize_component_match_result(
        ComponentType.FULL,
        result,
        params,
        value_chunks=[],
        best_value_len=0,
    )
    assert finalized.best_match_node == live_root


def test_stale_handle_operations_raise_key_error_without_poisoning_the_core():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _tree_core()
    stale_root = core.root_node_handle()
    core.reset()
    live_root = core.root_node_handle()
    empty = torch.empty(0, dtype=torch.int64)

    operations = {
        "inc_lock_ref": lambda: core.inc_lock_ref(stale_root),
        "dec_lock_ref": lambda: core.dec_lock_ref(stale_root, DecLockRefParams()),
        "dec_swa_lock_only": lambda: core.dec_swa_lock_only(
            stale_root, DecLockRefParams()
        ),
        "evict_device_leaf": lambda: core.evict_device_leaf(stale_root, False),
        "drop_subtree_no_host": lambda: core.drop_subtree_no_host(stale_root),
        "demote": lambda: core.demote(stale_root),
        "is_full_device_evicted": lambda: core.is_full_device_evicted(stale_root),
        "collect_full_device_indices/from": lambda: core.collect_full_device_indices(
            stale_root, live_root
        ),
        "collect_full_device_indices/until": lambda: core.collect_full_device_indices(
            live_root, stale_root
        ),
        "insert_host": lambda: core.insert_host(
            stale_root, _key([1]), empty, ["0" * 64]
        ),
        "build_backup_spec": lambda: core.build_backup_spec(stale_root),
        "build_storage_backup_spec": lambda: core.build_storage_backup_spec(
            stale_root, False
        ),
        "build_hicache_transfers": lambda: core.build_hicache_transfers(
            ComponentType.FULL, stale_root, CacheTransferPhase.BACKUP_STORAGE
        ),
        "commit_backup": lambda: core.commit_backup(stale_root, empty, {}),
        "commit_hicache_transfers": lambda: core.commit_hicache_transfers(
            stale_root,
            CacheTransferPhase.BACKUP_HOST,
            {},
            cache_actions=[],
        ),
        "commit_load_back": lambda: core.commit_load_back(
            stale_root, empty, PoolTransfer(name=PoolName.KV), {}
        ),
        "build_load_back_spec": lambda: core.build_load_back_spec(stale_root),
        "build_external_linker_offload_transfers": lambda: (
            core.build_external_linker_offload_transfers(stale_root)
        ),
        "mark_external_cache_stored_path/from": lambda: (
            core.mark_external_cache_stored_path(stale_root, live_root)
        ),
        "mark_external_cache_stored_path/until": lambda: (
            core.mark_external_cache_stored_path(live_root, stale_root)
        ),
        "mark_external_linker_offload_pending": lambda: (
            core.mark_external_linker_offload_pending(stale_root)
        ),
        "finish_external_linker_offload": lambda: core.finish_external_linker_offload(
            [live_root, stale_root], live_root, True
        ),
        "evict_excess_path_states": lambda: core.evict_excess_path_states(
            stale_root, {}, {}
        ),
        "inc_host_lock_ref": lambda: core.inc_host_lock_ref(stale_root),
        "dec_host_lock_ref": lambda: core.dec_host_lock_ref(
            stale_root, DecLockRefParams()
        ),
        "mark_write_through_pending": lambda: core.mark_write_through_pending(
            [stale_root], stale_root
        ),
        "finish_write_through": lambda: core.finish_write_through(
            [stale_root], stale_root
        ),
        "finish_load_back": lambda: core.finish_load_back(stale_root),
        "get_component_device_value": lambda: core.get_component_device_value(
            stale_root, ComponentType.FULL
        ),
        "component_has_host_value_only": lambda: core.component_has_host_value_only(
            stale_root, ComponentType.FULL
        ),
        "get_hash_values": lambda: core.get_hash_values(stale_root),
        "dfs_weight_order": lambda: core.dfs_weight_order([stale_root]),
    }
    for name, operation in operations.items():
        with pytest.raises(KeyError) as exc_info:
            operation()
        assert exc_info.value.args == (stale_root,), name
        assert core.is_root(live_root), name


def test_stale_handles_nested_in_transfer_results_do_not_poison_the_core():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _tree_core()
    stale_root = core.root_node_handle()
    core.reset()
    live_root = core.root_node_handle()
    stale_transfer = PoolTransfer(name=PoolName.KV, nodes_to_load=[stale_root])

    operations = (
        lambda: core.commit_hicache_transfers(
            live_root,
            CacheTransferPhase.LOAD_BACK,
            {ComponentType.FULL: [stale_transfer]},
            cache_actions=[],
        ),
        lambda: core.commit_hicache_transfers(
            live_root,
            CacheTransferPhase.PREFETCH,
            {},
            cache_actions=[],
            insert_result=InsertResult(prefix_len=0, inserted_host_node=stale_root),
        ),
        lambda: core.commit_load_back(
            live_root,
            torch.empty(0, dtype=torch.int64),
            stale_transfer,
            {},
        ),
    )
    for operation in operations:
        with pytest.raises(KeyError) as exc_info:
            operation()
        assert exc_info.value.args == (stale_root,)
        assert core.is_root(live_root)


def test_stale_handle_component_access_does_not_poison_the_core():
    core = _tree_core(
        tree_components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=8,
    )
    stale_root = core.root_node_handle()
    core.reset()
    live_root = core.root_node_handle()

    operations = (
        lambda: core.set_component_device_value(
            stale_root, ComponentType.SWA, torch.empty(0, dtype=torch.int64)
        ),
        lambda: core.get_component_device_value(stale_root, ComponentType.SWA),
    )
    for operation in operations:
        with pytest.raises(KeyError) as exc_info:
            operation()
        assert exc_info.value.args == (stale_root,)
        assert core.is_root(live_root)


def test_dfs_weight_order_groups_the_heaviest_subtree_first():
    core = _tree_core()
    _insert(core, [1, 10], [10, 11])
    _insert(core, [1, 11], [10, 12])
    _insert(core, [2, 20], [20, 21])

    branch_a = core.match_prefix(MatchPrefixParams(key=_key([1, 99]))).last_device_node
    leaf_a1 = core.match_prefix(MatchPrefixParams(key=_key([1, 10]))).last_device_node
    leaf_a2 = core.match_prefix(MatchPrefixParams(key=_key([1, 11]))).last_device_node
    leaf_b = core.match_prefix(MatchPrefixParams(key=_key([2, 20]))).last_device_node

    assert core.dfs_weight_order([leaf_b, leaf_a2, leaf_a1, leaf_a1, branch_a]) == [
        2,
        3,
        1,
        4,
        0,
    ]
    assert core.dfs_weight_order([leaf_b, leaf_a2]) == [1, 0]


def test_get_hash_values_round_trips_through_insert_host():
    core = _tree_core()
    root = core.match_prefix(MatchPrefixParams(key=_key([99]))).best_match_node
    result = core.insert_host(
        root, _key([1]), torch.tensor([100], dtype=torch.int64), ["h0"]
    )
    assert core.get_hash_values(result.inserted_host_node) == ["h0"]
    # A never-hashed device node reads back empty.
    _insert(core, [5], [50])
    device_node = core.match_prefix(MatchPrefixParams(key=_key([5]))).best_match_node
    assert core.get_hash_values(device_node) == []


def test_insert_coerces_a_none_priority():
    core = _tree_core()
    result = _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2]),
            value=torch.tensor([10, 11], dtype=torch.int64),
            priority=None,
        ),
    )
    assert result.prefix_len == 0


def test_extension_insert_frees_the_duplicate_overlap():
    core = _tree_core()
    _insert(core, [1, 2, 3], [10, 11, 12])
    result = _insert(core, [1, 2, 3, 4, 5], [20, 21, 22, 13, 14])
    assert result.prefix_len == 3
    # The overlap's fresh indices are duplicates: freed, not stored.
    assert len(result.cache_actions) == 1
    action = result.cache_actions[0]
    assert isinstance(action, FreeDeviceKV)
    assert torch.cat(action.indices).tolist() == [20, 21, 22]
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 4, 5])))
    assert matched.device_indices.tolist() == [10, 11, 12, 13, 14]


def test_lock_and_unlock_move_tokens_between_protected_and_evictable():
    core = _tree_core()
    _insert(core, [1, 2, 3], [10, 11, 12])
    _insert(core, [1, 2, 3, 4, 5], [20, 21, 22, 13, 14])
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 4, 5])))
    lock = core.inc_lock_ref(matched.best_match_node)
    assert core.protected_size() == 5
    assert core.evictable_size() == 0
    core.dec_lock_ref(matched.best_match_node, lock.to_dec_params())
    assert core.protected_size() == 0
    assert core.evictable_size() == 5


def test_full_eviction_walk_drains_the_tree():
    core = _tree_core()
    _insert(core, [1, 2, 3], [10, 11, 12])
    _insert(core, [1, 2, 3, 4, 5], [20, 21, 22, 13, 14])
    tracker = {ComponentType.FULL: 0}
    device_frees: dict = {}
    host_frees: dict = {}
    core.evict_device_start(ComponentType.FULL, 100)
    evicted = 0
    while True:
        step = core.evict_device_next_node(ComponentType.FULL, tracker)
        node = step.node_id
        _accumulate_step(step, tracker, device_frees, host_frees)
        if node is None:
            break
        leaf_step = core.evict_device_leaf(node, is_write_back=False)
        _accumulate_step(leaf_step, tracker, device_frees, host_frees)
        evicted += 1
    core.evict_device_end(ComponentType.FULL)
    assert evicted == 2
    assert tracker == {ComponentType.FULL: 5}
    assert core.evictable_size() == 0
    freed = torch.cat(device_frees[ComponentType.FULL])
    assert sorted(freed.tolist()) == [10, 11, 12, 13, 14]
    assert host_frees == {}


def test_insert_suspends_at_a_backup_barrier_through_the_binding():
    core = _tree_core()
    core.write_through_threshold = 2
    core.set_hicache_enabled()
    _insert(core, [1, 2, 3], [10, 11, 12])
    core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))

    step = core.begin_insert(
        InsertParams(
            key=_key([1, 2, 3, 4, 5]),
            value=torch.tensor([20, 21, 22, 13, 14], dtype=torch.int64),
        )
    )
    # The crossing node's backup is a barrier: the walk stays suspended in Rust.
    assert step.result is None
    assert core.has_ongoing_insert()
    assert [type(a).__name__ for a in step.actions] == ["FreeDeviceKV", "BackupKV"]

    done = core.resume_insert()
    assert done.actions == []
    assert done.result is not None
    assert done.result.prefix_len == 3
    assert not core.has_ongoing_insert()
    assert core.end_insert() == []
    core.sanity_check([], [])


def test_configuration_reads_the_locked_rust_state():
    core = _tree_core()
    core._binding.set_hicache_enabled()
    core._binding.set_is_write_back(True)
    core._binding.set_write_through_threshold(7)
    core._binding.set_enable_storage(True)
    assert core.enable_hicache is True
    assert core.is_write_back is True
    assert core.write_through_threshold == 7
    assert core.enable_storage is True

    swa_core = _swa_tree_core()
    swa_core._binding.set_has_swa_host_pool()
    assert swa_core.has_swa_host_pool is True


def test_external_cache_linker_enablement_and_component_guard():
    core = _tree_core()
    assert core.enable_external_cache_linker is False
    core.enable_external_cache_linker = True
    assert core.enable_external_cache_linker is True
    core.enable_external_cache_linker = False
    assert core.enable_external_cache_linker is False

    mamba_core = _mamba_tree_core()
    with pytest.raises(AssertionError, match="(?i)mamba"):
        mamba_core.enable_external_cache_linker = True
    assert mamba_core.enable_external_cache_linker is False


def test_sanity_check_passes_after_the_full_flow():
    core = _tree_core()
    _insert(core, [1, 2, 3], [10, 11, 12])
    _insert(core, [1, 2, 3, 4, 5], [20, 21, 22, 13, 14])
    core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 4, 5])))
    core.sanity_check([], [])


def test_sanity_check_maps_invariant_failures_to_assertion_error():
    core = _tree_core()
    _insert(core, [1], [10])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node

    with pytest.raises(
        AssertionError, match=r"(?s)Sanity check FAILED.*load_back node 8 lock_ref=0"
    ):
        core.sanity_check([], [(8, leaf)])

    # A reported invariant failure does not poison the binding mutex.
    core.sanity_check([], [])


def test_short_value_tensor_raises_value_error():
    binding = _binding()
    params = mem_cache.InsertParamsBinding(
        key=array("q", [1, 2, 3]),
        value=torch.tensor([10, 11], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="shorter than the aligned key length"):
        binding.insert(params)


def test_binding_stays_usable_after_a_failed_insert():
    binding = _binding()
    with pytest.raises(ValueError):
        binding.insert(
            mem_cache.InsertParamsBinding(
                key=array("q", [1, 2, 3]),
                value=torch.tensor([10], dtype=torch.int64),
            )
        )
    result = binding.insert(
        mem_cache.InsertParamsBinding(
            key=array("q", [1, 2, 3]),
            value=torch.tensor([10, 11, 12], dtype=torch.int64),
        )
    )
    assert result.prefix_len == 0
    matched = binding.match_prefix(mem_cache.MatchParamsBinding(array("q", [1, 2, 3])))
    assert matched.device_indices.tolist() == [10, 11, 12]


@pytest.mark.parametrize("prior_hash", ["abcd", "z" * 64])
def test_hash_boundary_rejects_malformed_prior_hash(prior_hash):
    with pytest.raises(ValueError, match="64-character hexadecimal digest"):
        mem_cache.get_hash_str(array("q", [1, 2]), prior_hash, 2)


@pytest.mark.parametrize("token_id", [-1, 1 << 32])
def test_hash_boundary_rejects_token_ids_outside_uint32(token_id):
    with pytest.raises(ValueError, match="does not fit in uint32"):
        mem_cache.get_hash_str(array("q", [token_id]), None, 1)


def test_hash_boundary_rejects_zero_page_size():
    with pytest.raises(ValueError, match="page_size must be positive"):
        mem_cache.get_hash_str(array("q", [1, 2]), None, 0)


def test_binding_rejects_zero_page_size_before_core_construction():
    with pytest.raises(ValueError, match="page_size must be at least 1"):
        _binding(page_size=0)


def test_binding_rejects_unknown_eviction_policy_before_core_construction():
    with pytest.raises(ValueError, match="Unknown eviction policy: clock"):
        _binding(eviction_policy="clock")


def test_poisoned_binding_refuses_to_reuse_the_core():
    binding = _binding()
    root = binding.root_node_handle()

    # Reading a backup spec from the value-less root deliberately trips a native
    # invariant while the binding owns the mutex.
    with pytest.raises(BaseException) as initial_panic:
        binding.build_backup_spec(root)
    assert initial_panic.type.__name__ == "PanicException"

    # The guard must fail closed instead of handing potentially partial state to
    # the next operation through PoisonError::into_inner().
    with pytest.raises(BaseException) as poisoned:
        binding.root_node_handle()
    assert poisoned.type.__name__ == "PanicException"
    assert "Rust TreeCore mutex poisoned" in str(poisoned.value)


def test_extra_key_isolates_namespaces():
    core = _tree_core()
    result = _pump_insert(
        core,
        InsertParams(
            key=RadixKey(array("q", [1, 2, 3]), extra_key="salt"),
            value=torch.tensor([10, 11, 12], dtype=torch.int64),
        ),
    )
    assert result.prefix_len == 0
    salted = core.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), extra_key="salt"))
    )
    assert salted.device_indices.tolist() == [10, 11, 12]
    unsalted = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert unsalted.device_indices.numel() == 0
    other = core.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3]), extra_key="other"))
    )
    assert other.device_indices.numel() == 0
    assert core.prefetch_anchor_info(salted.best_match_node) == ("salt", None)
    assert core.prefetch_anchor_info(core.root_node_handle()) == (None, None)


def test_cache_salt_is_supported_by_all_key_entry_points():
    core = _tree_core()
    tokens = array("q", [1, 2])
    first_key = RadixKey(tokens, extra_key="bc", cache_salt="a")
    second_key = RadixKey(tokens, extra_key="c", cache_salt="ab")
    _pump_insert(
        core,
        InsertParams(key=first_key, value=torch.tensor([10, 11], dtype=torch.int64)),
    )
    _pump_insert(
        core,
        InsertParams(key=second_key, value=torch.tensor([20, 21], dtype=torch.int64)),
    )

    assert core.match_prefix(
        MatchPrefixParams(key=first_key)
    ).device_indices.tolist() == [
        10,
        11,
    ]
    assert core.match_prefix(
        MatchPrefixParams(key=second_key)
    ).device_indices.tolist() == [
        20,
        21,
    ]
    assert (
        core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).device_indices.numel()
        == 0
    )

    host_core = _tree_core()
    host_core.set_hicache_enabled()
    result = host_core.insert_host(
        host_core.root_node_handle(),
        first_key,
        torch.tensor([100, 101], dtype=torch.int64),
        ["h0", "h1"],
    )
    assert result.inserted_host_node is not None
    host_match = host_core.match_prefix(MatchPrefixParams(key=first_key))
    assert host_match.host_hit_length == 2
    assert host_core.prefetch_anchor_info(host_match.best_match_node) == ("bc", "a")
    with pytest.raises(RuntimeError, match="does not match non-root anchor"):
        host_core.insert_host(
            host_match.best_match_node,
            RadixKey(array("q", [3, 4]), extra_key="bc", cache_salt="other"),
            torch.tensor([102, 103], dtype=torch.int64),
            ["h2", "h3"],
        )


def test_session_radix_cache_is_rejected():
    with pytest.raises(ValueError, match="enable-session-radix-cache"):
        _tree_core(enable_session_radix_cache=True)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"tree_components": (ComponentType.FULL, ComponentType.C128)},
            "components: C128",
        ),
        (
            {"component_registry_override": {ComponentType.FULL: object}},
            "component_registry_override",
        ),
    ],
)
def test_unsupported_component_configuration_is_rejected(params, message):
    with pytest.raises(ValueError, match=message):
        _tree_core(**params)


def test_page_size_two_drops_the_ragged_tail():
    core = _tree_core(page_size=2)
    result = _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2, 3, 4, 5]),
            value=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64),
        ),
    )
    assert result.prefix_len == 0
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 4, 5])))
    assert matched.device_indices.tolist() == [10, 11, 12, 13]


def test_insert_value_none_materializes_the_token_ids():
    core = _tree_core()
    result = _pump_insert(core, InsertParams(key=_key([1, 2, 3])))
    assert result.prefix_len == 0
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert matched.device_indices.tolist() == [1, 2, 3]


@pytest.mark.parametrize("swa", [False, True])
def test_empty_match_result_is_root_anchored(swa):
    core = _swa_tree_core() if swa else _tree_core()
    empty = core.empty_match_result
    assert empty.device_indices.numel() == 0
    assert empty.host_hit_length == 0
    probe = core.match_prefix(MatchPrefixParams(key=_key([9])))
    assert empty.best_match_node == probe.best_match_node
    assert empty.last_device_node == probe.last_device_node
    assert empty.last_host_node == probe.last_host_node
    core.dec_lock_ref(empty.best_match_node, DecLockRefParams())
    core.dec_host_lock_ref(empty.best_match_node, DecLockRefParams())
    released = core.dec_swa_lock_only(empty.best_match_node, DecLockRefParams())
    assert not released.device_frees
    assert not released.host_frees
    core.sanity_check([], [])


def test_set_hicache_enabled_marks_the_tree():
    core = _tree_core()
    core.set_hicache_enabled()
    assert core.enable_hicache


def test_hicache_write_through_and_load_back_round_trip():
    core = _tree_core()
    core.set_hicache_enabled()
    _insert(core, [1, 2], [10, 11])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    # Write-through: back the leaf up host-side, then demote it to host-only.
    device_value, comp_xfers = core.build_backup_spec(leaf)
    assert device_value.tolist() == [10, 11]
    assert comp_xfers == {}
    core.mark_write_through_pending([leaf], ack_id=leaf)
    core.commit_backup(leaf, torch.tensor([100, 101], dtype=torch.int64), comp_xfers)
    core.finish_write_through([leaf], leaf)
    tracker = {ComponentType.FULL: 0}
    device_frees, host_frees = {}, {}
    _accumulate_step(core.demote(leaf), tracker, device_frees, host_frees)
    assert tracker[ComponentType.FULL] == 2
    assert [t.tolist() for t in device_frees[ComponentType.FULL]] == [[10, 11]]
    assert core.component_has_host_value_only(leaf, ComponentType.FULL)
    # Load back host -> device; the match then serves device indices again.
    kv_xfer, comp_xfers = core.build_load_back_spec(leaf)
    assert kv_xfer.name == PoolName.KV
    assert kv_xfer.host_indices.tolist() == [100, 101]
    assert kv_xfer.nodes_to_load == [leaf]
    actions = core.commit_load_back(
        leaf, torch.tensor([50, 51], dtype=torch.int64), kv_xfer, comp_xfers
    )
    assert actions == []
    result = core.match_prefix(MatchPrefixParams(key=_key([1, 2])))
    assert result.device_indices.tolist() == [50, 51]
    core.finish_load_back(leaf)
    core.sanity_check([], [])


def test_cache_tracks_one_write_through_ack_across_rust_nodes():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    core = _tree_core()
    _insert(core, [1], [10])
    _insert(core, [1, 2], [10, 11])
    parent = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    cache = SimpleNamespace(tree_core=core, ongoing_write_through={})

    # Child-first in, ancestors-first out: the publish side links every store
    # event to its parent, and component transfer order is not tree order.
    UnifiedRadixCache._track_write_through_node(
        cache,
        leaf,
        lock_params=None,
        publish_node_ids=[leaf, parent],
    )

    assert cache.ongoing_write_through[leaf].publish_node_ids == [parent, leaf]
    core.finish_write_through([parent, leaf], ack_id=leaf)
    core.sanity_check([], [])


def test_invalid_demote_states_raise_assertion_error():
    core = _tree_core()
    core.set_hicache_enabled()
    _insert(core, [1], [10])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node

    with pytest.raises(AssertionError):
        core.demote(leaf)

    core.commit_backup(leaf, torch.tensor([100], dtype=torch.int64), {})
    tracker = {ComponentType.FULL: 0}
    _accumulate_step(core.demote(leaf), tracker, {}, {})
    with pytest.raises(AssertionError):
        core.demote(leaf)


def test_write_through_load_back_is_unpinned_and_refreshes_duplicate_tracking():
    core = _tree_core()
    core.set_hicache_enabled()
    root = core.root_node_handle()
    leaf = core.insert_host(
        root, _key([1]), torch.tensor([100], dtype=torch.int64), ["h0"]
    ).inserted_host_node
    assert leaf is not None

    kv_xfer, comp_xfers = core.build_load_back_spec(leaf)
    core.commit_load_back(
        leaf, torch.tensor([50], dtype=torch.int64), kv_xfer, comp_xfers
    )

    # Write-through load-back does not pin Full KV against device eviction.
    core.evict_device_start(ComponentType.FULL, 1)
    candidate = core.evict_device_next_node(
        ComponentType.FULL, {ComponentType.FULL: 0}
    ).node_id
    core.evict_device_end(ComponentType.FULL)
    assert candidate == leaf

    # insert_host created no stale duplicate entry, so this checks the ack refresh.
    core.finish_load_back(leaf)
    core.sanity_check([], [])


def test_insert_host_extends_the_backuped_path():
    core = _tree_core()
    core.set_hicache_enabled()
    _insert(core, [1], [10])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node
    core.commit_backup(leaf, torch.tensor([100], dtype=torch.int64), {})
    result = core.insert_host(
        leaf, _key([2, 3]), torch.tensor([101, 102], dtype=torch.int64), []
    )
    assert result.prefix_len == 0
    assert result.total_len == 2
    assert result.inserted_host_node is not None
    match = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert match.host_hit_length == 2


def test_insert_host_reports_a_dropped_write_through_suffix():
    core = _tree_core()
    _insert(core, [1], [10])
    parent = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node

    result = core.insert_host(
        parent, _key([2]), torch.tensor([100], dtype=torch.int64), ["h0"]
    )

    assert result.prefix_len == 0
    assert result.total_len == 1
    assert result.inserted_host_node is None
    assert result.host_insert_dropped


@pytest.mark.parametrize(
    "swa, missing_receipt", [(False, False), (True, False), (True, True)]
)
def test_host_lock_refs_round_trip(swa, missing_receipt):
    core = _swa_tree_core() if swa else _tree_core()
    core.set_hicache_enabled()
    # The Full-only insertion is not a resumable SWA match until SWA is set below.
    leaf = _insert(core, [1], [10]).last_device_node
    component_transfers = {}
    if swa:
        core.has_swa_host_pool = True
        core.set_component_device_value(
            leaf, ComponentType.SWA, torch.tensor([20], dtype=torch.int64)
        )
        component_transfers[ComponentType.SWA] = [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.tensor([200], dtype=torch.int64),
            )
        ]
    core.commit_backup(
        leaf, torch.tensor([100], dtype=torch.int64), component_transfers
    )
    core.mark_write_through_pending([leaf], leaf)
    core.finish_write_through([leaf], leaf)
    device_lock = core.inc_lock_ref(leaf)
    host_lock = core.inc_host_lock_ref(leaf)
    assert not host_lock.component_lock_uuids
    assert not device_lock.component_host_lock_uuids
    if swa:
        assert host_lock.component_host_lock_uuids == {ComponentType.SWA: None}
    params = host_lock.to_dec_params()
    if missing_receipt:
        params.component_host_lock_uuids.clear()
        with pytest.raises(RuntimeError, match="no entry found for key") as error:
            core.dec_host_lock_ref(leaf, params)
        assert type(error.value.__cause__).__name__ == "PanicException"
        return  # A Rust ownership violation poisons the core.
    core.dec_host_lock_ref(leaf, params)
    if swa:
        assert core.swa_protected_size() == 1
    core.dec_lock_ref(leaf, device_lock.to_dec_params())
    core.sanity_check([], [])


def test_drive_host_eviction_frees_the_demoted_leaf():
    core = _tree_core()
    core.set_hicache_enabled()
    _insert(core, [1], [10])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node
    core.commit_backup(leaf, torch.tensor([100], dtype=torch.int64), {})
    _accumulate_step(core.demote(leaf), {}, {}, {})
    tracker = {ComponentType.FULL: 0}
    device_frees, host_frees = {}, {}
    _accumulate_step(
        core.drive_host_eviction(ComponentType.FULL, 1),
        tracker,
        device_frees,
        host_frees,
    )
    assert tracker[ComponentType.FULL] == 1
    assert [t.tolist() for t in host_frees[ComponentType.FULL]] == [[100]]
    # The host-only leaf is gone: the key no longer matches anywhere.
    result = core.match_prefix(MatchPrefixParams(key=_key([1])))
    assert result.host_hit_length == 0
    core.sanity_check([], [])


def test_host_duplicate_reclaim_override_preserves_normal_host_eviction():
    core = _tree_core()
    core.set_hicache_enabled()
    core.is_write_back = True
    duplicate = _insert(core, [1], [10]).last_device_node
    core.commit_backup(duplicate, torch.tensor([100], dtype=torch.int64), {})
    core.mark_write_through_pending([duplicate], duplicate)
    core.finish_write_through([duplicate], duplicate)
    core.insert_host(
        core.root_node_handle(),
        _key([2]),
        torch.tensor([200], dtype=torch.int64),
        ["h0"],
    )

    host_frees = {}
    with envs.SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM.override(True):
        _accumulate_step(
            core.drive_host_eviction(ComponentType.FULL, 1), {}, {}, host_frees
        )
    assert torch.cat(host_frees[ComponentType.FULL]).tolist() == [200]
    assert core.is_backuped(duplicate)
    core.sanity_check([], [])

    # Read the override on every eviction call, including after construction.
    host_frees = {}
    with envs.SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM.override(False):
        _accumulate_step(
            core.drive_host_eviction(ComponentType.FULL, 1), {}, {}, host_frees
        )
    assert torch.cat(host_frees[ComponentType.FULL]).tolist() == [100]
    assert not core.is_backuped(duplicate)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("backup_again", [False, True])
def test_full_host_duplicates_preserve_backup_order(backend, backup_again):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend):
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                page_size=1,
                tree_components=(ComponentType.FULL,),
            )
        )
    core = cache.tree_core
    core.set_hicache_enabled()
    core.is_write_back = True
    nodes = []
    for index in range(3):
        node = _insert(core, [index + 1], [index + 101]).last_device_node
        core.commit_backup(node, torch.tensor([index + 1001]), {})
        core.mark_write_through_pending([node], node)
        core.finish_write_through([node], node)
        nodes.append(node)

    expected = [1001, 1002, 1003]
    if backup_again:
        expected.append(2001)
    for index, expected_slot in enumerate(expected):
        host_frees = {}
        _accumulate_step(
            core.drive_host_eviction(ComponentType.FULL, 1), {}, {}, host_frees
        )
        assert torch.cat(host_frees[ComponentType.FULL]).tolist() == [expected_slot]
        if index == 0 and backup_again:
            # A new backup rejoins after existing duplicates; it must not take
            # the removed entry's old position or reorder the surviving nodes.
            core.commit_backup(nodes[0], torch.tensor([2001]), {})
            core.mark_write_through_pending([nodes[0]], nodes[0])
            core.finish_write_through([nodes[0]], nodes[0])
    assert all(not core.is_backuped(node) for node in nodes)
    core.sanity_check([], [])


@pytest.mark.parametrize("backed_up", [False, True])
def test_device_eviction_counts_only_full_tokens_without_a_host_copy(backed_up):
    core = _mamba_tree_core()
    core.set_hicache_enabled()
    leaf = _mamba_insert(core, [1, 2, 3], [10, 11, 12], 7).last_device_node
    if backed_up:
        core.commit_backup(leaf, torch.tensor([100, 101, 102]), {})

    step = core.evict_device_leaf(leaf, is_write_back=False)
    unbacked_tokens = step.unbacked_tokens
    tracker = {}
    _accumulate_step(step, tracker, {}, {})
    assert tracker == {ComponentType.FULL: 3, ComponentType.MAMBA: 1}
    assert unbacked_tokens == (0 if backed_up else 3)

    # Counters belong to a single step, never to the next eviction walk.
    core.evict_device_start(ComponentType.FULL, 1)
    step = core.evict_device_next_node(ComponentType.FULL, {})
    assert step.unbacked_tokens == 0
    _accumulate_step(step, {}, {}, {})
    core.evict_device_end(ComponentType.FULL)
    core.sanity_check([], [])


def test_events_disabled_take_events_is_empty():
    core = _tree_core()
    _insert(core, [1, 2], [10, 11])
    assert core.take_events() == []


def test_insert_emits_block_stored_events():
    core = _tree_core(enable_kv_cache_events=True, page_size=2)
    _insert(core, [1, 2, 7, 8], [10, 11, 12, 13])
    hashes = [
        hash_str_to_int64(h)
        for h in mem_cache.get_hash_str(array("q", [1, 2, 7, 8]), None, 2)
    ]
    assert core.take_events() == [
        BlockStored(
            block_hashes=hashes,
            parent_block_hash=None,
            token_ids=[1, 2, 7, 8],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
        ),
    ]
    assert core.take_events() == []


def test_salted_events_match_python_hash_and_metadata_contract():
    core = _tree_core(enable_kv_cache_events=True, page_size=2)
    key = RadixKey(array("q", [1, 2, 7, 8]), cache_salt="tenant-a")
    _pump_insert(
        core,
        InsertParams(
            key=key,
            value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        ),
    )
    seed = hashlib.sha256(b"sglang-cache-salt-v1\0tenant-a").hexdigest()
    hashes = [
        hash_str_to_int64(value)
        for value in mem_cache.get_hash_str(array("q", [1, 2, 7, 8]), seed, 2)
    ]
    assert core.take_events() == [
        BlockStored(
            block_hashes=hashes,
            parent_block_hash=None,
            token_ids=[1, 2, 7, 8],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            cache_salt="tenant-a",
        )
    ]

    tracker = {ComponentType.FULL: 0}
    core.evict_device_start(ComponentType.FULL, 4)
    candidate = core.evict_device_next_node(ComponentType.FULL, tracker).node_id
    assert candidate is not None
    evicted = core.evict_device_leaf(candidate, is_write_back=False)
    evicted.device_frees.clear()
    evicted.host_frees.clear()
    core.evict_device_end(ComponentType.FULL)
    assert core.take_events() == [
        BlockRemoved(block_hashes=hashes, medium=StorageMedium.GPU)
    ]


def test_salted_eagle_events_match_the_bigram_hash_contract():
    core = _tree_core(enable_kv_cache_events=True, page_size=2, is_eagle=True)
    raw_tokens = array("q", [1, 2, 3, 4, 5])
    key = RadixKey(raw_tokens, cache_salt="tenant-a", is_bigram=True)
    _pump_insert(
        core,
        InsertParams(
            key=key,
            value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        ),
    )
    seed = hashlib.sha256(b"sglang-cache-salt-v1\0tenant-a").hexdigest()
    hashes = [
        hash_str_to_int64(value)
        for value in mem_cache.get_hash_str(raw_tokens, seed, 2, is_bigram=True)
    ]
    assert core.take_events() == [
        BlockStored(
            block_hashes=hashes,
            parent_block_hash=None,
            token_ids=[(1, 2), (2, 3), (3, 4), (4, 5)],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            cache_salt="tenant-a",
        )
    ]


def test_demote_emits_block_removed():
    core = _tree_core(enable_kv_cache_events=True)
    core.set_hicache_enabled()
    _insert(core, [1, 2], [10, 11])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    core.commit_backup(leaf, torch.tensor([100, 101], dtype=torch.int64), {})
    core.take_events()
    _accumulate_step(core.demote(leaf), {}, {}, {})
    hashes = [
        hash_str_to_int64(h)
        for h in mem_cache.get_hash_str(array("q", [1, 2]), None, 1)
    ]
    assert core.take_events() == [
        BlockRemoved(block_hashes=hashes, medium=StorageMedium.GPU)
    ]


def test_all_cleared_event_crosses_the_binding():
    core = _tree_core(enable_kv_cache_events=True)
    core._record_all_cleared_event()
    assert core.take_events() == [AllBlocksCleared()]


def test_match_result_mamba_fields_are_inert_without_mamba():
    core = _tree_core()
    _insert(core, [1, 2], [10, 11])
    result = core.match_prefix(MatchPrefixParams(key=_key([1, 2])))
    assert result.mamba_host_hit_length == 0
    assert result.mamba_branching_seqlen is None


def test_storage_backup_spec_round_trips_the_backuped_node():
    core = _tree_core(page_size=2)
    core.set_hicache_enabled()
    core.enable_storage = True
    key = RadixKey(
        array("q", [1, 2, 7, 8]), extra_key="adapter-a", cache_salt="tenant-a"
    )
    for length in (2, 4):
        _pump_insert(
            core,
            InsertParams(key=key[:length], value=torch.arange(10, 10 + length)),
        )
    parent = core.match_prefix(MatchPrefixParams(key=key[:2])).best_match_node
    child = core.match_prefix(MatchPrefixParams(key=key)).best_match_node
    core.commit_backup(parent, torch.tensor([100, 101], dtype=torch.int64), {})
    core.commit_backup(child, torch.tensor([102, 103], dtype=torch.int64), {})

    spec = core.build_storage_backup_spec(child, pass_prefix_keys=True)
    assert spec.host_value.tolist() == [102, 103]
    assert spec.token_ids == array("q", [7, 8])
    hashes = get_storage_hash_str(key, page_size=2)
    assert spec.prefix_keys == hashes[:1]
    assert spec.hash_value == hashes[1:]
    assert spec.comp_xfers == {}


def test_prefetch_node_accessors_round_trip():
    core = _tree_core(page_size=2)
    core.set_hicache_enabled()
    core.enable_storage = True
    _insert(core, [1, 2], [10, 11])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node

    assert not core.is_backuped(leaf)
    assert not core.is_root(leaf)
    assert (
        core.get_last_hash_value(leaf)
        == (mem_cache.get_hash_str(array("q", [1, 2]), None, 2)[-1])
    )
    assert core.get_prefix_hash_values(leaf) == []

    core.commit_backup(leaf, torch.tensor([100, 101], dtype=torch.int64), {})
    assert core.is_backuped(leaf)


def test_storage_backup_spec_is_none_for_an_unbackuped_node():
    core = _tree_core()
    _insert(core, [1, 2], [10, 11])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    assert core.build_storage_backup_spec(leaf, pass_prefix_keys=False) is None


def test_build_hicache_transfers_routes_the_backup_storage_phase():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _tree_core()
    _insert(core, [1, 2], [10, 11])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    # The FULL component has no storage sidecar; the kv transfer is built by
    # the controller from the spec instead.
    assert (
        core.build_hicache_transfers(
            ComponentType.FULL, leaf, CacheTransferPhase.BACKUP_STORAGE
        )
        is None
    )


def _canary_rows(core, *, unlocked_only=False, swa_resident_only=False):
    walk = core.walk_for_kv_canary(
        unlocked_only=unlocked_only, swa_resident_only=swa_resident_only
    )
    return sorted(
        zip(
            walk.slot_indices.tolist(),
            walk.positions.tolist(),
            walk.prev_slot_indices.tolist(),
        )
    )


def test_walk_for_kv_canary_emits_chained_rows():
    core = _tree_core()
    _insert(core, [1, 2, 3], [10, 11, 12])
    _insert(core, [1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert _canary_rows(core) == [
        (10, 0, -1),
        (11, 1, 10),
        (12, 2, 11),
        (13, 3, 12),
        (14, 4, 13),
    ]


def test_walk_for_kv_canary_unlocked_only_skips_locked_nodes_but_keeps_the_chain():
    core = _tree_core()
    _insert(core, [1, 2, 3], [10, 11, 12])
    _insert(core, [1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    locked = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3]))).best_match_node
    core.inc_lock_ref(locked)
    assert _canary_rows(core, unlocked_only=True) == [(13, 3, 12), (14, 4, 13)]


def test_walk_for_kv_canary_skips_demoted_nodes():
    core = _tree_core()
    core.set_hicache_enabled()
    _insert(core, [1, 2], [10, 11])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    core.commit_backup(leaf, torch.tensor([100, 101], dtype=torch.int64), {})
    _accumulate_step(core.demote(leaf), {}, {}, {})
    assert _canary_rows(core) == []


def test_walk_for_kv_canary_swa_filter_is_inert_without_the_swa_component():
    core = _tree_core()
    _insert(core, [1, 2], [10, 11])
    assert _canary_rows(core, swa_resident_only=True) == [(10, 0, -1), (11, 1, 10)]


def test_empty_keys_cross_the_binding():
    assert mem_cache.MatchParamsBinding(array("q")).key == []
    assert mem_cache.MatchParamsBinding([]).key == []


def test_empty_cache_salt_uses_the_default_namespace_at_the_binding():
    binding = _binding()
    binding.insert(
        mem_cache.InsertParamsBinding(
            key=array("q", [1]),
            value=torch.tensor([10], dtype=torch.int64),
            cache_salt="",
        )
    )
    result = binding.match_prefix(mem_cache.MatchParamsBinding(array("q", [1])))
    assert result.device_indices.tolist() == [10]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_core_resolves_the_current_device():
    core = _tree_core(
        token_to_kv_pool_allocator=SimpleNamespace(device="cuda"),
    )
    assert core.device == torch.device("cuda", torch.cuda.current_device())
    result = _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2, 3]),
            value=torch.tensor([10, 11, 12], dtype=torch.int64, device=core.device),
        ),
    )
    assert result.prefix_len == 0
    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3])))
    assert matched.device_indices.device == core.device
    assert matched.device_indices.tolist() == [10, 11, 12]
    # The value=None fallback also lands on the resolved device.
    fallback = _pump_insert(core, InsertParams(key=_key([7, 8])))
    assert fallback.prefix_len == 0
    matched = core.match_prefix(MatchPrefixParams(key=_key([7, 8])))
    assert matched.device_indices.tolist() == [7, 8]


def test_unsupported_component_sets_are_rejected():
    with pytest.raises(ValueError, match="component sets are supported"):
        mem_cache.RustUnifiedTreeCoreBinding(
            mem_cache.TreeCoreInitParamsBinding(), [int(ComponentType.SWA)]
        )
    with pytest.raises(ValueError, match="component sets are supported"):
        mem_cache.RustUnifiedTreeCoreBinding(
            mem_cache.TreeCoreInitParamsBinding(), [int(ComponentType.MAMBA)]
        )


def test_swa_requires_the_sliding_window_size():
    with pytest.raises(ValueError, match="requires swa_sliding_window_size"):
        mem_cache.RustUnifiedTreeCoreBinding(
            mem_cache.TreeCoreInitParamsBinding(),
            [int(ComponentType.FULL), int(ComponentType.SWA)],
        )


def test_enable_hicache_constructs():
    mem_cache.RustUnifiedTreeCoreBinding(
        mem_cache.TreeCoreInitParamsBinding(enable_hicache=True),
        [int(ComponentType.FULL)],
    )


def test_is_write_back_constructs():
    mem_cache.RustUnifiedTreeCoreBinding(
        mem_cache.TreeCoreInitParamsBinding(is_write_back=True),
        [int(ComponentType.FULL)],
    )


def test_write_back_eviction_backs_up_then_drop_subtree_falls_back():
    core = _tree_core()
    core.is_write_back = True
    core.set_hicache_enabled()
    _insert(core, [1, 2], [10, 11])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    tracker = {ComponentType.FULL: 0}
    device_frees, host_frees = {}, {}
    # The unbacked leaf earns a backup action; nothing is freed yet.
    leaf_step = core.evict_device_leaf(leaf, is_write_back=True)
    backup = leaf_step.backup_kv
    _accumulate_step(leaf_step, tracker, device_frees, host_frees)
    assert backup == BackupKV([leaf])
    assert device_frees == {} and host_frees == {}
    # Host pressure: the backup failed, so the subtree drop keeps eviction moving.
    drop_step = core.drop_subtree_no_host(leaf)
    dropped = drop_step.is_dropped
    _accumulate_step(drop_step, tracker, device_frees, host_frees)
    assert dropped
    assert tracker[ComponentType.FULL] == 2
    assert [t.tolist() for t in device_frees[ComponentType.FULL]] == [[10, 11]]
    result = core.match_prefix(MatchPrefixParams(key=_key([1, 2])))
    assert result.device_indices.numel() == 0
    core.sanity_check([], [])


# ==== SWA wiring ====


def _swa_tree_core(window: int = 8, **params_overrides) -> RustUnifiedTreeCore:
    return _tree_core(
        tree_components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=window,
        **params_overrides,
    )


def _swa_transfer_core(
    backend, unified=False, mamba=False, page_size=1, window=16, is_eagle=False
):
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
        UnifiedSWAAllocatorBase,
    )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
    from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
    from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    allocator = Mock(
        spec=UnifiedSWAAllocatorBase if unified else SWATokenToKVPoolAllocator
    )
    allocator.device = torch.device("cpu")
    allocator.swa_req_ring = False
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=Mock(spec=HybridReqToTokenPool) if mamba else None,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        is_eagle=is_eagle,
        tree_components=(ComponentType.FULL, ComponentType.SWA)
        + ((ComponentType.MAMBA,) if mamba else ()),
        sliding_window_size=window,
    )
    with get_context().override_server_args(
        _mamba_cache_chunk_size=256, mamba_max_states_per_path=-1
    ):
        if backend == "rust":
            core = RustUnifiedTreeCore(params)
        else:
            cache = SimpleNamespace(
                token_to_kv_pool_allocator=allocator, enable_session_radix_cache=False
            )
            components = {
                ComponentType.FULL: FullComponent(cache, params),
                ComponentType.SWA: SWAComponent(cache, params),
            }
            if mamba:
                components[ComponentType.MAMBA] = MambaComponent(cache, params)
            core = UnifiedTreeCore(params, components)
            cache.tree_core = core
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    return core, allocator


def _hybrid_transfer_order_fixture(backend):
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core, _ = _swa_transfer_core(backend, mamba=True)
    root = core.root_node_handle()
    a = core.insert_host(root, _key([10, 11]), torch.tensor([100, 101]), ["a0", "a1"])
    b = core.insert_host(root, _key([20, 21]), torch.tensor([200, 201]), ["b0", "b1"])
    # The pools can have different LRU order: SWA oldest=A, Mamba oldest=B.
    for result, component, pool, indices in (
        (a, ComponentType.SWA, PoolName.SWA, [1, 2]),
        (b, ComponentType.SWA, PoolName.SWA, [3, 4]),
        (b, ComponentType.MAMBA, PoolName.MAMBA, [12]),
        (a, ComponentType.MAMBA, PoolName.MAMBA, [11]),
    ):
        actions = []
        core.commit_hicache_transfers(
            root,
            CacheTransferPhase.PREFETCH,
            {component: [PoolTransfer(name=pool, host_indices=torch.tensor(indices))]},
            cache_actions=actions,
            insert_result=result,
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=2, extra_pool_hit_pages={pool: len(indices)}
            ),
        )
        assert actions == []
    c = _mamba_insert(core, [30, 31], [300, 301], 13)
    for action in c.cache_actions:
        if isinstance(action, SWARebuild):
            core.set_component_device_value(
                action.node_id, ComponentType.SWA, action.source_value
            )
    core.sanity_check([], [])
    return core, a.inserted_host_node, c.last_device_node


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("planner", ["backup", "load_back", "storage"])
def test_hybrid_transfer_planners_preserve_component_order(backend, planner):
    core, host_node, device_node = _hybrid_transfer_order_fixture(backend)
    # Each native plan creates a fresh map. Its randomized iteration must not
    # leak into allocation/reclamation order across calls or TP ranks.
    for _ in range(64):
        if planner == "backup":
            _, transfers = core.build_backup_spec(device_node)
        elif planner == "load_back":
            _, transfers = core.build_load_back_spec(host_node)
        else:
            transfers = core.build_storage_backup_spec(host_node, False).comp_xfers
        assert list(transfers) == [ComponentType.SWA, ComponentType.MAMBA]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_hybrid_backup_pool_pressure_preserves_host_victim(backend):
    from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry

    core, _, device_node = _hybrid_transfer_order_fixture(backend)
    _, transfers = core.build_backup_spec(device_node)
    free_slots = {
        ct: [] for ct in (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
    }
    freed_full = []

    def allocate(count, component):
        slots = free_slots[component]
        if len(slots) < count:
            return None
        indices = torch.tensor(slots[:count], dtype=torch.int64)
        del slots[:count]
        return indices

    def release(indices, component):
        free_slots[component].extend(indices.tolist())
        return len(indices)

    def reclaim(count, component):
        result = core.drive_host_eviction(component, count)
        assert not result.device_frees
        for ct, tensors in result.host_frees.items():
            for indices in tensors:
                release(indices, ct)
                if ct == ComponentType.FULL:
                    freed_full.extend(indices.tolist())
        result.host_frees.clear()

    entries = []
    for ct, name in (
        (ComponentType.FULL, PoolName.KV),
        (ComponentType.SWA, PoolName.SWA),
        (ComponentType.MAMBA, PoolName.MAMBA),
    ):
        # Simulate only physical host allocation; tree eviction and the host
        # group's allocation/reclaim/rollback flow are the production code.
        pool = Mock(can_use_write_back_jit=False)
        pool.alloc.side_effect = lambda count, ct=ct: allocate(count, ct)
        pool.free.side_effect = lambda indices, ct=ct: release(indices, ct)
        entries.append(
            PoolEntry(
                name=name,
                host_pool=pool,
                device_pool=None,
                layer_mapper=lambda _: 0,
                host_evict_fn=lambda count, ct=ct: reclaim(count, ct),
            )
        )
    resolved = HostPoolGroup(entries).resolve_host_transfers(
        [transfer for xfers in transfers.values() for transfer in xfers]
    )
    assert resolved is not None
    # #40512 allocates by pool name, so Mamba pressure frees B's complete host
    # leaf first, satisfying both side pools regardless of transfer-map order.
    assert freed_full == [200, 201]
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("guard", ["none", "host_lock", "pending_dma"])
def test_swa_host_pressure_retains_device_resident_backups(backend, guard):
    core, allocator = _swa_transfer_core(backend)
    allocator.translate_loc_from_full_to_swa.side_effect = lambda values: values + 100
    core.is_write_back = True
    nodes = []
    for tokens, values in (([1, 2], [10, 11]), ([1, 2, 3], [20, 21, 12])):
        result = _insert(core, tokens, values)
        for action in result.cache_actions:
            if isinstance(action, SWARebuild):
                core.set_component_device_value(
                    action.node_id,
                    ComponentType.SWA,
                    allocator.translate_loc_from_full_to_swa(action.source_value),
                )
            else:
                assert isinstance(action, (FreeDeviceKV, FreeDeviceKVFullOnly))
        nodes.append(result.last_device_node)
    for node in nodes:
        full, transfers = core.build_backup_spec(node)
        for component_transfers in transfers.values():
            for transfer in component_transfers:
                transfer.host_indices = transfer.device_indices + 1000
        core.commit_backup(node, full + 1000, transfers)
        core.mark_write_through_pending([node], node)
        core.finish_write_through([node], node)

    if guard == "host_lock":
        lock = core.inc_host_lock_ref(nodes[-1])
    elif guard == "pending_dma":
        core.mark_write_through_pending(nodes, nodes[-1])
    tracker, device_frees, host_frees = {}, {}, {}
    _accumulate_step(
        core.drive_host_eviction(ComponentType.SWA, 2),
        tracker,
        device_frees,
        host_frees,
    )
    assert tracker.get(ComponentType.SWA, 0) == 0
    assert not device_frees and not host_frees
    if guard == "host_lock":
        core.dec_host_lock_ref(nodes[-1], lock.to_dec_params())
    elif guard == "pending_dma":
        core.finish_write_through(nodes, nodes[-1])

    # Demotion makes the preserved SWA host copy the only copy. Ordinary host
    # eviction must still reclaim that host-only tree under pressure.
    for node in reversed(nodes):
        _accumulate_step(core.demote(node), {}, {}, {})
        assert core.component_has_host_value_only(node, ComponentType.SWA)
    host_frees = {}
    _accumulate_step(core.drive_host_eviction(ComponentType.SWA, 3), {}, {}, host_frees)
    assert sum(t.numel() for t in host_frees[ComponentType.SWA]) == 3
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("resident_full", [(), (0,), (0, 1, 2, 3)])
def test_swa_load_back_preserves_full_anchors_across_holes(backend, resident_full):
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.pool_host.group import PoolEntry

    core, _ = _swa_transfer_core(backend)
    nodes = []
    boundaries = (0, 2, 5, 6, 8)
    for start, end in pairwise(boundaries):
        node = _insert(
            core, list(range(end)), list(range(10, 10 + end))
        ).last_device_node
        nodes.append(node)
        core.set_component_device_value(
            node, ComponentType.SWA, torch.arange(50 + start, 50 + end)
        )
        core.commit_backup(
            node,
            torch.arange(100 + start, 100 + end),
            {
                ComponentType.SWA: [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=torch.arange(300 + start, 300 + end),
                        nodes_to_load=[node],
                    )
                ]
            },
        )
    for node in reversed(nodes):
        _accumulate_step(core.demote(node), {}, {}, {})
    # SWA is resident in the three-token middle node, leaving a hole in its
    # load list even when that node still needs a Full load.
    core.set_component_device_value(nodes[1], ComponentType.SWA, torch.arange(52, 55))
    for index in resident_full:
        start, end = boundaries[index : index + 2]
        core.commit_load_back(
            nodes[index],
            torch.arange(10 + start, 10 + end),
            PoolTransfer(
                name=PoolName.KV,
                host_indices=torch.arange(100 + start, 100 + end),
                nodes_to_load=[nodes[index]],
            ),
            {},
        )
        core.finish_load_back(nodes[index])

    kv, auxiliary = core.build_load_back_spec(nodes[-1])
    (swa,) = auxiliary[ComponentType.SWA]
    assert swa.nodes_to_load == [nodes[0], nodes[2], nodes[3]]
    assert swa.host_indices.tolist() == [300, 301, 305, 306, 307]
    expected = {
        (): [slice(0, 2), slice(5, 6), slice(6, 8)],
        (0,): [torch.tensor([10, 11]), slice(3, 4), slice(4, 6)],
        (0, 1, 2, 3): [
            torch.tensor([10, 11]),
            torch.tensor([15]),
            torch.tensor([16, 17]),
        ],
    }[resident_full]
    assert swa.anchor_index_parts is not None
    assert len(swa.anchor_index_parts) == len(expected)
    for actual, wanted in zip(swa.anchor_index_parts, expected):
        if isinstance(wanted, slice):
            assert actual == wanted
        else:
            torch.testing.assert_close(actual, wanted)

    # Exercise the consumer too: new Full rows and resident virtual IDs must
    # bind the same five SWA rows, without consuming the resident-SWA hole.
    bind = Mock(side_effect=lambda indices: indices + 1000)
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        entry_map={
            PoolName.SWA: PoolEntry(
                name=PoolName.SWA,
                host_pool=SimpleNamespace(),
                device_pool=SimpleNamespace(),
                layer_mapper=lambda i: i,
                device_indices_from_anchor_fn=bind,
                device_free_fn=Mock(),
            )
        }
    )
    loaded = torch.arange(200, 200 + len(kv.host_indices))
    assert controller._resolve_device_transfers([swa], loaded) is not None
    expected_bound = {
        (): [1200, 1201, 1205, 1206, 1207],
        (0,): [1010, 1011, 1203, 1204, 1205],
        (0, 1, 2, 3): [1010, 1011, 1015, 1016, 1017],
    }[resident_full]
    assert swa.device_indices.tolist() == expected_bound
    assert bind.call_count == 1


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("entrypoint", ["backup_spec", "component_transfer"])
def test_swa_backup_resolves_relocated_full_virtual_ids(backend, unified, entrypoint):
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core, allocator = _swa_transfer_core(backend, unified=unified)
    mapping = torch.arange(64) + 100
    allocator.translate_loc_from_full_to_swa.side_effect = lambda indices: mapping[
        indices
    ]
    values = [11, 7, 20, 4, 9, 13]
    nodes = []
    for start, end in ((0, 2), (2, 5), (5, 6)):
        node = _insert(core, list(range(end)), values[:end]).last_device_node
        nodes.append(node)
        core.set_component_device_value(
            node, ComponentType.SWA, mapping[torch.tensor(values[start:end])].clone()
        )
    # A hosted middle node is excluded from the next SWA backup.
    core.commit_backup(
        nodes[1],
        torch.arange(302, 305),
        {
            ComponentType.SWA: [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.arange(402, 405),
                    nodes_to_load=[nodes[1]],
                )
            ]
        },
    )
    # Relocation changes kernel-facing SWA addresses while tree-owned Full
    # virtual IDs and cached SWA physical snapshots stay unchanged.
    mapping[torch.tensor([11, 7, 13])] = torch.tensor([511, 407, 613])
    allocator.translate_loc_from_full_to_swa.reset_mock()
    if entrypoint == "backup_spec":
        full, auxiliary = core.build_backup_spec(nodes[-1])
        assert full.tolist() == [13]
        (swa,) = auxiliary[ComponentType.SWA]
    else:
        (swa,) = core.build_hicache_transfers(
            ComponentType.SWA, nodes[-1], CacheTransferPhase.BACKUP_HOST
        )
    assert swa.nodes_to_load == [nodes[0], nodes[2]]
    assert swa.device_indices.dtype == torch.int64
    assert swa.device_indices.tolist() == (
        [511, 407, 613] if unified else [111, 107, 113]
    )
    if unified:
        allocator.translate_loc_from_full_to_swa.assert_called_once()
        assert allocator.translate_loc_from_full_to_swa.call_args.args[0].tolist() == [
            11,
            7,
            13,
        ]
    else:
        allocator.translate_loc_from_full_to_swa.assert_not_called()
    assert core.get_component_device_value(nodes[0], ComponentType.SWA).tolist() == [
        111,
        107,
    ]


def test_swa_core_rejects_a_missing_or_non_positive_window():
    """A zero window can never fill, so no boundary uuid would ever be stamped;
    the adapter refuses it up front instead of letting the core misbehave later."""
    for window in (None, 0, -1):
        with pytest.raises(ValueError, match="positive sliding_window_size"):
            _swa_tree_core(window=window)


def test_write_back_load_back_ignores_auxiliary_nodes_for_pending_ownership():
    core = _swa_tree_core(window=4)
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    core.is_write_back = True
    root = core.root_node_handle()
    shared = core.insert_host(
        root, _key([1]), torch.tensor([100], dtype=torch.int64), ["h0"]
    ).inserted_host_node
    anchor = core.insert_host(
        root,
        _key([1, 2]),
        torch.tensor([100, 101], dtype=torch.int64),
        ["h0", "h1"],
    ).inserted_host_node
    assert shared is not None and anchor is not None

    core.commit_backup(
        shared,
        torch.empty(0, dtype=torch.int64),
        {
            ComponentType.SWA: [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.tensor([200], dtype=torch.int64),
                )
            ]
        },
    )
    core.commit_load_back(
        shared,
        torch.tensor([10], dtype=torch.int64),
        PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([100], dtype=torch.int64),
            nodes_to_load=[shared],
        ),
        {},
    )

    # The first Full load is genuinely pinned while awaiting its own ack.
    core.evict_device_start(ComponentType.FULL, 1)
    candidate = core.evict_device_next_node(
        ComponentType.FULL, {ComponentType.FULL: 0}
    ).node_id
    core.evict_device_end(ComponentType.FULL)
    assert candidate is None

    # Loading shared's SWA under another anchor must not claim its Full pin.
    core.commit_load_back(
        anchor,
        torch.tensor([11], dtype=torch.int64),
        PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([101], dtype=torch.int64),
            nodes_to_load=[anchor],
        ),
        {
            ComponentType.SWA: [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.tensor([200], dtype=torch.int64),
                    device_indices=torch.tensor([20], dtype=torch.int64),
                    nodes_to_load=[shared],
                )
            ]
        },
    )
    assert core.get_component_device_value(shared, ComponentType.SWA).tolist() == [20]

    core.finish_load_back(anchor)
    core.finish_load_back(shared)
    core.sanity_check([], [])


def _swa_cache(window: int = 8, page_size: int = 1, backend="rust", ring=False):
    """A real UnifiedRadixCache and SWA allocator, defaulting to the Rust core."""
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", page_size=page_size)
    )
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    kv_pool = SWAKVPool(
        size=64,
        size_swa=64,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        device="cpu",
    )
    if ring:
        kv_pool.swa_req_ring_size = window
    allocator = SWATokenToKVPoolAllocator(
        size=64,
        size_swa=64,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
        req_to_token_pool=req_to_token_pool,
    )
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend):
        cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                disable=False,
                sliding_window_size=window,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
            )
        )
    return cache, allocator


def _complete_backup(core, node, full_offset=1000, aux_offset=1000):
    full, aux = core.build_backup_spec(node)
    for transfers in aux.values():
        for transfer in transfers:
            transfer.host_indices = transfer.device_indices + aux_offset
    core.commit_backup(node, full + full_offset, aux)
    core.mark_write_through_pending([node], node)
    core.finish_write_through([node], node)
    return full + full_offset, aux


def _swa_mamba_cache(backend, window):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    _, allocator = _swa_cache(window=window, backend=backend)
    request_pool = Mock(spec=HybridReqToTokenPool)
    request_pool.mamba_allocator = Mock()
    with (
        get_context().override_server_args(
            _mamba_cache_chunk_size=256, mamba_max_states_per_path=-1
        ),
        envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend),
    ):
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=request_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                sliding_window_size=window,
                tree_components=(
                    ComponentType.FULL,
                    ComponentType.SWA,
                    ComponentType.MAMBA,
                ),
            )
        )
    return cache, allocator


def _internal_swa_write_back_case(
    backend, page_size=1, window=8, with_mamba=False, prefix_nodes=1
):
    cache, allocator = (
        _swa_mamba_cache(backend, window)
        if with_mamba
        else _swa_cache(window=window, page_size=page_size, backend=backend)
    )
    assert cache._tree_core_backend == backend
    segment = 2 * page_size
    child_length = math.ceil(window / page_size) * page_size
    prefix_length = prefix_nodes * segment
    total_length = prefix_length + child_length
    if page_size == 1:
        values = allocator.alloc(total_length)
    else:
        # Bulk page allocation is CPU-safe; alloc_extend runs a GPU kernel.
        values = allocator.full_attn_allocator.alloc(total_length)
        swa_values = allocator.swa_attn_allocator.alloc(total_length)
        allocator.set_full_to_swa_mapping(values, swa_values)
    sizes = list(range(segment, prefix_length + 1, segment)) + [len(values)]
    nodes = []
    previous = 0
    for index, size in enumerate(sizes):
        nodes.append(
            cache.insert(
                InsertParams(
                    key=_key(list(range(size))),
                    value=values[:size],
                    prev_prefix_len=previous,
                    mamba_value=torch.tensor([201 + index]) if with_mamba else None,
                )
            ).last_device_node
        )
        previous = size
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.has_swa_host_pool = True
    cache.is_write_back = True
    cache.cache_controller = SimpleNamespace(write_policy="write_back")
    cache._build_backup_sidecar = Mock(return_value=[])
    # The child spans a complete window, so its request lock leaves the older
    # internal SWA segments eligible while preventing a whole-leaf eviction.
    leaf_lock = cache.inc_lock_ref(nodes[-1]).to_dec_params()
    return SimpleNamespace(
        cache=cache,
        core=cache.tree_core,
        nodes=nodes,
        sizes=sizes,
        segment=segment,
        leaf_lock=leaf_lock,
        full={
            node: cache.tree_core.get_component_device_value(
                node, ComponentType.FULL
            ).clone()
            for node in nodes
        },
        swa={
            node: cache.tree_core.get_component_device_value(
                node, ComponentType.SWA
            ).clone()
            for node in nodes
        },
    )


def _mock_swa_write_back_io(case, backup="success"):
    cache, core = case.cache, case.core
    case.events, case.submissions, case.freed = [], [], []
    host_pool = Mock()
    host_pool.available_size.return_value = 0 if backup == "host_pressure" else 64
    cache.components[ComponentType.SWA]._swa_kv_pool_host = host_pool
    cache.host_pool_group = Mock()
    cache.host_pool_group.get_pool.return_value = host_pool

    def assert_resident(node, transfers):
        assert torch.equal(
            core.get_component_device_value(node, ComponentType.FULL),
            case.full[node],
        )
        for transfer in transfers[ComponentType.SWA]:
            assert torch.equal(
                torch.cat(
                    [
                        core.get_component_device_value(source, ComponentType.SWA)
                        for source in transfer.nodes_to_load
                    ]
                ),
                transfer.device_indices,
            )

    def evict_host(count, component):
        assert component == ComponentType.SWA
        case.events.append(("host_evict", count))
        host_pool.available_size.return_value = count
        return count

    def write(node, full, transfers, sidecars):
        assert sidecars == []
        assert_resident(node, transfers)
        case.events.append(("write", node))
        case.submissions.append((node, full.clone(), transfers))
        if backup == "raised":
            raise RuntimeError("DMA submission failed")
        if backup == "failed":
            return None
        for component_transfers in transfers.values():
            for transfer in component_transfers:
                transfer.host_indices = transfer.device_indices + 20000
        return full + 10000

    def acknowledge(write_back=False):
        assert write_back
        for node, _, transfers in case.submissions:
            if node in cache.ongoing_write_through:
                assert_resident(node, transfers)
                case.events.append(("ack", node))
        if backup == "ack_raised":
            raise RuntimeError("DMA completion failed")
        for ack_id in list(cache.ongoing_write_through):
            cache._finish_write_through_ack(ack_id)

    original_free = cache._free_values

    def free_values(device_frees, host_frees):
        assert not host_frees
        for component, tensors in device_frees.items():
            if tensors:
                case.freed.append((component, torch.cat(tensors).tolist()))
                case.events.append(("free", component))
        original_free(device_frees, host_frees)

    cache.evict_host = evict_host
    cache._execute_kv_backup = write
    cache.writing_check = acknowledge
    cache._free_values = free_values


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize(
    "page_size,window,with_mamba", [(1, 8, False), (4, 5, False), (1, 8, True)]
)
@pytest.mark.parametrize(
    "backup", ["success", "host_pressure", "failed", "raised", "ack_raised"]
)
def test_internal_swa_write_back_preserves_window_until_ack(
    backend, page_size, window, with_mamba, backup
):
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    # Real allocator, tree, backup planning, publication and eviction driver.
    # The physical allocation/DMA boundary is simulated for CPU coverage.
    case = _internal_swa_write_back_case(backend, page_size, window, with_mamba)
    _mock_swa_write_back_io(case, backup)
    cache, core = case.cache, case.core
    parent, leaf = case.nodes
    params = EvictParams(num_tokens=0, swa_num_tokens=case.segment)
    if backup in ("raised", "ack_raised"):
        with pytest.raises(RuntimeError, match="DMA .* failed"):
            cache.evict(params)
        assert [event[0] for event in case.events] == (
            ["write"] if backup == "raised" else ["write", "ack"]
        )
        assert not case.freed
        assert torch.equal(
            core.get_component_device_value(parent, ComponentType.SWA),
            case.swa[parent],
        )
        return

    result = cache.evict(params)
    assert result.swa_num_tokens_evicted == case.segment
    assert result.num_tokens_evicted == 0
    assert result.mamba_num_evicted == int(with_mamba)
    if with_mamba:
        assert core.get_component_device_value(parent, ComponentType.MAMBA) is None
    assert (ComponentType.SWA, case.swa[parent].tolist()) in case.freed
    assert not any(ct == ComponentType.FULL for ct, _ in case.freed)
    assert core.get_component_device_value(parent, ComponentType.SWA) is None
    assert torch.equal(
        core.get_component_device_value(parent, ComponentType.FULL), case.full[parent]
    )
    assert torch.equal(
        core.get_component_device_value(leaf, ComponentType.SWA), case.swa[leaf]
    )
    operations = [event[0] for event in case.events]
    if backup == "failed":
        assert operations == ["write"] + ["free"] * len(case.freed)
        assert not core.component_has_host_value_only(parent, ComponentType.SWA)
    else:
        prefix = ["host_evict"] if backup == "host_pressure" else []
        assert operations == prefix + ["write", "ack"] + ["free"] * len(case.freed)
        assert core.component_has_host_value_only(parent, ComponentType.SWA)
        matched = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(case.segment))))
        )
        assert matched.best_match_node == parent
        assert matched.device_indices.numel() == 0
        assert matched.full_kv_hit_length == case.segment
        assert matched.swa_host_hit_length == case.segment
        assert matched.mamba_host_hit_length == int(with_mamba)
        (transfer,) = core.build_hicache_transfers(
            ComponentType.SWA, parent, CacheTransferPhase.LOAD_BACK
        )
        assert transfer.nodes_to_load == [parent]
        assert torch.equal(transfer.host_indices, case.swa[parent] + 20000)
    cache.dec_lock_ref(leaf, case.leaf_lock)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_internal_swa_write_back_reserves_entire_unbacked_window(backend):
    case = _internal_swa_write_back_case(backend, prefix_nodes=2)
    cache, core = case.cache, case.core
    first, victim, leaf = case.nodes
    # Refresh only the first segment; the next internal segment now heads the
    # SWA LRU, and its backup window includes both unbacked prefix segments.
    cache.match_prefix(MatchPrefixParams(key=_key(list(range(case.segment)))))
    _mock_swa_write_back_io(case, "host_pressure")
    result = cache.evict(EvictParams(num_tokens=0, swa_num_tokens=case.segment))
    assert result.swa_num_tokens_evicted == case.segment
    assert case.events[:2] == [("host_evict", 2 * case.segment), ("write", victim)]
    (submission,) = case.submissions
    (transfer,) = submission[2][ComponentType.SWA]
    assert transfer.nodes_to_load == [first, victim]
    assert transfer.device_indices.numel() == 2 * case.segment
    assert case.freed == [(ComponentType.SWA, case.swa[victim].tolist())]
    assert torch.equal(
        core.get_component_device_value(first, ComponentType.SWA), case.swa[first]
    )
    assert core.component_has_host_value_only(victim, ComponentType.SWA)
    cache.dec_lock_ref(leaf, case.leaf_lock)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("guard", ["request", "pending_dma"])
def test_internal_swa_write_back_respects_active_window_locks(backend, guard):
    case = _internal_swa_write_back_case(backend)
    _mock_swa_write_back_io(case)
    cache, core = case.cache, case.core
    parent, leaf = case.nodes
    if guard == "request":
        lock = cache.inc_lock_ref(parent).to_dec_params()
    else:
        # A normal asynchronous backup takes real component locks and leaves
        # its source pinned until publication of the physical completion ACK.
        assert (
            cache._execute_and_commit_kv_backup(
                BackupKV(node_ids=[parent]), write_back=False
            )
            == case.segment
        )
        assert parent in cache.ongoing_write_through
    submissions = len(case.submissions)
    tracker = {ct: 0 for ct in cache.tree_components}
    cache._evict_components(
        {
            ct: case.segment if ct == ComponentType.SWA else 0
            for ct in cache.tree_components
        },
        tracker,
    )
    assert tracker[ComponentType.SWA] == 0 and not case.freed
    assert len(case.submissions) == submissions
    assert torch.equal(
        core.get_component_device_value(parent, ComponentType.SWA), case.swa[parent]
    )
    if guard == "request":
        cache.dec_lock_ref(parent, lock)
    else:
        cache._finish_write_through_ack(parent)
    assert (
        cache.evict(
            EvictParams(num_tokens=0, swa_num_tokens=case.segment)
        ).swa_num_tokens_evicted
        == case.segment
    )
    assert core.component_has_host_value_only(parent, ComponentType.SWA)
    cache.dec_lock_ref(leaf, case.leaf_lock)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_internal_swa_write_back_follows_runtime_policy_update(backend):
    from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment

    case = _internal_swa_write_back_case(backend)
    cache, core = case.cache, case.core
    _mock_swa_write_back_io(case)
    cache.is_write_back = False
    cache.cache_controller.write_policy = "write_through"
    cache.cache_controller.storage_backend_type = "file"
    cache.enable_storage = True
    cache.write_backup_storage = Mock()
    success, _ = StorageAttachment(cache).attach(
        "file", hicache_write_policy="write_back"
    )
    assert success and cache.is_write_back
    assert cache._tree_core_backend == backend
    assert (
        cache.evict(
            EvictParams(num_tokens=0, swa_num_tokens=case.segment)
        ).swa_num_tokens_evicted
        == case.segment
    )
    assert core.component_has_host_value_only(case.nodes[0], ComponentType.SWA)
    cache.write_backup_storage.assert_called_once_with(case.nodes[0])
    cache.dec_lock_ref(case.nodes[-1], case.leaf_lock)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("component", [ComponentType.MAMBA, ComponentType.SWA])
@pytest.mark.parametrize("pin_ancestor", [False, True])
def test_aux_host_reclaim_uses_its_own_load_back_sources(
    backend, component, pin_ancestor
):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if component == ComponentType.MAMBA:
        with (
            get_context().override_server_args(
                _mamba_cache_chunk_size=256, mamba_max_states_per_path=-1
            ),
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend),
        ):
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=True,
                    req_to_token_pool=Mock(spec=HybridReqToTokenPool),
                    token_to_kv_pool_allocator=None,
                    page_size=1,
                    tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                )
            )
        core = cache.tree_core
        parent = _mamba_insert(core, [1, 2], [101, 102], 201).last_device_node
        child = _mamba_insert(
            core, [1, 2, 3, 4], [101, 102, 103, 104], 202
        ).last_device_node
        loaded_full = torch.tensor([301, 302, 303, 304])
        amount = 1
    else:
        cache, allocator = _swa_cache(window=4, backend=backend)
        core = cache.tree_core
        loaded_full = allocator.alloc(8)
        parent = cache.insert(
            InsertParams(key=_key(list(range(4))), value=loaded_full[:4])
        ).last_device_node
        child = cache.insert(
            InsertParams(key=_key(list(range(8))), value=loaded_full, prev_prefix_len=4)
        ).last_device_node
        core.has_swa_host_pool = True
        amount = 4

    core.set_hicache_enabled()
    core.is_write_back = True
    _, parent_aux = _complete_backup(core, parent)
    expected_free = parent_aux[component][0].host_indices.tolist()
    _complete_backup(core, child)
    for node in (child, parent):
        _accumulate_step(core.demote(node), {}, {}, {})
    assert core.component_has_host_value_only(parent, component)

    # Loading the child reads Full from both nodes but auxiliary data only
    # from the child. Follow the controller's lock/commit/relock sequence.
    host_lock = core.inc_host_lock_ref(child).to_dec_params()
    before_lock = core.inc_lock_ref(child).to_dec_params()
    kv, aux = core.build_load_back_spec(child)
    assert kv.nodes_to_load == [parent, child]
    assert aux[component][0].nodes_to_load == [child]
    for transfers in aux.values():
        for transfer in transfers:
            transfer.device_indices = transfer.host_indices + 2000
    core.dec_lock_ref(child, before_lock)
    core.commit_load_back(child, loaded_full, kv, aux)
    pending_lock = core.inc_lock_ref(child).to_dec_params()
    extra_lock = (
        core.inc_host_lock_ref(parent).to_dec_params() if pin_ancestor else None
    )
    core.sanity_check([], [(child, child)])

    tracker, host_frees = {}, {}
    _accumulate_step(
        core.drive_host_eviction(component, amount), tracker, {}, host_frees
    )
    assert tracker.get(component, 0) == (0 if pin_ancestor else amount)
    assert [value.tolist() for value in host_frees.get(component, [])] == (
        [] if pin_ancestor else [expected_free]
    )
    assert core.is_backuped(parent) and core.is_backuped(child)
    core.sanity_check([], [(child, child)])

    core.dec_lock_ref(child, pending_lock)
    core.dec_host_lock_ref(child, host_lock)
    core.finish_load_back(child)
    if extra_lock is not None:
        core.dec_host_lock_ref(parent, extra_lock)
    tracker, host_frees = {}, {}
    _accumulate_step(
        core.drive_host_eviction(component, amount), tracker, {}, host_frees
    )
    assert tracker.get(component, 0) == (amount if pin_ancestor else 0)
    assert [value.tolist() for value in host_frees.get(component, [])] == (
        [expected_free] if pin_ancestor else []
    )
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("component", [ComponentType.MAMBA, ComponentType.SWA])
@pytest.mark.parametrize("pin_ancestor", [False, True])
def test_full_load_back_does_not_pin_restored_ancestor_aux(
    backend, component, pin_ancestor
):
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if component == ComponentType.SWA:
        cache, allocator = _swa_cache(window=4, backend=backend)
    else:
        allocator = TokenToKVPoolAllocator(64, torch.bfloat16, "cpu", None, False)
        request_pool = Mock(spec=HybridReqToTokenPool)
        request_pool.mamba_allocator = Mock()
        with (
            get_context().override_server_args(
                _mamba_cache_chunk_size=256, mamba_max_states_per_path=-1
            ),
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend),
        ):
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=request_pool,
                    token_to_kv_pool_allocator=allocator,
                    page_size=1,
                    tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                )
            )
    core = cache.tree_core
    values = allocator.alloc(8)
    nodes = [
        cache.insert(
            InsertParams(
                key=_key(list(range(size))),
                value=values[:size],
                prev_prefix_len=size - 4,
                mamba_value=torch.tensor([slot])
                if component == ComponentType.MAMBA
                else None,
            )
        ).last_device_node
        for size, slot in ((4, 201), (8, 202))
    ]
    parent, child = nodes
    core.set_hicache_enabled()
    core.has_swa_host_pool = component == ComponentType.SWA
    core.is_write_back = True
    full_host = [
        _complete_backup(core, node, full_offset=1000, aux_offset=2000)[0]
        for node in nodes
    ]
    for node in (child, parent):
        step = core.demote(node)
        cache._free_values(step.device_frees, step.host_frees)
    assert core.component_has_host_value_only(parent, component)
    submissions = []

    def load(host_indices, node_id, extra_pools):
        assert node_id == child
        assert host_indices.tolist() == torch.cat(full_host).tolist()
        assert len(extra_pools) == 1 and extra_pools[0].nodes_to_load == [child]
        full = allocator.alloc(host_indices.numel())
        for transfer in extra_pools:
            transfer.device_indices = (
                allocator.alloc(transfer.host_indices.numel())
                if component == ComponentType.SWA
                else transfer.host_indices + 1000
            )
        submissions.append(node_id)
        return full

    cache.cache_controller = SimpleNamespace(load=load, write_policy="write_back")
    cache.load_back_threshold = 10
    cache._build_sidecar_transfers = Mock(return_value=[])
    assert cache.load_back(child, req=None)
    assert submissions == [child]
    assert core.get_component_device_value(parent, component) is None
    selected = core.get_component_device_value(child, component).tolist()
    core.sanity_check([], [(child, child)])

    # A separate completed request restores state outside the transfer's
    # auxiliary span after load_back has committed and re-locked the path.
    cache.insert(
        InsertParams(
            key=_key(list(range(4))),
            value=allocator.alloc(4),
            mamba_value=torch.tensor([211])
            if component == ComponentType.MAMBA
            else None,
        )
    )
    restored = core.get_component_device_value(parent, component).tolist()
    extra_lock = cache.inc_lock_ref(parent).to_dec_params() if pin_ancestor else None
    active = [(parent, parent)] if pin_ancestor else []
    core.sanity_check(active, [(child, child)])
    full_before = {
        node: core.get_component_device_value(node, ComponentType.FULL).tolist()
        for node in nodes
    }
    # SWA frees are Full-side IDs, which the allocator resolves through its
    # current Full-to-SWA mapping; Mamba frees are its own state slots.
    expected_free = full_before[parent] if component == ComponentType.SWA else restored
    tracker, frees = {component: 0}, {}
    core.evict_device_start(component, len(restored))
    try:
        step = core.evict_device_next_node(component, tracker)
        assert step.node_id is None and step.mamba_backup_node_id is None
        _accumulate_step(step, tracker, frees, {})
    finally:
        core.evict_device_end(component)
    assert [value.tolist() for value in frees.get(component, [])] == (
        [] if pin_ancestor else [expected_free]
    )
    assert core.get_component_device_value(child, component).tolist() == selected
    for node in nodes:
        assert (
            core.get_component_device_value(node, ComponentType.FULL).tolist()
            == full_before[node]
        )
    core.sanity_check(active, [(child, child)])

    event = Mock()
    cache.cache_controller.ack_load_queue = [
        SimpleNamespace(
            finish_event=event,
            node_ids=[child],
            num_tokens_by_pool={},
            num_bytes=0,
            timing_enabled=False,
        )
    ]
    cache.loading_check(finish_count=1)
    event.synchronize.assert_called_once_with()
    core.sanity_check(active, [])
    core.evict_device_start(component, 1)
    try:
        step = core.evict_device_next_node(component, {component: 0})
        assert step.node_id == child
        assert not step.device_frees
        _accumulate_step(step, {}, {}, {})
    finally:
        core.evict_device_end(component)
    if extra_lock is not None:
        cache.dec_lock_ref(parent, extra_lock)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_full_host_duplicates_follow_ack_order_after_pending_swa_split(backend):
    cache, allocator = _swa_cache(window=4, backend=backend)
    core = cache.tree_core
    values = allocator.alloc(12)
    nodes = [
        cache.insert(
            InsertParams(
                key=_key(list(range(size))),
                value=values[:size],
                prev_prefix_len=size - 4,
            )
        ).last_device_node
        for size in (4, 8, 12)
    ]
    parent, child, leaf = nodes
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    core.is_write_back = True
    for node in nodes:
        _complete_backup(core, node, full_offset=10000, aux_offset=20000)

    # Repairing SWA later must issue an async backup even in write-back mode.
    # Reclaim the two internal SWA states and the parent's Full host copy.
    tracker = {}
    core.evict_device_start(ComponentType.SWA, 8)
    try:
        for _ in range(2):
            step = core.evict_device_next_node(ComponentType.SWA, tracker)
            assert step.node_id is None
            for component, count in step.tracker.items():
                tracker[component] = tracker.get(component, 0) + count
            cache._free_values(step.device_frees, step.host_frees)
    finally:
        core.evict_device_end(ComponentType.SWA)
    assert tracker[ComponentType.SWA] == 8
    for component, amount in ((ComponentType.SWA, 8), (ComponentType.FULL, 4)):
        step = core.drive_host_eviction(component, amount)
        assert step.tracker[component] == amount
        _accumulate_step(step, {}, {}, {})
    assert not core.is_backuped(parent) and core.is_backuped(child)

    # Mock DMA submission only; insertion, backup publication, the pending
    # split and FIFO acknowledgment all use the real shared controller.
    cache.cache_controller = SimpleNamespace(write_policy="write_back")
    cache._build_backup_sidecar = Mock(return_value=[])
    submissions = []

    def write(node, device, aux, sidecars):
        submissions.append((node, device.numel()))
        for transfers in aux.values():
            for transfer in transfers:
                transfer.host_indices = transfer.device_indices + 30000
        return device + 40000

    cache._execute_kv_backup = write
    cache.insert(
        InsertParams(
            key=_key(list(range(8))),
            value=allocator.alloc(8),
            component_evicted_seqlens={ComponentType.SWA: 4},
        )
    )
    assert submissions == [(parent, 4), (child, 0)]
    assert set(cache.ongoing_write_through) == {parent, child}
    core.sanity_check([(node, node) for node in cache.ongoing_write_through], [])
    cache.match_prefix(MatchPrefixParams(key=_key([0, 1])))
    prefix, suffix = cache.ongoing_write_through[parent].publish_node_ids
    assert suffix == parent and prefix != parent
    core.sanity_check([(node, node) for node in cache.ongoing_write_through], [])
    for node in (parent, child):
        cache._finish_write_through_ack(node)
    core.sanity_check([], [])

    # The existing child keeps its place. New Full copies join at ack in
    # prefix/suffix order, including when the pending node was internal.
    expected = [
        (values[4:8] + 10000).tolist(),
        (values[:2] + 40000).tolist(),
        (values[2:4] + 40000).tolist(),
        (values[8:] + 10000).tolist(),
    ]
    for host_indices in expected:
        host_frees = {}
        _accumulate_step(
            core.drive_host_eviction(ComponentType.FULL, 1), {}, {}, host_frees
        )
        assert [value.tolist() for value in host_frees[ComponentType.FULL]] == [
            host_indices
        ]
    assert all(not core.is_backuped(node) for node in (prefix, parent, child, leaf))
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("pin_ancestor", [False, True])
def test_pending_swa_backup_does_not_pin_restored_ancestor_mamba(backend, pin_ancestor):
    cache, allocator = _swa_mamba_cache(backend, window=8)
    core = cache.tree_core
    values = allocator.alloc(12)
    nodes = [
        cache.insert(
            InsertParams(
                key=_key(list(range(size))),
                value=values[:size],
                prev_prefix_len=size - 4,
                mamba_value=torch.tensor([slot]),
            )
        ).last_device_node
        for size, slot in ((4, 201), (8, 202), (12, 203))
    ]
    parent, child, leaf = nodes
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    core.is_write_back = True
    for node in nodes:
        _complete_backup(core, node, full_offset=10000, aux_offset=20000)

    # SWA eviction also removes ancestor Mamba states. Repair SWA first;
    # a separate request below restores the ancestor's Mamba checkpoint.
    tracker = {}
    core.evict_device_start(ComponentType.SWA, 8)
    try:
        for _ in range(2):
            step = core.evict_device_next_node(ComponentType.SWA, tracker)
            assert step.node_id is None
            for component, count in step.tracker.items():
                tracker[component] = tracker.get(component, 0) + count
            cache._free_values(step.device_frees, step.host_frees)
    finally:
        core.evict_device_end(ComponentType.SWA)
    assert tracker[ComponentType.SWA] == 8
    step = core.drive_host_eviction(ComponentType.SWA, 8)
    assert step.tracker[ComponentType.SWA] == 8
    _accumulate_step(step, {}, {}, {})
    assert all(core.is_backuped(node) for node in nodes)
    cache.cache_controller = SimpleNamespace(write_policy="write_back")
    cache._build_backup_sidecar = Mock(return_value=[])
    submissions = []

    def write(node, full, aux, sidecars):
        submissions.append((node, full.numel(), aux))
        for transfers in aux.values():
            for transfer in transfers:
                transfer.host_indices = transfer.device_indices + 30000
        return full + 40000

    cache._execute_kv_backup = write
    cache.insert(
        InsertParams(
            key=_key(list(range(8))),
            value=allocator.alloc(8),
            component_evicted_seqlens={ComponentType.SWA: 0},
            mamba_value=torch.tensor([202]),
        )
    )
    assert len(submissions) == 1
    node, full_rows, aux = submissions[0]
    assert node == child and full_rows == 0
    assert aux[ComponentType.SWA][0].nodes_to_load == [parent, child]
    assert aux[ComponentType.MAMBA][0].device_indices.tolist() == [202]
    assert list(cache.ongoing_write_through) == [child]
    assert cache.ongoing_write_through[child].publish_node_ids == [parent, child]
    assert core.get_component_device_value(parent, ComponentType.MAMBA) is None

    cache.insert(
        InsertParams(
            key=_key(list(range(4))),
            value=allocator.alloc(4),
            mamba_value=torch.tensor([211]),
        )
    )
    assert len(submissions) == 1
    assert core.get_component_device_value(parent, ComponentType.MAMBA).tolist() == [
        211
    ]
    leaf_lock = cache.inc_lock_ref(leaf).to_dec_params()
    extra_lock = cache.inc_lock_ref(parent).to_dec_params() if pin_ancestor else None
    active = [(leaf, leaf)] + ([(parent, parent)] if pin_ancestor else [])
    core.sanity_check([(child, child)] + active, [])
    retained = {
        (node, component): core.get_component_device_value(node, component).tolist()
        for node in nodes
        for component in (ComponentType.FULL, ComponentType.SWA)
    }

    def evict_one_mamba_state():
        tracker, frees = {ComponentType.MAMBA: 0}, {}
        core.evict_device_start(ComponentType.MAMBA, 1)
        try:
            step = core.evict_device_next_node(ComponentType.MAMBA, tracker)
            assert step.node_id is None and step.mamba_backup_node_id is None
            _accumulate_step(step, tracker, frees, {})
        finally:
            core.evict_device_end(ComponentType.MAMBA)
        return [value.tolist() for value in frees.get(ComponentType.MAMBA, [])]

    assert evict_one_mamba_state() == ([] if pin_ancestor else [[211]])
    # The actual Mamba DMA source remains protected by its own device lock.
    assert core.get_component_device_value(child, ComponentType.MAMBA).tolist() == [202]
    core.sanity_check([(child, child)] + active, [])
    for (node, component), indices in retained.items():
        assert core.get_component_device_value(node, component).tolist() == indices
    cache._finish_write_through_ack(child)
    assert evict_one_mamba_state() == [[202]]
    core.sanity_check(active, [])
    cache.dec_lock_ref(leaf, leaf_lock)
    if extra_lock is not None:
        cache.dec_lock_ref(parent, extra_lock)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_failed_mamba_backup_drops_state_under_unrelated_pending_swa(backend):
    cache, allocator = _swa_mamba_cache(backend, window=12)
    core = cache.tree_core
    values = allocator.alloc(16)
    nodes = [
        cache.insert(
            InsertParams(
                key=_key(list(range(size))),
                value=values[:size],
                prev_prefix_len=size - 4,
                mamba_value=torch.tensor([slot]),
            )
        ).last_device_node
        for size, slot in ((4, 201), (8, 202), (12, 203), (16, 204))
    ]
    grandparent, parent, anchor, leaf = nodes
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    core.is_write_back = True
    cache.cache_controller = SimpleNamespace(write_policy="write_back")
    cache._build_backup_sidecar = Mock(return_value=[])
    host_pool = Mock()
    host_pool.available_size.return_value = 1
    cache.host_pool_group = Mock()
    cache.host_pool_group.get_pool.return_value = host_pool
    cache.components[ComponentType.MAMBA]._mamba_pool_host = host_pool
    submissions = []
    failed_node = None

    def write(node, full, aux, sidecars):
        submissions.append((node, full.numel(), aux))
        if node == failed_node:
            return None  # Full host allocation failed before submitting DMA.
        for transfers in aux.values():
            for transfer in transfers:
                transfer.host_indices = transfer.device_indices + 20000
        return full + 10000

    def complete_physical_writes(write_back=False):
        for ack_id in list(cache.ongoing_write_through):
            cache._finish_write_through_ack(ack_id)

    cache._execute_kv_backup = write
    cache.writing_check = complete_physical_writes

    def evict_component(component, count):
        tracker = {ct: 0 for ct in cache.tree_components}
        requested = {
            ct: count if ct == component else 0 for ct in cache.tree_components
        }
        cache._evict_components(requested, tracker)
        return tracker[component]

    # Reach the host topology through actual state backups and ACKs. Full
    # duplicate reclaim then leaves an unbacked grandparent above a backed
    # parent; no synthetic node flags or direct backup commits are needed.
    assert evict_component(ComponentType.MAMBA, 3) == 3
    assert [(node, count) for node, count, _ in submissions] == [
        (node, 4) for node in nodes[:3]
    ]
    assert cache.evict_host(4, ComponentType.FULL) == 4
    assert [core.is_backuped(node) for node in nodes] == [False, True, True, False]
    core.sanity_check([], [])

    assert evict_component(ComponentType.SWA, 12) == 12
    assert cache.evict_host(12, ComponentType.SWA) == 12
    cache.insert(
        InsertParams(
            key=_key(list(range(12))),
            value=allocator.alloc(12),
            mamba_value=torch.tensor([203]),
        )
    )
    # Backup ancestry stops at the backed parent, while the expanded SWA
    # window reaches the unbacked grandparent. Its pending mark belongs to SWA.
    assert list(cache.ongoing_write_through) == [anchor]
    assert cache.ongoing_write_through[anchor].publish_node_ids == nodes[:3]
    assert submissions[-1][2][ComponentType.SWA][0].nodes_to_load == nodes[:3]
    cache.insert(
        InsertParams(
            key=_key(list(range(4))),
            value=allocator.alloc(4),
            mamba_value=torch.tensor([211]),
        )
    )
    assert not core.is_backuped(grandparent)
    assert ComponentType.MAMBA in core.build_backup_spec(grandparent)[1]
    leaf_lock = cache.inc_lock_ref(leaf).to_dec_params()
    active = [(anchor, anchor), (leaf, leaf)]
    core.sanity_check(active, [])
    retained = {
        (node, ct): core.get_component_device_value(node, ct).clone()
        for node in nodes
        for ct in (ComponentType.FULL, ComponentType.SWA)
    }

    failed_node = grandparent
    assert evict_component(ComponentType.MAMBA, 1) == 1
    assert submissions[-1][:2] == (grandparent, 4)
    assert core.get_component_device_value(grandparent, ComponentType.MAMBA) is None
    # The completed backup and active request retain their other device values.
    for node, slot in ((anchor, 203), (leaf, 204)):
        assert core.get_component_device_value(node, ComponentType.MAMBA).tolist() == [
            slot
        ]
    for (node, ct), value in retained.items():
        assert torch.equal(core.get_component_device_value(node, ct), value)
    # #41092 drains the earlier SWA ACK before attempting the failed backup.
    assert not cache.ongoing_write_through
    core.sanity_check([(leaf, leaf)], [])
    cache.dec_lock_ref(leaf, leaf_lock)
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize(
    "ring,hicache,has_swa_host_pool",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
        (True, True, False),
    ],
)
def test_swa_match_uses_allocator_layout(backend, ring, hicache, has_swa_host_pool):
    # #38269: a request ring rebuilds its SWA window outside the tree, while
    # paged SWA requires resident state even if HiCache has no SWA host pool
    # (for example DSV4.1 encoder replay with DSpark, #38798).
    cache, allocator = _swa_cache(backend=backend, ring=ring)
    assert type(cache.tree_core).__name__ == (
        "RustUnifiedTreeCore" if backend == "rust" else "UnifiedTreeCore"
    )
    indices = allocator.alloc(12)
    cache.insert(
        InsertParams(
            key=_key(list(range(12))),
            value=indices,
            component_evicted_seqlens={ComponentType.SWA: 8},
        )
    )
    if hicache:
        cache.tree_core.set_hicache_enabled()
    cache.tree_core.has_swa_host_pool = has_swa_host_pool
    matched = cache.match_prefix(MatchPrefixParams(key=_key(list(range(8)))))
    assert matched.full_kv_hit_length == 8
    assert matched.device_indices.tolist() == (indices[:8].tolist() if ring else [])


def test_buffer_backup_snapshot_round_trips_and_detects_a_split():
    core = _tree_core()
    core.enable_storage = True
    key = RadixKey(array("q", [1, 2]), extra_key="adapter-a", cache_salt="tenant-a")
    inserted = _pump_insert(
        core,
        InsertParams(key=key, value=torch.tensor([10, 11], dtype=torch.int64)),
    )
    leaf = inserted.last_device_node

    snapshot = core.snapshot_buffer_backup(leaf, pass_prefix_keys=True)
    assert snapshot.node_id == leaf
    assert snapshot.parent_is_root
    assert snapshot.key.token_ids == array("q", [1, 2])
    assert snapshot.key.extra_key == "adapter-a"
    assert snapshot.key.cache_salt == "tenant-a"
    assert not snapshot.key.is_bigram
    assert snapshot.prefix_keys == []
    assert core.validate_buffer_backup(leaf, len(snapshot.key)) is not None

    _pump_insert(
        core,
        InsertParams(
            key=RadixKey(
                array("q", [1, 9]), extra_key="adapter-a", cache_salt="tenant-a"
            ),
            value=torch.tensor([12, 13], dtype=torch.int64),
        ),
    )
    assert core.validate_buffer_backup(leaf, len(snapshot.key)) is None

    core.reset()
    assert core.snapshot_buffer_backup(leaf, pass_prefix_keys=True) is None
    assert core.validate_buffer_backup(leaf, len(snapshot.key)) is None


def test_buffer_backup_snapshot_preserves_bigram_keys():
    core = _tree_core(is_eagle=True)
    core.enable_storage = True
    inserted = _insert(core, [1, 2, 3], [10, 11])

    snapshot = core.snapshot_buffer_backup(
        inserted.last_device_node, pass_prefix_keys=False
    )
    assert snapshot.key.token_ids == array("q", [1, 2, 3])
    assert snapshot.key.is_bigram


def test_swa_core_builds_with_a_window():
    core = _swa_tree_core(window=8)
    result = _insert(core, [1, 2, 3], [10, 11, 12])
    assert result.prefix_len == 0
    # The in-window new leaf asks for one SWA rebuild.
    (action,) = result.cache_actions
    assert isinstance(action, SWARebuild)
    assert action.source_value.tolist() == [10, 11, 12]


def test_swa_load_back_missing_value_raises_assertion_error():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _swa_tree_core(window=4)
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    inserted = _insert(core, [1], [10])
    node = inserted.cache_actions[0].node_id

    with pytest.raises(AssertionError):
        core.build_hicache_transfers(
            ComponentType.SWA, node, CacheTransferPhase.LOAD_BACK
        )
    with pytest.raises(AssertionError):
        core.build_load_back_spec(node)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("is_bigram", [False, True])
def test_swa_window_repair_with_page_rounded_window(backend, is_bigram):
    core, _ = _swa_transfer_core(backend, page_size=4, window=5, is_eagle=is_bigram)
    key = RadixKey(
        array("q", range(16 + is_bigram)),
        extra_key="adapter-a",
        cache_salt="tenant-a",
        is_bigram=is_bigram,
    )
    full_values = torch.arange(10, 26, dtype=torch.int64)
    _pump_insert(
        core,
        InsertParams(
            key=key,
            value=full_values,
            component_evicted_seqlens={ComponentType.SWA: 16},
        ),
    )
    # A five-token window stages two four-token pages. Bigrams add one raw
    # boundary token but preserve the logical repair span [8, 16).
    ranges = core.swa_tombstone_ranges(key, 8, 16)
    assert ranges == [(8, 16)]
    assert core.attach_swa_window(key, 8, 16, torch.arange(50, 58)) == []
    matched_len, leaf, _ = core.match_full_device_prefix(key)
    assert matched_len == 16
    assert torch.equal(
        core.collect_full_device_indices(leaf, core.root_node_handle()), full_values
    )
    assert core.get_component_device_value(leaf, ComponentType.SWA).tolist() == list(
        range(50, 58)
    )
    assert core.swa_tombstone_ranges(key, 0, 16) == [(0, 8)]
    assert core.full_evictable_size() == 16
    assert core.swa_evictable_size() == 8
    core.sanity_check([], [])


@pytest.mark.parametrize("page_size", [1, 2])
@pytest.mark.parametrize("is_bigram", [False, True])
def test_swa_window_repair_preserves_full_slots_and_key_namespace(page_size, is_bigram):
    core = _swa_tree_core(page_size=page_size, is_eagle=is_bigram)
    key = RadixKey(
        array("q", range(8 + is_bigram)),
        extra_key="adapter-a",
        cache_salt="tenant-a",
        is_bigram=is_bigram,
    )
    _pump_insert(
        core,
        InsertParams(
            key=key,
            value=torch.arange(10, 18, dtype=torch.int64),
            component_evicted_seqlens={ComponentType.SWA: 8},
        ),
    )
    assert core.swa_tombstone_ranges(key, 0, 8) == [(0, 8)]
    values = torch.arange(50, 54, dtype=torch.int64)
    assert core.attach_swa_window(key, 2, 6, values) == []
    values.fill_(-1)

    # Slicing and namespace conversion must retain logical bigram positions.
    full_values = []
    for end in (2, 6, 8):
        _, node, _ = core.match_full_device_prefix(key[:end])
        full_values.extend(
            core.get_component_device_value(node, ComponentType.FULL).tolist()
        )
        swa = core.get_component_device_value(node, ComponentType.SWA)
        if end == 6:
            assert swa.tolist() == [50, 51, 52, 53]
        else:
            assert swa is None
    assert full_values == list(range(10, 18))
    assert core.swa_tombstone_ranges(key, 0, 8) == [(0, 2), (6, 8)]
    assert core.swa_tombstone_ranges(key, 1, 7) == [(1, 2), (6, 7)]
    for extra_key, cache_salt in (("adapter-b", "tenant-a"), ("adapter-a", "tenant-b")):
        other_key = RadixKey(
            key.token_ids,
            extra_key=extra_key,
            cache_salt=cache_salt,
            is_bigram=is_bigram,
        )
        assert core.swa_tombstone_ranges(other_key, 0, 8) == []
    assert core.full_evictable_size() == 8
    assert core.swa_evictable_size() == 4
    core.sanity_check([], [])


def test_swa_window_repair_returns_split_actions_and_balances_existing_locks():
    core = _swa_tree_core()
    core.set_hicache_enabled()
    key = _key(list(range(8)))
    _pump_insert(
        core,
        InsertParams(
            key=key,
            value=torch.arange(10, 18, dtype=torch.int64),
            component_evicted_seqlens={ComponentType.SWA: 8},
        ),
    )
    _, leaf, _ = core.match_full_device_prefix(key)
    receipt = core.inc_lock_ref(leaf)
    core.mark_write_through_pending([leaf], ack_id=leaf)
    actions = core.attach_swa_window(key, 2, 6, torch.arange(50, 54))
    assert len(actions) == 2
    for action in actions:
        assert isinstance(action, ReplaceWriteThroughOnNodeSplit)
        assert action.ack_id == leaf
        assert action.old_node_id == leaf
        assert action.new_child_node_id == leaf
    assert actions[0].new_node_id != actions[1].new_node_id
    assert core.swa_protected_size() == 4
    assert core.swa_evictable_size() == 0
    core.finish_write_through([action.new_node_id for action in actions] + [leaf], leaf)
    core.dec_lock_ref(leaf, receipt.to_dec_params())
    assert core.swa_protected_size() == 0
    assert core.swa_evictable_size() == 4
    core.sanity_check([], [])


def test_swa_window_repair_rejects_invalid_publication_without_partial_changes():
    core = _swa_tree_core()
    key = _key(list(range(8)))
    _pump_insert(
        core,
        InsertParams(
            key=key,
            value=torch.arange(10, 18, dtype=torch.int64),
            component_evicted_seqlens={ComponentType.SWA: 8},
        ),
    )
    core.attach_swa_window(key, 2, 6, torch.arange(50, 54))
    with pytest.raises(AssertionError, match="over live SWA"):
        core.attach_swa_window(key, 0, 6, torch.arange(60, 66))
    with pytest.raises(AssertionError, match="int64"):
        core.attach_swa_window(key, 0, 2, torch.zeros(2, dtype=torch.int32))
    assert core.swa_tombstone_ranges(key, 0, 8) == [(0, 2), (6, 8)]
    assert core.swa_evictable_size() == 4
    core.sanity_check([], [])


def test_swa_straddling_insert_crosses_the_boundary_actions():
    core = _swa_tree_core(window=8)
    _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    result = _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2, 3, 4]),
            value=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
            component_evicted_seqlens={ComponentType.SWA: 2},
        ),
    )
    free_tail, rebuild, free_duplicates = result.cache_actions
    assert isinstance(free_tail, FreeDeviceKVFullOnly)
    assert free_tail.indices[0].tolist() == [12, 13]
    assert isinstance(rebuild, SWARebuild)
    assert rebuild.source_value.tolist() == [22, 23]
    # Below the floor the duplicate's SWA peers are gone: full side only.
    assert isinstance(free_duplicates, FreeDeviceKVFullOnly)
    assert free_duplicates.indices[0].tolist() == [20, 21]


def test_every_pool_name_crosses_the_prefetch_commit_boundary():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _tree_core()
    core.set_hicache_enabled()
    anchor = core.match_prefix(MatchPrefixParams(key=_key([99]))).best_match_node
    # Sidecar pools (e.g. the EAGLE draft KV) report hit pages through the
    # commit's pool_storage_result; every python pool name must parse.
    for name in PoolName:
        core.commit_hicache_transfers(
            anchor,
            CacheTransferPhase.PREFETCH,
            {},
            cache_actions=[],
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=0, extra_pool_hit_pages={name: 1}
            ),
        )


def _mamba_tree_core(
    page_size: int = 1,
    mamba_max_states_per_path: int = -1,
    **params_overrides,
) -> RustUnifiedTreeCore:
    with get_context().override_server_args(
        _mamba_cache_chunk_size=256,
        mamba_max_states_per_path=mamba_max_states_per_path,
    ):
        return _tree_core(
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            page_size=page_size,
            **params_overrides,
        )


def _mamba_tree_core_with_cap(cap: int) -> RustUnifiedTreeCore:
    return _mamba_tree_core(mamba_max_states_per_path=cap)


def _hybrid_swa_mamba_tree_core(window: int) -> RustUnifiedTreeCore:
    with get_context().override_server_args(
        _mamba_cache_chunk_size=256,
        mamba_max_states_per_path=-1,
    ):
        return _tree_core(
            tree_components=(
                ComponentType.FULL,
                ComponentType.SWA,
                ComponentType.MAMBA,
            ),
            sliding_window_size=window,
        )


def _mamba_insert(core, token_ids, indices, mamba_slot):
    return _pump_insert(
        core,
        InsertParams(
            key=_key(token_ids),
            value=torch.tensor(indices, dtype=torch.int64),
            mamba_value=torch.tensor([mamba_slot], dtype=torch.int64),
        ),
    )


def test_kv_canary_rows_exclude_mamba_slots():
    core = _mamba_tree_core()
    _mamba_insert(core, [1, 2], [10, 11], 7)
    # The canary walk emits FULL slots only; the mamba state slot never appears.
    assert _canary_rows(core) == [(10, 0, -1), (11, 1, 10)]


def test_component_set_guard_accepts_the_mamba_set():
    mem_cache.RustUnifiedTreeCoreBinding(
        mem_cache.TreeCoreInitParamsBinding(mamba_cache_chunk_size=256),
        [int(ComponentType.FULL), int(ComponentType.MAMBA)],
    )


def test_component_set_guard_accepts_the_hybrid_swa_mamba_set():
    mem_cache.RustUnifiedTreeCoreBinding(
        mem_cache.TreeCoreInitParamsBinding(
            swa_sliding_window_size=8, mamba_cache_chunk_size=256
        ),
        [
            int(ComponentType.FULL),
            int(ComponentType.SWA),
            int(ComponentType.MAMBA),
        ],
    )


def test_skipped_mamba_lock_survives_swa_only_release_through_the_adapter():
    core = _hybrid_swa_mamba_tree_core(window=2)
    inserted = _mamba_insert(core, [1, 2], [10, 11], 7)
    for action in inserted.cache_actions:
        if isinstance(action, SWARebuild):
            core.set_component_device_value(
                action.node_id, ComponentType.SWA, action.source_value
            )
    node = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node

    owner = core.inc_lock_ref(node)
    holder = core.inc_lock_ref(node, skip_lock_components=(ComponentType.MAMBA,))
    assert ComponentType.MAMBA not in owner.skipped_lock_components
    assert ComponentType.MAMBA in holder.skipped_lock_components
    assert core.mamba_protected_size() == 1

    # The holder's receipt says it never took mamba: its early SWA release
    # must leave the owner's mamba lock alone.
    released = core.dec_swa_lock_only(node, holder.to_dec_params())
    assert dict(released.device_frees) == {}
    assert dict(released.host_frees) == {}
    assert core.mamba_protected_size() == 1

    core.dec_lock_ref(node, holder.to_dec_params(), skip_swa=True)
    assert core.mamba_protected_size() == 1
    core.dec_lock_ref(node, owner.to_dec_params())
    assert core.protected_size() == 0
    assert core.swa_protected_size() == 0
    assert core.mamba_protected_size() == 0


def test_component_set_guard_still_rejects_invalid_sets():
    for components in (
        [ComponentType.MAMBA],
        [ComponentType.SWA, ComponentType.MAMBA],
        [ComponentType.MAMBA, ComponentType.FULL],
    ):
        with pytest.raises(ValueError, match="component sets"):
            mem_cache.RustUnifiedTreeCoreBinding(
                mem_cache.TreeCoreInitParamsBinding(
                    swa_sliding_window_size=8, mamba_cache_chunk_size=256
                ),
                [int(component) for component in components],
            )


def test_mamba_requires_the_chunk_size_through_the_binding():
    with pytest.raises(ValueError, match="requires mamba_cache_chunk_size"):
        mem_cache.RustUnifiedTreeCoreBinding(
            mem_cache.TreeCoreInitParamsBinding(),
            [int(ComponentType.FULL), int(ComponentType.MAMBA)],
        )


def test_mamba_tree_round_trips_through_the_adapter():
    core = _mamba_tree_core()
    result = _mamba_insert(core, [1, 2], [10, 11], 7)
    assert result.prefix_len == 0
    assert not result.mamba_exist

    matched = core.match_prefix(MatchPrefixParams(key=_key([1, 2])))
    assert matched.device_indices.tolist() == [10, 11]
    assert matched.mamba_host_hit_length == 0
    assert core.mamba_evictable_size() == 1
    assert core.all_mamba_values_flatten().tolist() == [7]

    # A reinsert keeps the slot and flags the caller to free the donation.
    result = _mamba_insert(core, [1, 2], [10, 11], 8)
    assert result.mamba_exist
    assert core.all_mamba_values_flatten().tolist() == [7]

    lock = core.inc_lock_ref(matched.best_match_node)
    assert core.mamba_protected_size() == 1
    assert core.mamba_evictable_size() == 0
    core.dec_lock_ref(matched.best_match_node, lock.to_dec_params())
    assert core.mamba_protected_size() == 0
    core.sanity_check([], [])


def test_mamba_eviction_walk_frees_slots_through_the_adapter():
    core = _mamba_tree_core()
    _mamba_insert(core, [1], [10], 7)
    internal = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node
    _mamba_insert(core, [1, 2], [10, 11], 8)
    tracker = {ComponentType.MAMBA: 0}
    device_frees: dict = {}
    host_frees: dict = {}
    core.evict_device_start(ComponentType.MAMBA, 2)
    step = core.evict_device_next_node(ComponentType.MAMBA, tracker)
    assert step.node_id is None
    assert step.made_progress
    assert step.unbacked_tokens == 0  # Only an auxiliary state was dropped.
    _accumulate_step(step, tracker, device_frees, host_frees)

    step = core.evict_device_next_node(ComponentType.MAMBA, tracker)
    leaf = step.node_id
    _accumulate_step(step, tracker, device_frees, host_frees)
    assert leaf is not None
    core.evict_device_end(ComponentType.MAMBA)
    assert tracker[ComponentType.MAMBA] == 1
    assert torch.cat(device_frees[ComponentType.MAMBA]).tolist() == [7]
    assert core.mamba_evictable_size() == 1

    # A pre-eviction node handle still lock-round-trips: the segment lock
    # counts the tombstone and the paired release takes it back exactly.
    lock = core.inc_lock_ref(internal)
    core.dec_lock_ref(internal, lock.to_dec_params())
    core.sanity_check([], [])


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize(
    "backup", ["success", "host_pressure", "failed", "raised", "ack_raised"]
)
def test_internal_mamba_write_back_preserves_state_until_ack(backend, backup):
    from collections import defaultdict

    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase
    from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
    from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    # Real components, tree, backup planner/commit and controller eviction. Only
    # allocation and DMA completion are simulated so this also runs on CPU.
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.enable_session_radix_cache = False
    cache.token_to_kv_pool_allocator = None
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=Mock(spec=HybridReqToTokenPool),
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
    )
    with get_context().override_server_args(
        _mamba_cache_chunk_size=256, mamba_max_states_per_path=-1
    ):
        cache.components = {
            ComponentType.FULL: FullComponent(cache, params),
            ComponentType.MAMBA: MambaComponent(cache, params),
        }
        core = (
            RustUnifiedTreeCore(params)
            if backend == "rust"
            else UnifiedTreeCore(params, cache.components)
        )
    cache.tree_core = core
    for component in cache.components.values():
        component.tree_core = core
    core.set_hicache_enabled()
    core.is_write_back = True
    cache.host_memory_mode = "cache"
    cache.buffer_pipeline = None
    cache.cache_controller = SimpleNamespace(write_policy="write_back")
    cache.ongoing_write_through = {}
    cache._build_backup_sidecar = Mock(return_value=[])
    host_pool = Mock()
    host_pool.available_size.return_value = 0 if backup == "host_pressure" else 1
    cache.components[ComponentType.MAMBA]._mamba_pool_host = host_pool
    cache.host_pool_group = Mock()
    cache.host_pool_group.get_pool.return_value = host_pool
    _mamba_insert(core, [1], [10], 7)
    node = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node
    _mamba_insert(core, [1, 2], [10, 11], 8)
    events = []

    def assert_state_resident():
        assert core.get_component_device_value(node, ComponentType.MAMBA).tolist() == [
            7
        ]
        assert core.get_component_device_value(node, ComponentType.FULL).tolist() == [
            10
        ]

    def evict_host(count, component):
        assert (count, component) == (1, ComponentType.MAMBA)
        assert_state_resident()
        events.append("host_evict")
        host_pool.available_size.return_value = 1
        return 1

    def write(node_id, values, transfers, sidecars):
        assert node_id == node and values.tolist() == [10] and sidecars == []
        assert_state_resident()
        events.append("write")
        if backup == "raised":
            raise RuntimeError("DMA submission failed")
        if backup == "failed":
            return None
        (state,) = transfers[ComponentType.MAMBA]
        assert state.device_indices.tolist() == [7]
        state.host_indices = torch.tensor([70], dtype=torch.int64)
        return torch.tensor([100], dtype=torch.int64)

    def acknowledge(write_back):
        assert write_back
        assert_state_resident()
        assert node in cache.ongoing_write_through
        events.append("ack")
        if backup == "ack_raised":
            raise RuntimeError("DMA completion failed")
        core.finish_write_through([node], ack_id=node)
        cache.ongoing_write_through.clear()

    def free_values(device_frees, host_frees):
        assert not host_frees
        assert [t.tolist() for t in device_frees.pop(ComponentType.MAMBA)] == [[7]]
        assert not device_frees
        events.append("free")

    cache.evict_host = evict_host
    cache._execute_kv_backup = write
    cache.writing_check = acknowledge
    cache._free_values = free_values
    tracker = defaultdict(int)
    core.evict_device_start(ComponentType.MAMBA, 1)
    try:
        if backup in ("raised", "ack_raised"):
            with pytest.raises(RuntimeError, match="DMA .* failed"):
                cache._evict_device_next_node(ComponentType.MAMBA, tracker)
            assert events == (["write"] if backup == "raised" else ["write", "ack"])
            assert not tracker[ComponentType.MAMBA]
            assert_state_resident()
            return
        assert cache._evict_device_next_node(ComponentType.MAMBA, tracker) == (
            None,
            True,
        )
    finally:
        core.evict_device_end(ComponentType.MAMBA)
    assert tracker[ComponentType.MAMBA] == 1 and tracker[ComponentType.FULL] == 0
    assert core.get_component_device_value(node, ComponentType.FULL).tolist() == [10]
    assert core.get_component_device_value(node, ComponentType.MAMBA) is None
    if backup == "failed":
        assert events == ["write", "free"]
        assert not core.component_has_host_value_only(node, ComponentType.MAMBA)
    else:
        expected = ["write", "ack", "free"]
        assert events == (
            ["host_evict"] + expected if backup == "host_pressure" else expected
        )
        matched = core.match_prefix(MatchPrefixParams(key=_key([1])))
        assert matched.best_match_node == node and matched.mamba_host_hit_length == 1
        (state,) = core.build_hicache_transfers(
            ComponentType.MAMBA, node, CacheTransferPhase.LOAD_BACK
        )
        assert state.host_indices.tolist() == [70] and state.nodes_to_load == [node]
    core.sanity_check([], [])


def test_mamba_path_cap_evicts_excess_states_through_the_adapter():
    from collections import defaultdict

    from sglang.srt.mem_cache.unified_cache.cache_action import (
        MambaEvictExcessPathStates,
    )

    core = _mamba_tree_core_with_cap(1)
    _mamba_insert(core, [1], [10], 7)
    _mamba_insert(core, [1, 2], [10, 11], 8)
    result = _mamba_insert(core, [1, 2, 3], [10, 11, 12], 9)
    (action,) = [
        a for a in result.cache_actions if isinstance(a, MambaEvictExcessPathStates)
    ]
    device_frees, host_frees = defaultdict(list), defaultdict(list)
    core.evict_excess_path_states(action.tail_node_id, device_frees, host_frees)
    # The two shallow states free; the tail's survives the soft cap.
    assert sorted(t.item() for t in device_frees[ComponentType.MAMBA]) == [7, 8]
    assert host_frees == {}
    assert core.mamba_evictable_size() == 1


def test_eagle_with_mamba_falls_back_to_the_unigram_binding():
    core = _mamba_tree_core(is_eagle=True)
    assert core.is_eagle is False
    assert type(core._binding._inner) is mem_cache.RustUnifiedTreeCoreBinding


def test_mamba_prefetch_commit_round_trips_through_the_adapter():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _mamba_tree_core()
    core.set_hicache_enabled()
    root = core.match_prefix(MatchPrefixParams(key=_key([99]))).best_match_node
    insert_result = core.insert_host(
        root, _key([1]), torch.tensor([100], dtype=torch.int64), ["h0"]
    )

    def commit(host_indices, loaded_pages):
        actions = []
        core.commit_hicache_transfers(
            root,
            CacheTransferPhase.PREFETCH,
            {
                ComponentType.MAMBA: [
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=torch.tensor(host_indices, dtype=torch.int64),
                    )
                ]
            },
            cache_actions=actions,
            insert_result=insert_result,
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=1, extra_pool_hit_pages={PoolName.MAMBA: loaded_pages}
            ),
        )
        return actions

    # The loaded buffer attaches to the inserted node.
    assert commit([50], loaded_pages=1) == []
    assert not insert_result.mamba_exist

    # A second buffer cannot attach: it frees and flags the caller.
    (free,) = commit([51], loaded_pages=1)
    assert isinstance(free, FreeComponentHostSlot)
    assert free.host_indices[0].tolist() == [51]
    assert insert_result.mamba_exist

    # The hosted slot now publishes to storage keyed by the trailing hash.
    (xfer,) = core.build_hicache_transfers(
        ComponentType.MAMBA, root_child(core), CacheTransferPhase.BACKUP_STORAGE
    )
    assert xfer.keys == ["h0"]
    assert xfer.hit_policy == PoolHitPolicy.TRAILING_PAGES


def root_child(core):
    """The single inserted node under the default root."""
    return core.match_prefix(MatchPrefixParams(key=_key([1]))).last_host_node


def test_split_of_a_write_through_pending_node_crosses_the_replace_action():
    core = _tree_core()
    core.set_hicache_enabled()
    _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 4]))).best_match_node
    core.mark_write_through_pending([leaf], ack_id=leaf)
    # A divergent prefix splits the pending node; the publish list must follow.
    result = _insert(core, [1, 2], [10, 11])
    (replace,) = [
        action
        for action in result.cache_actions
        if isinstance(action, ReplaceWriteThroughOnNodeSplit)
    ]
    assert replace.ack_id == leaf
    assert replace.old_node_id == leaf
    assert replace.new_child_node_id == leaf
    assert replace.new_node_id != leaf


def test_write_through_threshold_assignment_reaches_the_core():
    core = _tree_core()
    core.set_hicache_enabled()
    # HiCache init lowers the threshold after construction; the second hit on
    # the same prefix must then emit the write-through backup.
    core.write_through_threshold = 2
    assert _insert(core, [1, 2], [10, 11]).cache_actions == []
    result = _insert(core, [1, 2], [10, 11])
    assert any(isinstance(action, BackupKV) for action in result.cache_actions)


def test_swa_prefetch_commit_end_to_end():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _swa_tree_core(window=4)
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    anchor = core.match_prefix(MatchPrefixParams(key=_key([99]))).best_match_node

    # Without planned staging the SWA pool takes no part in the fetch.
    assert (
        core.build_hicache_transfers(
            ComponentType.SWA, anchor, CacheTransferPhase.PREFETCH
        )
        is None
    )

    # The build carries the planned staging as placeholder keys, trailing-pages
    # policy; the host buffer is attached once the hit is known.
    (xfer,) = core.build_hicache_transfers(
        ComponentType.SWA,
        anchor,
        CacheTransferPhase.PREFETCH,
        staging_tokens=2,
    )
    assert xfer.name == PoolName.SWA
    assert xfer.keys == ["__placeholder__", "__placeholder__"]
    assert xfer.hit_policy == PoolHitPolicy.TRAILING_PAGES
    assert xfer.host_indices is None

    # The prefetched suffix lands as one host node; its SWA host is a tombstone.
    insert_result = core.insert_host(
        anchor,
        _key([1, 2, 3]),
        torch.tensor([100, 101, 102], dtype=torch.int64),
        ["h0", "h1", "h2"],
    )
    assert insert_result.total_len == 3
    assert insert_result.inserted_host_node is not None

    def commit(host_indices, loaded_pages):
        actions = []
        core.commit_hicache_transfers(
            anchor,
            CacheTransferPhase.PREFETCH,
            {
                ComponentType.SWA: [
                    PoolTransfer(
                        name=PoolName.SWA,
                        host_indices=torch.tensor(host_indices, dtype=torch.int64),
                    )
                ]
            },
            cache_actions=actions,
            insert_result=insert_result,
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=3, extra_pool_hit_pages={PoolName.SWA: loaded_pages}
            ),
        )
        return actions

    # Underloaded window (1 of 2 pages): all-or-nothing frees the whole buffer.
    (free,) = commit([30, 31], loaded_pages=1)
    assert isinstance(free, FreeComponentHostSlot)
    assert free.component_type == ComponentType.SWA
    assert free.host_indices[0].tolist() == [30, 31]

    # A full window splits the partially covered node and attaches its tail.
    assert commit([40, 41], loaded_pages=2) == []

    # The window is hosted now: a re-prefetched buffer releases instead.
    (release,) = commit([50, 51], loaded_pages=2)
    assert isinstance(release, FreeComponentHostSlot)
    assert release.host_indices[0].tolist() == [50, 51]


def test_swa_locked_overlap_defers_through_the_recover_action():
    core = _swa_tree_core(window=8)
    first = _insert(core, [1, 2], [10, 11])
    node = first.cache_actions[0].node_id
    core.inc_lock_ref(node)
    result = _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2]),
            value=torch.tensor([20, 21], dtype=torch.int64),
        ),
    )
    (recover,) = result.cache_actions
    assert isinstance(recover, RecoverSWAWithLockedFull)
    assert recover.node_id == node
    assert recover.kept_full.tolist() == [10, 11]
    assert recover.incoming_full.tolist() == [20, 21]


def test_component_device_value_round_trips():
    core = _swa_tree_core(window=8)
    first = _insert(core, [1, 2], [10, 11])
    node = first.cache_actions[0].node_id
    assert core.get_component_device_value(node, ComponentType.SWA) is None
    core.set_component_device_value(
        node, ComponentType.SWA, torch.tensor([50, 51], dtype=torch.int64)
    )
    stored = core.get_component_device_value(node, ComponentType.SWA)
    assert stored.tolist() == [50, 51]


def test_lock_uuid_round_trips_through_dec_lock_ref():
    core = _swa_tree_core(window=2)
    first = _insert(core, [1, 2, 3], [10, 11, 12])
    # The window cap split the leaf: rebuild the in-window nodes' SWA values.
    for action in first.cache_actions:
        core.set_component_device_value(
            action.node_id,
            ComponentType.SWA,
            torch.arange(50, 50 + action.source_value.numel(), dtype=torch.int64),
        )
    node = first.cache_actions[-1].node_id
    result = core.inc_lock_ref(node)
    assert result.component_lock_uuids[ComponentType.SWA] is not None
    assert not result.component_host_lock_uuids
    # The locked window is protected SWA accounting, visible through the binding.
    assert core.swa_protected_size() == 2
    assert core.swa_evictable_size() == 1
    core.dec_lock_ref(
        node,
        result.to_dec_params(),
    )
    # The uuid-bounded release returned the window to evictable.
    assert core.swa_protected_size() == 0
    assert core.swa_evictable_size() == 3
    # A repeat acquire reuses the stamped uuid.
    again = core.inc_lock_ref(node)
    assert (
        again.component_lock_uuids[ComponentType.SWA]
        == result.component_lock_uuids[ComponentType.SWA]
    )
    skipped = core.inc_lock_ref(node, skip_lock_components=(ComponentType.SWA,))
    assert ComponentType.SWA not in skipped.component_lock_uuids
    core.dec_lock_ref(node, skipped.to_dec_params())
    assert core.swa_protected_size() == 2
    core.dec_lock_ref(node, again.to_dec_params())
    core.sanity_check([], [])


@pytest.mark.parametrize("swa_only", [False, True])
@pytest.mark.parametrize("missing_receipt", [False, True])
def test_swa_tombstones_cross_the_binding_and_release_balanced(
    swa_only, missing_receipt
):
    core = _swa_tree_core(window=8)
    _insert(core, [1, 2], [10, 11])
    second = _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    leaf = second.cache_actions[-1].node_id
    # Only the leaf carries SWA; the ancestor tombstone is counted too, and
    # the under-window walk reaches the root without stamping a uuid.
    core.set_component_device_value(
        leaf, ComponentType.SWA, torch.tensor([52, 53], dtype=torch.int64)
    )
    result = core.inc_lock_ref(leaf)
    assert result.component_lock_uuids[ComponentType.SWA] is None
    assert core.swa_protected_size() == 2
    release = core.dec_swa_lock_only if swa_only else core.dec_lock_ref
    if missing_receipt:
        missing = DecLockRefParams(node_id=leaf)
        if swa_only:
            # Early release always walks SWA, even if the receipt marks it skipped.
            missing.skipped_lock_components = (ComponentType.SWA,)
        with pytest.raises(RuntimeError, match="no entry found for key") as error:
            release(leaf, missing)
        assert type(error.value.__cause__).__name__ == "PanicException"
        return  # A Rust ownership violation poisons the core.
    release(leaf, result.to_dec_params())
    assert core.swa_protected_size() == 0
    if swa_only:
        core.dec_lock_ref(leaf, DecLockRefParams(node_id=leaf), skip_swa=True)
    core.sanity_check([], [])


def test_dec_swa_lock_only_frees_flow_after_the_full_release():
    core = _swa_tree_core(window=2)
    first = _insert(core, [1, 2], [10, 11])
    node = first.cache_actions[0].node_id
    core.set_component_device_value(
        node, ComponentType.SWA, torch.tensor([50, 51], dtype=torch.int64)
    )
    result = core.inc_lock_ref(node)
    # A non-None boundary: the window fills at the locked node itself.
    assert result.component_lock_uuids[ComponentType.SWA] is not None
    # The FULL lock releases first (skip_swa), then the early window release
    # finds a fully unlocked device leaf and evicts it in place.
    core.dec_lock_ref(node, result.to_dec_params(), skip_swa=True)
    device_frees: dict = {}
    host_frees: dict = {}
    _accumulate_step(
        core.dec_swa_lock_only(node, result.to_dec_params()),
        {},
        device_frees,
        host_frees,
    )
    assert [t.tolist() for t in device_frees[ComponentType.SWA]] == [[10, 11]]
    assert core.get_component_device_value(node, ComponentType.SWA) is None


def test_dec_swa_lock_only_releases_once_and_a_repeat_dies_loud():
    core = _swa_tree_core(window=2)
    first = _insert(core, [1, 2, 3], [10, 11, 12])
    for action in first.cache_actions:
        core.set_component_device_value(
            action.node_id,
            ComponentType.SWA,
            torch.arange(50, 50 + action.source_value.numel(), dtype=torch.int64),
        )
    node = first.cache_actions[-1].node_id
    result = core.inc_lock_ref(node)
    device_frees: dict = {}
    host_frees: dict = {}
    _accumulate_step(
        core.dec_swa_lock_only(node, result.to_dec_params()),
        {},
        device_frees,
        host_frees,
    )
    # The FULL lock still protects the path: the SWA release frees nothing
    # and the rebuilt values survive.
    assert device_frees == {}
    assert core.get_component_device_value(node, ComponentType.SWA) is not None
    # A repeat release of the same window is a protocol violation and dies
    # at the segment instead of silently walking it.
    with pytest.raises(BaseException, match="SWA window release hit lock_ref=0"):
        core.dec_swa_lock_only(node, result.to_dec_params())


def test_swa_rebuild_applies_through_the_python_allocator():
    cache, allocator = _swa_cache(window=8)
    full = allocator.alloc(4)
    result = cache.insert(InsertParams(key=_key([1, 2, 3, 4]), value=full))
    assert result.prefix_len == 0
    node = cache.match_prefix(MatchPrefixParams(key=_key([1, 2, 3, 4]))).best_match_node
    assert node != 0, "the SWA-covered match must reach the leaf"
    # The cache executed SWARebuild through the allocator: the node holds the
    # full slice's SWA translation.
    stored = cache.tree_core.get_component_device_value(node, ComponentType.SWA)
    expected = allocator.translate_loc_from_full_to_swa(full)
    assert stored is not None
    assert stored.tolist() == expected.tolist()
    assert (allocator.full_to_swa_index_mapping[full.to(torch.int64)] > 0).all()


def test_recover_with_locked_full_applies_through_the_python_allocator():
    cache, allocator = _swa_cache(window=8)
    kept = allocator.alloc(2)
    cache.insert(InsertParams(key=_key([1, 2]), value=kept))
    node = cache.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    assert node != 0
    lock = cache.inc_lock_ref(node)
    # The decode advanced past the window: the SWA lock releases early, then
    # window eviction tombstones the SWA slot under the FULL lock (the state a
    # locked-full overlap recovers from); its frees return to the allocator.
    cache.dec_swa_lock_only(node, lock.to_dec_params())
    tracker = {ComponentType.FULL: 0, ComponentType.SWA: 0}
    device_frees: dict = {}
    host_frees: dict = {}
    cache.tree_core.evict_device_start(ComponentType.SWA, 100)
    step = cache.tree_core.evict_device_next_node(ComponentType.SWA, tracker)
    assert step.node_id is None
    _accumulate_step(step, tracker, device_frees, host_frees)
    cache.tree_core.evict_device_end(ComponentType.SWA)
    for freed in device_frees[ComponentType.SWA]:
        allocator.free_swa(freed)
    assert cache.tree_core.get_component_device_value(node, ComponentType.SWA) is None
    incoming = allocator.alloc(2)
    before_free = allocator.full_attn_allocator.available_size()
    cache.components[ComponentType.SWA].apply_component_action(
        RecoverSWAWithLockedFull(node_id=node, kept_full=kept, incoming_full=incoming)
    )
    # The locked full keeps its slots, remapped onto the incoming full's SWA
    # translation; the incoming full is freed back to the allocator.
    stored = cache.tree_core.get_component_device_value(node, ComponentType.SWA)
    assert stored.tolist() == allocator.translate_loc_from_full_to_swa(kept).tolist()
    assert (allocator.full_to_swa_index_mapping[incoming.to(torch.int64)] == 0).all()
    assert (
        allocator.full_attn_allocator.available_size() == before_free + incoming.numel()
    )


# ==== Bigram (EAGLE) wiring ====


def _bigram_tree_core(**params_overrides) -> RustUnifiedTreeCore:
    # Without mamba, the core honors is_eagle and selects the bigram binding.
    params = dict(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        is_eagle=True,
        tree_components=(ComponentType.FULL,),
    )
    params.update(params_overrides)
    return RustUnifiedTreeCore(CacheInitParams(**params))


def _bigram_key(token_ids: list[int]) -> RadixKey:
    return RadixKey(array("q", token_ids), is_bigram=True)


def test_bigram_insert_then_a_longer_match_returns_the_inserted_prefix():
    core = _bigram_tree_core()
    # 4 raw tokens = 3 bigram atoms, so the value carries 3 indices.
    result = _pump_insert(
        core,
        InsertParams(
            key=_bigram_key([1, 2, 3, 4]),
            value=torch.tensor([10, 11, 12], dtype=torch.int64),
        ),
    )
    assert result.prefix_len == 0
    matched = core.match_prefix(MatchPrefixParams(key=_bigram_key([1, 2, 3, 4, 5])))
    assert matched.device_indices.tolist() == [10, 11, 12]


def test_bigram_match_diverges_on_the_pair_not_the_token():
    core = _bigram_tree_core()
    _pump_insert(
        core,
        InsertParams(
            key=_bigram_key([1, 2, 3, 4]),
            value=torch.tensor([10, 11, 12], dtype=torch.int64),
        ),
    )
    # (1, 2) matches; (2, 9) diverges from (2, 3) despite the shared token 2.
    matched = core.match_prefix(MatchPrefixParams(key=_bigram_key([1, 2, 9])))
    assert matched.device_indices.tolist() == [10]


def test_bigram_empty_and_single_token_keys_match_nothing():
    core = _bigram_tree_core()
    _pump_insert(
        core,
        InsertParams(
            key=_bigram_key([1, 2, 3]),
            value=torch.tensor([10, 11], dtype=torch.int64),
        ),
    )
    empty = core.match_prefix(MatchPrefixParams(key=_bigram_key([])))
    assert empty.device_indices.numel() == 0
    single = core.match_prefix(MatchPrefixParams(key=_bigram_key([1])))
    assert single.device_indices.numel() == 0


def test_bigram_insert_truncates_a_raw_length_value_to_the_bigram_count():
    core = _bigram_tree_core()
    result = _pump_insert(
        core,
        InsertParams(
            key=_bigram_key([1, 2, 3]),
            value=torch.tensor([10, 11, 12], dtype=torch.int64),
        ),
    )
    assert result.prefix_len == 0
    matched = core.match_prefix(MatchPrefixParams(key=_bigram_key([1, 2, 3])))
    assert matched.device_indices.tolist() == [10, 11]


def test_bigram_insert_value_shorter_than_the_bigram_count_raises():
    core = _bigram_tree_core()
    with pytest.raises(ValueError, match="shorter than the aligned key length"):
        _pump_insert(
            core,
            InsertParams(
                key=_bigram_key([1, 2, 3, 4]),
                value=torch.tensor([10, 11], dtype=torch.int64),
            ),
        )


def test_stale_inspection_handles_raise_key_error_or_report_absence():
    from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

    from sglang.srt.mem_cache.unified_cache.components import EvictLayer

    core = RustUnifiedTreeCoreInspector(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    stale_root = core.root_node_handle()
    core.reset()
    live_root = core.root_node_handle()

    operations = {
        "get_parent_node_id": lambda: core.get_parent_node_id(stale_root),
        "get_child_node_ids": lambda: core.get_child_node_ids(stale_root),
        "get_node_key_length": lambda: core.get_node_key_length(stale_root),
        "get_node_token_ids": lambda: core.get_node_token_ids(stale_root),
        "is_node_key_bigram": lambda: core.is_node_key_bigram(stale_root),
        "get_component_host_value": lambda: core.get_component_host_value(
            stale_root, ComponentType.FULL
        ),
        "get_component_device_lock_ref": lambda: core.get_component_device_lock_ref(
            stale_root, ComponentType.FULL
        ),
        "get_node_hit_count": lambda: core.get_node_hit_count(stale_root),
        "get_write_through_pending_id": lambda: core.get_write_through_pending_id(
            stale_root
        ),
        "is_external_cache_stored": lambda: core.is_external_cache_stored(stale_root),
        "is_node_in_device_lru": lambda: core.is_node_in_device_lru(
            stale_root, ComponentType.FULL
        ),
        "is_node_in_host_lru": lambda: core.is_node_in_host_lru(
            stale_root, ComponentType.FULL
        ),
        "is_device_leaf": lambda: core.is_device_leaf(stale_root),
        "set_node_hash_values": lambda: core.set_node_hash_values(stale_root, None),
        "set_component_device_value_raw": lambda: core.set_component_device_value_raw(
            stale_root, ComponentType.FULL, None
        ),
        "set_component_host_value_raw": lambda: core.set_component_host_value_raw(
            stale_root, ComponentType.FULL, None
        ),
        "set_component_device_lock_ref": lambda: core.set_component_device_lock_ref(
            stale_root, ComponentType.FULL, 0
        ),
        "remove_node_from_device_lru": lambda: core.remove_node_from_device_lru(
            stale_root, ComponentType.FULL
        ),
        "insert_node_into_host_lru": lambda: core.insert_node_into_host_lru(
            stale_root, ComponentType.FULL
        ),
        "update_duplicate_tracking": lambda: core.update_duplicate_tracking(stale_root),
        "evict_component": lambda: core.evict_component(
            stale_root, ComponentType.FULL, EvictLayer.DEVICE
        ),
        "validate_cascade_evict": lambda: core.validate_cascade_evict(
            stale_root, ComponentType.FULL, EvictLayer.DEVICE
        ),
        "cleanup_tombstone_ancestors": lambda: core.cleanup_tombstone_ancestors(
            stale_root
        ),
        "build_backup_node_ids": lambda: core.build_backup_node_ids(stale_root),
    }
    for name, operation in operations.items():
        with pytest.raises(KeyError) as exc_info:
            operation()
        assert exc_info.value.args == (stale_root,), name
        assert core.is_root(live_root), name

    disabled_component_operations = {
        "get_component_host_value": lambda: core.get_component_host_value(
            stale_root, ComponentType.SWA
        ),
        "get_component_device_lock_ref": lambda: core.get_component_device_lock_ref(
            stale_root, ComponentType.SWA
        ),
        "set_component_device_value_raw": lambda: core.set_component_device_value_raw(
            stale_root, ComponentType.SWA, None
        ),
        "set_component_host_value_raw": lambda: core.set_component_host_value_raw(
            stale_root, ComponentType.SWA, None
        ),
        "set_component_device_lock_ref": lambda: core.set_component_device_lock_ref(
            stale_root, ComponentType.SWA, 0
        ),
        "remove_node_from_device_lru": lambda: core.remove_node_from_device_lru(
            stale_root, ComponentType.SWA
        ),
        "insert_node_into_host_lru": lambda: core.insert_node_into_host_lru(
            stale_root, ComponentType.SWA
        ),
        "evict_component": lambda: core.evict_component(
            stale_root, ComponentType.SWA, EvictLayer.DEVICE
        ),
        "validate_cascade_evict": lambda: core.validate_cascade_evict(
            stale_root, ComponentType.SWA, EvictLayer.DEVICE
        ),
    }
    for name, operation in disabled_component_operations.items():
        with pytest.raises(KeyError) as exc_info:
            operation()
        assert exc_info.value.args == (stale_root,), name
        assert core.is_root(live_root), name

    assert not core.contains_node(stale_root)
    assert not core.is_device_evictable_leaf(stale_root)
    assert not core.is_host_evictable_leaf(stale_root)
    assert not core.is_node_in_device_lru(stale_root, ComponentType.SWA)
    assert not core.is_node_in_host_lru(stale_root, ComponentType.SWA)


# ---- SWA branching-point caching ----


def _swa_hicache_core(window: int = 8) -> RustUnifiedTreeCore:
    core = _swa_tree_core(window=window)
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    return core


def test_insert_reports_whether_it_reached_the_swa_branch_boundary():
    for branching_seqlen, expected in [(2, True), (3, False), (None, False)]:
        core = _swa_hicache_core()
        result = _pump_insert(
            core,
            InsertParams(
                key=_key([1, 2]),
                value=torch.tensor([10, 11], dtype=torch.int64),
                swa_branching_seqlen=branching_seqlen,
            ),
        )
        assert result.swa_branch_inserted is expected, branching_seqlen


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
