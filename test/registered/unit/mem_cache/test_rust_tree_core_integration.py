"""Integration tests driving the real compiled Rust mem_cache extension."""

import hashlib
import shutil
import sys
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=17, suite="base-a-test-cpu")

if shutil.which("cargo") is None:
    pytest.skip("the rust backend builds with cargo", allow_module_level=True)

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    StorageMedium,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
from sglang.srt.mem_cache.rust_tree_core.extension import bindings as mem_cache
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
from sglang.srt.mem_cache.utils import hash_str_to_int64
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


def test_stale_handle_operations_raise_key_error_without_poisoning_the_core():
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase

    core = _tree_core()
    stale_root = core.root_node_handle()
    core.reset()
    live_root = core.root_node_handle()

    operations = (
        lambda: core.demote(stale_root),
        lambda: core.build_hicache_transfers(
            ComponentType.FULL, stale_root, CacheTransferPhase.BACKUP_STORAGE
        ),
        lambda: core.build_load_back_spec(stale_root),
        lambda: core.get_hash_values(stale_root),
        lambda: core.dfs_weight_order([stale_root]),
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
    core.inc_lock_ref(matched.best_match_node)
    assert core.protected_size() == 5
    assert core.evictable_size() == 0
    core.dec_lock_ref(matched.best_match_node)
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


def test_external_cache_linker_is_rejected():
    core = _tree_core()
    assert core.enable_external_cache_linker is False
    with pytest.raises(ValueError, match="External cache linker"):
        core.enable_external_cache_linker = True
    assert core.enable_external_cache_linker is False


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


def test_empty_match_result_is_root_anchored():
    core = _tree_core()
    empty = core.empty_match_result
    assert empty.device_indices.numel() == 0
    assert empty.host_hit_length == 0
    probe = core.match_prefix(MatchPrefixParams(key=_key([9])))
    assert empty.best_match_node == probe.best_match_node
    assert empty.last_device_node == probe.last_device_node
    assert empty.last_host_node == probe.last_host_node


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


def test_host_lock_refs_round_trip():
    core = _tree_core()
    core.set_hicache_enabled()
    _insert(core, [1], [10])
    leaf = core.match_prefix(MatchPrefixParams(key=_key([1]))).best_match_node
    core.commit_backup(leaf, torch.tensor([100], dtype=torch.int64), {})
    core.inc_host_lock_ref(leaf)
    core.dec_host_lock_ref(leaf)
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
        BlockStoredWithMetadata(
            block_hashes=hashes,
            parent_block_hash=None,
            token_ids=[1, 2, 7, 8],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            metadata=BlockStoredMetadata(cache_salt="tenant-a"),
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
        BlockStoredWithMetadata(
            block_hashes=hashes,
            parent_block_hash=None,
            token_ids=[(1, 2), (2, 3), (3, 4), (4, 5)],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            metadata=BlockStoredMetadata(cache_salt="tenant-a"),
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
    _insert(core, [1, 2], [10, 11])
    _insert(core, [1, 2, 7, 8], [10, 11, 12, 13])
    parent = core.match_prefix(MatchPrefixParams(key=_key([1, 2]))).best_match_node
    child = core.match_prefix(MatchPrefixParams(key=_key([1, 2, 7, 8]))).best_match_node
    core.commit_backup(parent, torch.tensor([100, 101], dtype=torch.int64), {})
    core.commit_backup(child, torch.tensor([102, 103], dtype=torch.int64), {})

    spec = core.build_storage_backup_spec(child, pass_prefix_keys=True)
    assert spec.host_value.tolist() == [102, 103]
    assert spec.token_ids == array("q", [7, 8])
    parent_hashes = mem_cache.get_hash_str(array("q", [1, 2]), None, 2)
    assert spec.prefix_keys == parent_hashes
    assert spec.hash_value == mem_cache.get_hash_str(
        array("q", [7, 8]), parent_hashes[-1], 2
    )
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


def test_swa_without_a_window_is_rejected_through_the_adapter():
    with pytest.raises(ValueError, match="requires swa_sliding_window_size"):
        _tree_core(tree_components=(ComponentType.FULL, ComponentType.SWA))


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


def _swa_cache(window: int = 8, page_size: int = 1):
    """A real UnifiedRadixCache on the Rust tree core with a real SWA allocator."""
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
    allocator = SWATokenToKVPoolAllocator(
        size=64,
        size_swa=64,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
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


def test_swa_straddling_insert_crosses_the_boundary_actions():
    core = _swa_tree_core(window=8)
    _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    result = _pump_insert(
        core,
        InsertParams(
            key=_key([1, 2, 3, 4]),
            value=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
            swa_evicted_seqlen=2,
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
    skipped = core.inc_lock_ref(node, skip_lock_components=(ComponentType.MAMBA,))
    assert skipped.skip_lock_node_ids == {ComponentType.MAMBA: {node}}
    assert core.mamba_protected_size() == 1

    released = core.dec_swa_lock_only(
        node,
        skipped.swa_uuid_for_lock,
        skip_lock_node_ids=skipped.skip_lock_node_ids,
    )
    assert dict(released.device_frees) == {}
    assert dict(released.host_frees) == {}
    assert core.mamba_protected_size() == 1

    core.dec_lock_ref(node, skipped.to_dec_params(), skip_swa=True)
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
    _accumulate_step(step, tracker, device_frees, host_frees)

    step = core.evict_device_next_node(ComponentType.MAMBA, tracker)
    leaf = step.node_id
    _accumulate_step(step, tracker, device_frees, host_frees)
    assert leaf is not None
    core.evict_device_end(ComponentType.MAMBA)
    assert tracker[ComponentType.MAMBA] == 1
    assert torch.cat(device_frees[ComponentType.MAMBA]).tolist() == [7]
    assert core.mamba_evictable_size() == 1

    # A pre-eviction node handle locked after the tombstoning lands in the
    # skip map, and the replay keeps the release off it.
    lock = core.inc_lock_ref(internal)
    assert internal in lock.skip_lock_node_ids[ComponentType.MAMBA]
    core.dec_lock_ref(internal, lock.to_dec_params())
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
    assert type(core._binding) is mem_cache.RustUnifiedTreeCoreBinding


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

    # The build wraps the host buffer with placeholder keys, trailing-pages policy.
    (xfer,) = core.build_hicache_transfers(
        ComponentType.SWA,
        anchor,
        CacheTransferPhase.PREFETCH,
        host_indices=torch.tensor([30, 31], dtype=torch.int64),
    )
    assert xfer.name == PoolName.SWA
    assert xfer.keys == ["__placeholder__", "__placeholder__"]
    assert xfer.hit_policy == PoolHitPolicy.TRAILING_PAGES
    assert xfer.host_indices.tolist() == [30, 31]

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
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams

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
    assert result.swa_uuid_for_lock is not None
    assert result.swa_uuid_for_host_lock is None
    # The locked window is protected SWA accounting, visible through the binding.
    assert core.swa_protected_size() == 2
    assert core.swa_evictable_size() == 1
    core.dec_lock_ref(
        node,
        DecLockRefParams(
            swa_uuid_for_lock=result.swa_uuid_for_lock,
            skip_lock_node_ids=result.skip_lock_node_ids,
        ),
    )
    # The uuid-bounded release returned the window to evictable.
    assert core.swa_protected_size() == 0
    assert core.swa_evictable_size() == 3
    # A repeat acquire reuses the stamped uuid.
    again = core.inc_lock_ref(node)
    assert again.swa_uuid_for_lock == result.swa_uuid_for_lock


def test_swa_skip_map_crosses_the_binding_and_replays():
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams

    core = _swa_tree_core(window=8)
    _insert(core, [1, 2], [10, 11])
    second = _insert(core, [1, 2, 3, 4], [10, 11, 12, 13])
    leaf = second.cache_actions[-1].node_id
    # Only the leaf carries SWA; its ancestor is recorded as a tombstone skip.
    core.set_component_device_value(
        leaf, ComponentType.SWA, torch.tensor([52, 53], dtype=torch.int64)
    )
    result = core.inc_lock_ref(leaf)
    assert result.skip_lock_node_ids[ComponentType.SWA]
    core.dec_lock_ref(
        leaf,
        DecLockRefParams(
            swa_uuid_for_lock=result.swa_uuid_for_lock,
            skip_lock_node_ids=result.skip_lock_node_ids,
        ),
    )
    assert core.swa_protected_size() == 0


def test_dec_swa_lock_only_frees_flow_after_the_full_release():
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams

    core = _swa_tree_core(window=2)
    first = _insert(core, [1, 2], [10, 11])
    node = first.cache_actions[0].node_id
    core.set_component_device_value(
        node, ComponentType.SWA, torch.tensor([50, 51], dtype=torch.int64)
    )
    result = core.inc_lock_ref(node)
    # The FULL lock releases first (skip_swa), then the early window release
    # finds a fully unlocked device leaf and evicts it in place.
    core.dec_lock_ref(
        node,
        DecLockRefParams(skip_lock_node_ids=result.skip_lock_node_ids),
        skip_swa=True,
    )
    device_frees: dict = {}
    host_frees: dict = {}
    _accumulate_step(
        core.dec_swa_lock_only(node, result.swa_uuid_for_lock),
        {},
        device_frees,
        host_frees,
    )
    assert [t.tolist() for t in device_frees[ComponentType.SWA]] == [[10, 11]]
    assert core.get_component_device_value(node, ComponentType.SWA) is None


def test_dec_swa_lock_only_returns_the_window_frees():
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
        core.dec_swa_lock_only(node, result.swa_uuid_for_lock),
        {},
        device_frees,
        host_frees,
    )
    # The FULL lock still protects the path: the SWA release frees nothing and
    # the rebuilt values survive; a repeat release is a no-op.
    assert device_frees == {}
    assert core.get_component_device_value(node, ComponentType.SWA) is not None
    _accumulate_step(
        core.dec_swa_lock_only(node, result.swa_uuid_for_lock),
        {},
        device_frees,
        host_frees,
    )
    assert device_frees == {}


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
    cache.dec_swa_lock_only(node, lock.swa_uuid_for_lock)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
