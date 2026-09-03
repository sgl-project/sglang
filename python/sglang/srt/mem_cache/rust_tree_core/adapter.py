"""The Rust TreeCore adapter: satisfies ``UnifiedTreeCoreInterface`` over the
``mem_cache`` extension's ``RustUnifiedTreeCoreBinding``."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    StorageMedium,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_tree_core.extension import bindings
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    MambaEvictExcessPathStates,
    RebuildFullToSWAMapping,
    RecoverSWAWithLockedFull,
    ReplaceWriteThroughOnNodeSplit,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import StorageBackupSpec
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
    BufferBackupState,
    DecSwaLockOnlyResult,
    DemoteResult,
    DriveHostEvictionResult,
    DropSubtreeNoHostResult,
    EvictDeviceLeafResult,
    EvictDeviceNextNodeResult,
    InsertStepResult,
    NodeId,
    RadixCacheWalkResult,
    UnifiedTreeCoreInterface,
)
from sglang.srt.runtime_context import get_exec, mamba_cache_chunk_size

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import PoolTransferResult
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


def _radix_key_buffer(key: RadixKey) -> array:
    """The key's token ids honoring `limit`; view-independent since the
    binding derives its own atoms."""
    token_ids = key.raw_token_ids()
    assert isinstance(token_ids, array) and token_ids.typecode == "q", (
        f"tree keys must carry array('q') token ids, got {type(token_ids).__name__}"
    )
    return token_ids


def _kv_event_from_tagged(event: tuple):
    """Build the Python KV cache event for one of the binding's tagged tuples."""
    tag = event[0]
    if tag == "block_stored":
        event_args = dict(
            block_hashes=event[1],
            parent_block_hash=event[2],
            token_ids=event[3],
            block_size=event[4],
            lora_id=None,
            medium=StorageMedium(event[5]),
        )
        if event[6] is None:
            return BlockStored(**event_args)
        return BlockStoredWithMetadata(
            **event_args,
            metadata=BlockStoredMetadata(cache_salt=event[6]),
        )
    if tag == "block_removed":
        return BlockRemoved(block_hashes=event[1], medium=StorageMedium(event[2]))
    if tag == "all_blocks_cleared":
        return AllBlocksCleared()
    raise ValueError(f"unknown kv event tag: {tag}")


def _cache_action_from_tagged(action: tuple) -> CacheAction:
    """Build the Python CacheAction for one of the binding's tagged tuples."""
    tag = action[0]
    if tag == "free_device_kv":
        return FreeDeviceKV(indices=list(action[1]))
    if tag == "free_device_kv_full_only":
        return FreeDeviceKVFullOnly(indices=list(action[1]))
    if tag == "backup_kv":
        return BackupKV(node_ids=list(action[1]))
    if tag == "mamba_evict_excess_path_states":
        return MambaEvictExcessPathStates(tail_node_id=action[1])
    if tag == "replace_write_through_on_node_split":
        return ReplaceWriteThroughOnNodeSplit(
            ack_id=action[1],
            old_node_id=action[2],
            new_node_id=action[3],
            new_child_node_id=action[4],
        )
    if tag == "free_component_device_slot":
        return FreeComponentDeviceSlot(
            component_type=ComponentType(action[1]), indices=list(action[2])
        )
    if tag == "free_component_host_slot":
        return FreeComponentHostSlot(
            component_type=ComponentType(action[1]), host_indices=list(action[2])
        )
    if tag == "rebuild_full_to_swa_mapping":
        return RebuildFullToSWAMapping(
            full_indices=list(action[1]), swa_indices=list(action[2])
        )
    if tag == "recover_swa_with_locked_full":
        return RecoverSWAWithLockedFull(
            node_id=action[1], kept_full=action[2], incoming_full=action[3]
        )
    if tag == "swa_rebuild":
        return SWARebuild(node_id=action[1], source_value=action[2])
    raise ValueError(f"unknown cache action tag: {tag}")


def _cache_actions_from_tagged(actions: Sequence[tuple]) -> list[CacheAction]:
    """Build the Python CacheActions for the binding's tagged tuples, in order."""
    return [_cache_action_from_tagged(action) for action in actions]


def _inc_lock_ref_result_from_binding(result) -> IncLockRefResult:
    return IncLockRefResult(
        delta=result.delta,
        swa_uuid_for_lock=result.swa_uuid_for_lock,
        swa_uuid_for_host_lock=result.swa_uuid_for_host_lock,
        skip_lock_node_ids=_skip_lock_node_ids_from_binding(result.skip_lock_node_ids),
    )


def _transfer_to_binding(transfer: PoolTransfer) -> tuple:
    """The binding's (name, host_indices, device_indices, nodes_to_load, keys,
    hit_policy) tuple."""
    return (
        transfer.name.value,
        transfer.host_indices,
        transfer.device_indices,
        transfer.nodes_to_load,
        transfer.keys,
        transfer.hit_policy.value,
    )


def _transfer_from_binding(transfer: tuple) -> PoolTransfer:
    """Build the Python PoolTransfer for one of the binding's transfer tuples."""
    name, host_indices, device_indices, nodes_to_load, keys, hit_policy = transfer
    return PoolTransfer(
        name=PoolName(name),
        host_indices=host_indices,
        device_indices=device_indices,
        keys=keys,
        hit_policy=PoolHitPolicy(hit_policy),
        nodes_to_load=nodes_to_load,
    )


def _comp_xfers_to_binding(
    comp_xfers: dict[ComponentType, list[PoolTransfer]],
) -> dict[int, list[tuple]]:
    """Rekey per-component transfers by the binding's component values."""
    return {
        int(ct): [_transfer_to_binding(x) for x in xfers]
        for ct, xfers in comp_xfers.items()
    }


def _comp_xfers_from_binding(
    comp_xfers: dict[int, list[tuple]],
) -> dict[ComponentType, list[PoolTransfer]]:
    """Rekey the binding's per-component transfer tuples by ComponentType."""
    return {
        ComponentType(ct): [_transfer_from_binding(x) for x in xfers]
        for ct, xfers in comp_xfers.items()
    }


def _insert_step_from_binding(step) -> InsertStepResult:
    """Build the interface step for the binding's step (result on the final one)."""
    result = None
    if step.result is not None:
        # A stepped insert delivers all actions through steps, never the result.
        assert not step.result.cache_actions
        result = InsertResult(
            prefix_len=step.result.prefix_len,
            last_device_node=step.result.last_device_node,
            mamba_exist=step.result.mamba_exist,
            host_insert_dropped=step.result.host_insert_dropped,
            adopted_ranges=(
                {
                    ComponentType(component_type): list(ranges)
                    for component_type, ranges in step.result.adopted_ranges.items()
                }
                if step.result.adopted_ranges is not None
                else None
            ),
        )
    return InsertStepResult(
        actions=_cache_actions_from_tagged(step.actions), result=result
    )


def _match_result_from_binding(result) -> MatchResult:
    """Build the Python MatchResult for the binding's match result."""
    return MatchResult(
        device_indices=result.device_indices,
        last_device_node=result.last_device_node_id,
        last_host_node=result.last_host_node_id,
        best_match_node=result.best_match_node_id,
        host_hit_length=result.host_hit_length,
        swa_host_hit_length=result.swa_host_hit_length,
        mamba_host_hit_length=result.mamba_host_hit_length,
        mamba_branching_seqlen=result.mamba_branching_seqlen,
        full_kv_hit_length=result.full_kv_hit_length,
        cache_actions=_cache_actions_from_tagged(result.cache_actions),
    )


def _skip_lock_node_ids_from_binding(
    skip_lock_node_ids: dict[int, set[int]],
) -> dict[ComponentType, set[int]]:
    """Rekey the binding's component-value skip map by ComponentType."""
    return {
        ComponentType(component): set(node_ids)
        for component, node_ids in skip_lock_node_ids.items()
    }


def _skip_lock_node_ids_to_binding(
    skip_lock_node_ids: dict[ComponentType, set[int]],
) -> dict[int, set[int]]:
    """Rekey a ComponentType skip map by the binding's component values."""
    return {
        int(component): set(node_ids)
        for component, node_ids in skip_lock_node_ids.items()
    }


def _tracker_to_binding(tracker: dict[ComponentType, int]) -> dict[int, int]:
    """Rekey a ComponentType tracker by the binding's component values."""
    return {int(component): freed for component, freed in tracker.items()}


def _fill_evict_result(binding_result, result):
    """Map a binding eviction step into an interface step result; both carry
    this step's per-component deltas and freed tensors."""
    for component, delta in binding_result.tracker.items():
        result.tracker[ComponentType(component)] = delta
    for component, tensors in binding_result.new_device_frees.items():
        result.device_frees[ComponentType(component)].extend(tensors)
    for component, tensors in binding_result.new_host_frees.items():
        result.host_frees[ComponentType(component)].extend(tensors)
    return result


class _RustKVCacheEventRecorder:
    """Expose the Rust event queue through the Python recorder interface."""

    def __init__(self, binding, enabled: bool):
        self._binding = binding
        self.enabled = enabled

    def record_all_cleared(self) -> None:
        self._binding.record_all_cleared_event()

    def take(self) -> list:
        return [_kv_event_from_tagged(event) for event in self._binding.take_events()]


class RustUnifiedTreeCore(UnifiedTreeCoreInterface):
    """A TreeCore backed by the Rust extension binding."""

    _bindings = bindings

    def __init__(self, params: CacheInitParams):
        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)

        # TODO(Jialin): Port session-reference-aware TreeCore support from #29173.
        if params.enable_session_radix_cache:
            raise ValueError(
                "--enable-session-radix-cache is not supported by the Rust TreeCore"
            )

        # TODO(Jialin): Port custom component registration from #25754 and
        # C128 support from #33676.
        unsupported_components = set(self.tree_components) - {
            ComponentType.FULL,
            ComponentType.SWA,
            ComponentType.MAMBA,
        }
        if unsupported_components:
            names = ", ".join(
                sorted(component.name for component in unsupported_components)
            )
            raise ValueError(f"Rust TreeCore does not support components: {names}")
        if params.component_registry_override:
            raise ValueError(
                "Rust TreeCore does not support component_registry_override"
            )

        self._page_size = params.page_size
        self.is_eagle = (
            params.is_eagle and ComponentType.MAMBA not in self.tree_components
        )

        # ``device`` is derived from the construction-time allocator; the
        # allocator/pool themselves are owned by the cache, not the tree.
        if params.token_to_kv_pool_allocator:
            device = torch.device(params.token_to_kv_pool_allocator.device)
            # A bare "cuda" means the process's current device, not cuda:0.
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.enable_kv_cache_events = params.enable_kv_cache_events
        has_mamba = ComponentType.MAMBA in self.tree_components
        mamba_max_states_per_path = (
            get_exec().mamba.mamba_max_states_per_path if has_mamba else -1
        )

        self._binding = self._binding_class()(
            self._bindings.TreeCoreInitParamsBinding(
                eviction_policy=params.eviction_policy,
                page_size=params.page_size,
                is_write_back=False,
                enable_hicache=False,
                write_through_threshold=256,
                device=str(self.device),
                swa_sliding_window_size=params.sliding_window_size,
                enable_kv_cache_events=params.enable_kv_cache_events,
                mamba_cache_chunk_size=(
                    mamba_cache_chunk_size() if has_mamba else None
                ),
                mamba_max_states_per_path=(
                    mamba_max_states_per_path
                    if mamba_max_states_per_path >= 0
                    else None
                ),
            ),
            [int(component) for component in self.tree_components],
        )
        self.kv_events = _RustKVCacheEventRecorder(
            self._binding, params.enable_kv_cache_events
        )
        # The default-root empty result, prebuilt once from the binding.
        self._empty_match_result = _match_result_from_binding(
            self._binding.empty_match_result()
        )

    def _binding_class(self) -> type:
        """The extension binding class this core constructs."""
        if self.is_eagle:
            return self._bindings.RustBigramUnifiedTreeCoreBinding
        return self._bindings.RustUnifiedTreeCoreBinding

    # ==== Tree API ====

    def reset(self) -> None:
        self._binding.reset()
        # Node handles are never re-minted, so the fresh root gets a new one.
        self._empty_match_result = _match_result_from_binding(
            self._binding.empty_match_result()
        )

    def node_by_id(self, node_id: NodeId) -> UnifiedTreeNode:
        # TODO(Jialin): Move the remaining Python-node consumers to
        # backend-neutral APIs: sessions (#29173), C128 (#33676).
        raise NotImplementedError("node_by_id: not yet ported to the Rust tree core")

    @property
    def root_node(self) -> UnifiedTreeNode:
        raise NotImplementedError("root_node: not yet ported to the Rust tree core")

    def inc_lock_ref(
        self,
        node_id: NodeId,
        skip_lock_components: Sequence[ComponentType] = (),
    ) -> IncLockRefResult:
        result = self._binding.inc_lock_ref(
            node_id, [int(component) for component in skip_lock_components]
        )
        return _inc_lock_ref_result_from_binding(result)

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        binding_params = (
            self._bindings.DecLockRefParamsBinding(
                swa_uuid_for_lock=params.swa_uuid_for_lock,
                swa_uuid_for_host_lock=params.swa_uuid_for_host_lock,
                skip_lock_node_ids=_skip_lock_node_ids_to_binding(
                    params.skip_lock_node_ids
                ),
            )
            if params is not None
            else None
        )
        self._binding.dec_lock_ref(node_id, binding_params, skip_swa)
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        skip_lock_node_ids: Optional[dict] = None,
    ) -> DecSwaLockOnlyResult:
        result = DecSwaLockOnlyResult()
        new_device_frees, new_host_frees = self._binding.dec_swa_lock_only(
            node_id,
            swa_uuid_for_lock,
            (
                _skip_lock_node_ids_to_binding(skip_lock_node_ids)
                if skip_lock_node_ids
                else None
            ),
        )
        for component, tensors in new_device_frees.items():
            result.device_frees[ComponentType(component)].extend(tensors)
        for component, tensors in new_host_frees.items():
            result.host_frees[ComponentType(component)].extend(tensors)
        return result

    # ==== Device eviction (driven step-wise by the Controller's evict()) ====

    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:
        self._binding.evict_device_start(int(component_type), request_cnt)

    def evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> EvictDeviceNextNodeResult:
        binding_result = self._binding.evict_device_next_node(
            int(component_type), _tracker_to_binding(tracker)
        )
        result = EvictDeviceNextNodeResult(
            node_id=binding_result.node_id,
            made_progress=binding_result.made_progress,
        )
        return _fill_evict_result(binding_result, result)

    def evict_device_leaf(
        self, node_id: NodeId, is_write_back: bool
    ) -> EvictDeviceLeafResult:
        # The binding reads is_write_back from the core's construction config.
        assert is_write_back == self.is_write_back, (
            "is_write_back must match the core's construction config"
        )
        binding_result = self._binding.evict_device_leaf(node_id)
        backup = binding_result.backup_kv
        result = EvictDeviceLeafResult(
            backup_kv=_cache_action_from_tagged(backup) if backup is not None else None
        )
        return _fill_evict_result(binding_result, result)

    def demote(self, node_id: NodeId) -> DemoteResult:
        binding_result = self._binding.demote(node_id)
        return _fill_evict_result(binding_result, DemoteResult())

    def evict_device_end(self, component_type: ComponentType) -> None:
        self._binding.evict_device_end(int(component_type))

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        result = self._binding.inc_host_lock_ref(node_id)
        return IncLockRefResult(
            delta=result.delta,
            swa_uuid_for_lock=result.swa_uuid_for_lock,
            swa_uuid_for_host_lock=result.swa_uuid_for_host_lock,
            skip_lock_node_ids=_skip_lock_node_ids_from_binding(
                result.skip_lock_node_ids
            ),
        )

    def dec_host_lock_ref(
        self, node_id: NodeId, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        binding_params = (
            self._bindings.DecLockRefParamsBinding(
                swa_uuid_for_lock=params.swa_uuid_for_lock,
                swa_uuid_for_host_lock=params.swa_uuid_for_host_lock,
                skip_lock_node_ids=_skip_lock_node_ids_to_binding(
                    params.skip_lock_node_ids
                ),
            )
            if params is not None
            else None
        )
        self._binding.dec_host_lock_ref(node_id, binding_params)
        return DecLockRefResult()

    def evictable_size(self) -> int:
        return self._binding.evictable_size()

    def protected_size(self) -> int:
        return self._binding.protected_size()

    def component_evictable_size(self, component_type: ComponentType) -> int:
        return self._binding.component_evictable_size(int(component_type))

    def full_evictable_size(self) -> int:
        return self._binding.full_evictable_size()

    def full_protected_size(self) -> int:
        return self._binding.full_protected_size()

    def swa_evictable_size(self) -> int:
        return self._binding.component_evictable_size(int(ComponentType.SWA))

    def mamba_evictable_size(self) -> int:
        return self._binding.component_evictable_size(int(ComponentType.MAMBA))

    def swa_protected_size(self) -> int:
        return self._binding.component_protected_size(int(ComponentType.SWA))

    def mamba_protected_size(self) -> int:
        return self._binding.component_protected_size(int(ComponentType.MAMBA))

    def total_size(self) -> tuple[int, int]:
        return self._binding.total_size()

    def all_values_flatten(self) -> torch.Tensor:
        return self._binding.all_values_flatten()

    def walk_for_kv_canary(
        self, unlocked_only: bool, swa_resident_only: bool
    ) -> RadixCacheWalkResult:
        result = self._binding.walk_for_kv_canary(unlocked_only, swa_resident_only)
        return RadixCacheWalkResult(
            slot_indices=result.slot_indices,
            positions=result.positions,
            prev_slot_indices=result.prev_slot_indices,
        )

    def _record_all_cleared_event(self) -> None:
        self.kv_events.record_all_cleared()

    def take_events(self) -> list:
        return self.kv_events.take()

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._binding.all_mamba_values_flatten()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        result = self._binding.match_prefix(
            self._bindings.MatchParamsBinding(
                key=_radix_key_buffer(key),
                extra_key=key.extra_key,
                cache_salt=key.cache_salt,
            )
        )
        return _match_result_from_binding(result)

    @property
    def empty_match_result(self) -> MatchResult:
        return self._empty_match_result

    def is_full_device_evicted(self, node_id: NodeId) -> bool:
        return self._binding.is_full_device_evicted(node_id)

    def collect_full_device_indices(
        self, from_node_id: NodeId, until_node_id: NodeId
    ) -> torch.Tensor:
        return self._binding.collect_full_device_indices(from_node_id, until_node_id)

    def begin_insert(self, params: InsertParams) -> InsertStepResult:
        key = params.key
        key_buffer = _radix_key_buffer(key)
        value = params.value
        if value is None:
            # The binding always receives a value tensor; fall back to the
            # token ids materialized on the core's device.
            value = torch.tensor(key_buffer, dtype=torch.int64, device=self.device)
        step = self._binding.begin_insert(
            self._bindings.InsertParamsBinding(
                key=key_buffer,
                value=value,
                extra_key=key.extra_key,
                cache_salt=key.cache_salt,
                mamba_value=params.mamba_value,
                prev_prefix_len=params.prev_prefix_len,
                swa_evicted_seqlen=params.swa_evicted_seqlen,
                chunked=params.chunked,
                priority=0 if params.priority is None else params.priority,
                track_adopted_ranges=params.track_adopted_ranges,
            )
        )
        return _insert_step_from_binding(step)

    def resume_insert(self) -> InsertStepResult:
        return _insert_step_from_binding(self._binding.resume_insert())

    def has_ongoing_insert(self) -> bool:
        return self._binding.has_ongoing_insert()

    def end_insert(self) -> list[CacheAction | ComponentAction]:
        return _cache_actions_from_tagged(self._binding.end_insert())

    def drive_host_eviction(
        self, component_type: ComponentType, num_tokens: int
    ) -> DriveHostEvictionResult:
        binding_result = self._binding.drive_host_eviction(
            int(component_type), num_tokens
        )
        return _fill_evict_result(binding_result, DriveHostEvictionResult())

    def evict_excess_path_states(
        self,
        tail_node_id: NodeId,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        binding_result = self._binding.evict_excess_path_states(tail_node_id)
        for component, tensors in binding_result.new_device_frees.items():
            device_frees[ComponentType(component)].extend(tensors)
        for component, tensors in binding_result.new_host_frees.items():
            host_frees[ComponentType(component)].extend(tensors)

    # ==== HiCache ====

    def set_hicache_enabled(self) -> None:
        self._binding.set_hicache_enabled()

    @property
    def page_size(self) -> int:
        # Read-only: the Rust core freezes it at construction.
        return self._page_size

    @property
    def enable_hicache(self) -> bool:
        return self._binding.enable_hicache()

    @property
    def has_swa_host_pool(self) -> bool:
        return self._binding.has_swa_host_pool()

    @has_swa_host_pool.setter
    def has_swa_host_pool(self, value: bool) -> None:
        # The Rust core has no unset path; reject a True -> False transition.
        assert value or not self.has_swa_host_pool
        if value:
            self._binding.set_has_swa_host_pool()

    @property
    def write_through_threshold(self) -> int:
        return self._binding.write_through_threshold()

    @write_through_threshold.setter
    def write_through_threshold(self, value: int) -> None:
        # The cache assigns tree_core.write_through_threshold at HiCache init.
        self._binding.set_write_through_threshold(value)

    @property
    def is_write_back(self) -> bool:
        return self._binding.is_write_back()

    @is_write_back.setter
    def is_write_back(self, value: bool) -> None:
        # The cache assigns tree_core.is_write_back at HiCache init; forward it.
        self._binding.set_is_write_back(value)

    @property
    def enable_storage(self) -> bool:
        return self._binding.enable_storage()

    @enable_storage.setter
    def enable_storage(self, value: bool) -> None:
        # The cache assigns tree_core.enable_storage at storage init; forward it.
        self._binding.set_enable_storage(value)

    @property
    def enable_external_cache_linker(self) -> bool:
        return False

    @enable_external_cache_linker.setter
    def enable_external_cache_linker(self, value: bool) -> None:
        # TODO(Jialin): Port external cache linker support from #37091 and #37151.
        if value:
            raise ValueError(
                "External cache linker is not supported by the Rust TreeCore"
            )

    def insert_host(
        self,
        node_id: NodeId,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        result = self._binding.insert_host(
            node_id,
            key.extra_key,
            _radix_key_buffer(key),
            host_value,
            list(hash_value),
            key.cache_salt,
        )
        return InsertResult(
            prefix_len=result.prefix_len,
            total_len=result.total_len,
            last_device_node=result.last_device_node,
            inserted_host_node=result.inserted_host_node,
            host_insert_dropped=result.host_insert_dropped,
            mamba_exist=result.mamba_exist,
            cache_actions=_cache_actions_from_tagged(result.cache_actions),
        )

    def build_backup_spec(
        self, node_id: NodeId
    ) -> tuple[torch.Tensor, dict[ComponentType, list[PoolTransfer]]]:
        device_value, comp_xfers = self._binding.build_backup_spec(node_id)
        return device_value, _comp_xfers_from_binding(comp_xfers)

    def build_storage_backup_spec(
        self, node_id: NodeId, pass_prefix_keys: bool
    ) -> Optional[StorageBackupSpec]:
        spec = self._binding.build_storage_backup_spec(node_id, pass_prefix_keys)
        if spec is None:
            return None
        # Token ids cross the boundary as raw int64 bytes, not per-token ints.
        token_ids = array("q")
        token_ids.frombytes(spec.token_ids)
        return StorageBackupSpec(
            host_value=spec.host_value,
            token_ids=token_ids,
            hash_value=spec.hash_value,
            prefix_keys=spec.prefix_keys,
            comp_xfers=_comp_xfers_from_binding(spec.comp_xfers),
        )

    def build_hicache_transfers(
        self,
        component_type: ComponentType,
        node_id: NodeId,
        phase: CacheTransferPhase,
        *,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        transfers = self._binding.build_hicache_transfers(
            int(component_type),
            node_id,
            phase.value,
            host_indices,
            # TODO: Forward token ids when Rust Mamba prefetch consumes them.
            None,
            prefetch_tokens,
            last_hash,
        )
        if transfers is None:
            return None
        return [_transfer_from_binding(transfer) for transfer in transfers]

    def build_load_back_spec(
        self, node_id: NodeId, req: Optional[Req] = None
    ) -> tuple[PoolTransfer, dict[ComponentType, list[PoolTransfer]]]:
        # Component hooks take primitives, not Req: extract its fields here.
        mamba_pool_idx = req.kv.mamba_pool_idx if req is not None else None
        kv_xfer, comp_xfers = self._binding.build_load_back_spec(
            node_id, mamba_pool_idx
        )
        return _transfer_from_binding(kv_xfer), _comp_xfers_from_binding(comp_xfers)

    def prefetch_anchor_info(
        self, node_id: NodeId
    ) -> tuple[Optional[str], Optional[str]]:
        return self._binding.prefetch_anchor_info(node_id)

    def is_backuped(self, node_id: NodeId) -> bool:
        return self._binding.node_backuped(node_id)

    def is_root(self, node_id: NodeId) -> bool:
        return self._binding.is_root(node_id)

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        return self._binding.get_last_hash_value(node_id)

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        return self._binding.get_prefix_hash_values(node_id)

    def get_hash_values(self, node_id: NodeId) -> list[str]:
        return self._binding.get_hash_values(node_id)

    def snapshot_buffer_backup(
        self, node_id: NodeId, pass_prefix_keys: bool
    ) -> Optional[BufferBackupSnapshot]:
        snapshot = self._binding.snapshot_buffer_backup(node_id, pass_prefix_keys)
        if snapshot is None:
            return None
        token_ids = array("q")
        token_ids.frombytes(snapshot.key_token_ids)
        return BufferBackupSnapshot(
            node_id=snapshot.node_id,
            parent_node_id=snapshot.parent_node_id,
            parent_is_root=snapshot.parent_is_root,
            parent_last_hash=snapshot.parent_last_hash,
            hash_values=snapshot.hash_values,
            key=RadixKey(
                token_ids,
                extra_key=snapshot.extra_key,
                is_bigram=snapshot.is_bigram,
                cache_salt=snapshot.cache_salt,
            ),
            prefix_keys=snapshot.prefix_keys,
        )

    def validate_buffer_backup(
        self, node_id: NodeId, expected_key_length: int
    ) -> Optional[BufferBackupState]:
        state = self._binding.validate_buffer_backup(node_id, expected_key_length)
        if state is None:
            return None
        return BufferBackupState(
            parent_node_id=state.parent_node_id,
            parent_is_root=state.parent_is_root,
            parent_last_hash=state.parent_last_hash,
        )

    def backfill_missing_hash_values(self) -> int:
        return self._binding.backfill_missing_hash_values()

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        return self._binding.root_node_handle(extra_key)

    def dfs_weight_order(self, node_ids: Sequence[NodeId]) -> list[int]:
        return self._binding.dfs_weight_order(list(node_ids))

    def commit_hicache_transfers(
        self,
        node_id: NodeId,
        phase: CacheTransferPhase,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        actions, mamba_exist = self._binding.commit_hicache_transfers(
            node_id,
            phase.value,
            _comp_xfers_to_binding(comp_xfers),
            (
                None
                if insert_result is None
                else (
                    insert_result.total_len,
                    insert_result.inserted_host_node,
                    insert_result.mamba_exist,
                )
            ),
            (
                None
                if pool_storage_result is None
                else (
                    pool_storage_result.kv_hit_pages,
                    dict(pool_storage_result.extra_pool_hit_pages),
                )
            ),
        )
        if insert_result is not None and mamba_exist is not None:
            insert_result.mamba_exist = mamba_exist
        cache_actions.extend(_cache_actions_from_tagged(actions))

    def commit_backup(
        self,
        node_id: NodeId,
        host_indices: torch.Tensor,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> None:
        self._binding.commit_backup(
            node_id, host_indices, _comp_xfers_to_binding(comp_xfers)
        )

    def commit_load_back(
        self,
        node_id: NodeId,
        device_indices: torch.Tensor,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[CacheAction | ComponentAction]:
        actions = self._binding.commit_load_back(
            node_id,
            device_indices,
            _transfer_to_binding(kv_xfer),
            _comp_xfers_to_binding(comp_xfers),
        )
        return _cache_actions_from_tagged(actions)

    def drop_subtree_no_host(self, node_id: NodeId) -> DropSubtreeNoHostResult:
        binding_result = self._binding.drop_subtree_no_host(node_id)
        result = DropSubtreeNoHostResult(is_dropped=binding_result.dropped)
        return _fill_evict_result(binding_result, result)

    def mark_write_through_pending(self, node_id: NodeId) -> None:
        self._binding.mark_write_through_pending(node_id)

    def finish_write_through(self, node_ids: list[NodeId], ack_id: int) -> None:
        self._binding.finish_write_through(list(node_ids), ack_id)

    def finish_load_back(self, anchor_node_id: NodeId) -> None:
        self._binding.finish_load_back(anchor_node_id)

    @property
    def write_back_duplicate_reclaim_digest(self) -> int:
        return self._binding.write_back_duplicate_reclaim_digest()

    def set_component_device_value(
        self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
    ) -> None:
        self._binding.set_component_device_value(
            node_id, int(component_type), value.to(torch.int64)
        )

    def get_component_device_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        return self._binding.get_component_device_value(node_id, int(component_type))

    def component_has_host_value_only(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        return self._binding.component_has_host_value_only(node_id, int(component_type))

    # ==== Others ====

    def sanity_check(
        self,
        ongoing_write_through: list[tuple[int, NodeId]],
        ongoing_load_back: list[tuple[int, NodeId]],
    ) -> None:
        self._binding.sanity_check(ongoing_write_through, ongoing_load_back)

    def pretty_print(self) -> None:
        self._binding.pretty_print()
