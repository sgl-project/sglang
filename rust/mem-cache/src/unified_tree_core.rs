//! The radix prefix tree of cached KV.

use crate::components::{self, FullComponent, TreeComponent};
use crate::components::{
    BASE_COMPONENT_TYPE, ComponentType, FULL, MAMBA, NUM_COMPONENT_TYPES, SWA,
};
use crate::node::ChildKeyType;
use crate::node::EvictableNodeSet;
use crate::node::Node;
use crate::node::NodeArena;
use crate::node::{NUM_VALUE_SLOTS, NodeId, NodeIdx_, ValueSlotIdx};
use crate::unified_lru_list::UnifiedLRUList;
use crate::unified_lru_list::{EvictionStrategy, PriorityKey, get_eviction_strategy};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

// ---- interface types ----

/// Result of `inc_lock_ref`, handed back to the matching `dec_lock_ref`.
#[derive(Default)]
pub struct IncLockRefResult {
    /// Tokens newly protected (moved out of evictable) by this lock.
    pub delta: Option<usize>,
    /// SWA lock-window uuid minted/reused by the device lock walk.
    pub swa_uuid_for_lock: Option<i64>,
    /// SWA lock-window uuid minted/reused by the host lock walk.
    pub swa_uuid_for_host_lock: Option<i64>,
    /// Per-component nodes that were tombstones at acquire time; replayed at
    /// release so the unlock skips them.
    pub skip_lock_node_ids: HashMap<ComponentType, HashSet<NodeId>>,
}

/// Params for `dec_lock_ref`.
#[derive(Default)]
pub struct DecLockRefParams {
    /// SWA lock-window uuid the device unlock stops at, from the matching acquire.
    pub swa_uuid_for_lock: Option<i64>,
    /// SWA lock-window uuid the host unlock stops at, from the matching acquire.
    pub swa_uuid_for_host_lock: Option<i64>,
    /// Per-component nodes the unlock walk skips (from the matching acquire).
    pub skip_lock_node_ids: HashMap<ComponentType, HashSet<NodeId>>,
}

/// Result of `dec_lock_ref`.
#[derive(Default)]
pub struct DecLockRefResult {}

/// Result of a prefix match.
pub struct MatchResult {
    /// Device KV indices matched by the common prefix.
    pub device_indices: Tensor,
    /// Last matched node still resident on device.
    pub last_device_node_id: NodeId,
    /// Last matched node on host; equals `last_device_node_id` without HiCache.
    pub last_host_node_id: NodeId,
    /// Deepest node accepted by all component validators; anchors host->device load-back.
    pub best_match_node_id: NodeId,
    /// Full-KV tokens that hit on host and must be loaded back to device.
    pub host_hit_length: usize,
    /// SWA tokens that hit on host (within the sliding window) and will be
    /// loaded back into the SWA device pool.
    pub swa_host_hit_length: usize,
    /// Mamba slots that hit on host and will be loaded back; 0 or 1.
    pub mamba_host_hit_length: usize,
    /// The longest chunk-aligned position that could have hit if a mamba state existed.
    pub mamba_branching_seqlen: Option<usize>,
    /// Longest Full-KV prefix available on either device or host, independent
    /// of other components.
    pub full_kv_hit_length: usize,
    /// Actions for the controller to apply.
    pub cache_actions: Vec<CacheAction>,
}

/// Params for a prefix match; the key is borrowed from the caller.
pub struct MatchPrefixParams<'k, K: ChildKeyType> {
    /// The query key (already page-typed; bigram conversion happens at the boundary).
    pub key: &'k K,
    /// Namespace of the query; picks the named subtree root.
    pub extra_key: Option<&'k str>,
}

/// Params for an insert; the key is borrowed from the caller.
pub struct InsertParams<'k, K: ChildKeyType> {
    /// The insert key (already page-typed; bigram conversion happens at the boundary).
    pub key: &'k K,
    /// Namespace of the insert; picks the named subtree root.
    pub extra_key: Option<&'k str>,
    /// Device KV indices covering the key, one row per atom.
    pub value: Tensor,
    /// Tokens of this request already cached before the insert (the duplicate
    /// window starts past them).
    pub prev_prefix_len: usize,
    /// The request's SWA-evicted prefix boundary; SWA data below it stays tombstoned.
    pub swa_evicted_seqlen: usize,
    /// The donated mamba slot for the insert target leaf; None on non-mamba trees.
    pub mamba_value: Option<Tensor>,
    /// Whether this is a chunked-prefill insert (no hit-count bump).
    pub chunked: bool,
    /// Eviction priority floor applied along the walked path.
    pub priority: i64,
}

/// Result of an insert.
#[derive(Default)]
pub struct InsertResult {
    /// Tokens of the insert key that overlapped existing nodes.
    pub prefix_len: usize,
    /// The inserted key's full (page-aligned) length.
    pub total_len: usize,
    /// Whether the cache holds Mamba state covering the inserted sequence;
    /// vacuously true for an empty insert.
    pub mamba_exist: bool,
    /// The deepest host-backed node an insert_host attached or matched.
    pub inserted_host_node: Option<NodeId>,
    /// Actions for the controller to apply.
    pub cache_actions: Vec<CacheAction>,
}

/// One step of a resumable insert: the Controller executes `actions`, then
/// resumes while `result` is None; `result` is set on the final step.
pub struct InsertStepResult {
    pub actions: Vec<CacheAction>,
    pub result: Option<InsertResult>,
}

// WALK (one node per step) -> COMMIT (leaf + commit hooks) -> TAIL (refresh + backup).
pub enum InsertPhase {
    Walk,
    Commit,
    Tail,
}

/// In-flight resumable-insert state persisted across step barriers; owns its
/// key/value/params snapshot so the walk survives across boundary calls.
pub struct InsertWalkState<K: ChildKeyType> {
    phase: InsertPhase,
    node_id: NodeIdx_,
    /// The full page-aligned insert key; `total_prefix_length` is the walk cursor.
    key: K,
    aligned_key_len: usize,
    value: Tensor,
    extra_key: Option<String>,
    prev_prefix_len: usize,
    swa_evicted_seqlen: usize,
    mamba_value: Option<Tensor>,
    chunked: bool,
    priority: i64,
    total_prefix_length: usize,
    is_new_leaf: bool,
    target_node_id: Option<NodeIdx_>,
    result: Option<InsertResult>,
    /// Emitted actions awaiting the next barrier flush (or the final step).
    pending_actions: Vec<CacheAction>,
}

/// Result of a KV-canary walk: parallel per-slot rows over the tree's FULL device slots.
pub struct KvCanaryWalkResult {
    /// Device slot index of each emitted token.
    pub slot_indices: Vec<i64>,
    /// Token depth from the root for each emitted slot.
    pub positions: Vec<i64>,
    /// The preceding device slot on the path (-1 at a chain start).
    pub prev_slot_indices: Vec<i64>,
}

/// A queued cache IO action.
pub enum CacheAction {
    /// Duplicate device KV slices the cache frees after the insert.
    FreeDeviceKV(Vec<Tensor>),
    /// A device->host backup work item (the write-through threshold fired).
    BackupKV(BackupKV),
    /// Replace the pending write-through node on a node split:
    ///
    ///     parent -> node    =>    parent -> new_node -> new_child
    ///
    /// old_node_id (the pre-split node) is replaced by new_node_id + new_child_node_id.
    ReplaceWriteThroughOnNodeSplit {
        ack_id: usize,
        old_node_id: NodeId,
        new_node_id: NodeId,
        new_child_node_id: NodeId,
    },
    /// Per-path Mamba state-cap eviction from the tail's root path; applied at
    /// the insert's commit barrier, after the walk-time backups whose
    /// write-through locks shield the backed-up chain.
    MambaEvictExcessPathStates { tail_node_id: NodeId },
    /// Free only the given component's device KV slots.
    FreeComponentDeviceSlot {
        component_type: ComponentType,
        indices: Vec<Tensor>,
    },
    /// Free the given component's host KV pages.
    FreeComponentHostSlot {
        component_type: ComponentType,
        host_indices: Vec<Tensor>,
    },
    /// Rebuild the SWA allocator's full->swa index mapping for loaded chunks.
    RebuildFullToSwaMapping {
        full_indices: Vec<Tensor>,
        swa_indices: Vec<Tensor>,
    },
    /// Recover an SWA tombstone whose full is locked: keep the locked full, remap
    /// it onto the incoming full's SWA translation, and free only the incoming full.
    RecoverSwaWithLockedFull {
        node_id: NodeId,
        kept_full: Tensor,
        incoming_full: Tensor,
    },
    /// Rebuild a node's SWA value by translating its source full value, then store it.
    SwaRebuild {
        node_id: NodeId,
        source_value: Tensor,
    },
}

/// A HiCache pool transfer descriptor.
#[derive(Default)]
pub struct PoolTransfer {
    /// The pool this transfer targets.
    pub name: PoolName,
    /// Host-side indices for the device<->host path.
    pub host_indices: Option<Tensor>,
    /// Device-side indices, filled in once the transfer lands on device.
    pub device_indices: Option<Tensor>,
    /// Per-page storage keys for the host<->storage path.
    pub keys: Option<Vec<String>>,
    /// How storage prefix-matches this pool's pages.
    pub hit_policy: PoolHitPolicy,
    /// The nodes a load-back restores, ancestors first (external handles).
    pub nodes_to_load: Option<Vec<NodeId>>,
}

/// Hit policy for storage's per-pool prefix matching.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum PoolHitPolicy {
    /// Every page in the hit range must exist.
    #[default]
    AllPages,
    /// Only the last N pages must exist (window/state pools).
    TrailingPages,
}

impl PoolHitPolicy {
    /// The python PoolHitPolicy enum value.
    pub fn as_str(self) -> &'static str {
        todo!()
    }
}

/// Well-known pool names used as PoolTransfer identifiers.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum PoolName {
    #[default]
    Kv,
    Mamba,
    Swa,
    Indexer,
    DeepseekV4C4,
    DeepseekV4C4Indexer,
    DeepseekV4C128,
    DeepseekV4C4State,
    DeepseekV4C4IndexerState,
    DeepseekV4C128State,
    Draft,
}

/// Result of a HiCache pool transfer.
#[derive(Default)]
pub struct PoolTransferResult {
    /// Pages of the KV pool the storage transfer completed.
    pub kv_hit_pages: usize,
    /// Completed pages per auxiliary pool.
    pub extra_pool_hit_pages: HashMap<PoolName, usize>,
}

/// A device->host backup work item for the cache to execute.
#[derive(Default)]
pub struct BackupKV {
    /// Backup these nodes device->host in order, stopping at the first failure; the
    /// caller orders them parent-before-child for write-through and child-first for
    /// write-back. External handles: the list crosses to the orchestrator.
    pub node_ids: Vec<NodeId>,
}

/// A device->storage backup spec.
#[derive(Default)]
pub struct StorageBackupSpec {
    /// The node's FULL host value (the storage write's source indices).
    pub host_value: Tensor,
    /// Raw token ids spanned by the node's key.
    pub token_ids: Vec<i64>,
    /// The node's per-page hash chain (the storage keys).
    pub hash_value: Option<Vec<String>>,
    /// Ancestor-chain hashes, root-to-parent, when requested.
    pub prefix_keys: Option<Vec<String>>,
    /// Auxiliary per-component transfers riding the same storage write.
    pub comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>>,
}

/// Which storage layer(s) an eviction targets.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum EvictLayer {
    Device,
    Host,
    All,
}

impl EvictLayer {
    /// Whether this target includes `layer` (the Python IntFlag `in` membership).
    pub fn contains(self, layer: EvictLayer) -> bool {
        todo!()
    }
}

/// The request fields load-back planning reads.
#[derive(Default)]
pub struct Req {
    /// Mamba pool slot backing the request, when one is assigned.
    pub mamba_pool_idx: Option<Tensor>,
}

/// When the LRU is refreshed during a tree walk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LRURefreshPhase {
    Walkdown,
    MatchEnd,
    InsertEnd,
}

/// Direction of a HiCache transfer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CacheTransferPhase {
    BackupHost,
    LoadBack,
    BackupStorage,
    Prefetch,
}

/// Per-component tree-wide bookkeeping (device-tier accounting and walk state).
#[derive(Default)]
pub struct ComponentState {
    /// Evictable device token count.
    pub(crate) evictable_size: usize,
    /// Locked (protected) device token count.
    pub(crate) protected_size: usize,
    /// Whether a device-eviction walk is between start and end.
    pub(crate) is_evict_device_ongoing: bool,
    /// The walk's resume point, captured at return time since the returned
    /// leaf may be freed: the leaf's parent for Full, the LRU predecessor for
    /// SWA and Mamba.
    pub(crate) evict_device_cursor: Option<NodeIdx_>,
    /// Token budget for the current eviction walk.
    pub(crate) evict_device_request_cnt: usize,
}

/// Tree-construction parameters: the tree-consumed slice of the cache's init params.
pub struct CacheInitParams {
    /// Eviction-policy name resolved into the tree's strategy.
    pub eviction_policy: String,
    /// Atoms per radix page; children are keyed by their key's first page.
    pub page_size: usize,
    /// Whether the cache runs the write-back (vs write-through) policy.
    pub is_write_back: bool,
    /// Whether the host tier (HiCache) is wired.
    pub enable_hicache: bool,
    /// Hit count at which a node earns a host write-through backup.
    pub write_through_threshold: i64,
    /// Device the KV indices live on.
    pub device: Device,
    /// SWA sliding window size in tokens; None when SWA is disabled.
    pub swa_sliding_window_size: Option<usize>,
    /// Whether the cache wired a host SWA pool (HiCache).
    pub has_swa_host_pool: bool,
    /// Whether tree mutations emit BlockStored/BlockRemoved events.
    pub enable_kv_cache_events: bool,
    /// Chunk alignment for the mamba branching seqlen; None when Mamba is disabled.
    pub mamba_cache_chunk_size: Option<usize>,
    /// Per-root-path cap on cached Mamba states; None means unlimited.
    pub mamba_max_states_per_path: Option<usize>,
}

impl Default for CacheInitParams {
    fn default() -> Self {
        todo!()
    }
}

/// Radix tree of cached token prefixes; each node carries its KV per component.
/// A single eviction step's outputs: this step's per-component evicted
/// counts (deltas, for the Controller to accumulate) and freed tensors.
#[derive(Default, Debug)]
pub struct EvictionStepResult {
    pub tracker: HashMap<ComponentType, usize>,
    pub device_frees: HashMap<ComponentType, Vec<Tensor>>,
    pub host_frees: HashMap<ComponentType, Vec<Tensor>>,
}

/// The radix tree mechanism: owns the tree structure, per-node values, the
/// per-component LRUs, the size/leaf bookkeeping, and the component drivers,
/// plus `reset()`.
pub struct UnifiedTreeCore<K: ChildKeyType> {
    pub(crate) arena: NodeArena<K>,
    /// Ordered component registry; each driver reports its own type.
    components: Vec<Arc<dyn TreeComponent<K> + Send + Sync>>,
    /// Prebuilt per-type driver lookup, indexed by `ComponentType::idx`.
    components_by_type: [Option<Arc<dyn TreeComponent<K> + Send + Sync>>; NUM_COMPONENT_TYPES],
    /// Per-component bookkeeping, indexed by `ComponentType::idx`.
    pub(crate) component_states: [ComponentState; NUM_COMPONENT_TYPES],
    /// Nodes currently eligible for device eviction (D-leaves).
    pub(crate) evictable_device_leaves: EvictableNodeSet,
    /// Nodes currently eligible for host eviction (H-leaves).
    pub(crate) evictable_host_leaves: EvictableNodeSet,
    /// Per-slot LRU lists, indexed by `ValueSlotIdx::idx`.
    pub(crate) lru_lists: [UnifiedLRUList; NUM_VALUE_SLOTS],
    /// Device-eviction candidates; the lowest priority is popped first.
    pub(crate) full_evict_device_heap: BinaryHeap<Reverse<(PriorityKey, NodeIdx_)>>,
    /// Eviction-priority strategy; lower priority evicts first.
    pub(crate) eviction_strategy: Box<dyn EvictionStrategy<K> + Send>,
    /// Atoms per radix page; children are keyed by their key's first page.
    pub(crate) page_size: usize,
    /// Whether the cache runs the write-back (vs write-through) policy.
    pub(crate) is_write_back: bool,
    /// Whether the host tier (HiCache) is wired.
    pub(crate) enable_hicache: bool,
    /// Whether the storage tier (L3) is wired; gates page-hash computation.
    pub(crate) enable_storage: bool,
    /// Whether the cache wired a host SWA pool (HiCache).
    pub(crate) has_swa_host_pool: bool,
    /// Whether tree mutations emit BlockStored/BlockRemoved events.
    pub(crate) enable_kv_cache_events: bool,
    /// Queued placement events, drained by take_events.
    pub(crate) kv_event_queue: Vec<KvCacheEvent<K::Atom>>,
    /// Hit count at which a node earns a host write-through backup.
    pub(crate) write_through_threshold: i64,

    /// Monotonic source for SWA lock-window uuids.
    pub(crate) swa_uuid_counter: i64,
    /// Device the KV indices live on.
    pub(crate) device: Device,
    /// Shared empty device-index tensor (an empty match's indices).
    pub(crate) empty_device_indices: Tensor,
    /// The single in-flight resumable insert, if suspended at a barrier.
    ongoing_insert_walk_state: Option<InsertWalkState<K>>,
}

impl<K: ChildKeyType> UnifiedTreeCore<K> {
    /// Build a tree core for the given component types with a fresh arena.
    /// Fresh per-slot LRU lists.
    pub(crate) fn new_lru_lists() -> [UnifiedLRUList; NUM_VALUE_SLOTS] {
        std::array::from_fn(|i| UnifiedLRUList::new(ValueSlotIdx::from_idx(i)))
    }

    /// The component's device-tier LRU list.
    pub(crate) fn device_lru_list(&self, component_type: ComponentType) -> &UnifiedLRUList {
        todo!()
    }

    /// The component's device-tier LRU list, mutable.
    pub(crate) fn device_lru_list_mut(
        &mut self,
        component_type: ComponentType,
    ) -> &mut UnifiedLRUList {
        todo!()
    }

    /// The component's host-tier LRU list.
    pub(crate) fn host_lru_list(&self, component_type: ComponentType) -> &UnifiedLRUList {
        todo!()
    }

    /// The component's host-tier LRU list, mutable.
    pub(crate) fn host_lru_list_mut(
        &mut self,
        component_type: ComponentType,
    ) -> &mut UnifiedLRUList {
        todo!()
    }

    /// The LRU list gated by the slot's lock.
    pub(crate) fn lru_list_(&self, slot: ValueSlotIdx) -> &UnifiedLRUList {
        todo!()
    }

    /// The LRU list gated by the slot's lock, mutable.
    pub(crate) fn lru_list_mut_(&mut self, slot: ValueSlotIdx) -> &mut UnifiedLRUList {
        todo!()
    }

    /// The component's device LRU list, mutable, paired with the arena the
    /// reset walks read.
    pub(crate) fn device_lru_list_mut_with_arena(
        &mut self,
        component_type: ComponentType,
    ) -> (&mut UnifiedLRUList, &NodeArena<K>) {
        todo!()
    }

    /// The component's tree-wide bookkeeping state.
    pub(crate) fn component_state(&self, component_type: ComponentType) -> &ComponentState {
        todo!()
    }

    /// The component's mutable tree-wide bookkeeping state.
    pub(crate) fn component_state_mut(
        &mut self,
        component_type: ComponentType,
    ) -> &mut ComponentState {
        todo!()
    }

    /// The component's evictable device-token count.
    pub(crate) fn evictable_size_(&self, component_type: ComponentType) -> usize {
        // def evictable_size(self) -> int:
        //         return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)
        todo!()
    }

    /// The component's protected (locked) device-token count.
    pub(crate) fn protected_size_(&self, component_type: ComponentType) -> usize {
        // def protected_size(self) -> int:
        //         return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)
        todo!()
    }

    /// Begin the component's device-eviction bookkeeping for up to
    /// `request_cnt` tokens; panics if a walk is already in progress.
    pub(crate) fn set_evict_device_start(
        &mut self,
        component_type: ComponentType,
        request_cnt: usize,
    ) {
        todo!()
    }

    /// Finish the component's device-eviction bookkeeping; panics if no walk
    /// is in progress.
    pub(crate) fn set_evict_device_end(&mut self, component_type: ComponentType) {
        todo!()
    }

    /// Add newly evictable device tokens to the component's evictable size.
    pub(crate) fn inc_evictable_size(&mut self, component_type: ComponentType, tokens: usize) {
        todo!()
    }

    /// Subtract freed device tokens from the component's evictable size.
    pub(crate) fn dec_evictable_size(&mut self, component_type: ComponentType, tokens: usize) {
        todo!()
    }

    /// Add newly locked device tokens to the component's protected size.
    pub(crate) fn inc_protected_size(&mut self, component_type: ComponentType, tokens: usize) {
        todo!()
    }

    /// Subtract unlocked device tokens from the component's protected size.
    pub(crate) fn dec_protected_size(&mut self, component_type: ComponentType, tokens: usize) {
        todo!()
    }

    pub fn new(params: CacheInitParams, component_types: Vec<ComponentType>) -> Self {
        todo!()
    }

    /// Rebuild the root, LRUs, sizes, evictable-leaf sets, and the empty
    /// match result.
    pub fn reset(&mut self) {
        // def reset(self) -> None:
        //         """Rebuild the root, LRUs, sizes, evictable-leaf sets, and the empty
        //         match result."""
        //         # Maintains the NodeId -> active tree node mapping.
        //         self._node_arena: dict[NodeId, UnifiedTreeNode] = {}
        //
        //         # The single in-flight resumable insert, if suspended at a barrier.
        //         self._ongoing_insert_walk_state: Optional[_InsertWalkState] = None
        //
        //         self.root_node = self._new_node()
        //         self.root_node.priority = -sys.maxsize
        //         self.root_node.key = RadixKey(array("q"), None)
        //         self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        //         self.root_node.hash_value = []
        //         for ct in self.component_types:
        //             self.root_node.component_data[ct].lock_ref = 1
        //
        //         self.component_evictable_size_ = {ct: 0 for ct in self.component_types}
        //         self.component_protected_size_ = {ct: 0 for ct in self.component_types}
        //
        //         self.lru_lists = {
        //             ct: UnifiedLRUList(ct, self.component_types) for ct in self.component_types
        //         }
        //
        //         self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        //         self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        //         self.host_lru_lists = {
        //             ct: UnifiedLRUList(ct, self.component_types, use_host_ptr=True)
        //             for ct in self.component_types
        //         }
        //
        //         self._empty_match_result = MatchResult(
        //             device_indices=torch.empty(
        //                 (0,),
        //                 dtype=torch.int64,
        //                 device=self.device,
        //             ),
        //             last_device_node=self.root_node.id,
        //             last_host_node=self.root_node.id,
        //             best_match_node=self.root_node.id,
        //             cache_actions=[],
        //         )
        todo!()
    }

    /// Create a keyed, parented node not yet in its parent's child map;
    /// `creation_counter` None keeps the fresh allocation stamp.
    pub fn new_node_(
        &mut self,
        key: K,
        parent_id: NodeIdx_,
        priority: i64,
        hit_count: i64,
        creation_counter: Option<i64>,
        extra_key: Option<&str>,
    ) -> NodeIdx_ {
        // def _new_node(self, priority: int = 0) -> UnifiedTreeNode:
        //         """Create and register a tree node in the arena."""
        //         node = UnifiedTreeNode(self.component_types, priority=priority)
        //         self._register_node(node)
        //         return node
        todo!()
    }

    /// Mint the next SWA lock-window uuid.
    pub(crate) fn next_swa_uuid_(&mut self) -> i64 {
        todo!()
    }

    /// Bump the reference count on a node's component locks.
    pub fn inc_lock_ref(&mut self, node_id: NodeId) -> IncLockRefResult {
        // def inc_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        //         node = self.node_by_id(node_id)
        //         result = IncLockRefResult()
        //         for component in self.components:
        //             result = component.acquire_component_lock(node=node, result=result)
        //         self._update_evictable_leaf_sets(node)
        //         return result
        todo!()
    }

    /// Decrease the reference count on a node's component locks.
    pub fn dec_lock_ref(
        &mut self,
        node_id: NodeId,
        params: Option<&DecLockRefParams>,
        skip_swa: bool,
    ) -> DecLockRefResult {
        // def dec_lock_ref(
        //         self,
        //         node_id: NodeId,
        //         params: Optional[DecLockRefParams] = None,
        //         skip_swa: bool = False,
        //     ) -> DecLockRefResult:
        //         node = self.node_by_id(node_id)
        //         for component in self.components:
        //             if skip_swa and component.component_type == ComponentType.SWA:
        //                 continue
        //             component.release_component_lock(node=node, params=params)
        //         self._update_evictable_leaf_sets(node)
        //         # TODO: delta is not aggregated from components; no caller uses it yet.
        //         return DecLockRefResult()
        todo!()
    }

    /// Early-release the SWA portion of a request's tree lock, plus any
    /// strictly-lower-priority locks (e.g. Mamba) co-located on the node.
    pub fn dec_swa_lock_only(
        &mut self,
        node_id: NodeId,
        swa_uuid_for_lock: Option<i64>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        // def dec_swa_lock_only(
        //         self, node_id: NodeId, swa_uuid_for_lock: Optional[int]
        //     ) -> DecSwaLockOnlyResult:
        //         """Early-release the SWA portion of a request's tree lock, plus any
        //         strictly-lower-priority locks (e.g. Mamba) co-located on the node."""
        //         result = DecSwaLockOnlyResult()
        //         node = self.node_by_id(node_id)
        //         swa_component = self.components_by_type.get(ComponentType.SWA)
        //         if swa_component is None:
        //             return result
        //         swa_component.release_window_lock(
        //             node, swa_uuid_for_lock, result.device_frees, result.host_frees
        //         )
        //
        //         # Drop strictly-lower-priority locks (e.g. Mamba) co-located on the node.
        //         swa_priority = swa_component.eviction_priority(is_leaf=False)
        //         dec_params = DecLockRefParams(swa_uuid_for_lock=swa_uuid_for_lock)
        //         for comp in self.components:
        //             if comp.eviction_priority(is_leaf=False) < swa_priority:
        //                 comp.release_component_lock(node, dec_params)
        //         return result
        todo!()
    }

    /// Evict shallow Mamba device checkpoints beyond the per-path cap on the
    /// tail's root path; the mamba component drives the walk.
    pub fn evict_excess_path_states(&mut self, tail_node_id: NodeId) -> EvictionStepResult {
        todo!()
    }

    /// Bump the reference count on a node's host-side component locks.
    pub fn inc_host_lock_ref(&mut self, node_id: NodeId) -> IncLockRefResult {
        // def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        //         node = self.node_by_id(node_id)
        //         result = IncLockRefResult()
        //         for component in self.components:
        //             result = component.acquire_component_lock(
        //                 node=node, result=result, lock_host=True
        //             )
        //         self._update_evictable_leaf_sets(node)
        //         return result
        todo!()
    }

    /// Decrease the reference count on a node's host-side component locks.
    pub fn dec_host_lock_ref(
        &mut self,
        node_id: NodeId,
        params: Option<&DecLockRefParams>,
    ) -> DecLockRefResult {
        // def dec_host_lock_ref(
        //         self, node_id: NodeId, params: Optional[DecLockRefParams] = None
        //     ) -> DecLockRefResult:
        //         node = self.node_by_id(node_id)
        //         for component in self.components:
        //             component.release_component_lock(node=node, params=params, lock_host=True)
        //         self._update_evictable_leaf_sets(node)
        //         return DecLockRefResult()
        todo!()
    }

    /// Match a key against the tree; returns device indices + boundary NodeIds.
    pub fn match_prefix(&mut self, params: &MatchPrefixParams<'_, K>) -> MatchResult {
        // def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        //         key = params.key
        //         key, _ = key.maybe_to_bigram_view(self.is_eagle)
        //         if len(key) == 0:
        //             return self._empty_match_result
        //         key = key.page_aligned(self.page_size)
        //         if len(key) == 0:
        //             return self._empty_match_result
        //
        //         (
        //             value,
        //             best_match_node,
        //             best_match_device_node,
        //             best_match_device_value_len,
        //             full_kv_hit_length,
        //             action,
        //         ) = self._match_prefix_helper(key)
        //         return self._match_post_processor(
        //             params,
        //             value,
        //             best_match_node,
        //             best_match_device_node,
        //             best_match_device_value_len,
        //             full_kv_hit_length,
        //             action,
        //         )
        todo!()
    }

    /// Walk the tree for `key`; returns matched value chunks, the best match,
    /// the best device-resident match, its device value length, and any split action.
    pub fn match_prefix_helper_(
        &mut self,
        root_id: NodeIdx_,
        extra_key: Option<&str>,
        key: &K,
        aligned_key_len: usize,
    ) -> (
        Vec<Tensor>,
        NodeIdx_,
        NodeIdx_,
        usize,
        usize,
        Option<CacheAction>,
    ) {
        // def _match_prefix_helper(self, key: RadixKey) -> tuple[
        //         list[torch.Tensor],
        //         UnifiedTreeNode,
        //         UnifiedTreeNode,
        //         int,
        //         int,
        //         Optional[CacheAction | ComponentAction],
        //     ]:
        //         # Non-HiCache mode has only device-resident matches, so the scheduler
        //         # device anchor follows the best match. In HiCache mode, host-backed
        //         # nodes can also match, so we separately track the best device-resident
        //         # match for scheduler prefix indices and locking.
        //         node = self.root_node
        //         child_key = key.child_key(self.page_size)
        //         value: list[torch.Tensor] = []
        //         best_match_node = node
        //         best_match_device_node = node
        //         best_match_device_value_len = 0
        //         full_kv_hit_length = 0
        //         action: Optional[CacheAction | ComponentAction] = None
        //         separate_device_match = self.enable_hicache
        //         if separate_device_match:
        //             validators = tuple(
        //                 comp.create_match_validator() for comp in self.components
        //             )
        //             device_validators = tuple(
        //                 comp.create_match_validator(match_device_only=True)
        //                 for comp in self.components
        //             )
        //         else:
        //             validators = tuple(
        //                 comp.create_match_validator(match_device_only=True)
        //                 for comp in self.components
        //             )
        //
        //         def _all_valid(validators, node):
        //             return all([v(node) for v in validators])
        //
        //         def _update_best_if_valid(node):
        //             nonlocal best_match_node
        //             nonlocal best_match_device_value_len, best_match_device_node
        //             matched = _all_valid(validators, node)
        //             if matched:
        //                 best_match_node = node
        //
        //             if not separate_device_match:
        //                 if matched:
        //                     best_match_device_value_len = len(value)
        //                     best_match_device_node = node
        //                 return
        //             if _all_valid(device_validators, node):
        //                 best_match_device_value_len = len(value)
        //                 best_match_device_node = node
        //
        //         while len(key) > 0 and child_key in node.children:
        //             child = node.children[child_key]
        //
        //             # HiCache: dead node (evicted + not backuped) — stop traversal
        //             if child.evicted and not child.backuped:
        //                 break
        //
        //             prefix_len = child.key.match(key, page_size=self.page_size)
        //             full_kv_hit_length += prefix_len
        //             if prefix_len < len(child.key):
        //                 node, action = self._split_node(child.key, child, prefix_len)
        //                 if not node.evicted:
        //                     value.append(node.component_data[BASE_COMPONENT_TYPE].value)
        //                 _update_best_if_valid(node)
        //                 break
        //
        //             if not child.evicted:
        //                 value.append(child.component_data[BASE_COMPONENT_TYPE].value)
        //             node = child
        //             _update_best_if_valid(node)
        //             key = key[prefix_len:]
        //             if len(key):
        //                 child_key = key.child_key(self.page_size)
        //
        //         return (
        //             value,
        //             best_match_node,
        //             best_match_device_node,
        //             best_match_device_value_len,
        //             full_kv_hit_length,
        //             action,
        //         )
        todo!()
    }

    /// Assemble the MatchResult from the walk outputs.
    pub fn match_post_processor_(
        &mut self,
        params: &MatchPrefixParams<'_, K>,
        root_id: NodeIdx_,
        value: Vec<Tensor>,
        best_match_node_id: NodeIdx_,
        best_match_device_node_id: NodeIdx_,
        best_match_device_value_len: usize,
        full_kv_hit_length: usize,
        action: Option<CacheAction>,
    ) -> MatchResult {
        // def _match_post_processor(
        //         self,
        //         params: MatchPrefixParams,
        //         value: list[torch.Tensor],
        //         best_match_node: UnifiedTreeNode,
        //         best_match_device_node: UnifiedTreeNode,
        //         best_match_device_value_len: int,
        //         full_kv_hit_length: int,
        //         action: Optional[CacheAction | ComponentAction],
        //     ) -> MatchResult:
        //         node_update = best_match_node
        //         for comp in self.components:
        //             if comp.component_type == BASE_COMPONENT_TYPE:
        //                 continue  # Full uses last_access_time, not LRU
        //             comp.refresh_lru(LRURefreshPhase.MATCH_END, node_update, self.root_node)
        //
        //         cur_time = get_and_increase_time_counter()
        //         while node_update:
        //             node_update.last_access_time = cur_time
        //             cur_time -= 0.00001
        //             node_update = node_update.parent
        //
        //         # last_host_node will be used as the starting node for the subsequent
        //         # `prefetch_from_storage` flow. We directly use best_match_node here,
        //         # because best_match_node represents the node where all components
        //         # have reached consensus on both device & host availability.
        //         last_host_node = (
        //             best_match_node if self.enable_hicache else best_match_device_node
        //         )
        //
        //         if best_match_device_value_len > 0:
        //             device_indices = torch.cat(value[:best_match_device_value_len])
        //         else:
        //             device_indices = self._empty_match_result.device_indices
        //         result = MatchResult(
        //             device_indices=device_indices,
        //             last_device_node=best_match_device_node,
        //             last_host_node=last_host_node,
        //             best_match_node=best_match_node,
        //             host_hit_length=0,
        //             full_kv_hit_length=full_kv_hit_length,
        //         )
        //
        //         for component in self.components:
        //             result = component.finalize_match_result_in_tree_core(
        //                 result=result,
        //                 params=params,
        //                 value_chunks=value,
        //                 best_value_len=best_match_device_value_len,
        //             )
        //         # Expose only NodeIds outside TreeCore.
        //         return result._replace(
        //             last_device_node=result.last_device_node.id,
        //             last_host_node=result.last_host_node.id,
        //             best_match_node=result.best_match_node.id,
        //             cache_actions=[action] if action is not None else [],
        //         )
        todo!()
    }

    /// An empty match: no device indices, every boundary anchored at the root.
    pub fn empty_match_result(&self) -> MatchResult {
        // def empty_match_result(self) -> MatchResult:
        //         """A shared empty MatchResult (empty device indices + boundary NodeIds)."""
        //         return self._empty_match_result
        todo!()
    }

    /// Whether the node's FULL device value has been evicted.
    pub fn is_full_device_evicted(&self, node_id: NodeId) -> bool {
        // def is_full_device_evicted(self, node_id: NodeId) -> bool:
        //         """Whether the node's FULL device value has been evicted."""
        //         return self.node_by_id(node_id).evicted
        todo!()
    }

    /// Concatenate FULL device values from ``from_node`` up to (exclusive)
    /// ``until_node``, in root order; empty tensor if the path is empty.
    pub fn collect_full_device_indices(
        &self,
        from_node_id: NodeId,
        until_node_id: NodeId,
    ) -> Tensor {
        // def collect_full_device_indices(
        //         self, from_node_id: NodeId, until_node_id: NodeId
        //     ) -> torch.Tensor:
        //         """Concatenate FULL device values from ``from_node`` up to (exclusive)
        //         ``until_node``, in root order; empty tensor if the path is empty."""
        //         until_node = self.node_by_id(until_node_id)
        //         prefix_chunks: list[torch.Tensor] = []
        //         node = self.node_by_id(from_node_id)
        //         while node is not until_node:
        //             value = node.component_data[BASE_COMPONENT_TYPE].value
        //             assert value is not None
        //             prefix_chunks.append(value)
        //             node = node.parent
        //         if not prefix_chunks:
        //             return self._empty_match_result.device_indices
        //         prefix_chunks.reverse()
        //         return torch.cat(prefix_chunks)
        todo!()
    }

    /// Refresh a node's access tick and component LRU positions.
    pub fn touch_node_(&mut self, node_id: NodeIdx_) {
        // def _touch_node(self, node: UnifiedTreeNode):
        //         node.last_access_time = get_and_increase_time_counter()
        //         if node != self.root_node:
        //             for comp in self.components:
        //                 if comp.component_type == BASE_COMPONENT_TYPE:
        //                     continue
        //                 comp.refresh_lru(LRURefreshPhase.WALKDOWN, node, self.root_node)
        todo!()
    }

    /// Increment hit count; check whether a write backup should be fired.
    pub fn inc_hit_count_and_check_(&mut self, node_id: NodeIdx_, chunked: bool) -> bool {
        // def _inc_hit_count_and_check(
        //         self, node: UnifiedTreeNode, chunked: bool = False
        //     ) -> bool:
        //         """Increment hit count; check whether a write backup should be fired."""
        //         if node.evicted or chunked:
        //             return False
        //         if self.is_write_back:
        //             return False
        //         node.hit_count += 1
        //         return (
        //             self.enable_hicache
        //             and not node.backuped
        //             and node.hit_count >= self.write_through_threshold
        //         )
        todo!()
    }

    /// Insert device values to the tree per the provided key.
    pub fn insert(&mut self, params: &InsertParams<'_, K>) -> InsertResult {
        todo!()
    }

    /// Start the insert, running to its first barrier or completion.
    pub fn begin_insert(&mut self, params: &InsertParams<'_, K>) -> InsertStepResult {
        // def begin_insert(self, params: InsertParams) -> InsertStepResult:
        //         """Start the insert, running to its first barrier or completion."""
        //         # Insert walks are single-flight; a live walk means re-entrancy.
        //         assert self._ongoing_insert_walk_state is None, "concurrent insert walks"
        //         key = params.key
        //         value = params.value
        //         key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        //         key = key.page_aligned(self.page_size)
        //         if value is not None:
        //             value = value[: len(key)]
        //         else:
        //             value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)
        //
        //         priority = params.priority
        //         if priority is None:
        //             priority = 0
        //         self._touch_node(self.root_node)
        //         self.root_node.priority = max(self.root_node.priority, priority)
        //         if len(key) == 0:
        //             return InsertStepResult(
        //                 actions=[], result=InsertResult(prefix_len=0, mamba_exist=True)
        //             )
        //
        //         self._ongoing_insert_walk_state = _InsertWalkState(
        //             phase=_InsertPhase.WALK,
        //             node=self.root_node,
        //             key=key,
        //             value=value,
        //             params=params,
        //             priority=priority,
        //         )
        //         return self._advance_insert()
        todo!()
    }

    /// Continue the suspended insert after its step actions were executed.
    pub fn resume_insert(&mut self) -> InsertStepResult {
        // def resume_insert(self) -> InsertStepResult:
        //         """Continue the suspended insert after its step actions were executed."""
        //         assert self._ongoing_insert_walk_state is not None, "no in-flight insert"
        //         return self._advance_insert()
        todo!()
    }

    /// Whether an insert walk is suspended at a barrier.
    pub fn has_ongoing_insert(&self) -> bool {
        // def has_ongoing_insert(self) -> bool:
        //         """Whether an insert walk is suspended at a barrier."""
        //         return self._ongoing_insert_walk_state is not None
        todo!()
    }

    /// Finish the insert (idempotent); returns still-pending actions to drain.
    pub fn end_insert(&mut self) -> Vec<CacheAction> {
        // def end_insert(self) -> list[CacheAction | ComponentAction]:
        //         """Finish the insert (idempotent); returns still-pending actions to drain."""
        //         state = self._ongoing_insert_walk_state
        //         self._ongoing_insert_walk_state = None
        //         return state.pending_actions if state is not None else []
        todo!()
    }

    /// Run the in-flight insert to its next barrier or to completion.
    fn advance_insert_(&mut self) -> InsertStepResult {
        // def _advance_insert(self) -> InsertStepResult:
        //         """Run the in-flight insert to its next barrier or to completion."""
        //         state = self._ongoing_insert_walk_state
        //         while True:
        //             flushed_len = len(state.pending_actions)
        //             if state.phase is _InsertPhase.WALK:
        //                 self._insert_walk_step(state)
        //             elif state.phase is _InsertPhase.COMMIT:
        //                 self._insert_commit_step(state)
        //             elif state.phase is _InsertPhase.TAIL:
        //                 self._insert_tail_step(state)
        //                 self._ongoing_insert_walk_state = None
        //                 return InsertStepResult(
        //                     actions=state.pending_actions, result=state.result
        //                 )
        //             else:
        //                 raise AssertionError(f"unsupported insert phase: {state.phase}")
        //             new_actions = state.pending_actions[flushed_len:]
        //             # Suspend only when a step emitted a non-deferrable action.
        //             if new_actions and not all(map(self._is_deferrable_action, new_actions)):
        //                 flushed = state.pending_actions
        //                 state.pending_actions = []
        //                 return InsertStepResult(actions=flushed)
        todo!()
    }

    /// Fire-and-forget actions safe to batch until the next barrier.
    fn is_deferrable_action_(action: &CacheAction) -> bool {
        // def _is_deferrable_action(action: CacheAction | ComponentAction) -> bool:
        //         """Fire-and-forget actions safe to batch until the next barrier."""
        //         return isinstance(action, (FreeDeviceKV, ReplaceWriteThroughOnNodeSplit))
        todo!()
    }

    /// Process one walked node, appending its barrier actions to the state.
    fn insert_walk_step_(&mut self, state: &mut InsertWalkState<K>) {
        // def _insert_walk_step(self, state: _InsertWalkState) -> None:
        //         """Process one walked node, appending its barrier actions to the state."""
        //         key = state.key
        //         child_key = key.child_key(self.page_size) if len(key) else None
        //         if child_key not in state.node.children:
        //             state.phase = _InsertPhase.COMMIT
        //             return
        //         step_actions = state.pending_actions
        //         node = state.node.children[child_key]
        //         self._touch_node(node)
        //         prefix_len = node.key.match(key, page_size=self.page_size)
        //         if prefix_len < len(node.key):
        //             node, action = self._split_node(node.key, node, prefix_len)
        //             if action is not None:
        //                 step_actions.append(action)
        //         node.priority = max(node.priority, state.priority)
        //
        //         if node.evicted:
        //             self._unevict_node_on_insert(node, state.value[:prefix_len])
        //             # FULL was restored from the request's fresh KV. Aux
        //             # components (e.g. SWA) may still hold tombstones and need
        //             # to rebuild their value from the same slice.
        //             for component in self.components:
        //                 if component.component_type == BASE_COMPONENT_TYPE:
        //                     continue
        //                 component.recover_after_unevict(
        //                     node=node,
        //                     prefix_len=prefix_len,
        //                     total_prefix_len=state.total_prefix_length,
        //                     params=state.params,
        //                     cache_actions=step_actions,
        //                 )
        //         else:
        //             value_slice = state.value[:prefix_len]
        //             consumed_from = prefix_len
        //             # Let each component claim ownership of overlapping KV slots
        //             for component in self.components:
        //                 comp_consumed_from = component.update_component_on_insert_overlap(
        //                     node=node,
        //                     prefix_len=prefix_len,
        //                     total_prefix_len=state.total_prefix_length,
        //                     value_slice=value_slice,
        //                     params=state.params,
        //                     cache_actions=step_actions,
        //                 )
        //                 consumed_from = min(consumed_from, comp_consumed_from)
        //
        //             dup_start = max(0, state.params.prev_prefix_len - state.total_prefix_length)
        //             if dup_start < consumed_from:
        //                 step_actions.append(
        //                     FreeDeviceKV([value_slice[dup_start:consumed_from]])
        //                 )
        //
        //         if self._inc_hit_count_and_check(node, state.params.chunked):
        //             step_actions.append(self._build_backup_kv_action(node))
        //         state.node = node
        //         state.total_prefix_length += prefix_len
        //         state.key = key[prefix_len:]
        //         state.value = state.value[prefix_len:]
        todo!()
    }

    /// Create the tail leaf and run the component commit hooks.
    fn insert_commit_step_(&mut self, state: &mut InsertWalkState<K>) {
        // def _insert_commit_step(self, state: _InsertWalkState) -> None:
        //         """Create the tail leaf and run the component commit hooks."""
        //         # Create new leaf for remaining suffix. A leaf survives on its Full
        //         # value alone; auxiliary components (SWA, Mamba) may legitimately hold
        //         # only a tombstone for this span (e.g. the whole leaf is outside the SWA
        //         # window). Materialize it anyway so the Full KV stays cacheable.
        //         if len(state.key):
        //             state.target_node = self._add_new_node(
        //                 state.node, state.key, state.value, priority=state.priority
        //             )
        //             state.is_new_leaf = True
        //         else:
        //             state.target_node = state.node
        //
        //         # Finalize: let each component attach its data to the target node.
        //         # e.g. Mamba attaches mamba_value to the leaf node
        //         # All hooks run before their emitted actions execute; an action failure
        //         # fail-stops the process, so partial-commit state is never observed.
        //         state.result = InsertResult(prefix_len=state.total_prefix_length)
        //         for component in self.components:
        //             component.commit_insert_component_data(
        //                 node=state.target_node,
        //                 is_new_leaf=state.is_new_leaf,
        //                 params=state.params,
        //                 result=state.result,
        //                 cache_actions=state.pending_actions,
        //             )
        //         state.phase = _InsertPhase.TAIL
        todo!()
    }

    /// Refresh the LRUs and append the terminal new-leaf backup.
    fn insert_tail_step_(&mut self, state: &mut InsertWalkState<K>) {
        // def _insert_tail_step(self, state: _InsertWalkState) -> None:
        //         """Refresh the LRUs and append the terminal new-leaf backup."""
        //         if state.target_node is not self.root_node:
        //             for component in self.components:
        //                 if component.component_type == BASE_COMPONENT_TYPE:
        //                     continue
        //                 component.refresh_lru(
        //                     LRURefreshPhase.INSERT_END, state.target_node, self.root_node
        //                 )
        //
        //         if state.is_new_leaf and self._inc_hit_count_and_check(
        //             state.target_node, state.params.chunked
        //         ):
        //             state.pending_actions.append(
        //                 self._build_backup_kv_action(state.target_node)
        //             )
        todo!()
    }

    /// Split `child` at `split_len`; returns the new prefix node and any split action.
    pub fn split_node_(
        &mut self,
        child_id: NodeIdx_,
        split_len: usize,
    ) -> (NodeIdx_, Option<CacheAction>) {
        // def _split_node(
        //         self, key: RadixKey, child: UnifiedTreeNode, split_len: int
        //     ) -> tuple[UnifiedTreeNode, Optional[CacheAction | ComponentAction]]:
        //         new_node = self._new_node(priority=child.priority)
        //         new_node.children = {key[split_len:].child_key(self.page_size): child}
        //         new_node.parent = child.parent
        //         new_node.key = child.key[:split_len]
        //         new_node.hit_count = child.hit_count
        //         new_node.creation_time = child.creation_time
        //
        //         self._for_each_component_lru(child, UnifiedLRUList.remove_node)
        //
        //         child.parent = new_node
        //         child.key = child.key[split_len:]
        //         new_node.hash_value, child.hash_value = split_node_hash_value(
        //             child.hash_value, split_len, self.page_size
        //         )
        //
        //         for component in self.components:
        //             component.redistribute_on_node_split(new_parent=new_node, child=child)
        //         new_node.parent.children[key.child_key(self.page_size)] = new_node
        //
        //         # A split of a backuped node tells the cache to fix its publish list.
        //         action: Optional[CacheAction | ComponentAction] = None
        //         if child.write_through_pending_id is not None:
        //             ack_id = child.write_through_pending_id
        //             new_node.write_through_pending_id = ack_id
        //             action = ReplaceWriteThroughOnNodeSplit(
        //                 ack_id=ack_id,
        //                 old_node_id=child.id,
        //                 new_node_id=new_node.id,
        //                 new_child_node_id=child.id,
        //             )
        //
        //         self._for_each_component_lru(
        //             new_node, UnifiedLRUList.insert_mru, skip_existing=True
        //         )
        //         self._for_each_component_lru(
        //             child, UnifiedLRUList.insert_mru, skip_existing=True
        //         )
        //         child.last_access_time = get_and_increase_time_counter()
        //
        //         self._update_evictable_leaf_sets(new_node)
        //         self._update_evictable_leaf_sets(child)
        //         return new_node, action
        todo!()
    }

    /// Create a leaf holding `value` under `parent`.
    pub fn add_new_node_(
        &mut self,
        parent_id: NodeIdx_,
        key: K,
        value: &Tensor,
        priority: i64,
        extra_key: Option<&str>,
    ) -> NodeIdx_ {
        // def _add_new_node(
        //         self,
        //         parent: UnifiedTreeNode,
        //         key: RadixKey,
        //         value: torch.Tensor,
        //         priority: int = 0,
        //     ) -> UnifiedTreeNode:
        //         new_node = self._new_node(priority=priority)
        //         new_node.parent = parent
        //         new_node.key = key
        //         new_node.component_data[BASE_COMPONENT_TYPE].value = value.clone()
        //         parent.children[key.child_key(self.page_size)] = new_node
        //         self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)
        //         if self.enable_storage:
        //             new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
        //
        //         self._update_evictable_leaf_sets(new_node)
        //         self._update_evictable_leaf_sets(parent)
        //         self._record_store_event(new_node)
        //         return new_node
        todo!()
    }

    /// Restore an evicted node's Full device value from fresh KV indices
    /// during insert.
    pub fn unevict_node_on_insert_(&mut self, node_id: NodeIdx_, fresh_value: &Tensor) {
        // def _unevict_node_on_insert(
        //         self, node: UnifiedTreeNode, fresh_value: torch.Tensor
        //     ) -> None:
        //         """Restore an evicted node's Full device value from fresh KV indices
        //         during insert."""
        //         ct = BASE_COMPONENT_TYPE
        //         cd = node.component_data[ct]
        //         assert cd.value is None
        //         n = len(fresh_value)
        //         cd.value = fresh_value.clone()
        //         self.component_evictable_size_[ct] += n
        //         self._update_evictable_leaf_sets(node)
        //         if node.parent is not None:
        //             self._update_evictable_leaf_sets(node.parent)
        //         self._record_store_event(node, medium=StorageMedium.GPU)
        todo!()
    }

    /// Update both device and host leaf sets for a node.
    pub(crate) fn update_evictable_leaf_sets_(&mut self, node_id: NodeIdx_) {
        // def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        //         """Update both device and host leaf sets for a node."""
        //         if self._is_device_leaf(node):
        //             self.evictable_device_leaves.add(node)
        //         else:
        //             self.evictable_device_leaves.discard(node)
        //
        //         if self._is_host_leaf(node):
        //             self.evictable_host_leaves.add(node)
        //         else:
        //             self.evictable_host_leaves.discard(node)
        todo!()
    }

    /// Apply lru_op to each aux component's LRU that has data on this node.
    /// If skip_existing=True, skip components already in the target LRU list.
    pub(crate) fn for_each_component_lru_(
        &mut self,
        node_id: NodeIdx_,
        lru_op: &mut dyn FnMut(&mut UnifiedLRUList, NodeIdx_),
        target: EvictLayer,
        skip_existing: bool,
    ) {
        // def _for_each_component_lru(
        //         self,
        //         node: UnifiedTreeNode,
        //         lru_op,
        //         target: EvictLayer = EvictLayer.DEVICE,
        //         skip_existing: bool = False,
        //     ):
        //         """Apply lru_op to each aux component's LRU that has data on this node.
        //         If skip_existing=True, skip components already in the target LRU list."""
        //         lru_dict = self.host_lru_lists if target is EvictLayer.HOST else self.lru_lists
        //         for ct in self.component_types:
        //             if ct == BASE_COMPONENT_TYPE:
        //                 continue  # Full uses leaf sets, not LRU
        //             cd = node.component_data[ct]
        //             if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
        //                 lru = lru_dict[ct]
        //                 if skip_existing and lru.in_list(node):
        //                     continue
        //                 lru_op(lru, node)
        todo!()
    }

    /// Register a component driver into the ordered fan-out list and the
    /// by-type lookup slot; rejects duplicates.
    pub(crate) fn register_component_(
        &mut self,
        component: Arc<dyn TreeComponent<K> + Send + Sync>,
    ) {
        todo!()
    }

    /// Panics if the component is not enabled, matching the python KeyError.
    fn assert_component_enabled_(&self, component_type: ComponentType) {
        todo!()
    }

    /// The component driver for `component_type`; panics if not enabled.
    fn component_by_type_(
        &self,
        component_type: ComponentType,
    ) -> Arc<dyn TreeComponent<K> + Send + Sync> {
        todo!()
    }

    /// The component driver for `component_type`, or None when not enabled.
    fn try_component_by_type_(
        &self,
        component_type: ComponentType,
    ) -> Option<Arc<dyn TreeComponent<K> + Send + Sync>> {
        todo!()
    }

    /// Begin a component's device-eviction walk for up to request_cnt tokens.
    pub fn evict_device_start(&mut self, component_type: ComponentType, request_cnt: usize) {
        // def evict_device_start(
        //         self, component_type: ComponentType, request_cnt: int
        //     ) -> None:
        //         """Begin a component's device-eviction walk for up to request_cnt tokens."""
        //         self.components_by_type[component_type].evict_device_start(request_cnt)
        todo!()
    }

    /// Return the next device leaf to evict for a component, or None when
    /// done. The walk's budget gate reads running totals from *baseline*;
    /// the result carries only this step's deltas.
    pub fn evict_device_next_node(
        &mut self,
        component_type: ComponentType,
        baseline: &HashMap<ComponentType, usize>,
    ) -> (Option<NodeId>, EvictionStepResult) {
        // def evict_device_next_node(
        //         self, component_type: ComponentType, tracker: dict[ComponentType, int]
        //     ) -> EvictDeviceNextNodeResult:
        //         """Return the next device leaf to evict for a component, or None when done."""
        //         result = EvictDeviceNextNodeResult()
        //         # The walk reads running totals for its doneness check; the result
        //         # carries only this step's delta.
        //         updated_tracker = defaultdict(int, tracker)
        //         result.node_id = self.components_by_type[component_type].evict_device_next_node(
        //             updated_tracker, result.device_frees, result.host_frees
        //         )
        //         for ct, n in updated_tracker.items():
        //             delta = n - tracker.get(ct, 0)
        //             if delta:
        //                 result.tracker[ct] = delta
        //         return result
        todo!()
    }

    /// Finish a component's device-eviction walk.
    pub fn evict_device_end(&mut self, component_type: ComponentType) {
        // def evict_device_end(self, component_type: ComponentType) -> None:
        //         """Finish a component's device-eviction walk."""
        //         self.components_by_type[component_type].evict_device_end()
        todo!()
    }

    /// Evict one device leaf (demote if backuped, delete if write-through);
    /// for an unbacked write-back node, return the BackupKV for the cache to
    /// execute and then demote, else None.
    pub fn evict_device_leaf(
        &mut self,
        node_id: NodeId,
        is_write_back: bool,
    ) -> (Option<BackupKV>, EvictionStepResult) {
        // def evict_device_leaf(
        //         self, node_id: NodeId, is_write_back: bool
        //     ) -> EvictDeviceLeafResult:
        //         """Evict one device leaf (demote if backuped, delete if write-through);
        //         for an unbacked write-back node, the result carries the BackupKV for
        //         the cache to execute and then demote."""
        //         result = EvictDeviceLeafResult()
        //         node = self.node_by_id(node_id)
        //         assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        //         if not node.backuped:
        //             if is_write_back:
        //                 result.backup_kv = self._build_backup_kv_action(node, write_back=True)
        //                 return result
        //             # Write-through: node has no backup, delete entirely.
        //             self._delete_unbacked_device_leaf(
        //                 node,
        //                 result.tracker,
        //                 device_frees=result.device_frees,
        //                 host_frees=result.host_frees,
        //             )
        //             return result
        //         self._demote(
        //             node,
        //             result.tracker,
        //             device_frees=result.device_frees,
        //             host_frees=result.host_frees,
        //         )
        //         return result
        todo!()
    }

    /// Write-back fallback when a D-leaf's D->H backup fails under host
    /// memory pressure: drop the subtree rooted at the unbacked leaf so
    /// device eviction keeps making progress instead of leaving its KV
    /// unevictable until host space frees up.
    pub fn drop_subtree_no_host(&mut self, node_id: NodeId) -> (bool, EvictionStepResult) {
        // def drop_subtree_no_host(self, node_id: NodeId) -> DropSubtreeNoHostResult:
        //         """Write-back fallback when a D-leaf's D->H backup fails under host
        //         memory pressure: drop the subtree rooted at the unbacked leaf so
        //         device eviction keeps making progress instead of leaving its KV
        //         unevictable until host space frees up."""
        //         result = DropSubtreeNoHostResult(is_dropped=False)
        //         node = self.node_by_id(node_id)
        //         assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        //         # A failed backup never issues the D->H copy, so the subtree root has
        //         # no host state and no in-flight DMA reading its device slots.
        //         assert not node.backuped and node.write_through_pending_id is None
        //         if any(cd.host_lock_ref > 0 for cd in node.component_data):
        //             return result
        //         descendants: list[UnifiedTreeNode] = []
        //         stack = list(node.children.values())
        //         while stack:
        //             cur = stack.pop()
        //             if any(
        //                 cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
        //             ):
        //                 return result
        //             descendants.append(cur)
        //             stack.extend(cur.children.values())
        //         for desc in reversed(descendants):
        //             # Host-only by construction: a device descendant would contradict
        //             # this node being a D-leaf, and D-leaves evict before ancestors.
        //             assert desc.evicted and desc.backuped, f"node {desc.id} not host-only"
        //             assert desc.write_through_pending_id is None
        //             self._release_all_component_layers(
        //                 desc,
        //                 StorageMedium.CPU,
        //                 result.tracker,
        //                 result.device_frees,
        //                 result.host_frees,
        //             )
        //             self._remove_leaf_from_parent(desc)
        //         self._delete_unbacked_device_leaf(
        //             node,
        //             result.tracker,
        //             device_frees=result.device_frees,
        //             host_frees=result.host_frees,
        //         )
        //         result.is_dropped = True
        //         return result
        todo!()
    }

    /// Free every component layer on the node and detach it from the LRU
    /// lists and evictable leaf sets.
    pub fn release_all_component_layers_(
        &mut self,
        node_id: NodeIdx_,
        medium: StorageMedium,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        // def _release_all_component_layers(
        //         self,
        //         node: UnifiedTreeNode,
        //         medium: StorageMedium,
        //         tracker: dict[ComponentType, int],
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //     ) -> None:
        //         """Free every component layer on the node and detach it from the LRU
        //         lists and evictable leaf sets."""
        //         self._record_remove_event(node, medium=medium)
        //         for comp in self.components:
        //             self._evict_component_and_detach_lru(
        //                 node,
        //                 comp,
        //                 target=EvictLayer.ALL,
        //                 tracker=tracker,
        //                 device_frees=device_frees,
        //                 host_frees=host_frees,
        //             )
        //         self.evictable_device_leaves.discard(node)
        //         self.evictable_host_leaves.discard(node)
        todo!()
    }

    /// Delete a device leaf that has no host backup, freeing all layers.
    pub fn delete_unbacked_device_leaf_(
        &mut self,
        node_id: NodeIdx_,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        // def _delete_unbacked_device_leaf(
        //         self,
        //         node: UnifiedTreeNode,
        //         tracker: dict[ComponentType, int],
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //     ) -> None:
        //         """Delete a device leaf that has no host backup, freeing all layers."""
        //         self._release_all_component_layers(
        //             node, StorageMedium.GPU, tracker, device_frees, host_frees
        //         )
        //         parent = node.parent
        //         self._remove_leaf_from_parent(node)
        //         self._update_evictable_leaf_sets(parent)
        //         self._iteratively_delete_tombstone_leaf(
        //             node, tracker, device_frees=device_frees, host_frees=host_frees
        //         )
        todo!()
    }

    /// Evict a component's host-side resources; no-op if the component is absent.
    pub fn drive_host_eviction(
        &mut self,
        component_type: ComponentType,
        num_tokens: usize,
    ) -> EvictionStepResult {
        // def drive_host_eviction(
        //         self, component_type: ComponentType, num_tokens: int
        //     ) -> DriveHostEvictionResult:
        //         """Evict a component's host-side resources; no-op if the component is absent."""
        //         result = DriveHostEvictionResult()
        //         comp = self.components_by_type.get(component_type)
        //         if comp is not None:
        //             comp.drive_host_eviction(
        //                 num_tokens,
        //                 result.tracker,
        //                 result.device_frees,
        //                 result.host_frees,
        //             )
        //         return result
        todo!()
    }

    /// Atomically evict all components on a host leaf.
    ///
    /// All freed tokens are accumulated into *tracker*.
    pub fn evict_host_leaf_(
        &mut self,
        node_id: NodeIdx_,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        // def _evict_host_leaf(
        //         self,
        //         node: UnifiedTreeNode,
        //         tracker: dict[ComponentType, int],
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //     ) -> None:
        //         """Atomically evict all components on a host leaf.
        //
        //         All freed tokens are accumulated into *tracker*."""
        //         assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"
        //
        //         self._record_remove_event(node, medium=StorageMedium.CPU)
        //         for comp in self.components:
        //             _, hf = self._evict_component_and_detach_lru(
        //                 node,
        //                 comp,
        //                 target=EvictLayer.ALL,
        //                 tracker=None,
        //                 device_frees=device_frees,
        //                 host_frees=host_frees,
        //             )
        //             tracker[comp.component_type] += hf
        //         self.evictable_host_leaves.discard(node)
        //         self._remove_leaf_from_parent(node)
        //         self._iteratively_delete_tombstone_leaf(node, tracker, device_frees, host_frees)
        todo!()
    }

    /// Release a node's device KV once its host copy exists; the node stays in the
    /// tree, now host-only.
    pub fn demote(&mut self, node_id: NodeId) -> EvictionStepResult {
        // def demote(self, node_id: NodeId) -> DemoteResult:
        //         """Release a node's device KV once its host copy exists; the node stays in the
        //         tree, now host-only."""
        //         result = DemoteResult()
        //         self._demote(
        //             self.node_by_id(node_id),
        //             result.tracker,
        //             result.device_frees,
        //             result.host_frees,
        //         )
        //         return result
        todo!()
    }

    /// Drop a backed-up node's device value, keeping the host copy.
    pub fn demote_(
        &mut self,
        node_id: NodeIdx_,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        // def demote(self, node_id: NodeId) -> DemoteResult:
        //         """Release a node's device KV once its host copy exists; the node stays in the
        //         tree, now host-only."""
        //         result = DemoteResult()
        //         self._demote(
        //             self.node_by_id(node_id),
        //             result.tracker,
        //             result.device_frees,
        //             result.host_frees,
        //         )
        //         return result
        todo!()
    }

    /// Cascade eviction from trigger to lower-or-equal priority components.
    pub fn cascade_evict_(
        &mut self,
        node_id: NodeIdx_,
        trigger_component_type: ComponentType,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) {
        // def _cascade_evict(
        //         self,
        //         node: UnifiedTreeNode,
        //         trigger: TreeComponent,
        //         tracker: dict[ComponentType, int],
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //         target: EvictLayer = EvictLayer.DEVICE,
        //     ):
        //         """Cascade eviction from trigger to lower-or-equal priority components."""
        //
        //         is_leaf = False
        //         if target == EvictLayer.DEVICE:
        //             is_leaf = node in self.evictable_device_leaves
        //         elif target == EvictLayer.HOST:
        //             is_leaf = node in self.evictable_host_leaves
        //
        //         trigger_priority = trigger.eviction_priority(is_leaf)
        //
        //         for comp in self.components:
        //             if comp.eviction_priority(is_leaf) <= trigger_priority:
        //                 if comp is not trigger and comp.node_has_component_data(node, target):
        //                     cd = node.component_data[comp.component_type]
        //                     # A comp whose TRUE internal priority outranks the trigger
        //                     # is only in this loop because leaf-collapse flattened
        //                     # priorities; a lock on it is a legit pin and must be
        //                     # spared. A lock on a strictly-lower-priority tier is a
        //                     # real strand — fall through to the assert below.
        //                     if comp.eviction_priority(
        //                         is_leaf=False
        //                     ) >= trigger.eviction_priority(is_leaf=False):
        //                         if EvictLayer.DEVICE in target and cd.lock_ref != 0:
        //                             continue
        //                         if EvictLayer.HOST in target and cd.host_lock_ref != 0:
        //                             continue
        //                     if EvictLayer.DEVICE in target:
        //                         assert cd.lock_ref == 0
        //                     if EvictLayer.HOST in target:
        //                         assert cd.host_lock_ref == 0
        //                     self._evict_component_and_detach_lru(
        //                         node,
        //                         comp,
        //                         target=target,
        //                         tracker=tracker,
        //                         device_frees=device_frees,
        //                         host_frees=host_frees,
        //                     )
        //
        //         # Now that all components (including SWA which depends on Full.value)
        //         # have been freed, we can safely tombstone Full.value.
        //         # This is deferred from evict_component because free_swa needs it.
        //         if (
        //             target is EvictLayer.DEVICE
        //             and trigger.component_type == BASE_COMPONENT_TYPE
        //         ):
        //             node.component_data[trigger.component_type].value = None
        //
        //         self._update_evictable_leaf_sets(node)
        todo!()
    }

    /// Unlink a leaf from its parent.
    pub fn remove_leaf_from_parent_(&mut self, node_id: NodeIdx_) {
        // def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        //         key = node.key.child_key(self.page_size)
        //         v = node.parent.children.pop(key, None)
        //         assert v == node
        //         self._unregister_node(node)
        todo!()
    }

    /// Evict one component on the node and detach its LRU entries.
    pub fn evict_component_and_detach_lru_(
        &mut self,
        node_id: NodeIdx_,
        component_type: ComponentType,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
        tracker: Option<&mut HashMap<ComponentType, usize>>,
    ) -> (usize, usize) {
        // def _evict_component_and_detach_lru(
        //         self,
        //         node: UnifiedTreeNode,
        //         comp: TreeComponent,
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //         target: EvictLayer = EvictLayer.DEVICE,
        //         tracker: Optional[dict[ComponentType, int]] = None,
        //     ) -> tuple[int, int]:
        //         device_freed, host_freed = comp.evict_component(
        //             node, target=target, device_frees=device_frees, host_frees=host_frees
        //         )
        //         if tracker is not None:
        //             if EvictLayer.DEVICE in target:
        //                 tracker[comp.component_type] += device_freed
        //             elif EvictLayer.HOST in target:
        //                 tracker[comp.component_type] += host_freed
        //
        //         # Detach from the appropriate LRU list(s)
        //         ct = comp.component_type
        //         for layer, lru_lists in (
        //             (EvictLayer.DEVICE, self.lru_lists),
        //             (EvictLayer.HOST, self.host_lru_lists),
        //         ):
        //             if layer in target:
        //                 lru = lru_lists[ct]
        //                 if lru.in_list(node):
        //                     lru.remove_node(node)
        //         return device_freed, host_freed
        todo!()
    }

    /// Walk up from *deleted_node* and cascade-delete childless ancestors.
    ///
    /// Only the Full (base) component decides whether a node survives:
    ///   - Full device present  → keep as D-leaf
    ///   - Full host present    → keep as H-leaf
    ///   - neither              → evict all remaining data, delete, continue up
    pub fn iteratively_delete_tombstone_leaf_(
        &mut self,
        deleted_node_parent_id: NodeIdx_,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        // def _iteratively_delete_tombstone_leaf(
        //         self,
        //         deleted_node: UnifiedTreeNode,
        //         tracker: dict[ComponentType, int],
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //     ):
        //         """Walk up from *deleted_node* and cascade-delete childless ancestors.
        //
        //         Only the Full (base) component decides whether a node survives:
        //           - Full device present  → keep as D-leaf
        //           - Full host present    → keep as H-leaf
        //           - neither              → evict all remaining data, delete, continue up
        //         """
        //         ct = BASE_COMPONENT_TYPE
        //         cur = deleted_node.parent
        //         while cur != self.root_node and len(cur.children) == 0:
        //             if any(
        //                 cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
        //             ):
        //                 break
        //
        //             has_device = cur.component_data[ct].value is not None
        //             has_host = cur.component_data[ct].host_value is not None
        //
        //             if has_device:
        //                 self._update_evictable_leaf_sets(cur)
        //                 break
        //
        //             # Full device absent — clean up orphaned aux device data.
        //             for comp in self.components_by_type.values():
        //                 if comp.node_has_component_data(cur):
        //                     self._evict_component_and_detach_lru(
        //                         cur,
        //                         comp,
        //                         target=EvictLayer.DEVICE,
        //                         tracker=tracker,
        //                         device_frees=device_frees,
        //                         host_frees=host_frees,
        //                     )
        //
        //             if has_host:
        //                 self._update_evictable_leaf_sets(cur)
        //                 break
        //
        //             # Full absent on both layers — evict remaining host data, delete.
        //             for comp in self.components_by_type.values():
        //                 if comp.node_has_component_data(cur, target=EvictLayer.HOST):
        //                     self._evict_component_and_detach_lru(
        //                         cur,
        //                         comp,
        //                         target=EvictLayer.HOST,
        //                         tracker=tracker,
        //                         device_frees=device_frees,
        //                         host_frees=host_frees,
        //                     )
        //
        //             self.evictable_host_leaves.discard(cur)
        //             self._remove_leaf_from_parent(cur)
        //             parent = cur.parent
        //             self._update_evictable_leaf_sets(parent)
        //             cur = parent
        todo!()
    }

    /// D-leaf: Full device value present, no child with Full KV on device,
    /// unlocked, not root.
    ///
    /// Only the Full (base) component is required; auxiliary components
    /// (Mamba, SWA) are not mandatory for D-leaf membership.
    pub(crate) fn is_device_leaf_(&self, node: &Node<K>) -> bool {
        // def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        //         """D-leaf: Full device value present, no child with Full KV on device,
        //         unlocked, not root.
        //
        //         Only the Full (base) component is required; auxiliary components
        //         (Mamba, SWA) are not mandatory for D-leaf membership."""
        //         ct = BASE_COMPONENT_TYPE
        //         if node is self.root_node or node.evicted:
        //             return False
        //         if any(cd.lock_ref > 0 for cd in node.component_data):
        //             return False
        //         if any(
        //             child.component_data[ct].value is not None
        //             for child in node.children.values()
        //         ):
        //             return False
        //         return True
        todo!()
    }

    /// H-leaf: evicted, Full host value present, no children, unlocked, not root.
    ///
    /// Only the Full (base) component host_value is required; auxiliary
    /// components are not mandatory for H-leaf membership.
    fn is_host_leaf_(&self, node: &Node<K>) -> bool {
        // def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        //         """H-leaf: evicted, Full host value present, no children, unlocked, not root.
        //
        //         Only the Full (base) component host_value is required; auxiliary
        //         components are not mandatory for H-leaf membership."""
        //         if node is self.root_node or not node.evicted:
        //             return False
        //         if not node.backuped:
        //             return False
        //         if any(cd.host_lock_ref > 0 for cd in node.component_data):
        //             return False
        //         if len(node.children) > 0:
        //             return False
        //         return True
        todo!()
    }

    /// Mark the host tier (HiCache) as wired.
    pub fn set_hicache_enabled(&mut self) {
        // def set_hicache_enabled(self) -> None:
        //         self.enable_hicache = True
        todo!()
    }

    /// Whether the storage tier (L3) is wired; storage attaches after tree construction.
    pub fn set_enable_storage(&mut self, value: bool) {
        todo!()
    }

    // ==== KV cache placement events ====

    /// Queue one BlockStored per page of the node's key; hashes lazily if needed.
    fn record_store_event_(&mut self, node_id: NodeIdx_, medium: StorageMedium) {
        todo!()
    }

    /// Queue one BlockRemoved carrying all the node's page hashes; hashes lazily if needed.
    fn record_remove_event_(&mut self, node_id: NodeIdx_, medium: StorageMedium) {
        todo!()
    }

    /// Queue the all-cleared marker.
    pub fn record_all_cleared_event(&mut self) {
        todo!()
    }

    /// Take all queued events, leaving the queue empty.
    pub fn take_events(&mut self) -> Vec<KvCacheEvent<K::Atom>> {
        todo!()
    }

    /// Mark the SWA host pool as wired; pools attach after tree construction.
    pub fn set_has_swa_host_pool(&mut self) {
        todo!()
    }

    /// Insert a host-side (backuped) tree path descending from the given node.
    pub fn insert_host(
        &mut self,
        node_id: NodeId,
        extra_key: Option<&str>,
        key: K,
        host_value: Tensor,
        hash_value: Vec<String>,
    ) -> InsertResult {
        // def insert_host(
        //         self,
        //         node_id: NodeId,
        //         key: RadixKey,
        //         host_value: torch.Tensor,
        //         hash_value: list[str],
        //     ) -> InsertResult:
        //         """Insert a host-side (backuped) tree path descending from the given node."""
        //         node = self.node_by_id(node_id)
        //         total_len = len(key)
        //         self._touch_node(node)
        //         if total_len == 0:
        //             return InsertResult(prefix_len=0, mamba_exist=True)
        //
        //         child_key = key.child_key(self.page_size)
        //         matched_length = 0
        //         cache_actions: list[CacheAction | ComponentAction] = []
        //         while len(key) > 0 and child_key in node.children:
        //             node = node.children[child_key]
        //             self._touch_node(node)
        //             prefix_len = node.key.match(key, page_size=self.page_size)
        //
        //             key = key[prefix_len:]
        //             host_value = host_value[prefix_len:]
        //             hash_value = hash_value[prefix_len // self.page_size :]
        //             matched_length += prefix_len
        //
        //             if prefix_len < len(node.key):
        //                 node, action = self._split_node(node.key, node, prefix_len)
        //                 if action is not None:
        //                     cache_actions.append(action)
        //
        //             if len(key):
        //                 child_key = key.child_key(self.page_size)
        //
        //         result = InsertResult(
        //             prefix_len=matched_length,
        //             total_len=total_len,
        //             cache_actions=cache_actions,
        //         )
        //         if len(key) == 0:
        //             if (
        //                 node is not self.root_node
        //                 and node.component_data[BASE_COMPONENT_TYPE].host_value is not None
        //             ):
        //                 result.inserted_host_node = node.id
        //             return result
        //
        //         # Drop the refill only under write-through (a non-write-back policy).
        //         if node is not self.root_node and not node.backuped and not self.is_write_back:
        //             logger.info(
        //                 "HiCache prefetch dropped %d-token refill under un-backed-up node %d",
        //                 len(host_value),
        //                 node.id,
        //             )
        //             result.host_insert_dropped = True
        //             return result
        //
        //         new_node = self._new_node(priority=node.priority)
        //         new_node.parent = node
        //         new_node.key = key
        //         new_node.hash_value = hash_value
        //         new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        //         node.children[child_key] = new_node
        //         self._update_evictable_leaf_sets(new_node)
        //         self._update_evictable_leaf_sets(node)
        //         result.inserted_host_node = new_node.id
        //         return result
        todo!()
    }

    /// Read a node's device->host backup spec (device value + component transfers) now.
    pub fn build_backup_spec(
        &self,
        node_id: NodeId,
    ) -> (Tensor, HashMap<ComponentType, Vec<PoolTransfer>>) {
        // def build_backup_spec(self, node_id: NodeId):
        //         """Read a node's device->host backup spec (device value + component transfers) now."""
        //         return self._build_backup_spec(self.node_by_id(node_id))
        todo!()
    }

    /// Gather device value backup spec.
    pub fn build_backup_spec_(
        &self,
        node: &Node<K>,
    ) -> (Tensor, HashMap<ComponentType, Vec<PoolTransfer>>) {
        // def build_backup_spec(self, node_id: NodeId):
        //         """Read a node's device->host backup spec (device value + component transfers) now."""
        //         return self._build_backup_spec(self.node_by_id(node_id))
        todo!()
    }

    /// Gather a node's device->storage backup spec; None if the node is not backuped.
    pub fn build_storage_backup_spec(
        &self,
        node_id: NodeId,
        pass_prefix_keys: bool,
    ) -> Option<StorageBackupSpec> {
        // def build_storage_backup_spec(
        //         self, node_id: NodeId, pass_prefix_keys: bool
        //     ) -> Optional[StorageBackupSpec]:
        //         """Gather a node's device->storage backup spec; None if the node is not backuped."""
        //         node = self.node_by_id(node_id)
        //         if not node.backuped:
        //             return None
        //         prefix_keys = None
        //         if pass_prefix_keys:
        //             prefix_keys = node.get_prefix_hash_values(node.parent)
        //         comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        //         for comp in self.components:
        //             if comp.component_type == BASE_COMPONENT_TYPE:
        //                 continue
        //             transfers = comp.build_hicache_transfers(
        //                 node, CacheTransferPhase.BACKUP_STORAGE
        //             )
        //             if transfers:
        //                 comp_xfers[comp.component_type] = transfers
        //         return StorageBackupSpec(
        //             host_value=node.component_data[BASE_COMPONENT_TYPE].host_value,
        //             token_ids=node.key.token_ids,
        //             hash_value=node.hash_value,
        //             prefix_keys=prefix_keys,
        //             comp_xfers=comp_xfers,
        //         )
        todo!()
    }

    /// Route a build_hicache_transfers call to the component for the given type.
    pub fn build_hicache_transfers(
        &self,
        component_type: ComponentType,
        node_id: NodeId,
        phase: CacheTransferPhase,
        host_indices: Option<Tensor>,
        token_ids: Option<&[i64]>,
        prefetch_tokens: usize,
        last_hash: Option<&str>,
    ) -> Option<Vec<PoolTransfer>> {
        // def build_hicache_transfers(
        //         self,
        //         component_type: ComponentType,
        //         node_id: NodeId,
        //         phase: CacheTransferPhase,
        //         *,
        //         host_indices: Optional[torch.Tensor] = None,
        //         token_ids: Optional[Sequence[int]] = None,
        //         prefetch_tokens: int = 0,
        //         last_hash: Optional[str] = None,
        //     ) -> Optional[list[PoolTransfer]]:
        //         """Route a build_hicache_transfers call to the component for the given type."""
        //         return self.components_by_type[component_type].build_hicache_transfers(
        //             self.node_by_id(node_id),
        //             phase,
        //             host_indices=host_indices,
        //             token_ids=token_ids,
        //             prefetch_tokens=prefetch_tokens,
        //             last_hash=last_hash,
        //         )
        todo!()
    }

    /// Build the H->D load-back KV transfer plus per-component aux transfers.
    pub fn build_load_back_spec(
        &self,
        node_id: NodeId,
        req: Option<&Req>,
    ) -> (PoolTransfer, HashMap<ComponentType, Vec<PoolTransfer>>) {
        // def build_load_back_spec(
        //         self, node_id: NodeId, req: Optional[Req] = None
        //     ) -> tuple[PoolTransfer, dict[ComponentType, list[PoolTransfer]]]:
        //         """Build the H->D load-back KV transfer plus per-component aux transfers."""
        //         # Component hooks take primitives, not Req: extract its fields here.
        //         mamba_pool_idx = req.mamba_pool_idx if req is not None else None
        //         node = self.node_by_id(node_id)
        //         kv_xfer = self.components_by_type[BASE_COMPONENT_TYPE].build_hicache_transfers(
        //             node, CacheTransferPhase.LOAD_BACK
        //         )[0]
        //         comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        //         for comp in self.components:
        //             if comp.component_type == BASE_COMPONENT_TYPE:
        //                 continue
        //             t = comp.build_hicache_transfers(
        //                 node, CacheTransferPhase.LOAD_BACK, mamba_pool_idx=mamba_pool_idx
        //             )
        //             if t:
        //                 comp_xfers[comp.component_type] = t
        //         return kv_xfer, comp_xfers
        todo!()
    }

    /// The anchor node's namespace; None for the default namespace.
    pub fn prefetch_anchor_info(&self, node_id: NodeId) -> Option<String> {
        // def prefetch_anchor_info(self, node_id: NodeId) -> Optional[str]:
        //         """The anchor node's key extra_key."""
        //         node = self.node_by_id(node_id)
        //         return node.key.extra_key if node.key else None
        todo!()
    }

    /// Whether the node's Full KV is present on host.
    pub fn node_backuped(&self, node_id: NodeId) -> bool {
        todo!()
    }

    /// Whether the node is a (default or named) root.
    pub fn is_root(&self, node_id: NodeId) -> bool {
        // def is_root(self, node_id: NodeId) -> bool:
        //         """Whether the node is the tree root."""
        //         return self.node_by_id(node_id) is self.root_node
        todo!()
    }

    /// The node's last page hash, or None when it was never hashed.
    pub fn get_last_hash_value(&self, node_id: NodeId) -> Option<String> {
        // def get_last_hash_value(self) -> Optional[str]:
        //         if self.hash_value is None or len(self.hash_value) == 0:
        //             return None
        //         return self.hash_value[-1]
        todo!()
    }

    /// The hash chain of the node's ancestors, in root-to-parent order.
    pub fn get_prefix_hash_values(&self, node_id: NodeId) -> Vec<String> {
        // def get_prefix_hash_values(self, node: UnifiedTreeNode) -> list[str]:
        //         if node is None or node.hash_value is None:
        //             return []
        //
        //         return node.get_prefix_hash_values(node.parent) + node.hash_value
        todo!()
    }

    /// The hash values owned by this node, excluding its ancestors.
    pub fn get_hash_values(&self, node_id: NodeId) -> Vec<String> {
        todo!()
    }

    /// The NodeId anchoring matches; the single root serves every namespace.
    pub fn root_node_handle(&self, _extra_key: Option<&str>) -> NodeId {
        todo!()
    }

    /// Build the backup action for a node and its unbacked ancestors.
    pub fn build_backup_kv_action_(&self, node: &Node<K>, write_back: bool) -> BackupKV {
        // def _build_backup_kv_action(
        //         self, node: UnifiedTreeNode, write_back: bool = False
        //     ) -> BackupKV:
        //         """Build the backup action for a node and its unbacked ancestors."""
        //         chain = [node]
        //         if not write_back:
        //             ancestor = node.parent
        //             while (
        //                 ancestor is not None
        //                 and ancestor is not self.root_node
        //                 and not ancestor.backuped
        //             ):
        //                 chain.append(ancestor)
        //                 ancestor = ancestor.parent
        //             # write_through: Ancestors first to preserve backup invariant
        //             chain.reverse()
        //         return BackupKV([target.id for target in chain])
        todo!()
    }

    /// Commit each component's HiCache transfers onto the node.
    pub fn commit_hicache_transfers(
        &mut self,
        node_id: NodeId,
        phase: CacheTransferPhase,
        comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>>,
        cache_actions: &mut Vec<CacheAction>,
        mut insert_result: Option<&mut InsertResult>,
        pool_storage_result: Option<&PoolTransferResult>,
    ) {
        // def commit_hicache_transfers(
        //         self,
        //         node_id: NodeId,
        //         phase: CacheTransferPhase,
        //         comp_xfers: dict[ComponentType, list[PoolTransfer]],
        //         *,
        //         cache_actions: list[CacheAction | ComponentAction],
        //         insert_result: Optional[InsertResult] = None,
        //         pool_storage_result: Optional[PoolTransferResult] = None,
        //     ) -> None:
        //         """Commit each component's HiCache transfers onto the node."""
        //         node = self.node_by_id(node_id)
        //         for ct, xfers in comp_xfers.items():
        //             self.components_by_type[ct].commit_hicache_transfer(
        //                 node,
        //                 phase,
        //                 xfers,
        //                 cache_actions=cache_actions,
        //                 insert_result=insert_result,
        //                 pool_storage_result=pool_storage_result,
        //             )
        todo!()
    }

    /// Commit a successful backup to the node.
    pub fn commit_backup(
        &mut self,
        node_id: NodeId,
        host_indices: Tensor,
        comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>>,
    ) {
        // def commit_backup(
        //         self,
        //         node_id: NodeId,
        //         host_indices: torch.Tensor,
        //         comp_xfers: dict[ComponentType, list[PoolTransfer]],
        //     ) -> None:
        //         """Commit a successful backup to the node."""
        //         node = self.node_by_id(node_id)
        //         cache_actions: list[CacheAction | ComponentAction] = []
        //         kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        //         self.components_by_type[BASE_COMPONENT_TYPE].commit_hicache_transfer(
        //             node,
        //             CacheTransferPhase.BACKUP_HOST,
        //             transfers=[kv_xfer],
        //             cache_actions=cache_actions,
        //         )
        //         for ct, xfers in comp_xfers.items():
        //             self.components_by_type[ct].commit_hicache_transfer(
        //                 node,
        //                 CacheTransferPhase.BACKUP_HOST,
        //                 transfers=xfers,
        //                 cache_actions=cache_actions,
        //             )
        //         assert not cache_actions
        todo!()
    }

    /// Commit a successful H->D load-back onto the node; the SWA full->swa mapping
    /// rebuild is deferred to the orchestration layer.
    pub fn commit_load_back(
        &mut self,
        node_id: NodeId,
        device_indices: Tensor,
        mut kv_xfer: PoolTransfer,
        comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>>,
    ) -> Vec<CacheAction> {
        // def commit_load_back(
        //         self,
        //         node_id: NodeId,
        //         device_indices: torch.Tensor,
        //         kv_xfer: PoolTransfer,
        //         comp_xfers: dict[ComponentType, list[PoolTransfer]],
        //     ) -> list[CacheAction | ComponentAction]:
        //         """Commit a successful H->D load-back onto the node; the SWA full->swa mapping
        //         rebuild is deferred to the orchestration layer."""
        //         node = self.node_by_id(node_id)
        //         cache_actions: list[CacheAction | ComponentAction] = []
        //         kv_xfer.device_indices = device_indices
        //         self.components_by_type[BASE_COMPONENT_TYPE].commit_hicache_transfer(
        //             node,
        //             CacheTransferPhase.LOAD_BACK,
        //             [kv_xfer],
        //             cache_actions=cache_actions,
        //         )
        //         for nid in kv_xfer.nodes_to_load or ():
        //             loaded = self.node_by_id(nid)
        //             self._record_store_event(loaded, medium=StorageMedium.GPU)
        //         for ct, xfers in comp_xfers.items():
        //             self.components_by_type[ct].commit_hicache_transfer(
        //                 node,
        //                 CacheTransferPhase.LOAD_BACK,
        //                 xfers,
        //                 cache_actions=cache_actions,
        //             )
        //         self._update_evictable_leaf_sets(node)
        //         return cache_actions
        todo!()
    }

    /// Mark a node as having an in-flight write-through backup.
    pub fn mark_write_through_pending(&mut self, node_id: NodeId) {
        // def mark_write_through_pending(self, node_id: NodeId) -> None:
        //         """Mark a node as having an in-flight write-through backup."""
        //         node = self.node_by_id(node_id)
        //         node.write_through_pending_id = node_id
        todo!()
    }

    /// Clear the write-through-pending mark (when it matches ack_id) and record the
    /// host store event for each acked node.
    pub fn finish_write_through(&mut self, node_ids: Vec<NodeId>, ack_id: usize) {
        // def finish_write_through(self, node_ids: list[NodeId], ack_id: int) -> None:
        //         """Clear the write-through-pending mark (when it matches ack_id) and record the
        //         host store event for each acked node."""
        //         for node_id in node_ids:
        //             node = self.node_by_id(node_id)
        //             if node.write_through_pending_id == ack_id:
        //                 node.write_through_pending_id = None
        //             self._record_store_event(node, medium=StorageMedium.CPU)
        todo!()
    }

    /// Store an auxiliary component's device value onto a node and restamp
    /// its LRU.
    pub fn set_component_device_value(
        &mut self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Tensor,
    ) {
        // def set_component_device_value(
        //         self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
        //     ) -> None:
        //         """Store an auxiliary component's device value onto a node."""
        //         # Full uses leaf sets, not LRU; its stores go through the insert paths.
        //         assert component_type != BASE_COMPONENT_TYPE
        //         node = self.node_by_id(node_id)
        //         node.component_data[component_type].value = value
        //         host_lru = self.host_lru_lists[component_type]
        //         if host_lru.in_list(node):
        //             host_lru.remove_node(node)
        //         self.lru_lists[component_type].insert_mru(node)
        //         self.component_evictable_size_[component_type] += len(value)
        todo!()
    }

    /// Slot-keyed aux store (internal): set the device value and restamp the LRU.
    pub(crate) fn set_component_device_value_(
        &mut self,
        node_id: NodeIdx_,
        component_type: ComponentType,
        value: Tensor,
    ) {
        // def set_component_device_value(
        //         self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
        //     ) -> None:
        //         """Store an auxiliary component's device value onto a node."""
        //         # Full uses leaf sets, not LRU; its stores go through the insert paths.
        //         assert component_type != BASE_COMPONENT_TYPE
        //         node = self.node_by_id(node_id)
        //         node.component_data[component_type].value = value
        //         host_lru = self.host_lru_lists[component_type]
        //         if host_lru.in_list(node):
        //             host_lru.remove_node(node)
        //         self.lru_lists[component_type].insert_mru(node)
        //         self.component_evictable_size_[component_type] += len(value)
        todo!()
    }

    /// The component's device value on the node, or None if evicted.
    pub fn get_component_device_value(
        &self,
        node_id: NodeId,
        component_type: ComponentType,
    ) -> Option<&Tensor> {
        // def get_component_device_value(
        //         self, node_id: NodeId, component_type: ComponentType
        //     ) -> Optional[torch.Tensor]:
        //         """The component's device value on the node, or None if evicted."""
        //         return self.node_by_id(node_id).component_data[component_type].value
        todo!()
    }

    /// Whether the component's data is device-evicted but host-backed.
    pub fn component_has_host_value_only(
        &self,
        node_id: NodeId,
        component_type: ComponentType,
    ) -> bool {
        // def component_has_host_value_only(
        //         self, node_id: NodeId, component_type: ComponentType
        //     ) -> bool:
        //         """Whether the component's data is device-evicted but host-backed."""
        //         cd = self.node_by_id(node_id).component_data[component_type]
        //         return cd.value is None and cd.host_value is not None
        todo!()
    }

    /// Verify tree-structure, leaf-set, LRU, size, and ongoing-op invariants; raise
    /// AssertionError on any violation. ongoing_* args are (id, node_id) pairs.
    pub fn sanity_check(
        &self,
        ongoing_write_through: &[(i64, NodeId)],
        ongoing_load_back: &[(i64, NodeId)],
    ) {
        // def sanity_check(
        //         self,
        //         ongoing_write_through: list[tuple[int, NodeId]],
        //         ongoing_load_back: list[tuple[int, NodeId]],
        //     ) -> None:
        //         """Verify tree-structure, leaf-set, LRU, size, and ongoing-op invariants; raise
        //         AssertionError on any violation. ongoing_* args are (id, node_id) pairs.
        //         """
        //         errors: list[str] = []
        //         E = errors.append
        //         all_nodes = self._collect_all_nodes()
        //         all_node_set = set(all_nodes)
        //         FCT = BASE_COMPONENT_TYPE
        //
        //         # ── PART 1: Tree Structure ──
        //         # Root state
        //         if self.root_node.component_data[FCT].value is None:
        //             E("[Root] root missing Full device value")
        //         if self.root_node.component_data[FCT].lock_ref <= 0:
        //             E(
        //                 f"[Root] root Full lock_ref={self.root_node.component_data[FCT].lock_ref}"
        //             )
        //         if self.root_node.parent is not None:
        //             E("[Root] root has a parent pointer")
        //         # Parent ↔ child bidirectional consistency
        //         for node in all_nodes:
        //             for child in node.children.values():
        //                 if child.parent is not node:
        //                     pid = child.parent.id if child.parent else None
        //                     E(f"[Tree] child {child.id} parent={pid}, expected {node.id}")
        //                 if child.key is None:
        //                     E(f"[Tree] node {child.id} has no key")
        //
        //         # ── PART 2: Per-node state machine and leaf qualification ──
        //         expected_dev_leaves: set[UnifiedTreeNode] = set()
        //         expected_hst_leaves: set[UnifiedTreeNode] = set()
        //
        //         for node in all_nodes:
        //             if node is self.root_node:
        //                 continue
        //             nid = node.id
        //             full_dev = node.component_data[FCT].value is not None
        //             full_hst = node.component_data[FCT].host_value is not None
        //
        //             # Full is the tree backbone, so aux data requires Full data.
        //             for ct in self.component_types:
        //                 if ct == FCT:
        //                     continue
        //                 cd = node.component_data[ct]
        //                 if cd.value is not None and not full_dev:
        //                     E(f"node {nid} {ct} device present but Full.value=None")
        //                 if cd.host_value is not None and not full_hst:
        //                     E(f"node {nid} {ct} host present but Full.host_value=None")
        //
        //             # Every node must keep Full data on at least one layer.
        //             if not full_dev and not full_hst:
        //                 E(f"node {nid} dead: no Full device and no Full host")
        //
        //             # Parent prefixes must keep data whenever the child does.
        //             if node.parent is not None and node.parent is not self.root_node:
        //                 p_dev = node.parent.component_data[FCT].value is not None
        //                 p_hst = node.parent.component_data[FCT].host_value is not None
        //                 if full_dev and not p_dev:
        //                     E(f"node {nid} device present but parent {node.parent.id} evicted")
        //                 if full_hst and not p_hst and not self.is_write_back:
        //                     E(f"node {nid} backed up but parent {node.parent.id} not backed up")
        //
        //             # Lock hierarchy and counters must stay sane.
        //             fl = node.component_data[FCT].lock_ref
        //             for ct in self.component_types:
        //                 cd = node.component_data[ct]
        //                 if cd.lock_ref < 0:
        //                     E(f"node {nid} {ct} lock_ref={cd.lock_ref}")
        //                 if cd.host_lock_ref < 0:
        //                     E(f"node {nid} {ct} host_lock_ref={cd.host_lock_ref}")
        //                 if ct != FCT and fl < cd.lock_ref:
        //                     E(f"node {nid} full_lock={fl} < {ct}_lock={cd.lock_ref}")
        //                 if cd.value is None and cd.lock_ref > 0:
        //                     E(f"node {nid} {ct} evicted but lock_ref={cd.lock_ref}")
        //
        //             # Collect expected leaf qualification (single pass)
        //             if self._is_device_leaf(node):
        //                 expected_dev_leaves.add(node)
        //             if self._is_host_leaf(node):
        //                 expected_hst_leaves.add(node)
        //
        //         # ── PART 3: Tracking structures ──
        //
        //         # Device leaf set must match the expected leaves.
        //         if self.evictable_device_leaves != expected_dev_leaves:
        //             extra = self.evictable_device_leaves - expected_dev_leaves
        //             missing = expected_dev_leaves - self.evictable_device_leaves
        //             if extra:
        //                 E(f"D-leaf extra: {[n.id for n in list(extra)[:5]]}")
        //             if missing:
        //                 E(f"D-leaf missing: {[n.id for n in list(missing)[:5]]}")
        //
        //         # Host leaf set must match the expected leaves.
        //         if self.evictable_host_leaves != expected_hst_leaves:
        //             extra = self.evictable_host_leaves - expected_hst_leaves
        //             missing = expected_hst_leaves - self.evictable_host_leaves
        //             if extra:
        //                 E(f"H-leaf extra: {[n.id for n in list(extra)[:5]]}")
        //             if missing:
        //                 E(f"H-leaf missing: {[n.id for n in list(missing)[:5]]}")
        //
        //         # D-leaf ∩ H-leaf = ∅
        //         overlap = self.evictable_device_leaves & self.evictable_host_leaves
        //         if overlap:
        //             E(
        //                 f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
        //             )
        //
        //         # Stale nodes: leaf sets must only contain tree-reachable nodes
        //         stale = self.evictable_device_leaves - all_node_set
        //         if stale:
        //             E(
        //                 f"{len(stale)} stale nodes in device_leaves: {[n.id for n in list(stale)[:5]]}"
        //             )
        //         stale = self.evictable_host_leaves - all_node_set
        //         if stale:
        //             E(
        //                 f"{len(stale)} stale nodes in host_leaves: {[n.id for n in list(stale)[:5]]}"
        //             )
        //
        //         # Per-component LRU tracking
        //         for ct in self.component_types:
        //             lru = self.lru_lists[ct]
        //             if ct == FCT:
        //                 # Full uses leaf sets, not LRU
        //                 if len(lru.cache) > 0:
        //                     E(f"Full device LRU not empty: {len(lru.cache)}")
        //                 if len(self.host_lru_lists[ct].cache) > 0:
        //                     E(f"Full host LRU not empty: {len(self.host_lru_lists[ct].cache)}")
        //             else:
        //                 # Aux device values must match the device LRU.
        //                 tree_ids = {
        //                     n.id
        //                     for n in all_nodes
        //                     if n is not self.root_node
        //                     and n.component_data[ct].value is not None
        //                 }
        //                 lru_ids = set(lru.cache.keys())
        //                 if tree_ids != lru_ids:
        //                     E(
        //                         f"{ct} device LRU: "
        //                         f"+tree={tree_ids - lru_ids}, +lru={lru_ids - tree_ids}"
        //                     )
        //                 # Aux host-only states must match the host LRU.
        //                 host_lru = self.host_lru_lists[ct]
        //                 s3_ids = {
        //                     n.id
        //                     for n in all_nodes
        //                     if n is not self.root_node
        //                     and n.component_data[ct].value is None
        //                     and n.component_data[ct].host_value is not None
        //                 }
        //                 host_lru_ids = set(host_lru.cache.keys())
        //                 if s3_ids != host_lru_ids:
        //                     E(
        //                         f"{ct} host LRU: "
        //                         f"+S3={s3_ids - host_lru_ids}, +lru={host_lru_ids - s3_ids}"
        //                     )
        //                 # The same aux node must not appear in both device and host LRU.
        //                 inv5_overlap = lru_ids & host_lru_ids
        //                 if inv5_overlap:
        //                     E(f"{ct} in both device and host LRU: {inv5_overlap}")
        //                 # Linked-list integrity
        //                 self._check_lru_linked_list(lru, ct, "device", errors)
        //                 self._check_lru_linked_list(host_lru, ct, "host", errors)
        //
        //         # ── PART 4: Size Accounting ──
        //         for ct in self.component_types:
        //             evictable = 0
        //             protected = 0
        //             for n in all_nodes:
        //                 if n is self.root_node:
        //                     continue
        //                 cd = n.component_data[ct]
        //                 if cd.value is not None:
        //                     toks = len(cd.value)
        //                     if cd.lock_ref > 0:
        //                         protected += toks
        //                     else:
        //                         evictable += toks
        //             if self.component_evictable_size_[ct] != evictable:
        //                 E(
        //                     f"[Size] {ct} evictable={self.component_evictable_size_[ct]} "
        //                     f"!= recomputed={evictable}"
        //                 )
        //             if self.component_protected_size_[ct] != protected:
        //                 E(
        //                     f"[Size] {ct} protected={self.component_protected_size_[ct]} "
        //                     f"!= recomputed={protected}"
        //                 )
        //
        //         # ── PART 5: Ongoing Operations ──
        //         for nid, node_id in ongoing_write_through:
        //             n = self._node_arena.get(node_id)
        //             if n is None or n not in all_node_set:
        //                 E(f"[Ongoing] write_through node {nid} not in tree")
        //             elif n.component_data[FCT].lock_ref <= 0:
        //                 E(
        //                     f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
        //                 )
        //         for nid, node_id in ongoing_load_back:
        //             n = self._node_arena.get(node_id)
        //             if n is None or n not in all_node_set:
        //                 E(f"[Ongoing] load_back node {nid} not in tree")
        //             elif n.component_data[FCT].lock_ref <= 0:
        //                 E(
        //                     f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
        //                 )
        //
        //         if errors:
        //             msg = (
        //                 f"Sanity check FAILED ({len(errors)} violations "
        //                 f"across {len(all_nodes)} nodes):\n"
        //                 + "\n".join(f"  {e}" for e in errors)
        //             )
        //             logger.error(msg)
        //             self.pretty_print()
        //             raise AssertionError(msg)
        todo!()
    }

    /// Every live node in the tree.
    pub fn collect_all_nodes_(&self) -> Vec<NodeIdx_> {
        // def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        //         nodes = []
        //         stack = [self.root_node]
        //         while stack:
        //             node = stack.pop()
        //             nodes.append(node)
        //             stack.extend(node.children.values())
        //         return nodes
        todo!()
    }

    /// Print the tree structure for debugging.
    pub fn pretty_print(&self) {
        // def pretty_print(self) -> None:
        //         stack = [(self.root_node, 0)]
        //         while stack:
        //             node, indent = stack.pop()
        //             component_str = " ".join(
        //                 f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
        //                 for ct in self.component_types
        //             )
        //             print(
        //                 " " * indent,
        //                 f"[{node.id}]",
        //                 len(node.key),
        //                 f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
        //                 component_str,
        //             )
        //             for child in node.children.values():
        //                 stack.append((child, indent + 2))
        todo!()
    }

    /// The pretty_print rendering: one indented
    /// `[id] key_len full_lock component=yes/no` line per node.
    fn pretty_format_(&self) -> String {
        todo!()
    }

    /// Evictable token count of the FULL (base) component.
    pub fn evictable_size(&self) -> usize {
        // def evictable_size(self) -> int:
        //         return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)
        todo!()
    }

    /// Protected (locked) token count of the FULL (base) component.
    pub fn protected_size(&self) -> usize {
        // def protected_size(self) -> int:
        //         return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)
        todo!()
    }

    /// Evictable token count for one component (0 if the component is absent).
    pub fn component_evictable_size(&self, component_type: ComponentType) -> usize {
        // def component_evictable_size(self, component_type: ComponentType) -> int:
        //         """Evictable token count for one component (0 if the component is absent)."""
        //         return self.component_evictable_size_.get(component_type, 0)
        todo!()
    }

    /// Protected token count for one component (0 if the component is absent).
    pub fn component_protected_size(&self, component_type: ComponentType) -> usize {
        todo!()
    }

    /// FULL component evictable token count.
    pub fn full_evictable_size(&self) -> usize {
        // def full_evictable_size(self) -> int:
        //         return self.evictable_size()
        todo!()
    }

    /// FULL component protected token count.
    pub fn full_protected_size(&self) -> usize {
        // def full_protected_size(self) -> int:
        //         return self.protected_size()
        todo!()
    }

    /// SWA component evictable token count.
    pub fn swa_evictable_size(&self) -> usize {
        // def swa_evictable_size(self) -> int:
        //         return self.component_evictable_size_.get(ComponentType.SWA, 0)
        todo!()
    }

    /// Mamba component evictable token count.
    pub fn mamba_evictable_size(&self) -> usize {
        // def mamba_evictable_size(self) -> int:
        //         return self.component_evictable_size_.get(ComponentType.MAMBA, 0)
        todo!()
    }

    /// SWA component protected token count.
    pub fn swa_protected_size(&self) -> usize {
        // def swa_protected_size(self) -> int:
        //         return self.component_protected_size_.get(ComponentType.SWA, 0)
        todo!()
    }

    /// Mamba component protected token count.
    pub fn mamba_protected_size(&self) -> usize {
        // def mamba_protected_size(self) -> int:
        //         return self.component_protected_size_.get(ComponentType.MAMBA, 0)
        todo!()
    }

    /// (full_tokens, aux_tokens) summed across the whole tree.
    pub fn total_size(&self) -> (usize, usize) {
        // def total_size(self) -> tuple[int, int]:
        //         total_size = 0
        //         total_aux_size = 0
        //         stack = [self.root_node]
        //         while stack:
        //             node = stack.pop()
        //             full_value = node.component_data[BASE_COMPONENT_TYPE].value
        //             if full_value is not None:
        //                 total_size += len(full_value)
        //             for ct in self.component_types:
        //                 if ct == BASE_COMPONENT_TYPE:
        //                     continue
        //                 value = node.component_data[ct].value
        //                 if value is not None:
        //                     total_aux_size += len(value)
        //             for child in node.children.values():
        //                 stack.append(child)
        //         return total_size, total_aux_size
        todo!()
    }

    /// Every FULL device value in the tree, concatenated.
    pub fn all_values_flatten(&self) -> Tensor {
        // def all_values_flatten(self) -> torch.Tensor:
        //         values = []
        //
        //         def _dfs(node: UnifiedTreeNode):
        //             for child in node.children.values():
        //                 v = child.component_data[BASE_COMPONENT_TYPE].value
        //                 if v is not None:
        //                     values.append(v)
        //                 _dfs(child)
        //
        //         _dfs(self.root_node)
        //         if values:
        //             return torch.cat(values)
        //         return torch.tensor([], dtype=torch.int64, device=self.device)
        todo!()
    }

    /// Flatten every FULL device slot into (slot, position, prev-slot) rows for the KV-canary sweep.
    pub fn walk_for_kv_canary(
        &self,
        unlocked_only: bool,
        swa_resident_only: bool,
    ) -> KvCanaryWalkResult {
        // def walk_for_kv_canary(
        //         self, unlocked_only: bool, swa_resident_only: bool
        //     ) -> RadixCacheWalkResult:
        //         """Flatten every FULL device slot into (slot, position, prev-slot) rows for the KV-canary sweep."""
        //         slots: list[int] = []
        //         positions: list[int] = []
        //         prev_slots: list[int] = []
        //         swa_filter = swa_resident_only and ComponentType.SWA in self.components_by_type
        //
        //         def _dfs(node: UnifiedTreeNode, depth: int, parent_last_slot: int) -> None:
        //             value = node.component_data[BASE_COMPONENT_TYPE].value
        //             node_slots = value.tolist() if isinstance(value, torch.Tensor) else []
        //
        //             emit = node is not self.root_node
        //             if unlocked_only:
        //                 # Unified SWA owns an independent component lock. A node can still
        //                 # hold Full KV for a running request while its SWA slots are unused.
        //                 lock_ct = ComponentType.SWA if swa_filter else BASE_COMPONENT_TYPE
        //                 emit = emit and node.component_data[lock_ct].lock_ref == 0
        //             if swa_filter:
        //                 emit = emit and node.component_data[ComponentType.SWA].value is not None
        //
        //             # Skipped nodes still advance the chain/depth so descendants stay consistent.
        //             chain_last_slot = parent_last_slot
        //             for j, slot in enumerate(node_slots):
        //                 if emit:
        //                     slots.append(slot)
        //                     positions.append(depth + j)
        //                     prev_slots.append(parent_last_slot if j == 0 else node_slots[j - 1])
        //                 chain_last_slot = slot
        //
        //             # Device-evicted nodes hold no slots but still span their key length.
        //             if node is self.root_node or node.key is None:
        //                 child_depth = depth + len(node_slots)
        //             else:
        //                 child_depth = depth + len(node.key)
        //             for child in node.children.values():
        //                 _dfs(child, child_depth, chain_last_slot)
        //
        //         _dfs(self.root_node, 0, -1)
        //         return RadixCacheWalkResult(
        //             slot_indices=torch.tensor(slots, dtype=torch.int64),
        //             positions=torch.tensor(positions, dtype=torch.int64),
        //             prev_slot_indices=torch.tensor(prev_slots, dtype=torch.int64),
        //         )
        todo!()
    }

    /// Every Mamba device value in the tree, concatenated.
    pub fn all_mamba_values_flatten(&self) -> Tensor {
        // def all_mamba_values_flatten(self) -> torch.Tensor:
        //         return self._all_component_values_flatten(ComponentType.MAMBA)
        todo!()
    }
}

// ==== KV cache events ===================================================

/// Storage tier of a stored/removed block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageMedium {
    Gpu,
    Cpu,
}

impl StorageMedium {
    /// The python StorageMedium enum value.
    pub fn as_str(self) -> &'static str {
        todo!()
    }
}

/// A KV placement event; token ids carry one page's key atoms.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum KvCacheEvent<A> {
    BlockStored {
        block_hash: i64,
        parent_block_hash: Option<i64>,
        token_ids: Vec<A>,
        medium: StorageMedium,
    },
    BlockRemoved {
        block_hashes: Vec<i64>,
        medium: StorageMedium,
    },
    AllBlocksCleared,
}
