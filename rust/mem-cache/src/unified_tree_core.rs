//! The radix prefix tree of cached KV.
#![allow(unused_variables)]

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use tch::{Device, Kind, Tensor};

use crate::components::{self, FullComponent, MambaComponent, SwaComponent, TreeComponent};
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

// A 42-bit mask keeps digest multiplication by 1_000_003 within i64.
const COEXIST_RECLAIM_DIGEST_MULTIPLIER: i64 = 1_000_003;
const COEXIST_RECLAIM_DIGEST_MASK: i64 = (1 << 42) - 1;

fn next_coexist_reclaim_digest(current: i64, node_id: NodeId, component_idx: usize) -> i64 {
    let event = (node_id as i64 + 1) * NUM_COMPONENT_TYPES as i64 + component_idx as i64;
    (current * COEXIST_RECLAIM_DIGEST_MULTIPLIER + event) & COEXIST_RECLAIM_DIGEST_MASK
}

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
        match self {
            PoolHitPolicy::AllPages => "all_pages",
            PoolHitPolicy::TrailingPages => "trailing_pages",
        }
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
    DraftIndexer,
    DraftSwa,
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
        self == EvictLayer::All || self == layer
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
        CacheInitParams {
            eviction_policy: "lru".to_string(),
            page_size: 1,
            is_write_back: false,
            enable_hicache: false,
            write_through_threshold: 256,
            device: Device::Cpu,
            swa_sliding_window_size: None,
            has_swa_host_pool: false,
            enable_kv_cache_events: false,
            mamba_cache_chunk_size: None,
            mamba_max_states_per_path: None,
        }
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
    /// Full has no device LRU, so track nodes whose device and host values coexist.
    pub(crate) full_coexisting_host_nodes: EvictableNodeSet,
    pub(crate) write_back_coexist_reclaim_digest: i64,
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
        self.lru_list_(ValueSlotIdx::device(component_type))
    }

    /// The component's device-tier LRU list, mutable.
    pub(crate) fn device_lru_list_mut(
        &mut self,
        component_type: ComponentType,
    ) -> &mut UnifiedLRUList {
        self.lru_list_mut_(ValueSlotIdx::device(component_type))
    }

    /// The component's host-tier LRU list.
    pub(crate) fn host_lru_list(&self, component_type: ComponentType) -> &UnifiedLRUList {
        self.lru_list_(ValueSlotIdx::host(component_type))
    }

    /// The component's host-tier LRU list, mutable.
    pub(crate) fn host_lru_list_mut(
        &mut self,
        component_type: ComponentType,
    ) -> &mut UnifiedLRUList {
        self.lru_list_mut_(ValueSlotIdx::host(component_type))
    }

    /// The LRU list gated by the slot's lock.
    pub(crate) fn lru_list_(&self, slot: ValueSlotIdx) -> &UnifiedLRUList {
        &self.lru_lists[slot.idx()]
    }

    /// The LRU list gated by the slot's lock, mutable.
    pub(crate) fn lru_list_mut_(&mut self, slot: ValueSlotIdx) -> &mut UnifiedLRUList {
        &mut self.lru_lists[slot.idx()]
    }

    /// The component's device LRU list, mutable, paired with the arena the
    /// reset walks read.
    pub(crate) fn device_lru_list_mut_with_arena(
        &mut self,
        component_type: ComponentType,
    ) -> (&mut UnifiedLRUList, &NodeArena<K>) {
        (
            &mut self.lru_lists[ValueSlotIdx::device(component_type).idx()],
            &self.arena,
        )
    }

    /// The component's tree-wide bookkeeping state.
    pub(crate) fn component_state(&self, component_type: ComponentType) -> &ComponentState {
        &self.component_states[component_type.idx()]
    }

    /// The component's mutable tree-wide bookkeeping state.
    pub(crate) fn component_state_mut(
        &mut self,
        component_type: ComponentType,
    ) -> &mut ComponentState {
        &mut self.component_states[component_type.idx()]
    }

    /// The component's evictable device-token count.
    pub(crate) fn evictable_size_(&self, component_type: ComponentType) -> usize {
        self.component_state(component_type).evictable_size
    }

    /// The component's protected (locked) device-token count.
    pub(crate) fn protected_size_(&self, component_type: ComponentType) -> usize {
        self.component_state(component_type).protected_size
    }

    /// Begin the component's device-eviction bookkeeping for up to
    /// `request_cnt` tokens; panics if a walk is already in progress.
    pub(crate) fn set_evict_device_start(
        &mut self,
        component_type: ComponentType,
        request_cnt: usize,
    ) {
        let state = self.component_state_mut(component_type);
        assert!(
            !state.is_evict_device_ongoing,
            "{component_type:?} device eviction already in progress"
        );
        state.is_evict_device_ongoing = true;
        state.evict_device_request_cnt = request_cnt;
        state.evict_device_cursor = None;
    }

    /// Finish the component's device-eviction bookkeeping; panics if no walk
    /// is in progress.
    pub(crate) fn set_evict_device_end(&mut self, component_type: ComponentType) {
        let state = self.component_state_mut(component_type);
        assert!(
            state.is_evict_device_ongoing,
            "{component_type:?} device eviction not started"
        );
        state.is_evict_device_ongoing = false;
        state.evict_device_cursor = None;
    }

    /// Add newly evictable device tokens to the component's evictable size.
    pub(crate) fn inc_evictable_size(&mut self, component_type: ComponentType, tokens: usize) {
        self.component_state_mut(component_type).evictable_size += tokens;
    }

    /// Subtract freed device tokens from the component's evictable size.
    pub(crate) fn dec_evictable_size(&mut self, component_type: ComponentType, tokens: usize) {
        let state = self.component_state_mut(component_type);
        state.evictable_size = state.evictable_size.checked_sub(tokens).unwrap_or_else(|| {
            panic!("dec_evictable_size: {component_type:?} evictable size underflow")
        });
    }

    /// Add newly locked device tokens to the component's protected size.
    pub(crate) fn inc_protected_size(&mut self, component_type: ComponentType, tokens: usize) {
        self.component_state_mut(component_type).protected_size += tokens;
    }

    /// Subtract unlocked device tokens from the component's protected size.
    pub(crate) fn dec_protected_size(&mut self, component_type: ComponentType, tokens: usize) {
        let state = self.component_state_mut(component_type);
        state.protected_size = state.protected_size.checked_sub(tokens).unwrap_or_else(|| {
            panic!("dec_protected_size: {component_type:?} protected size underflow")
        });
    }

    pub fn new(params: CacheInitParams, component_types: Vec<ComponentType>) -> Self {
        assert!(
            !component_types.is_empty(),
            "at least one component type is required"
        );
        assert!(
            component_types.contains(&BASE_COMPONENT_TYPE),
            "the base (Full) component is required"
        );
        assert!(params.page_size >= 1, "page_size must be at least 1");
        let arena = NodeArena::new(component_types.clone(), params.page_size);
        let mut tree_core = UnifiedTreeCore {
            arena,
            components: Vec::new(),
            components_by_type: Default::default(),
            component_states: Default::default(),
            evictable_device_leaves: EvictableNodeSet::new(),
            evictable_host_leaves: EvictableNodeSet::new(),
            full_coexisting_host_nodes: EvictableNodeSet::new(),
            write_back_coexist_reclaim_digest: 0,
            // Disabled components keep harmless empty lists, like component_states.
            lru_lists: Self::new_lru_lists(),
            full_evict_device_heap: BinaryHeap::new(),
            eviction_strategy: get_eviction_strategy(&params.eviction_policy),
            page_size: params.page_size,
            is_write_back: params.is_write_back,
            enable_hicache: params.enable_hicache,
            enable_storage: false,
            has_swa_host_pool: params.has_swa_host_pool,
            enable_kv_cache_events: params.enable_kv_cache_events,
            kv_event_queue: Vec::new(),
            write_through_threshold: params.write_through_threshold,
            swa_uuid_counter: 1,
            device: params.device,
            empty_device_indices: Tensor::empty([0], (Kind::Int64, params.device)),
            ongoing_insert_walk_state: None,
        };
        for ct in &component_types {
            let component: Arc<dyn TreeComponent<K> + Send + Sync> = match ct {
                ComponentType::Full => Arc::new(FullComponent),
                ComponentType::Swa => Arc::new(SwaComponent::new(&params)),
                ComponentType::Mamba => Arc::new(MambaComponent::new(&params)),
            };
            tree_core.register_component_(component);
        }
        tree_core
    }

    /// Rebuild the root, LRUs, sizes, evictable-leaf sets, and the empty
    /// match result.
    pub fn reset(&mut self) {
        self.arena.reset();
        self.component_states = Default::default();
        self.evictable_device_leaves = EvictableNodeSet::new();
        self.evictable_host_leaves = EvictableNodeSet::new();
        self.full_coexisting_host_nodes = EvictableNodeSet::new();
        self.write_back_coexist_reclaim_digest = 0;
        self.lru_lists = Self::new_lru_lists();
        self.full_evict_device_heap.clear();
        self.ongoing_insert_walk_state = None;
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
        let new_node_id = self.arena.alloc_detached(priority);
        // Root children adopt the op namespace; deeper nodes inherit the parent's.
        let ns = if self.arena.node(parent_id).is_root() {
            extra_key.map(Arc::from)
        } else {
            self.arena.node(parent_id).extra_key.clone()
        };
        let new_node = self.arena.node_mut(new_node_id);
        new_node.key = key;
        new_node.parent = Some(parent_id);
        new_node.extra_key = ns;
        new_node.hit_count = hit_count;
        if let Some(creation_counter) = creation_counter {
            new_node.creation_counter = creation_counter;
        }
        new_node_id
    }

    /// Mint the next SWA lock-window uuid.
    pub(crate) fn next_swa_uuid_(&mut self) -> i64 {
        self.swa_uuid_counter += 1;
        self.swa_uuid_counter
    }

    /// Bump the reference count on a node's component locks.
    pub fn inc_lock_ref(&mut self, node_id: NodeId) -> IncLockRefResult {
        let node_id = self.arena.resolve(node_id);
        let mut result = IncLockRefResult::default();
        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            result = component
                .acquire_component_lock(self, node_id, result, /* lock_host = */ false);
        }
        self.update_evictable_leaf_sets_(node_id);
        result
    }

    /// Decrease the reference count on a node's component locks.
    pub fn dec_lock_ref(
        &mut self,
        node_id: NodeId,
        params: Option<&DecLockRefParams>,
        skip_swa: bool,
    ) -> DecLockRefResult {
        let node_id = self.arena.resolve(node_id);
        for i in 0..self.components.len() {
            if skip_swa && self.components[i].component_type() == SWA {
                continue;
            }
            let component = Arc::clone(&self.components[i]);
            component.release_component_lock(self, node_id, params, /* lock_host = */ false);
        }
        self.update_evictable_leaf_sets_(node_id);
        // TODO: delta is not aggregated from components; no caller uses it yet.
        DecLockRefResult::default()
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
        let node_id = self.arena.resolve(node_id);
        let Some(swa) = self.try_component_by_type_(SWA) else {
            return;
        };
        swa.release_window_lock(self, node_id, swa_uuid_for_lock, device_frees, host_frees);

        // Drop strictly-lower-priority locks (e.g. Mamba) co-located on the node.
        let swa_priority = swa.eviction_priority(/* is_leaf = */ false);
        let dec_params = DecLockRefParams {
            swa_uuid_for_lock,
            ..Default::default()
        };
        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            if component.eviction_priority(/* is_leaf = */ false) < swa_priority {
                component.release_component_lock(
                    self,
                    node_id,
                    Some(&dec_params),
                    /* lock_host = */ false,
                );
            }
        }
    }

    /// Evict shallow Mamba device checkpoints beyond the per-path cap on the
    /// tail's root path; the mamba component drives the walk.
    pub fn evict_excess_path_states(&mut self, tail_node_id: NodeId) -> EvictionStepResult {
        let tail_node_id = self.arena.resolve(tail_node_id);
        let mut result = EvictionStepResult::default();
        let component = self.component_by_type_(MAMBA);
        component.evict_excess_path_states(
            self,
            tail_node_id,
            &mut result.device_frees,
            &mut result.host_frees,
        );
        result
    }

    /// Bump the reference count on a node's host-side component locks.
    pub fn inc_host_lock_ref(&mut self, node_id: NodeId) -> IncLockRefResult {
        let node_id = self.arena.resolve(node_id);
        let mut result = IncLockRefResult::default();
        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            result = component
                .acquire_component_lock(self, node_id, result, /* lock_host = */ true);
        }
        self.update_evictable_leaf_sets_(node_id);
        result
    }

    /// Decrease the reference count on a node's host-side component locks.
    pub fn dec_host_lock_ref(
        &mut self,
        node_id: NodeId,
        params: Option<&DecLockRefParams>,
    ) -> DecLockRefResult {
        let node_id = self.arena.resolve(node_id);
        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            component.release_component_lock(self, node_id, params, /* lock_host = */ true);
        }
        self.update_evictable_leaf_sets_(node_id);
        DecLockRefResult::default()
    }

    /// Match a key against the tree; returns device indices + boundary NodeIds.
    pub fn match_prefix(&mut self, params: &MatchPrefixParams<'_, K>) -> MatchResult {
        // Bigram view conversion happens at the boundary; the key arrives typed.
        let aligned_key_len = params.key.atom_len() / self.page_size * self.page_size;
        if aligned_key_len == 0 {
            return self.empty_match_result();
        }
        // The walk reads only [0, aligned_key_len); the ragged tail never enters.
        let key = params.key;

        let root_id = self.arena.root();
        let (
            value,
            best_match_node_id,
            best_match_device_node_id,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        ) = self.match_prefix_helper_(root_id, params.extra_key, key, aligned_key_len);
        self.match_post_processor_(
            params,
            root_id,
            value,
            best_match_node_id,
            best_match_device_node_id,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        )
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
        // Non-HiCache mode has only device-resident matches, so the scheduler
        // device anchor follows the best match. In HiCache mode, host-backed
        // nodes can also match, so we separately track the best device-resident
        // match for scheduler prefix indices and locking.
        let mut node_id = root_id;
        // Walk cursor: atoms of `key` already matched.
        let mut offset = 0;
        let mut value: Vec<Tensor> = Vec::new();
        let mut best_match_node_id = node_id;
        let mut best_match_device_node_id = node_id;
        let mut best_match_device_value_len = 0;
        let mut full_kv_hit_length = 0;
        let mut action: Option<CacheAction> = None;
        let separate_device_match = self.enable_hicache;
        let mut validators = Vec::with_capacity(self.components.len());
        let mut device_validators = if separate_device_match {
            Vec::with_capacity(self.components.len())
        } else {
            Vec::new()
        };
        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            if separate_device_match {
                validators.push(
                    component.create_match_validator(self, /* match_device_only = */ false),
                );
                device_validators
                    .push(component.create_match_validator(self, /* match_device_only = */ true));
            } else {
                validators
                    .push(component.create_match_validator(self, /* match_device_only = */ true));
            }
        }

        fn update_best_if_valid<K: ChildKeyType>(
            tree: &UnifiedTreeCore<K>,
            node_id: NodeIdx_,
            value_len: usize,
            separate_device_match: bool,
            validators: &mut [Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool>],
            device_validators: &mut [Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool>],
            best_match_node_id: &mut NodeIdx_,
            best_match_device_node_id: &mut NodeIdx_,
            best_match_device_value_len: &mut usize,
        ) {
            // Every validator observes every node (stateful validators need the full walk).
            let matched = validators
                .iter_mut()
                .fold(true, |acc, validator| validator(tree, node_id) & acc);
            if matched {
                *best_match_node_id = node_id;
            }
            if !separate_device_match {
                if matched {
                    *best_match_device_value_len = value_len;
                    *best_match_device_node_id = node_id;
                }
                return;
            }
            if device_validators
                .iter_mut()
                .fold(true, |acc, validator| validator(tree, node_id) & acc)
            {
                *best_match_device_value_len = value_len;
                *best_match_device_node_id = node_id;
            }
        }

        while offset < aligned_key_len {
            let Some(child_id) =
                self.arena
                    .child_on_page(node_id, extra_key, key.page_at(offset, self.page_size))
            else {
                break;
            };
            let child = self.arena.node(child_id);
            // HiCache: a dead node (evicted and not backuped) stops the traversal.
            if child.evicted() && !child.backuped() {
                break;
            }
            let prefix_len = key.match_len(offset, &child.key, self.page_size);
            full_kv_hit_length += prefix_len;
            if prefix_len < child.key.atom_len() {
                let (split_node_id, split_action) = self.split_node_(child_id, prefix_len);
                node_id = split_node_id;
                action = split_action;
                let node = self.arena.node(node_id);
                if !node.evicted() {
                    value.push(node.device_value(FULL).shallow_clone());
                }
                update_best_if_valid(
                    self,
                    node_id,
                    value.len(),
                    separate_device_match,
                    &mut validators,
                    &mut device_validators,
                    &mut best_match_node_id,
                    &mut best_match_device_node_id,
                    &mut best_match_device_value_len,
                );
                break;
            }

            if !child.evicted() {
                value.push(child.device_value(FULL).shallow_clone());
            }
            node_id = child_id;
            update_best_if_valid(
                self,
                node_id,
                value.len(),
                separate_device_match,
                &mut validators,
                &mut device_validators,
                &mut best_match_node_id,
                &mut best_match_device_node_id,
                &mut best_match_device_value_len,
            );
            offset += prefix_len;
        }

        (
            value,
            best_match_node_id,
            best_match_device_node_id,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        )
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
        for i in 0..self.components.len() {
            // Full uses last_access ticks, not LRU.
            if self.components[i].component_type() == BASE_COMPONENT_TYPE {
                continue;
            }
            let component = Arc::clone(&self.components[i]);
            component.refresh_lru(self, LRURefreshPhase::MatchEnd, best_match_node_id);
        }

        // Re-stamp the matched path with fresh ticks, newest leaf-ward.
        let mut path = Vec::new();
        let mut cur = Some(best_match_node_id);
        while let Some(id) = cur {
            path.push(id);
            cur = self.arena.node(id).try_parent();
        }
        let newest_tick = self
            .arena
            .get_and_batch_bump_access_counter(path.len() as i64);
        for (i, id) in path.iter().enumerate() {
            self.arena.node_mut(*id).last_access_counter = newest_tick - i as i64;
        }

        // last_host_node will be used as the starting node for the subsequent
        // `prefetch_from_storage` flow. We directly use best_match_node here,
        // because best_match_node represents the node where all components
        // have reached consensus on both device & host availability.
        let last_host_node_id = if self.enable_hicache {
            best_match_node_id
        } else {
            best_match_device_node_id
        };

        let device_indices = if best_match_device_value_len > 0 {
            Tensor::cat(&value[..best_match_device_value_len], 0)
        } else {
            self.empty_device_indices.shallow_clone()
        };
        let mut result = MatchResult {
            device_indices,
            last_device_node_id: self.arena.node(best_match_device_node_id).id,
            last_host_node_id: self.arena.node(last_host_node_id).id,
            best_match_node_id: self.arena.node(best_match_node_id).id,
            host_hit_length: 0,
            mamba_host_hit_length: 0,
            mamba_branching_seqlen: None,
            swa_host_hit_length: 0,
            full_kv_hit_length,
            cache_actions: Vec::new(),
        };
        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            result = component.finalize_match_result_in_tree_core(
                self,
                result,
                params,
                &value,
                best_match_device_value_len,
            );
        }
        result.cache_actions = action.into_iter().collect();
        result
    }

    /// An empty match: no device indices, every boundary anchored at the root.
    pub fn empty_match_result(&self) -> MatchResult {
        let root_id = self.arena.node(self.arena.root()).id;
        MatchResult {
            device_indices: self.empty_device_indices.shallow_clone(),
            last_device_node_id: root_id,
            last_host_node_id: root_id,
            best_match_node_id: root_id,
            host_hit_length: 0,
            swa_host_hit_length: 0,
            full_kv_hit_length: 0,
            mamba_host_hit_length: 0,
            mamba_branching_seqlen: None,
            cache_actions: Vec::new(),
        }
    }

    /// Whether the node's FULL device value has been evicted.
    pub fn is_full_device_evicted(&self, node_id: NodeId) -> bool {
        let node_id = self.arena.resolve(node_id);
        self.arena.node(node_id).evicted()
    }

    /// Concatenate FULL device values from ``from_node`` up to (exclusive)
    /// ``until_node``, in root order; empty tensor if the path is empty.
    pub fn collect_full_device_indices(
        &self,
        from_node_id: NodeId,
        until_node_id: NodeId,
    ) -> Tensor {
        let from_node_id = self.arena.resolve(from_node_id);
        let until_node_id = self.arena.resolve(until_node_id);
        let mut prefix_chunks: Vec<Tensor> = Vec::new();
        let mut node_id = from_node_id;
        while node_id != until_node_id {
            let node = self.arena.node(node_id);
            prefix_chunks.push(node.device_value(FULL).shallow_clone());
            node_id = node.parent();
        }
        if prefix_chunks.is_empty() {
            return self.empty_device_indices.shallow_clone();
        }
        prefix_chunks.reverse();
        Tensor::cat(&prefix_chunks, 0)
    }

    /// Refresh a node's access tick and component LRU positions.
    pub fn touch_node_(&mut self, node_id: NodeIdx_) {
        let tick = self.arena.get_and_bump_access_counter();
        let node = self.arena.node_mut(node_id);
        node.last_access_counter = tick;
        if node.is_root() {
            return;
        }
        for i in 0..self.components.len() {
            // Full uses leaf sets, not LRU.
            if self.components[i].component_type() == BASE_COMPONENT_TYPE {
                continue;
            }
            let component = Arc::clone(&self.components[i]);
            component.refresh_lru(self, LRURefreshPhase::Walkdown, node_id);
        }
    }

    /// Increment hit count; check whether a write backup should be fired.
    pub fn inc_hit_count_and_check_(&mut self, node_id: NodeIdx_, chunked: bool) -> bool {
        let node = self.arena.node_mut(node_id);
        if node.evicted() || chunked {
            return false;
        }
        if self.is_write_back {
            return false;
        }
        node.hit_count += 1;
        self.enable_hicache && !node.backuped() && node.hit_count >= self.write_through_threshold
    }

    /// Insert device values to the tree per the provided key.
    pub fn insert(&mut self, params: &InsertParams<'_, K>) -> InsertResult {
        // Single-shot pump over the resumable walk: run every step inline and
        // fold the step actions into the result for the caller to apply.
        let mut actions = Vec::new();
        let mut step = self.begin_insert(params);
        loop {
            actions.append(&mut step.actions);
            if let Some(mut result) = step.result {
                result.cache_actions = actions;
                return result;
            }
            step = self.resume_insert();
        }
    }

    /// Start the insert, running to its first barrier or completion.
    pub fn begin_insert(&mut self, params: &InsertParams<'_, K>) -> InsertStepResult {
        // Insert walks are single-flight; a live walk means re-entrancy.
        assert!(
            self.ongoing_insert_walk_state.is_none(),
            "concurrent insert walks"
        );
        // Bigram view conversion happens at the boundary; the key arrives typed.
        let aligned_key_len = params.key.atom_len() / self.page_size * self.page_size;
        if aligned_key_len == 0 {
            // An empty insert still touches the root.
            let root_id = self.arena.root();
            self.touch_node_(root_id);
            {
                let node = self.arena.node_mut(root_id);
                node.priority = node.priority.max(params.priority);
            }
            return InsertStepResult {
                actions: Vec::new(),
                result: Some(InsertResult {
                    prefix_len: 0,
                    total_len: 0,
                    inserted_host_node: None,
                    mamba_exist: true,
                    cache_actions: Vec::new(),
                }),
            };
        }
        let root_id = self.arena.root();
        self.touch_node_(root_id);
        {
            let node = self.arena.node_mut(root_id);
            node.priority = node.priority.max(params.priority);
        }
        // The walk reads only [0, aligned_key_len); the ragged tail never enters.
        self.ongoing_insert_walk_state = Some(InsertWalkState {
            phase: InsertPhase::Walk,
            node_id: root_id,
            key: K::from(params.key.as_ref()[..aligned_key_len].to_vec()),
            aligned_key_len,
            value: params.value.narrow(0, 0, aligned_key_len as i64),
            extra_key: params.extra_key.map(str::to_owned),
            prev_prefix_len: params.prev_prefix_len,
            swa_evicted_seqlen: params.swa_evicted_seqlen,
            mamba_value: params.mamba_value.as_ref().map(Tensor::shallow_clone),
            chunked: params.chunked,
            priority: params.priority,
            total_prefix_length: 0,
            is_new_leaf: false,
            target_node_id: None,
            result: None,
            pending_actions: Vec::new(),
        });
        self.advance_insert_()
    }

    /// Continue the suspended insert after its step actions were executed.
    pub fn resume_insert(&mut self) -> InsertStepResult {
        assert!(
            self.ongoing_insert_walk_state.is_some(),
            "no in-flight insert"
        );
        self.advance_insert_()
    }

    /// Whether an insert walk is suspended at a barrier.
    pub fn has_ongoing_insert(&self) -> bool {
        self.ongoing_insert_walk_state.is_some()
    }

    /// Finish the insert (idempotent); returns still-pending actions to drain.
    pub fn end_insert(&mut self) -> Vec<CacheAction> {
        self.ongoing_insert_walk_state
            .take()
            .map(|state| state.pending_actions)
            .unwrap_or_default()
    }

    /// Run the in-flight insert to its next barrier or to completion.
    fn advance_insert_(&mut self) -> InsertStepResult {
        // The state moves out of self while steps run (they borrow the tree mutably).
        let mut state = self
            .ongoing_insert_walk_state
            .take()
            .expect("no in-flight insert");
        loop {
            let flushed_len = state.pending_actions.len();
            match state.phase {
                InsertPhase::Walk => self.insert_walk_step_(&mut state),
                InsertPhase::Commit => self.insert_commit_step_(&mut state),
                InsertPhase::Tail => {
                    self.insert_tail_step_(&mut state);
                    return InsertStepResult {
                        actions: state.pending_actions,
                        result: state.result,
                    };
                }
            }
            let new_actions = &state.pending_actions[flushed_len..];
            // Suspend only when a step emitted a non-deferrable action.
            if !new_actions.is_empty() && !new_actions.iter().all(Self::is_deferrable_action_) {
                let flushed = std::mem::take(&mut state.pending_actions);
                self.ongoing_insert_walk_state = Some(state);
                return InsertStepResult {
                    actions: flushed,
                    result: None,
                };
            }
        }
    }

    /// Fire-and-forget actions safe to batch until the next barrier.
    fn is_deferrable_action_(action: &CacheAction) -> bool {
        matches!(
            action,
            CacheAction::FreeDeviceKV(_) | CacheAction::ReplaceWriteThroughOnNodeSplit { .. }
        )
    }

    /// Process one walked node, appending its barrier actions to the state.
    fn insert_walk_step_(&mut self, state: &mut InsertWalkState<K>) {
        // Walk cursor: atoms of `key` already matched (also the running prefix length).
        let cursor = state.total_prefix_length;
        let child_id = if cursor < state.aligned_key_len {
            self.arena.child_on_page(
                state.node_id,
                state.extra_key.as_deref(),
                state.key.page_at(cursor, self.page_size),
            )
        } else {
            None
        };
        let Some(child_id) = child_id else {
            state.phase = InsertPhase::Commit;
            return;
        };
        let mut node_id = child_id;
        self.touch_node_(node_id);
        let node = self.arena.node(node_id);
        let prefix_len = state.key.match_len(cursor, &node.key, self.page_size);
        if prefix_len < node.key.atom_len() {
            let (split_node_id, action) = self.split_node_(node_id, prefix_len);
            node_id = split_node_id;
            if let Some(action) = action {
                state.pending_actions.push(action);
            }
        }
        {
            let node = self.arena.node_mut(node_id);
            node.priority = node.priority.max(state.priority);
        }

        let params = InsertParams {
            key: &state.key,
            extra_key: state.extra_key.as_deref(),
            value: state.value.shallow_clone(),
            prev_prefix_len: state.prev_prefix_len,
            swa_evicted_seqlen: state.swa_evicted_seqlen,
            mamba_value: state.mamba_value.as_ref().map(Tensor::shallow_clone),
            chunked: state.chunked,
            priority: state.priority,
        };
        if self.arena.node(node_id).evicted() {
            self.unevict_node_on_insert_(
                node_id,
                &state.value.narrow(0, cursor as i64, prefix_len as i64),
            );
            // FULL was restored from the request's fresh KV. Aux
            // components (e.g. SWA) may still hold tombstones and need
            // to rebuild their value from the same slice.
            for i in 0..self.components.len() {
                if self.components[i].component_type() == BASE_COMPONENT_TYPE {
                    continue;
                }
                let component = Arc::clone(&self.components[i]);
                component.recover_after_unevict(
                    self,
                    node_id,
                    prefix_len,
                    cursor,
                    &params,
                    &mut state.pending_actions,
                );
            }
        } else {
            let value_slice = state.value.narrow(0, cursor as i64, prefix_len as i64);
            let mut consumed_from = prefix_len;
            // Let each component claim ownership of overlapping KV slots.
            for i in 0..self.components.len() {
                let component = Arc::clone(&self.components[i]);
                let comp_consumed_from = component.update_component_on_insert_overlap(
                    self,
                    node_id,
                    prefix_len,
                    cursor,
                    value_slice.shallow_clone(),
                    &params,
                    &mut state.pending_actions,
                );
                consumed_from = consumed_from.min(comp_consumed_from);
            }

            let dup_start = state.prev_prefix_len.saturating_sub(cursor);
            if dup_start < consumed_from {
                state
                    .pending_actions
                    .push(CacheAction::FreeDeviceKV(vec![value_slice.narrow(
                        0,
                        dup_start as i64,
                        (consumed_from - dup_start) as i64,
                    )]));
            }
        }

        if self.inc_hit_count_and_check_(node_id, state.chunked) {
            let backup = self
                .build_backup_kv_action_(self.arena.node(node_id), /* write_back = */ false);
            state.pending_actions.push(CacheAction::BackupKV(backup));
        }
        state.node_id = node_id;
        state.total_prefix_length += prefix_len;
    }

    /// Create the tail leaf and run the component commit hooks.
    fn insert_commit_step_(&mut self, state: &mut InsertWalkState<K>) {
        // Create new leaf for remaining suffix. A leaf survives on its Full
        // value alone; auxiliary components (SWA, Mamba) may legitimately hold
        // only a tombstone for this span (e.g. the whole leaf is outside the SWA
        // window). Materialize it anyway so the Full KV stays cacheable.
        let target_node_id = if state.total_prefix_length < state.aligned_key_len {
            state.is_new_leaf = true;
            // The walk's only owned key: the unmatched suffix backing the new leaf.
            let leaf_value = state.value.narrow(
                0,
                state.total_prefix_length as i64,
                (state.aligned_key_len - state.total_prefix_length) as i64,
            );
            self.add_new_node_(
                state.node_id,
                K::from(
                    state.key.as_ref()[state.total_prefix_length..state.aligned_key_len].to_vec(),
                ),
                &leaf_value,
                state.priority,
                state.extra_key.as_deref(),
            )
        } else {
            state.node_id
        };
        state.target_node_id = Some(target_node_id);

        // Finalize: let each component attach its data to the target node.
        // e.g. Mamba attaches mamba_value to the leaf node
        // All hooks run before their emitted actions execute; an action failure
        // fail-stops the process, so partial-commit state is never observed.
        let mut result = InsertResult {
            prefix_len: state.total_prefix_length,
            total_len: 0,
            inserted_host_node: None,
            mamba_exist: false,
            cache_actions: Vec::new(),
        };
        let params = InsertParams {
            key: &state.key,
            extra_key: state.extra_key.as_deref(),
            value: state.value.shallow_clone(),
            prev_prefix_len: state.prev_prefix_len,
            swa_evicted_seqlen: state.swa_evicted_seqlen,
            mamba_value: state.mamba_value.as_ref().map(Tensor::shallow_clone),
            chunked: state.chunked,
            priority: state.priority,
        };
        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            component.commit_insert_component_data(
                self,
                target_node_id,
                state.is_new_leaf,
                &params,
                &mut result,
                &mut state.pending_actions,
            );
        }
        state.result = Some(result);
        state.phase = InsertPhase::Tail;
    }

    /// Refresh the LRUs and append the terminal new-leaf backup.
    fn insert_tail_step_(&mut self, state: &mut InsertWalkState<K>) {
        let target_node_id = state
            .target_node_id
            .expect("the commit step sets the target");
        if !self.arena.node(target_node_id).is_root() {
            for i in 0..self.components.len() {
                // Full uses leaf sets, not LRU.
                if self.components[i].component_type() == BASE_COMPONENT_TYPE {
                    continue;
                }
                let component = Arc::clone(&self.components[i]);
                component.refresh_lru(self, LRURefreshPhase::InsertEnd, target_node_id);
            }
        }

        if state.is_new_leaf && self.inc_hit_count_and_check_(target_node_id, state.chunked) {
            let backup = self.build_backup_kv_action_(
                self.arena.node(target_node_id),
                /* write_back = */ false,
            );
            state.pending_actions.push(CacheAction::BackupKV(backup));
        }
    }

    /// Split `child` at `split_len`; returns the new prefix node and any split action.
    pub fn split_node_(
        &mut self,
        child_id: NodeIdx_,
        split_len: usize,
    ) -> (NodeIdx_, Option<CacheAction>) {
        assert!(
            split_len > 0 && split_len.is_multiple_of(self.page_size),
            "split_node_: split_len {split_len} must be a nonzero page multiple"
        );
        let page_size = self.page_size;

        // The new node takes the child's prefix, link position, and stats.
        let child = self.arena.node(child_id);
        let parent_id = child.parent();
        let child_ns = child.extra_key.clone();
        let (key_head, key_tail) = child.key.split_at(split_len);
        // key_head keeps the original key's first page, which keys the parent's child map.
        let parent_map_key = key_head.child_key(page_size);
        let new_node_id = self.new_node_(
            key_head,
            parent_id,
            child.priority,
            child.hit_count,
            Some(child.creation_counter),
            child_ns.as_deref(),
        );
        self.arena
            .node_mut(new_node_id)
            .children
            .insert((child_ns.clone(), key_tail.child_key(page_size)), child_id);

        // The child's aux LRU cells detach while it is re-linked.
        self.for_each_component_lru_(
            child_id,
            &mut |lru, node_id| lru.remove_node(node_id),
            EvictLayer::Device,
            /* skip_existing = */ false,
        );

        let child = self.arena.node_mut(child_id);
        child.parent = Some(new_node_id);
        child.key = key_tail;
        let (new_node_hash, child_hash) =
            crate::node::split_node_hash_value(child.hash_value.take(), split_len, self.page_size);
        child.hash_value = child_hash;
        self.arena.node_mut(new_node_id).hash_value = new_node_hash;

        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            component.redistribute_on_node_split(self, new_node_id, child_id);
        }
        let replaced = self
            .arena
            .insert_child_edge(parent_id, parent_map_key, new_node_id);
        assert_eq!(
            replaced,
            Some(child_id),
            "split_node_: the parent's page entry must map to the split child"
        );

        // Preserve the load-back pin across a split.
        self.arena.node_mut(new_node_id).load_back_pending_id =
            self.arena.node(child_id).load_back_pending_id;

        // A split of a backuped node tells the cache to fix its publish list.
        let action = if let Some(ack_id) = self.arena.node(child_id).write_through_pending_id {
            self.arena.node_mut(new_node_id).write_through_pending_id = Some(ack_id);
            Some(CacheAction::ReplaceWriteThroughOnNodeSplit {
                ack_id,
                old_node_id: self.arena.node(child_id).id,
                new_node_id: self.arena.node(new_node_id).id,
                new_child_node_id: self.arena.node(child_id).id,
            })
        } else {
            None
        };

        self.for_each_component_lru_(
            new_node_id,
            &mut |lru, node_id| lru.insert_mru(node_id),
            EvictLayer::Device,
            /* skip_existing = */ true,
        );
        self.for_each_component_lru_(
            child_id,
            &mut |lru, node_id| lru.insert_mru(node_id),
            EvictLayer::Device,
            /* skip_existing = */ true,
        );
        let tick = self.arena.get_and_bump_access_counter();
        self.arena.node_mut(child_id).last_access_counter = tick;

        self.update_evictable_leaf_sets_(new_node_id);
        self.update_evictable_leaf_sets_(child_id);
        self.update_full_coexisting_host_tracking_(new_node_id);
        (new_node_id, action)
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
        let page_size = self.page_size;
        let child_map_key = key.child_key(page_size);
        let new_node_id = self.new_node_(
            key, parent_id, priority, /* hit_count = */ 0, /* creation_counter = */ None,
            extra_key,
        );
        self.arena.set_device_value(new_node_id, FULL, value.copy());
        let displaced = self
            .arena
            .insert_child_edge(parent_id, child_map_key, new_node_id);
        assert!(
            displaced.is_none(),
            "add_new_node_: parent {parent_id} already has a child on the new node's page"
        );
        self.inc_evictable_size(FULL, value.size()[0] as usize);
        if self.enable_storage {
            let hash_values = self.arena.compute_node_hash_values(new_node_id, page_size);
            self.arena.node_mut(new_node_id).hash_value = Some(hash_values);
        }

        self.update_evictable_leaf_sets_(new_node_id);
        self.update_evictable_leaf_sets_(parent_id);
        self.record_store_event_(new_node_id, StorageMedium::Gpu);
        new_node_id
    }

    /// Restore an evicted node's Full device value from fresh KV indices
    /// during insert.
    pub fn unevict_node_on_insert_(&mut self, node_id: NodeIdx_, fresh_value: &Tensor) {
        self.arena
            .set_device_value(node_id, FULL, fresh_value.copy());
        self.inc_evictable_size(FULL, fresh_value.size()[0] as usize);
        self.update_evictable_leaf_sets_(node_id);
        self.update_full_coexisting_host_tracking_(node_id);
        if let Some(parent_id) = self.arena.node(node_id).try_parent() {
            self.update_evictable_leaf_sets_(parent_id);
        }
        self.record_store_event_(node_id, StorageMedium::Gpu);
    }

    /// Update both device and host leaf sets for a node.
    pub(crate) fn update_evictable_leaf_sets_(&mut self, node_id: NodeIdx_) {
        let node = self.arena.node(node_id);
        let is_evictable_device_leaf = self.is_evictable_device_leaf_(node);
        let is_evictable_host_leaf = self.is_evictable_host_leaf_(node);
        if is_evictable_device_leaf {
            self.evictable_device_leaves.add(node_id);
        } else {
            self.evictable_device_leaves.discard(node_id);
        }
        if is_evictable_host_leaf {
            self.evictable_host_leaves.add(node_id);
        } else {
            self.evictable_host_leaves.discard(node_id);
        }
    }

    /// Refresh Full's lazily maintained device/host coexistence registry.
    pub(crate) fn update_full_coexisting_host_tracking_(&mut self, node_id: NodeIdx_) {
        if self.is_settled_full_coexisting_host_node_(self.arena.node(node_id)) {
            self.full_coexisting_host_nodes.add(node_id);
        } else {
            self.full_coexisting_host_nodes.discard(node_id);
        }
    }

    fn is_settled_full_coexisting_host_node_(&self, node: &Node<K>) -> bool {
        !node.is_root()
            && node.has_device_value(FULL)
            && node.has_host_value(FULL)
            && node.write_through_pending_id.is_none()
            && !node.is_load_back_pending()
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
        assert!(
            target != EvictLayer::All,
            "for_each_component_lru_: EvictLayer::All is not a single layer"
        );
        for i in 0..self.components.len() {
            let ct = self.components[i].component_type();
            // Full uses leaf sets, not LRU.
            if ct == BASE_COMPONENT_TYPE {
                continue;
            }
            let node = self.arena.node(node_id);
            let slot = if target == EvictLayer::Host {
                ValueSlotIdx::host(ct)
            } else {
                ValueSlotIdx::device(ct)
            };
            if !node.has_value_(slot) {
                continue;
            }
            let lru = self.lru_list_mut_(slot);
            if skip_existing && lru.in_list(Some(node_id)) {
                continue;
            }
            lru_op(lru, node_id);
        }
    }

    /// Register a component driver into the ordered fan-out list and the
    /// by-type lookup slot; rejects duplicates.
    pub(crate) fn register_component_(
        &mut self,
        component: Arc<dyn TreeComponent<K> + Send + Sync>,
    ) {
        let component_type = component.component_type();
        let slot = &mut self.components_by_type[component_type.idx()];
        assert!(
            slot.is_none(),
            "duplicate component type {component_type:?}"
        );
        *slot = Some(Arc::clone(&component));
        self.components.push(component);
    }

    /// Panics if the component is not enabled, matching the python KeyError.
    fn assert_component_enabled_(&self, component_type: ComponentType) {
        let _ = self.component_by_type_(component_type);
    }

    /// The component driver for `component_type`; panics if not enabled.
    fn component_by_type_(
        &self,
        component_type: ComponentType,
    ) -> Arc<dyn TreeComponent<K> + Send + Sync> {
        self.try_component_by_type_(component_type)
            .unwrap_or_else(|| panic!("{component_type:?} component is not enabled"))
    }

    /// The component driver for `component_type`, or None when not enabled.
    fn try_component_by_type_(
        &self,
        component_type: ComponentType,
    ) -> Option<Arc<dyn TreeComponent<K> + Send + Sync>> {
        // Cloning the Arc hands out an owned driver, leaving the registry unborrowed.
        self.components_by_type[component_type.idx()].clone()
    }

    /// Begin a component's device-eviction walk for up to request_cnt tokens.
    pub fn evict_device_start(&mut self, component_type: ComponentType, request_cnt: usize) {
        self.component_by_type_(component_type)
            .evict_device_start(self, request_cnt);
    }

    /// Return the next device leaf to evict for a component, or None when
    /// done. The walk's budget gate reads running totals from *baseline*;
    /// the result carries only this step's deltas.
    pub fn evict_device_next_node(
        &mut self,
        component_type: ComponentType,
        baseline: &HashMap<ComponentType, usize>,
    ) -> (Option<NodeId>, EvictionStepResult) {
        let mut tracker = baseline.clone();
        // The walk gates on the walked component's entry, so seed it.
        tracker.entry(component_type).or_insert(0);
        let mut result = EvictionStepResult::default();
        let node_id = self
            .component_by_type_(component_type)
            .evict_device_next_node(
                self,
                &mut tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
        for (ct, total) in tracker {
            let delta = total - baseline.get(&ct).copied().unwrap_or(0);
            if delta > 0 {
                result.tracker.insert(ct, delta);
            }
        }
        (node_id.map(|idx| self.arena.node(idx).id), result)
    }

    /// Finish a component's device-eviction walk.
    pub fn evict_device_end(&mut self, component_type: ComponentType) {
        self.component_by_type_(component_type)
            .evict_device_end(self);
    }

    /// Evict one device leaf (demote if backuped, delete if write-through);
    /// for an unbacked write-back node, return the BackupKV for the cache to
    /// execute and then demote, else None.
    pub fn evict_device_leaf(
        &mut self,
        node_id: NodeId,
        is_write_back: bool,
    ) -> (Option<BackupKV>, EvictionStepResult) {
        let node_id = self.arena.resolve(node_id);
        let mut result = EvictionStepResult::default();
        {
            let node = self.arena.node(node_id);
            assert!(
                self.is_evictable_device_leaf_(node),
                "node {node_id} is not a D-leaf"
            );
        }
        if self.arena.node(node_id).backuped() {
            self.demote_(
                node_id,
                &mut result.tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
            return (None, result);
        }
        if is_write_back {
            let backup = self
                .build_backup_kv_action_(self.arena.node(node_id), /* write_back = */ true);
            return (Some(backup), result);
        }

        // Write-through: node has no backup, delete entirely.
        self.delete_unbacked_device_leaf_(
            node_id,
            &mut result.tracker,
            &mut result.device_frees,
            &mut result.host_frees,
        );
        (None, result)
    }

    /// Write-back fallback when a D-leaf's D->H backup fails under host
    /// memory pressure: drop the subtree rooted at the unbacked leaf so
    /// device eviction keeps making progress instead of leaving its KV
    /// unevictable until host space frees up.
    pub fn drop_subtree_no_host(&mut self, node_id: NodeId) -> (bool, EvictionStepResult) {
        let node_id = self.arena.resolve(node_id);
        let mut result = EvictionStepResult::default();
        {
            let node = self.arena.node(node_id);
            assert!(
                self.is_evictable_device_leaf_(node),
                "node {node_id} is not a D-leaf"
            );
            // A failed backup never issues the D->H copy, so the subtree root has
            // no host state and no in-flight DMA reading its device slots.
            assert!(!node.backuped() && node.write_through_pending_id.is_none());
            if node.is_host_locked() {
                return (false, result);
            }
        }
        let mut descendants: Vec<NodeIdx_> = Vec::new();
        let mut stack: Vec<NodeIdx_> = self
            .arena
            .node(node_id)
            .children
            .values()
            .copied()
            .collect();
        while let Some(cur_id) = stack.pop() {
            let cur = self.arena.node(cur_id);
            if cur.is_device_locked() || cur.is_host_locked() {
                return (false, result);
            }
            descendants.push(cur_id);
            stack.extend(cur.children.values().copied());
        }
        for &desc_id in descendants.iter().rev() {
            {
                let desc = self.arena.node(desc_id);
                // Host-only by construction: a device descendant would contradict
                // this node being a D-leaf, and D-leaves evict before ancestors.
                assert!(
                    desc.evicted() && desc.backuped(),
                    "node {desc_id} not host-only"
                );
                assert!(desc.write_through_pending_id.is_none());
            }
            self.release_all_component_layers_(
                desc_id,
                StorageMedium::Cpu,
                &mut result.tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
            self.remove_leaf_from_parent_(desc_id);
        }
        self.delete_unbacked_device_leaf_(
            node_id,
            &mut result.tracker,
            &mut result.device_frees,
            &mut result.host_frees,
        );
        (true, result)
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
        self.record_remove_event_(node_id, medium);
        for i in 0..self.components.len() {
            let component_type = self.components[i].component_type();
            self.evict_component_and_detach_lru_(
                node_id,
                component_type,
                device_frees,
                host_frees,
                EvictLayer::All,
                Some(tracker),
            );
        }
        self.evictable_device_leaves.discard(node_id);
        self.evictable_host_leaves.discard(node_id);
    }

    /// Delete a device leaf that has no host backup, freeing all layers.
    pub fn delete_unbacked_device_leaf_(
        &mut self,
        node_id: NodeIdx_,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        self.release_all_component_layers_(
            node_id,
            StorageMedium::Gpu,
            tracker,
            device_frees,
            host_frees,
        );
        let parent = self.arena.node(node_id).parent();
        self.remove_leaf_from_parent_(node_id);
        self.update_evictable_leaf_sets_(parent);
        self.iteratively_delete_tombstone_leaf_(parent, tracker, device_frees, host_frees);
    }

    /// Evict a component's host-side resources.
    pub fn drive_host_eviction(
        &mut self,
        component_type: ComponentType,
        num_tokens: usize,
    ) -> EvictionStepResult {
        let mut result = EvictionStepResult::default();
        if let Some(component) = self.try_component_by_type_(component_type) {
            // The drive gates on the driven component's entry, so seed it.
            result.tracker.insert(component_type, 0);
            if self.is_write_back {
                component.reclaim_coexisting_host_values(
                    self,
                    num_tokens,
                    &mut result.tracker,
                    &mut result.device_frees,
                    &mut result.host_frees,
                );
            }
            component.drive_host_eviction(
                self,
                num_tokens,
                &mut result.tracker,
                &mut result.device_frees,
                &mut result.host_frees,
            );
        }
        result
    }

    pub(crate) fn can_reclaim_coexisting_host_value_(
        &self,
        node_id: NodeIdx_,
        component_type: ComponentType,
    ) -> bool {
        let node = self.arena.node(node_id);
        !node.is_root()
            && node.has_device_value(component_type)
            && node.has_host_value(component_type)
            && node.write_through_pending_id.is_none()
            && !node.is_load_back_pending()
            && node.host_lock_ref(component_type) == 0
    }

    /// Free one component's host value while its device value remains resident.
    pub(crate) fn release_coexisting_host_value_(
        &mut self,
        node_id: NodeIdx_,
        component_type: ComponentType,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        assert!(
            self.can_reclaim_coexisting_host_value_(node_id, component_type),
            "cannot reclaim coexisting {component_type:?} host value from node {node_id}"
        );
        if component_type == BASE_COMPONENT_TYPE {
            // BlockRemoved tracks Full host residency, not auxiliary slices.
            self.record_remove_event_(node_id, StorageMedium::Cpu);
        }
        self.evict_component_and_detach_lru_(
            node_id,
            component_type,
            device_frees,
            host_frees,
            EvictLayer::Host,
            Some(tracker),
        );
        let victim_id = self.arena.node(node_id).id;
        self.write_back_coexist_reclaim_digest = next_coexist_reclaim_digest(
            self.write_back_coexist_reclaim_digest,
            victim_id,
            component_type.idx(),
        );
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
        assert!(
            self.is_evictable_host_leaf_(self.arena.node(node_id)),
            "node {node_id} is not an H-leaf"
        );
        self.record_remove_event_(node_id, StorageMedium::Cpu);
        for i in 0..self.components.len() {
            let component_type = self.components[i].component_type();
            let (_, host_freed) = self.evict_component_and_detach_lru_(
                node_id,
                component_type,
                device_frees,
                host_frees,
                EvictLayer::All,
                None,
            );
            *tracker.entry(component_type).or_insert(0) += host_freed;
        }
        self.evictable_host_leaves.discard(node_id);
        let parent = self.arena.node(node_id).parent();
        self.remove_leaf_from_parent_(node_id);
        self.iteratively_delete_tombstone_leaf_(parent, tracker, device_frees, host_frees);
    }

    /// Release a node's device KV once its host copy exists; the node stays in the
    /// tree, now host-only.
    pub fn demote(&mut self, node_id: NodeId) -> EvictionStepResult {
        let node_id = self.arena.resolve(node_id);
        let mut result = EvictionStepResult::default();
        // Skip a deferred demote when a load-back now pins the device indices.
        if self.arena.node(node_id).is_load_back_pending() {
            return result;
        }
        self.demote_(
            node_id,
            &mut result.tracker,
            &mut result.device_frees,
            &mut result.host_frees,
        );
        result
    }

    /// Drop a backed-up node's device value, keeping the host copy.
    pub fn demote_(
        &mut self,
        node_id: NodeIdx_,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        {
            let node = self.arena.node(node_id);
            assert!(!node.evicted() && node.backuped());
        }
        self.evict_component_and_detach_lru_(
            node_id,
            BASE_COMPONENT_TYPE,
            device_frees,
            host_frees,
            EvictLayer::Device,
            Some(tracker),
        );
        self.cascade_evict_(
            node_id,
            BASE_COMPONENT_TYPE,
            tracker,
            device_frees,
            host_frees,
            EvictLayer::Device,
        );
        self.record_remove_event_(node_id, StorageMedium::Gpu);

        // after device eviction, insert aux components into host LRU.
        self.for_each_component_lru_(
            node_id,
            &mut UnifiedLRUList::insert_mru,
            EvictLayer::Host,
            /* skip_existing = */ true,
        );
        let parent = self.arena.node(node_id).parent();
        self.update_evictable_leaf_sets_(parent);
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
        let is_leaf = match target {
            EvictLayer::Device => self.evictable_device_leaves.contains(node_id),
            EvictLayer::Host => self.evictable_host_leaves.contains(node_id),
            EvictLayer::All => panic!("cascade_evict_: EvictLayer::All is not a single layer"),
        };

        let trigger_component = self.component_by_type_(trigger_component_type);
        let trigger_component_priority = trigger_component.eviction_priority(is_leaf);
        let trigger_component_internal_priority =
            trigger_component.eviction_priority(/* is_leaf = */ false);

        for i in 0..self.components.len() {
            let component = Arc::clone(&self.components[i]);
            let ct = component.component_type();
            if component.eviction_priority(is_leaf) > trigger_component_priority
                || ct == trigger_component_type
                || !components::node_has_component_data(&self.arena, node_id, ct, target)
            {
                continue;
            }
            let node = self.arena.node(node_id);
            let lock_ref = node.values[ct.idx()].lock_ref;
            let host_lock_ref = node.host_lock_ref(ct);
            // A comp whose TRUE internal priority outranks the trigger
            // is only in this loop because leaf-collapse flattened
            // priorities; a lock on it is a legit pin and must be
            // spared. A lock on a strictly-lower-priority tier is a
            // real strand — fall through to the assert below.
            if component.eviction_priority(/* is_leaf = */ false)
                >= trigger_component_internal_priority
            {
                if target == EvictLayer::Device && lock_ref != 0 {
                    continue;
                }
                if target == EvictLayer::Host && host_lock_ref != 0 {
                    continue;
                }
            }
            if target == EvictLayer::Device {
                assert!(
                    lock_ref == 0,
                    "cascade_evict_: a {ct:?} device lock strands node {node_id}"
                );
            }
            if target == EvictLayer::Host {
                assert!(
                    host_lock_ref == 0,
                    "cascade_evict_: a {ct:?} host lock strands node {node_id}"
                );
            }
            self.evict_component_and_detach_lru_(
                node_id,
                ct,
                device_frees,
                host_frees,
                target,
                Some(tracker),
            );
        }

        // Now that all components (including SWA which depends on Full.value)
        // have been freed, we can safely tombstone Full.value.
        // This is deferred from evict_component because free_swa needs it.
        if target == EvictLayer::Device && trigger_component_type == BASE_COMPONENT_TYPE {
            let _ = self.arena.take_device_value(node_id, FULL);
        }

        self.update_evictable_leaf_sets_(node_id);
    }

    /// Unlink a leaf from its parent.
    pub fn remove_leaf_from_parent_(&mut self, node_id: NodeIdx_) {
        // Arena slots are reused, so discard tracking before freeing the node.
        self.full_coexisting_host_nodes.discard(node_id);
        // The arena is the registry: freeing detaches by page key and recycles the slot.
        self.arena
            .free_leaf(node_id)
            .expect("remove_leaf_from_parent_: a deletable leaf");
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
        let component = self.component_by_type_(component_type);
        let (device_freed, host_freed) =
            component.evict_component(self, node_id, device_frees, host_frees, target);
        if let Some(tracker) = tracker {
            let freed = if target.contains(EvictLayer::Device) {
                device_freed
            } else {
                host_freed
            };
            *tracker.entry(component_type).or_insert(0) += freed;
        }

        // Detach from the targeted LRU list(s).
        if target.contains(EvictLayer::Device) {
            let lru = self.device_lru_list_mut(component_type);
            if lru.in_list(Some(node_id)) {
                lru.remove_node(node_id);
            }
        }
        if target.contains(EvictLayer::Host) {
            let lru = self.host_lru_list_mut(component_type);
            if lru.in_list(Some(node_id)) {
                lru.remove_node(node_id);
            }
        }
        (device_freed, host_freed)
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
        let ct = BASE_COMPONENT_TYPE;
        let mut cur = deleted_node_parent_id;
        loop {
            let node = self.arena.node(cur);
            if node.is_root() || !node.children.is_empty() {
                break;
            }
            if node.is_device_locked() || node.is_host_locked() {
                break;
            }
            let has_device = node.values[ct.idx()].value.is_some();
            let has_host = node.has_host_value(ct);

            if has_device {
                self.update_evictable_leaf_sets_(cur);
                break;
            }

            // Full device absent — clean up orphaned aux device data.
            for i in 0..self.components.len() {
                let component = Arc::clone(&self.components[i]);
                if self.arena.has_device_value(cur, component.component_type()) {
                    self.evict_component_and_detach_lru_(
                        cur,
                        component.component_type(),
                        device_frees,
                        host_frees,
                        EvictLayer::Device,
                        Some(tracker),
                    );
                }
            }

            if has_host {
                self.update_evictable_leaf_sets_(cur);
                break;
            }

            // Full absent on both layers — evict remaining host data, delete.
            for i in 0..self.components.len() {
                let component = Arc::clone(&self.components[i]);
                if self.arena.has_host_value(cur, component.component_type()) {
                    self.evict_component_and_detach_lru_(
                        cur,
                        component.component_type(),
                        device_frees,
                        host_frees,
                        EvictLayer::Host,
                        Some(tracker),
                    );
                }
            }

            self.evictable_host_leaves.discard(cur);
            let parent = self.arena.node(cur).parent();
            self.remove_leaf_from_parent_(cur);
            self.update_evictable_leaf_sets_(parent);
            cur = parent;
        }
    }

    /// Whether the node is an evictable Full device leaf.
    pub(crate) fn is_evictable_device_leaf_(&self, node: &Node<K>) -> bool {
        if node.is_root() || node.evicted() {
            return false;
        }
        if node.is_device_locked() {
            return false;
        }
        if node.is_load_back_pending() {
            return false;
        }
        if node
            .children
            .values()
            .any(|&child_id| self.arena.has_device_value(child_id, FULL))
        {
            return false;
        }
        true
    }

    /// Whether the node is an evictable Full host leaf.
    fn is_evictable_host_leaf_(&self, node: &Node<K>) -> bool {
        if node.is_root() || !node.evicted() {
            return false;
        }
        if !node.backuped() {
            return false;
        }
        if node.is_load_back_pending() {
            return false;
        }
        if node.is_host_locked() {
            return false;
        }
        if !node.children.is_empty() {
            return false;
        }
        true
    }

    /// Mark the host tier (HiCache) as wired.
    pub fn set_hicache_enabled(&mut self) {
        self.enable_hicache = true;
    }

    /// Whether the storage tier (L3) is wired; storage attaches after tree construction.
    pub fn set_enable_storage(&mut self, value: bool) {
        self.enable_storage = value;
    }

    // ==== KV cache placement events ====

    /// Queue one BlockStored per page of the node's key; hashes lazily if needed.
    fn record_store_event_(&mut self, node_id: NodeIdx_, medium: StorageMedium) {
        if !self.enable_kv_cache_events {
            return;
        }
        if self.arena.node(node_id).hash_value.is_none() {
            let hash_values = self.arena.compute_node_hash_values(node_id, self.page_size);
            self.arena.node_mut(node_id).hash_value = Some(hash_values);
        }
        let node = self.arena.node(node_id);
        let mut parent_block_hash = node.parent.and_then(|parent_id| {
            self.arena
                .node(parent_id)
                .get_last_hash_value()
                .map(crate::node::hash_str_to_int64)
        });
        let hash_value = node.hash_value.as_ref().expect("hashed above");
        let num_pages = node.key.atom_len().div_ceil(self.page_size);
        assert!(
            hash_value.len() >= num_pages,
            "store event: {} page hashes for {num_pages} pages",
            hash_value.len()
        );
        for (page, hash) in node.key.as_ref().chunks(self.page_size).zip(hash_value) {
            let block_hash = crate::node::hash_str_to_int64(hash);
            self.kv_event_queue.push(KvCacheEvent::BlockStored {
                block_hash,
                parent_block_hash,
                token_ids: page.to_vec(),
                medium,
            });
            parent_block_hash = Some(block_hash);
        }
    }

    /// Queue one BlockRemoved carrying all the node's page hashes; hashes lazily if needed.
    fn record_remove_event_(&mut self, node_id: NodeIdx_, medium: StorageMedium) {
        if !self.enable_kv_cache_events {
            return;
        }
        if self.arena.node(node_id).hash_value.is_none() {
            let hash_values = self.arena.compute_node_hash_values(node_id, self.page_size);
            self.arena.node_mut(node_id).hash_value = Some(hash_values);
        }
        let node = self.arena.node(node_id);
        let num_pages = node.key.atom_len().div_ceil(self.page_size);
        let block_hashes: Vec<i64> = node.hash_value.as_ref().expect("hashed above")[..num_pages]
            .iter()
            .map(|hash| crate::node::hash_str_to_int64(hash))
            .collect();
        if !block_hashes.is_empty() {
            self.kv_event_queue.push(KvCacheEvent::BlockRemoved {
                block_hashes,
                medium,
            });
        }
    }

    /// Queue the all-cleared marker.
    pub fn record_all_cleared_event(&mut self) {
        if self.enable_kv_cache_events {
            self.kv_event_queue.push(KvCacheEvent::AllBlocksCleared);
        }
    }

    /// Take all queued events, leaving the queue empty.
    pub fn take_events(&mut self) -> Vec<KvCacheEvent<K::Atom>> {
        std::mem::take(&mut self.kv_event_queue)
    }

    /// Mark the SWA host pool as wired; pools attach after tree construction.
    pub fn set_has_swa_host_pool(&mut self) {
        self.has_swa_host_pool = true;
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
        let total_len = key.atom_len();
        let mut node_id = self.arena.resolve(node_id);
        self.touch_node_(node_id);
        if total_len == 0 {
            return InsertResult {
                prefix_len: 0,
                total_len: 0,
                inserted_host_node: None,
                mamba_exist: true,
                cache_actions: Vec::new(),
            };
        }

        // Walk cursor: atoms of `key` already matched (also the running prefix length).
        let mut matched_length = 0;
        let mut cache_actions: Vec<CacheAction> = Vec::new();
        while matched_length < total_len {
            let Some(child_id) = self.arena.child_on_page(
                node_id,
                extra_key,
                key.page_at(matched_length, self.page_size),
            ) else {
                break;
            };
            node_id = child_id;
            self.touch_node_(node_id);
            let node = self.arena.node(node_id);
            let prefix_len = key.match_len(matched_length, &node.key, self.page_size);
            let node_key_len = node.key.atom_len();
            matched_length += prefix_len;

            if prefix_len < node_key_len {
                let (split_node_id, action) = self.split_node_(node_id, prefix_len);
                node_id = split_node_id;
                if let Some(action) = action {
                    cache_actions.push(action);
                }
            }
        }

        let mut result = InsertResult {
            prefix_len: matched_length,
            total_len,
            inserted_host_node: None,
            mamba_exist: false,
            cache_actions,
        };
        if matched_length == total_len {
            let node = self.arena.node(node_id);
            if !node.is_root() && node.has_host_value(FULL) {
                result.inserted_host_node = Some(self.arena.node(node_id).id);
            }
            return result;
        }

        let priority = self.arena.node(node_id).priority;
        let new_node_id = self.new_node_(
            key.suffix(matched_length),
            node_id,
            priority,
            /* hit_count = */ 0,
            /* creation_counter = */ None,
            extra_key,
        );
        {
            // The suffix moves into a right-sized list; the matched head drops.
            let mut hash_value = hash_value;
            let suffix = hash_value.split_off(matched_length / self.page_size);
            self.arena.node_mut(new_node_id).hash_value = Some(suffix);
        }
        self.arena.set_host_value(
            new_node_id,
            FULL,
            host_value
                .narrow(
                    0,
                    matched_length as i64,
                    (total_len - matched_length) as i64,
                )
                .copy(),
        );
        let child_map_key = self.arena.node(new_node_id).key.child_key(self.page_size);
        let displaced = self
            .arena
            .insert_child_edge(node_id, child_map_key, new_node_id);
        assert!(
            displaced.is_none(),
            "insert_host: parent {node_id} already has a child on the new node's page"
        );
        self.update_evictable_leaf_sets_(new_node_id);
        self.update_evictable_leaf_sets_(node_id);
        result.inserted_host_node = Some(self.arena.node(new_node_id).id);
        result
    }

    /// Read a node's device->host backup spec (device value + component transfers) now.
    pub fn build_backup_spec(
        &self,
        node_id: NodeId,
    ) -> (Tensor, HashMap<ComponentType, Vec<PoolTransfer>>) {
        self.build_backup_spec_(self.arena.node(self.arena.resolve(node_id)))
    }

    /// Gather device value backup spec.
    pub fn build_backup_spec_(
        &self,
        node: &Node<K>,
    ) -> (Tensor, HashMap<ComponentType, Vec<PoolTransfer>>) {
        // Overlapping backup chains may revisit a node whose Full KV already
        // has a host copy. Keep building transfers for auxiliary components,
        // but do not allocate and overwrite Full host KV a second time.
        let device_value = if node.backuped() {
            self.empty_device_indices.shallow_clone()
        } else {
            node.device_value(FULL).shallow_clone()
        };
        let mut comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>> = HashMap::new();
        for i in 0..self.components.len() {
            let component_type = self.components[i].component_type();
            if component_type == BASE_COMPONENT_TYPE {
                continue;
            }
            let transfers = self.components[i].build_hicache_transfers(
                self,
                node.idx,
                CacheTransferPhase::BackupHost,
                /* mamba_pool_idx = */ None,
                /* host_indices = */ None,
                /* token_ids = */ None,
                /* prefetch_tokens = */ 0,
                /* last_hash = */ None,
            );
            if let Some(transfers) = transfers
                && !transfers.is_empty()
            {
                comp_xfers.insert(component_type, transfers);
            }
        }
        (device_value, comp_xfers)
    }

    /// Gather a node's device->storage backup spec; None if the node is not backuped.
    pub fn build_storage_backup_spec(
        &self,
        node_id: NodeId,
        pass_prefix_keys: bool,
    ) -> Option<StorageBackupSpec> {
        let node_id = self.arena.resolve(node_id);
        let node = self.arena.node(node_id);
        if !node.backuped() {
            return None;
        }
        let prefix_keys = pass_prefix_keys.then(|| self.arena.prefix_hash_values(node.parent));
        let mut comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>> = HashMap::new();
        for i in 0..self.components.len() {
            let component_type = self.components[i].component_type();
            if component_type == BASE_COMPONENT_TYPE {
                continue;
            }
            let transfers = self.components[i].build_hicache_transfers(
                self,
                node_id,
                CacheTransferPhase::BackupStorage,
                /* mamba_pool_idx = */ None,
                /* host_indices = */ None,
                /* token_ids = */ None,
                /* prefetch_tokens = */ 0,
                /* last_hash = */ None,
            );
            if let Some(transfers) = transfers
                && !transfers.is_empty()
            {
                comp_xfers.insert(component_type, transfers);
            }
        }
        Some(StorageBackupSpec {
            host_value: node.host_value(FULL).shallow_clone(),
            token_ids: K::raw_token_ids(node.key.as_ref()).into_owned(),
            hash_value: node.hash_value.clone(),
            prefix_keys,
            comp_xfers,
        })
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
        let node_id = self.arena.resolve(node_id);
        self.component_by_type_(component_type)
            .build_hicache_transfers(
                self,
                node_id,
                phase,
                /* mamba_pool_idx = */ None,
                host_indices,
                token_ids,
                prefetch_tokens,
                last_hash,
            )
    }

    /// Build the H->D load-back KV transfer plus per-component aux transfers.
    pub fn build_load_back_spec(
        &self,
        node_id: NodeId,
        req: Option<&Req>,
    ) -> (PoolTransfer, HashMap<ComponentType, Vec<PoolTransfer>>) {
        let anchor_id = node_id;
        let node_id = self.arena.resolve(node_id);
        // Component hooks take primitives, not Req: extract its fields here.
        let mamba_pool_idx = req.and_then(|r| r.mamba_pool_idx.as_ref());
        let mut kv_transfers = self
            .component_by_type_(BASE_COMPONENT_TYPE)
            .build_hicache_transfers(
                self,
                node_id,
                CacheTransferPhase::LoadBack,
                /* mamba_pool_idx = */ None,
                /* host_indices = */ None,
                /* token_ids = */ None,
                /* prefetch_tokens = */ 0,
                /* last_hash = */ None,
            )
            .unwrap();
        let kv_xfer = kv_transfers.remove(0);
        let mut comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>> = HashMap::new();
        for i in 0..self.components.len() {
            let component_type = self.components[i].component_type();
            if component_type == BASE_COMPONENT_TYPE {
                continue;
            }
            let transfers = self.components[i].build_hicache_transfers(
                self,
                node_id,
                CacheTransferPhase::LoadBack,
                mamba_pool_idx.map(Tensor::shallow_clone),
                /* host_indices = */ None,
                /* token_ids = */ None,
                /* prefetch_tokens = */ 0,
                /* last_hash = */ None,
            );
            if let Some(transfers) = transfers
                && !transfers.is_empty()
            {
                comp_xfers.insert(component_type, transfers);
            }
        }
        // Reject transfers that would claim a node pinned by another load-back anchor.
        let any_foreign_pin = kv_xfer
            .nodes_to_load
            .iter()
            .chain(
                comp_xfers
                    .values()
                    .flatten()
                    .filter_map(|xfer| xfer.nodes_to_load.as_ref()),
            )
            .flatten()
            .any(|&pinned_id| {
                let pinned_idx = self.arena.resolve(pinned_id);
                self.arena
                    .node(pinned_idx)
                    .load_back_pending_id
                    .is_some_and(|id| id != anchor_id)
            });
        if any_foreign_pin {
            let empty_kv = PoolTransfer {
                name: PoolName::Kv,
                host_indices: Some(Tensor::empty([0], (Kind::Int64, tch::Device::Cpu))),
                nodes_to_load: Some(Vec::new()),
                ..Default::default()
            };
            return (empty_kv, HashMap::new());
        }
        (kv_xfer, comp_xfers)
    }

    /// The anchor node's namespace; None for the default namespace.
    pub fn prefetch_anchor_info(&self, node_id: NodeId) -> Option<String> {
        let node_id = self.arena.resolve(node_id);
        self.arena.node_extra_key(node_id).map(str::to_string)
    }

    /// Whether the node's Full KV is present on host.
    pub fn node_backuped(&self, node_id: NodeId) -> bool {
        let node_id = self.arena.resolve(node_id);
        self.arena.node(node_id).backuped()
    }

    /// Whether the node is a (default or named) root.
    pub fn is_root(&self, node_id: NodeId) -> bool {
        let node_id = self.arena.resolve(node_id);
        self.arena.node(node_id).is_root()
    }

    /// The node's last page hash, or None when it was never hashed.
    pub fn get_last_hash_value(&self, node_id: NodeId) -> Option<String> {
        let node_id = self.arena.resolve(node_id);
        self.arena
            .node(node_id)
            .get_last_hash_value()
            .map(str::to_string)
    }

    /// The hash chain of the node's ancestors, in root-to-parent order.
    pub fn get_prefix_hash_values(&self, node_id: NodeId) -> Vec<String> {
        let node_id = self.arena.resolve(node_id);
        self.arena
            .prefix_hash_values(self.arena.node(node_id).parent)
    }

    /// The hash values owned by this node, excluding its ancestors.
    pub fn get_hash_values(&self, node_id: NodeId) -> Vec<String> {
        let node_id = self.arena.resolve(node_id);
        self.arena
            .node(node_id)
            .hash_value
            .clone()
            .unwrap_or_default()
    }

    /// Hash every node built while storage was disabled.
    pub fn backfill_missing_hash_values(&mut self) -> usize {
        let root_id = self.arena.root();
        let mut filled = 0;
        for node_id in self.collect_all_nodes_() {
            if node_id == root_id || self.arena.node(node_id).hash_value.is_some() {
                continue;
            }
            let hash_values = self.arena.compute_node_hash_values(node_id, self.page_size);
            self.arena.node_mut(node_id).hash_value = Some(hash_values);
            filled += 1;
        }
        filled
    }

    /// The NodeId anchoring matches; the single root serves every namespace.
    pub fn root_node_handle(&self, _extra_key: Option<&str>) -> NodeId {
        self.arena.node(self.arena.root()).id
    }

    /// Build the backup action for a node and its unbacked ancestors.
    pub fn build_backup_kv_action_(&self, node: &Node<K>, write_back: bool) -> BackupKV {
        let mut chain = vec![node.id];
        if !write_back {
            let mut ancestor = node.try_parent();
            while let Some(ancestor_idx) = ancestor {
                let ancestor_node = self.arena.node(ancestor_idx);
                if ancestor_node.is_root() || ancestor_node.backuped() {
                    break;
                }
                chain.push(ancestor_node.id);
                ancestor = ancestor_node.try_parent();
            }
            // write_through: Ancestors first to preserve backup invariant
            chain.reverse();
        }
        BackupKV { node_ids: chain }
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
        let node_id = self.arena.resolve(node_id);
        for (component_type, transfers) in comp_xfers {
            self.component_by_type_(component_type)
                .commit_hicache_transfer(
                    self,
                    node_id,
                    phase,
                    transfers,
                    cache_actions,
                    insert_result.as_deref_mut(),
                    pool_storage_result,
                );
        }
    }

    /// Commit a successful backup to the node.
    pub fn commit_backup(
        &mut self,
        node_id: NodeId,
        host_indices: Tensor,
        comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>>,
    ) {
        let node_id = self.arena.resolve(node_id);
        let mut cache_actions: Vec<CacheAction> = Vec::new();
        if host_indices.numel() > 0 {
            let kv_xfer = PoolTransfer {
                name: PoolName::Kv,
                host_indices: Some(host_indices),
                ..Default::default()
            };
            self.component_by_type_(BASE_COMPONENT_TYPE)
                .commit_hicache_transfer(
                    self,
                    node_id,
                    CacheTransferPhase::BackupHost,
                    vec![kv_xfer],
                    &mut cache_actions,
                    /* insert_result = */ None,
                    /* pool_storage_result = */ None,
                );
        }
        for (component_type, transfers) in comp_xfers {
            self.component_by_type_(component_type)
                .commit_hicache_transfer(
                    self,
                    node_id,
                    CacheTransferPhase::BackupHost,
                    transfers,
                    &mut cache_actions,
                    /* insert_result = */ None,
                    /* pool_storage_result = */ None,
                );
        }
        assert!(cache_actions.is_empty()); // BACKUP_HOST emits no actions
        self.update_full_coexisting_host_tracking_(node_id);
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
        let anchor_id = node_id;
        let node_id = self.arena.resolve(node_id);
        let mut cache_actions: Vec<CacheAction> = Vec::new();
        kv_xfer.device_indices = Some(device_indices);
        let nodes_to_load = kv_xfer.nodes_to_load.clone();
        // Pin every host value read by the transfer until its ack.
        for pinned_id in nodes_to_load
            .iter()
            .chain(
                comp_xfers
                    .values()
                    .flatten()
                    .filter_map(|xfer| xfer.nodes_to_load.as_ref()),
            )
            .flatten()
            .copied()
        {
            let pinned_idx = self.arena.resolve(pinned_id);
            let pinned = self.arena.node_mut(pinned_idx);
            assert!(
                pinned.load_back_pending_id.is_none_or(|id| id == anchor_id),
                "node {pinned_id} pinned by load-back {:?}, new anchor {anchor_id}",
                pinned.load_back_pending_id
            );
            pinned.load_back_pending_id = Some(anchor_id);
            self.update_evictable_leaf_sets_(pinned_idx);
        }
        self.component_by_type_(BASE_COMPONENT_TYPE)
            .commit_hicache_transfer(
                self,
                node_id,
                CacheTransferPhase::LoadBack,
                vec![kv_xfer],
                &mut cache_actions,
                /* insert_result = */ None,
                /* pool_storage_result = */ None,
            );
        for loaded_id in nodes_to_load.unwrap_or_default() {
            let loaded_idx = self.arena.resolve(loaded_id);
            self.record_store_event_(loaded_idx, StorageMedium::Gpu);
        }
        for (component_type, transfers) in comp_xfers {
            self.component_by_type_(component_type)
                .commit_hicache_transfer(
                    self,
                    node_id,
                    CacheTransferPhase::LoadBack,
                    transfers,
                    &mut cache_actions,
                    /* insert_result = */ None,
                    /* pool_storage_result = */ None,
                );
        }
        self.update_evictable_leaf_sets_(node_id);
        cache_actions
    }

    /// Clear load-back pins on the anchor's root path.
    pub fn finish_load_back(&mut self, anchor_node_id: NodeId) {
        let mut node_id = Some(self.arena.resolve(anchor_node_id));
        while let Some(idx) = node_id {
            if self.arena.node(idx).is_root() {
                break;
            }
            if self.arena.node(idx).load_back_pending_id == Some(anchor_node_id) {
                self.arena.node_mut(idx).load_back_pending_id = None;
                self.update_full_coexisting_host_tracking_(idx);
                // The pin blocked leaf-set membership; re-evaluate it.
                self.update_evictable_leaf_sets_(idx);
            }
            node_id = self.arena.node(idx).try_parent();
        }
    }

    /// Mark a node as having an in-flight write-through backup.
    pub fn mark_write_through_pending(&mut self, node_id: NodeId) {
        let node_idx = self.arena.resolve(node_id);
        self.arena.node_mut(node_idx).write_through_pending_id = Some(node_id);
    }

    /// Clear the write-through-pending mark (when it matches ack_id) and record the
    /// host store event for each acked node.
    pub fn finish_write_through(&mut self, node_ids: Vec<NodeId>, ack_id: usize) {
        for node_id in node_ids {
            let node_idx = self.arena.resolve(node_id);
            let node = self.arena.node_mut(node_idx);
            if node.write_through_pending_id == Some(ack_id) {
                node.write_through_pending_id = None;
                self.update_full_coexisting_host_tracking_(node_idx);
            }
            self.record_store_event_(node_idx, StorageMedium::Cpu);
        }
    }

    /// Store an auxiliary component's device value onto a node and restamp
    /// its LRU.
    pub fn set_component_device_value(
        &mut self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Tensor,
    ) {
        self.assert_component_enabled_(component_type);
        let node_idx = self.arena.resolve(node_id);
        self.set_component_device_value_(node_idx, component_type, value);
    }

    /// Slot-keyed aux store (internal): set the device value and restamp the LRU.
    pub(crate) fn set_component_device_value_(
        &mut self,
        node_id: NodeIdx_,
        component_type: ComponentType,
        value: Tensor,
    ) {
        assert!(
            component_type != BASE_COMPONENT_TYPE,
            "set_component_device_value: auxiliary components only"
        );
        let tokens = value.size()[0] as usize;
        self.arena.set_device_value(node_id, component_type, value);
        let host_lru = self.host_lru_list_mut(component_type);
        if host_lru.in_list(Some(node_id)) {
            host_lru.remove_node(node_id);
        }
        self.device_lru_list_mut(component_type).insert_mru(node_id);
        self.inc_evictable_size(component_type, tokens);
    }

    /// The component's device value on the node, or None if evicted.
    pub fn get_component_device_value(
        &self,
        node_id: NodeId,
        component_type: ComponentType,
    ) -> Option<&Tensor> {
        self.assert_component_enabled_(component_type);
        self.arena
            .try_device_value(self.arena.resolve(node_id), component_type)
    }

    /// Whether the component's data is device-evicted but host-backed.
    pub fn component_has_host_value_only(
        &self,
        node_id: NodeId,
        component_type: ComponentType,
    ) -> bool {
        self.assert_component_enabled_(component_type);
        let node_idx = self.arena.resolve(node_id);
        !self.arena.has_device_value(node_idx, component_type)
            && self.arena.has_host_value(node_idx, component_type)
    }

    /// Verify tree-structure, leaf-set, LRU, size, and ongoing-op invariants; raise
    /// AssertionError on any violation. ongoing_* args are (id, node_id) pairs.
    pub fn sanity_check(
        &self,
        ongoing_write_through: &[(i64, NodeId)],
        ongoing_load_back: &[(i64, NodeId)],
    ) {
        let mut errors: Vec<String> = Vec::new();
        let all_nodes = self.collect_all_nodes_();
        let all_node_set: HashSet<NodeIdx_> = all_nodes.iter().copied().collect();

        // ── PART 1: Tree Structure ──
        // The single root: value-less, protected, parent-less, no node-level edges.
        let root_idx = self.arena.root();
        let root = self.arena.node(root_idx);
        for i in 0..self.components.len() {
            let ct = self.components[i].component_type();
            if root.values[ct.idx()].value.is_some() {
                errors.push(format!(
                    "[Root] root {root_idx} holds a {ct:?} device value"
                ));
            }
            if root.has_host_value(ct) {
                errors.push(format!("[Root] root {root_idx} holds a {ct:?} host value"));
            }
            if root.values[ct.idx()].lock_ref == 0 {
                errors.push(format!("[Root] root {root_idx} {ct:?} lock_ref=0"));
            }
        }
        if root.try_parent().is_some() {
            errors.push(format!("[Root] root {root_idx} has a parent pointer"));
        }
        // Leaf sets aside, every live arena slot must be tree-reachable.
        let orphans: Vec<NodeIdx_> = self
            .arena
            .live_ids()
            .filter(|id| !all_node_set.contains(id))
            .collect();
        if !orphans.is_empty() {
            errors.push(format!(
                "[Tree] {} orphaned live nodes: {:?}",
                orphans.len(),
                &orphans[..orphans.len().min(5)]
            ));
        }
        // Parent ↔ child bidirectional consistency
        for &node_id in &all_nodes {
            for ((edge_ns, edge_key), &child_id) in &self.arena.node(node_id).children {
                let child = self.arena.node(child_id);
                let child_parent = child.try_parent();
                if child_parent != Some(node_id) {
                    errors.push(format!(
                        "[Tree] child {child_id} parent={child_parent:?}, expected {node_id}"
                    ));
                }
                if child.key.atom_len() == 0 {
                    errors.push(format!("[Tree] node {child_id} has an empty key"));
                    continue;
                }
                if !child.key.atom_len().is_multiple_of(self.page_size) {
                    errors.push(format!("[Tree] node {child_id} key is not page-aligned"));
                    continue;
                }
                // The edge key must be the child's own namespaced child key.
                if *edge_key != child.key.child_key(self.page_size) {
                    errors.push(format!(
                        "[Tree] child {child_id} not mapped under its own child key"
                    ));
                }
                if *edge_ns != child.extra_key {
                    errors.push(format!(
                        "[Tree] child {child_id} namespace {:?} filed under {edge_ns:?}",
                        child.extra_key
                    ));
                }
                // Namespaces partition at the root; below it children inherit.
                if !self.arena.node(node_id).is_root()
                    && child.extra_key != self.arena.node(node_id).extra_key
                {
                    errors.push(format!(
                        "[Tree] child {child_id} namespace differs from its parent's"
                    ));
                }
                if let Some(value) = child.try_device_value(FULL)
                    && value.size()[0] as usize != child.key.atom_len()
                {
                    errors.push(format!(
                        "[Tree] node {child_id} Full value length {} != key length {}",
                        value.size()[0],
                        child.key.atom_len()
                    ));
                }
                if let Some(value) = child.try_host_value(FULL)
                    && value.size()[0] as usize != child.key.atom_len()
                {
                    errors.push(format!(
                        "[Tree] node {child_id} Full host value length {} != key length {}",
                        value.size()[0],
                        child.key.atom_len()
                    ));
                }
            }
        }

        // ── PART 2: Per-node state machine and leaf qualification ──
        let mut expected_dev_leaves: HashSet<NodeIdx_> = HashSet::new();
        let mut expected_hst_leaves: HashSet<NodeIdx_> = HashSet::new();
        let mut expected_full_coexisting_host_nodes: HashSet<NodeIdx_> = HashSet::new();

        for &node_id in &all_nodes {
            if self.arena.node(node_id).is_root() {
                continue;
            }
            let node = self.arena.node(node_id);
            let full_dev = node.has_device_value(FULL);
            let full_hst = node.has_host_value(FULL);

            // Full is the tree backbone, so aux data requires Full data.
            for i in 0..self.components.len() {
                let ct = self.components[i].component_type();
                if ct == BASE_COMPONENT_TYPE {
                    continue;
                }
                if node.values[ct.idx()].value.is_some() && !full_dev {
                    errors.push(format!(
                        "node {node_id} {ct:?} device present but Full.value=None"
                    ));
                }
                // Auxiliary host data may outlive Full host data under write-back.
                if node.has_host_value(ct) && !full_hst && !(self.is_write_back && full_dev) {
                    errors.push(format!(
                        "node {node_id} {ct:?} host present but Full.host_value=None"
                    ));
                }
            }

            // Every node must keep Full data on at least one layer.
            if !full_dev && !full_hst {
                errors.push(format!(
                    "node {node_id} dead: no Full device and no Full host"
                ));
            }

            // Parent prefixes must keep data whenever the child does.
            let parent_id = node.parent();
            if !self.arena.node(parent_id).is_root() {
                let parent = self.arena.node(parent_id);
                if full_dev && !parent.has_device_value(FULL) {
                    errors.push(format!(
                        "node {node_id} device present but parent {parent_id} evicted"
                    ));
                }
                if full_hst && !parent.has_host_value(FULL) && !self.is_write_back {
                    errors.push(format!(
                        "node {node_id} backed up but parent {parent_id} not backed up"
                    ));
                }
            }

            // Lock hierarchy must stay sane (the u32 counters cannot go negative).
            let full_lock = node.device_lock_ref(FULL);
            for i in 0..self.components.len() {
                let ct = self.components[i].component_type();
                let device_state = &node.values[ct.idx()];
                if ct != BASE_COMPONENT_TYPE && full_lock < device_state.lock_ref {
                    errors.push(format!(
                        "node {node_id} full_lock={full_lock} < {ct:?}_lock={}",
                        device_state.lock_ref
                    ));
                }
                if device_state.value.is_none() && device_state.lock_ref > 0 {
                    errors.push(format!(
                        "node {node_id} {ct:?} evicted but lock_ref={}",
                        device_state.lock_ref
                    ));
                }
            }

            // Collect expected leaf qualification (single pass)
            if self.is_evictable_device_leaf_(node) {
                expected_dev_leaves.insert(node_id);
            }
            if self.is_evictable_host_leaf_(node) {
                expected_hst_leaves.insert(node_id);
            }
            if self.is_settled_full_coexisting_host_node_(node) {
                expected_full_coexisting_host_nodes.insert(node_id);
            }
        }

        // ── PART 3: Tracking structures ──

        // Device leaf set must match the expected leaves.
        let device_leaves: HashSet<NodeIdx_> = self.evictable_device_leaves.iter().collect();
        if device_leaves != expected_dev_leaves {
            let extra: Vec<NodeIdx_> = device_leaves
                .difference(&expected_dev_leaves)
                .copied()
                .take(5)
                .collect();
            let missing: Vec<NodeIdx_> = expected_dev_leaves
                .difference(&device_leaves)
                .copied()
                .take(5)
                .collect();
            if !extra.is_empty() {
                errors.push(format!("D-leaf extra: {extra:?}"));
            }
            if !missing.is_empty() {
                errors.push(format!("D-leaf missing: {missing:?}"));
            }
        }

        // Host leaf set must match the expected leaves.
        let host_leaves: HashSet<NodeIdx_> = self.evictable_host_leaves.iter().collect();
        if host_leaves != expected_hst_leaves {
            let extra: Vec<NodeIdx_> = host_leaves
                .difference(&expected_hst_leaves)
                .copied()
                .take(5)
                .collect();
            let missing: Vec<NodeIdx_> = expected_hst_leaves
                .difference(&host_leaves)
                .copied()
                .take(5)
                .collect();
            if !extra.is_empty() {
                errors.push(format!("H-leaf extra: {extra:?}"));
            }
            if !missing.is_empty() {
                errors.push(format!("H-leaf missing: {missing:?}"));
            }
        }

        // Lazy tracking permits stale entries, but not missing or recycled ones.
        let full_coexisting_host_nodes: HashSet<NodeIdx_> =
            self.full_coexisting_host_nodes.iter().collect();
        let missing: Vec<NodeIdx_> = expected_full_coexisting_host_nodes
            .difference(&full_coexisting_host_nodes)
            .copied()
            .take(5)
            .collect();
        if !missing.is_empty() {
            errors.push(format!("Full host coexistence missing: {missing:?}"));
        }
        let ghosts: Vec<NodeIdx_> = full_coexisting_host_nodes
            .difference(&all_node_set)
            .copied()
            .take(5)
            .collect();
        if !ghosts.is_empty() {
            errors.push(format!("Full host coexistence ghosts: {ghosts:?}"));
        }

        // D-leaf ∩ H-leaf = ∅
        let overlap: Vec<NodeIdx_> = device_leaves.intersection(&host_leaves).copied().collect();
        if !overlap.is_empty() {
            errors.push(format!(
                "[Leaf] {} in both sets: {:?}",
                overlap.len(),
                &overlap[..overlap.len().min(5)]
            ));
        }

        // Stale nodes: leaf sets must only contain tree-reachable nodes
        let stale: Vec<NodeIdx_> = device_leaves.difference(&all_node_set).copied().collect();
        if !stale.is_empty() {
            errors.push(format!(
                "{} stale nodes in device_leaves: {:?}",
                stale.len(),
                &stale[..stale.len().min(5)]
            ));
        }
        let stale: Vec<NodeIdx_> = host_leaves.difference(&all_node_set).copied().collect();
        if !stale.is_empty() {
            errors.push(format!(
                "{} stale nodes in host_leaves: {:?}",
                stale.len(),
                &stale[..stale.len().min(5)]
            ));
        }

        // Per-component LRU tracking
        for i in 0..self.components.len() {
            let ct = self.components[i].component_type();
            let lru = self.device_lru_list(ct);
            let host_lru = self.host_lru_list(ct);
            if ct == BASE_COMPONENT_TYPE {
                // Full uses leaf sets, not LRU
                if lru.len() > 0 {
                    errors.push(format!("Full device LRU not empty: {}", lru.len()));
                }
                if host_lru.len() > 0 {
                    errors.push(format!("Full host LRU not empty: {}", host_lru.len()));
                }
                // Linked-list integrity
                lru.check_linked_list_(&format!("[device][{ct:?}]"), &mut errors);
                host_lru.check_linked_list_(&format!("[host][{ct:?}]"), &mut errors);
            } else {
                // Aux device values must match the device LRU; aux host-only
                // states must match the host LRU; never both at once.
                let mut device_count = 0;
                let mut host_only_count = 0;
                for &node_id in &all_nodes {
                    if self.arena.node(node_id).is_root() {
                        continue;
                    }
                    let node = self.arena.node(node_id);
                    let has_device = node.values[ct.idx()].value.is_some();
                    if has_device != lru.in_list(Some(node_id)) {
                        errors.push(format!(
                            "{ct:?} device LRU mismatch at node {node_id}: value={has_device} in_lru={}",
                            lru.in_list(Some(node_id))
                        ));
                    }
                    let host_only = !has_device && node.has_host_value(ct);
                    if host_only != host_lru.in_list(Some(node_id)) {
                        errors.push(format!(
                            "{ct:?} host LRU mismatch at node {node_id}: host_only={host_only} in_lru={}",
                            host_lru.in_list(Some(node_id))
                        ));
                    }
                    if lru.in_list(Some(node_id)) && host_lru.in_list(Some(node_id)) {
                        errors.push(format!("{ct:?} node {node_id} in both device and host LRU"));
                    }
                    device_count += has_device as usize;
                    host_only_count += host_only as usize;
                }
                if device_count != lru.len() {
                    errors.push(format!(
                        "{ct:?} device LRU: tree={device_count} != lru={}",
                        lru.len()
                    ));
                }
                if host_only_count != host_lru.len() {
                    errors.push(format!(
                        "{ct:?} host LRU: tree={host_only_count} != lru={}",
                        host_lru.len()
                    ));
                }
                // Linked-list integrity
                lru.check_linked_list_(&format!("[device][{ct:?}]"), &mut errors);
                host_lru.check_linked_list_(&format!("[host][{ct:?}]"), &mut errors);
            }
        }

        // ── PART 4: Size Accounting ──
        for i in 0..self.components.len() {
            let ct = self.components[i].component_type();
            let mut evictable = 0usize;
            let mut protected = 0usize;
            for &node_id in &all_nodes {
                if self.arena.node(node_id).is_root() {
                    continue;
                }
                let state = &self.arena.node(node_id).values[ct.idx()];
                if let Some(value) = &state.value {
                    let tokens = value.size()[0] as usize;
                    if state.lock_ref > 0 {
                        protected += tokens;
                    } else {
                        evictable += tokens;
                    }
                }
            }
            let recorded = self.component_state(ct);
            if recorded.evictable_size != evictable {
                errors.push(format!(
                    "[Size] {ct:?} evictable={} != recomputed={evictable}",
                    recorded.evictable_size
                ));
            }
            if recorded.protected_size != protected {
                errors.push(format!(
                    "[Size] {ct:?} protected={} != recomputed={protected}",
                    recorded.protected_size
                ));
            }
        }

        // ── PART 5: Ongoing Operations ──
        for &(op_id, node_id) in ongoing_write_through {
            match self.arena.try_resolve(node_id) {
                None => {
                    errors.push(format!("[Ongoing] write_through node {op_id} not in tree"));
                }
                Some(idx) if self.arena.device_lock_ref(idx, FULL) == 0 => {
                    errors.push(format!("[Ongoing] write_through node {op_id} lock_ref=0"));
                }
                Some(_) => {}
            }
        }
        for &(op_id, node_id) in ongoing_load_back {
            match self.arena.try_resolve(node_id) {
                None => {
                    errors.push(format!("[Ongoing] load_back node {op_id} not in tree"));
                }
                Some(idx) if self.arena.device_lock_ref(idx, FULL) == 0 => {
                    errors.push(format!("[Ongoing] load_back node {op_id} lock_ref=0"));
                }
                Some(_) => {}
            }
        }
        // Reject load-back pins that would survive their operation.
        let ongoing_load_ids: HashSet<NodeId> =
            ongoing_load_back.iter().map(|&(_, id)| id).collect();
        for &node_id in &all_nodes {
            let pending = self.arena.node(node_id).load_back_pending_id;
            if let Some(anchor) = pending
                && !ongoing_load_ids.contains(&anchor)
            {
                errors.push(format!(
                    "[Ongoing] node {node_id} load_back_pending_id={anchor} \
                     has no live load-back"
                ));
            }
        }

        if !errors.is_empty() {
            self.pretty_print();
            panic!(
                "Sanity check FAILED ({} violations across {} nodes):\n{}",
                errors.len(),
                all_nodes.len(),
                errors
                    .iter()
                    .map(|e| format!("  {e}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }

    /// Every live node in the tree.
    pub fn collect_all_nodes_(&self) -> Vec<NodeIdx_> {
        let mut nodes: Vec<NodeIdx_> = Vec::new();
        // The visited guard keeps a corrupted (cyclic) tree from hanging the walk.
        let mut visited: HashSet<NodeIdx_> = HashSet::new();
        let mut stack: Vec<NodeIdx_> = vec![self.arena.root()];
        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id) {
                continue;
            }
            nodes.push(node_id);
            stack.extend(self.arena.node(node_id).children.values().copied());
        }
        nodes
    }

    /// Print the tree structure for debugging.
    pub fn pretty_print(&self) {
        println!("{}", self.pretty_format_());
    }

    /// The pretty_print rendering: one indented
    /// `[id] key_len full_lock component=yes/no` line per node.
    fn pretty_format_(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        let mut visited: HashSet<NodeIdx_> = HashSet::new();
        let mut stack: Vec<(NodeIdx_, usize)> = vec![(self.arena.root(), 0)];
        while let Some((node_id, indent)) = stack.pop() {
            if !visited.insert(node_id) {
                continue;
            }
            let node = self.arena.node(node_id);
            let component_str = self
                .components
                .iter()
                .map(|component| {
                    let ct = component.component_type();
                    let state = if node.values[ct.idx()].value.is_some() {
                        "yes"
                    } else {
                        "no"
                    };
                    format!("{ct:?}={state}")
                })
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(format!(
                "{} [{}] {} full_lock={} {}",
                " ".repeat(indent),
                node.id,
                node.key.atom_len(),
                node.device_lock_ref(FULL),
                component_str
            ));
            stack.extend(node.children.values().map(|&child| (child, indent + 2)));
        }
        lines.join("\n")
    }

    /// Evictable token count of the FULL (base) component.
    pub fn evictable_size(&self) -> usize {
        self.evictable_size_(FULL)
    }

    /// Protected (locked) token count of the FULL (base) component.
    pub fn protected_size(&self) -> usize {
        self.protected_size_(FULL)
    }

    /// Evictable token count for one component (0 if the component is absent).
    pub fn component_evictable_size(&self, component_type: ComponentType) -> usize {
        self.try_component_by_type_(component_type)
            .map_or(0, |_| self.evictable_size_(component_type))
    }

    /// Protected token count for one component (0 if the component is absent).
    pub fn component_protected_size(&self, component_type: ComponentType) -> usize {
        self.try_component_by_type_(component_type)
            .map_or(0, |_| self.protected_size_(component_type))
    }

    /// FULL component evictable token count.
    pub fn full_evictable_size(&self) -> usize {
        self.evictable_size()
    }

    /// FULL component protected token count.
    pub fn full_protected_size(&self) -> usize {
        self.protected_size()
    }

    /// SWA component evictable token count.
    pub fn swa_evictable_size(&self) -> usize {
        self.evictable_size_(SWA)
    }

    /// Mamba component evictable token count.
    pub fn mamba_evictable_size(&self) -> usize {
        self.evictable_size_(MAMBA)
    }

    /// SWA component protected token count.
    pub fn swa_protected_size(&self) -> usize {
        self.protected_size_(SWA)
    }

    /// Mamba component protected token count.
    pub fn mamba_protected_size(&self) -> usize {
        self.protected_size_(MAMBA)
    }

    /// (full_tokens, aux_tokens) summed across the whole tree.
    pub fn total_size(&self) -> (usize, usize) {
        let mut total_size = 0;
        let mut total_aux_size = 0;
        let mut stack: Vec<NodeIdx_> = vec![self.arena.root()];
        while let Some(node_id) = stack.pop() {
            let node = self.arena.node(node_id);
            total_size += node.device_value_len(FULL);
            for i in 0..self.components.len() {
                let ct = self.components[i].component_type();
                if ct == BASE_COMPONENT_TYPE {
                    continue;
                }
                if let Some(value) = &node.values[ct.idx()].value {
                    total_aux_size += value.size()[0] as usize;
                }
            }
            stack.extend(self.arena.node(node_id).children.values().copied());
        }
        (total_size, total_aux_size)
    }

    /// Every FULL device value in the tree, concatenated.
    pub fn all_values_flatten(&self) -> Tensor {
        components::all_values_flatten(self, FULL)
    }

    /// Flatten every FULL device slot into (slot, position, prev-slot) rows for the KV-canary sweep.
    pub fn walk_for_kv_canary(
        &self,
        unlocked_only: bool,
        swa_resident_only: bool,
    ) -> KvCanaryWalkResult {
        let swa_filter = swa_resident_only && self.components_by_type[SWA.idx()].is_some();
        let mut slot_indices: Vec<i64> = Vec::new();
        let mut positions: Vec<i64> = Vec::new();
        let mut prev_slot_indices: Vec<i64> = Vec::new();
        // (node, is_root, atom depth from root, last device slot on the path above)
        let mut stack: Vec<(NodeIdx_, bool, i64, i64)> = vec![(self.arena.root(), true, 0, -1)];
        while let Some((node_id, is_root, depth, parent_last_slot)) = stack.pop() {
            let node = self.arena.node(node_id);
            let node_slots: Vec<i64> = node
                .try_device_value(FULL)
                .map(|value| {
                    Vec::<i64>::try_from(&value.to(Device::Cpu))
                        .expect("device values are 1-D int64 tensors")
                })
                .unwrap_or_default();

            let mut emit = !is_root;
            if unlocked_only {
                // Unified SWA owns an independent component lock. A node can still
                // hold Full KV for a running request while its SWA slots are unused.
                emit = emit
                    && if swa_filter {
                        node.device_lock_ref(SWA) == 0
                    } else {
                        node.device_lock_ref(FULL) == 0
                    };
            }
            if swa_filter {
                emit = emit && node.has_device_value(SWA);
            }

            // Skipped nodes still advance the chain/depth so descendants stay consistent.
            let mut chain_last_slot = parent_last_slot;
            for (j, &slot) in node_slots.iter().enumerate() {
                if emit {
                    slot_indices.push(slot);
                    positions.push(depth + j as i64);
                    prev_slot_indices.push(if j == 0 {
                        parent_last_slot
                    } else {
                        node_slots[j - 1]
                    });
                }
                chain_last_slot = slot;
            }

            // Device-evicted nodes hold no slots but still span their key length.
            let child_depth = depth + node.key.atom_len() as i64;
            for &child_id in node.children.values() {
                stack.push((child_id, false, child_depth, chain_last_slot));
            }
        }
        KvCanaryWalkResult {
            slot_indices,
            positions,
            prev_slot_indices,
        }
    }

    /// Every Mamba device value in the tree, concatenated.
    pub fn all_mamba_values_flatten(&self) -> Tensor {
        components::all_values_flatten(self, MAMBA)
    }
}

// KV cache placement events.

/// Storage tier of a stored/removed block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageMedium {
    Gpu,
    Cpu,
}

impl StorageMedium {
    /// The python StorageMedium enum value.
    pub fn as_str(self) -> &'static str {
        match self {
            StorageMedium::Gpu => "GPU",
            StorageMedium::Cpu => "CPU_PINNED",
        }
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
#[cfg(test)]
#[path = "tests/unified_tree_core.rs"]
mod tests;
