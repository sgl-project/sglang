use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use tch::Tensor;

use crate::component_type::{ComponentType, NUM_COMPONENT_TYPES};
use crate::error::{ChildKeyError, RadixCacheInitError};
use crate::tree_node_lru::{
    ComponentPoolState, EvictResult, FullLRUSlot, LRUData, LRUSlot, MambaLRUSlot, SwaLRUSlot,
    evict_full_value,
};

// TODO(Jialin): reuse evicted nodes by clearing fields in place instead of dropping;
// HashMap::clear() preserves the buffer and avoids reallocation on the next insert.
// TODO(Jialin): preallocate children HashMap via with_capacity(n) to avoid resize/rehash.
// TODO(Jialin): switch to hashbrown/ahash for faster non-cryptographic hashing.

/// Validated page size for the radix cache, rejected at construction if `0`.
#[derive(Debug, Clone, Copy)]
pub struct PageSize(usize);

impl PageSize {
    pub fn new(page_size: usize) -> Result<Self, RadixCacheInitError> {
        if page_size < 1 {
            return Err(RadixCacheInitError::InvalidPageSize {
                expected: ">= 1",
                got: page_size,
            });
        }
        Ok(Self(page_size))
    }

    pub fn get(&self) -> usize {
        self.0
    }
}

/// Trait for child key types that can be constructed from an atom slice.
///
/// `Atom` is the per-element type of the input key slice (`i64` for
/// single-token keys, `(i64, i64)` for EAGLE bigram keys). `Borrowed`
/// is the borrowed counterpart of the owned key (`[Atom]`).
pub trait ChildKeyType: Hash + Eq + Sized + Borrow<Self::Borrowed> {
    type Atom: Copy + Eq + Hash;
    type Borrowed: Hash + Eq + ToOwned<Owned = Self> + ?Sized;

    /// Extract a borrowed child key from an atom slice.
    fn make_child_key(
        key: &[Self::Atom],
        page_size: PageSize,
    ) -> Result<&Self::Borrowed, ChildKeyError>;
}

/// Page-aligned slice helper shared by all `ChildKeyType::make_child_key` impls.
fn page_aligned_slice<T>(key: &[T], page_size: PageSize) -> Result<&[T], ChildKeyError> {
    let ps = page_size.get();
    if key.len() < ps {
        return Err(ChildKeyError::SliceTooShort {
            page_size: ps,
            len: key.len(),
        });
    }
    Ok(&key[..ps])
}

impl ChildKeyType for Vec<i64> {
    type Atom = i64;
    type Borrowed = [i64];

    fn make_child_key(key: &[i64], page_size: PageSize) -> Result<&[i64], ChildKeyError> {
        page_aligned_slice(key, page_size)
    }
}

/// Bigram-keyed instantiation (EAGLE). Each atom is a `(t[i], t[i+1])` pair,
/// built from a raw `&[i64]` token slice via `windows(2).map(...)`.
impl ChildKeyType for Vec<(i64, i64)> {
    type Atom = (i64, i64);
    type Borrowed = [(i64, i64)];

    fn make_child_key(
        key: &[(i64, i64)],
        page_size: PageSize,
    ) -> Result<&[(i64, i64)], ChildKeyError> {
        page_aligned_slice(key, page_size)
    }
}

/// Index into a `TreeNodePool`.
///
/// Must NOT be held outside `TreeNodePool`: the freelist recycles indices, so an
/// evicted index may be reallocated to a new node (ABA problem). All node-referencing
/// state must be owned by `TreeNodePool` or its parent `RadixCache`.
pub type NodeIdx = usize;

/// Per-component per-node state. One entry per `ComponentType`.
///
/// `value: Option<Tensor>` semantics by component:
/// - FULL: `Some` for all live non-root nodes; `None` only on root. FULL
///   never tombstones — when FULL evicts a leaf, the whole node leaves.
/// - SWA: `Some` if the node sits within the sliding window with live SWA
///   KV; `None` otherwise (covers both never-populated and tombstoned —
///   `is_tombstone = value is None`).
/// - MAMBA: `Some` only on leaves with live Mamba state; `None` on
///   internal nodes and root.
///
/// Neither `Copy` nor `Clone` is derived (`tch::Tensor` is neither — GPU
/// resource copy semantics must be explicit).
#[derive(Default)]
pub struct ComponentNodeState {
    /// KV indices for this component on this node (None = no live value).
    pub value: Option<Tensor>,
    /// In-flight request count; while > 0 the slot is locked. Walk pattern
    /// is per-component (FULL: parents to root; SWA: until window-filled;
    /// Mamba: leaf only).
    pub lock_ref: u32,
    /// Intrusive LRU state for this component's slot. Mutated only via the
    /// `LRUSlot` trait's algorithm methods; `pub` so the trait's default
    /// `data` / `data_mut` accessors can route through it.
    pub lru_data: LRUData,
}

/// Radix tree node generic over child key type.
///
/// - `TreeNode<i64>`: page_size=1, children keyed by single token
/// - `TreeNode<Vec<i64>>`: page_size>1, children keyed by token page
///
/// Nodes are owned by a `TreeNodePool`. Parent and children are stored as
/// `NodeIdx` indices into the pool, not owned references.
pub struct TreeNode<K: ChildKeyType> {
    /// Atoms stored at this node (the edge label from parent to this node).
    /// Atom type follows `K::Atom` — `i64` for single-token keys, `(i64, i64)`
    /// for EAGLE bigram keys.
    key: Vec<K::Atom>,
    /// Index of parent node in the pool. None for the root.
    parent: Option<NodeIdx>,
    /// Children keyed by child key, values are indices into the pool.
    children: HashMap<K, NodeIdx>,
    /// Device-tier per-component node-level state.
    pub(crate) components: [ComponentNodeState; NUM_COMPONENT_TYPES],
    /// SWA's lock-walk boundary marker — `Some(uuid)` when this node is the
    /// boundary for one or more in-flight SWA acquires, `None` otherwise.
    /// Standalone field rather than per-component state since only the SWA
    /// acquire / release / split paths touch it.
    pub(crate) swa_uuid_for_lock: Option<u64>,
    /// Count of children holding a FULL device value. `is_evictable` treats a
    /// node with none as an (effective) leaf.
    pub(crate) num_children_with_device_full: u32,
}

impl<K: ChildKeyType> TreeNode<K> {
    /// Construct a root node — no parent, empty key, no value. FULL's
    /// `lock_ref` starts at 1 so the root is permanently protected; roots are
    /// never inserted into any LRU list.
    pub fn new_root() -> Self {
        let mut components: [ComponentNodeState; NUM_COMPONENT_TYPES] = Default::default();
        components[ComponentType::Full as usize].lock_ref = 1;
        Self {
            key: Vec::new(),
            parent: None,
            children: HashMap::new(),
            components,
            swa_uuid_for_lock: None,
            num_children_with_device_full: 0,
        }
    }

    /// Construct a non-root node with a known parent. `value` is `Option`
    /// because `split_node`'s transient intermediate and evicted children
    /// carry no value; the supplied value goes into FULL's slot (the canonical
    /// KV store). SWA/MAMBA values are populated by their `commit_insert_*` hooks.
    pub fn new_child(key: Vec<K::Atom>, parent_idx: NodeIdx, value: Option<Tensor>) -> Self {
        let mut components: [ComponentNodeState; NUM_COMPONENT_TYPES] = Default::default();
        components[ComponentType::Full as usize].value = value;
        Self {
            key,
            parent: Some(parent_idx),
            children: HashMap::new(),
            components,
            swa_uuid_for_lock: None,
            num_children_with_device_full: 0,
        }
    }

    /// Construct a sentinel node — empty, never reachable through tree
    /// traversal. Per-LRU sentinels are allocated at construction (and on
    /// `reset`) and never freed; `LRUSlot::init` cross-links their `lru_data`.
    pub fn new_sentinel() -> Self {
        Self {
            key: Vec::new(),
            parent: None,
            children: HashMap::new(),
            components: Default::default(),
            swa_uuid_for_lock: None,
            num_children_with_device_full: 0,
        }
    }

    /// The atoms stored at this node.
    pub fn key(&self) -> &[K::Atom] {
        &self.key
    }

    /// The parent index, or None for root.
    pub fn parent(&self) -> Option<NodeIdx> {
        self.parent
    }

    /// True iff this node is a namespace root (no parent). Roots are
    /// unevictable and are excluded from per-component LRU lists.
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// FULL's KV cache slot indices for this segment. None for root / evicted.
    /// SWA/MAMBA value access goes through `node.components[ct].value` directly.
    pub fn value(&self) -> Option<&Tensor> {
        self.components[ComponentType::Full as usize].value.as_ref()
    }

    /// Number of children.
    pub fn num_children(&self) -> usize {
        self.children.len()
    }

    /// True iff this node has no children. Leaves are FULL-evictable (subject
    /// to the lock_ref gate); internal nodes are SWA-tombstoneable but not
    /// FULL-evictable until they become leaves.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// FULL's lock-ref count: 0 means unlocked (eligible for eviction subject
    /// to leaf/value conditions), > 0 means in use. SWA/MAMBA lock_refs live in
    /// their own component slots (`node.components[ct].lock_ref`).
    pub fn lock_ref(&self) -> u32 {
        self.components[ComponentType::Full as usize].lock_ref
    }

    /// True iff any component's `lock_ref > 0`. `evict_leaf` asserts on this as
    /// defense-in-depth: `full_lock_ref == 0` should already imply the others
    /// are 0, so this catches an invariant break as a panic instead of a silent
    /// use-after-free of an orphaned non-FULL value.
    pub fn is_locked(&self) -> bool {
        self.components.iter().any(|c| c.lock_ref > 0)
    }

    /// SWA's lock-walk boundary uuid for this node. See the field doc.
    pub fn swa_uuid_for_lock(&self) -> Option<u64> {
        self.swa_uuid_for_lock
    }

    /// Set this node's SWA lock-walk boundary uuid (`None` clears). Production
    /// callers use `TreeNodePool::lazy_acquire_swa_uuid_for_lock`; this raw
    /// setter is for split-node transfer and tests.
    pub fn set_swa_uuid_for_lock(&mut self, uuid: Option<u64>) {
        self.swa_uuid_for_lock = uuid;
    }

    /// Extract the child key from this node's key (the first page_size atoms).
    pub fn child_key(&self, page_size: PageSize) -> Result<&K::Borrowed, ChildKeyError> {
        K::make_child_key(&self.key, page_size)
    }

    /// Lookup a child index by borrowed key. Zero allocation.
    pub fn get_child(&self, child_key: &K::Borrowed) -> Option<NodeIdx> {
        self.children.get(child_key).copied()
    }

    /// Insert a child index. Panics if the key already exists. Takes the key by
    /// value to avoid double-allocation when the caller already owns a `K`.
    pub fn insert_child(&mut self, child_key: K, child_idx: NodeIdx) {
        match self.children.entry(child_key) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(child_idx);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                #[allow(clippy::panic, reason = "callers must check child absence first")]
                {
                    panic!("insert_child: child_key already exists");
                }
            }
        }
    }

    /// Replace an existing child index. Panics if the key does not exist.
    /// Returns the old child index. Zero allocation.
    pub fn replace_child(&mut self, child_key: &K::Borrowed, child_idx: NodeIdx) -> NodeIdx {
        #[allow(clippy::expect_used, reason = "caller must check child presence first")]
        let slot = self
            .children
            .get_mut(child_key)
            .expect("replace_child: child_key does not exist");
        let old_idx = *slot;
        *slot = child_idx;
        old_idx
    }

    /// Remove a child by key. Panics if the key does not exist.
    /// Returns the removed child's NodeIdx.
    pub fn remove_child(&mut self, child_key: &K::Borrowed) -> NodeIdx {
        #[allow(clippy::expect_used, reason = "caller must check child presence first")]
        self.children
            .remove(child_key)
            .expect("remove_child: child_key does not exist")
    }

    /// Return the length of the longest common prefix between this node's key
    /// and `key`, rounded down to a multiple of `page_size`.
    pub fn match_key(&self, key: &[K::Atom], page_size: PageSize) -> usize {
        let ps = page_size.get();
        let max_pages = self.key.len().min(key.len()) / ps;
        if max_pages == 0 {
            return 0;
        }

        // Galloping + binary search over page-aligned chunks: long-context hits
        // are usually full-node matches, and slice equality uses libcore's
        // optimized comparison path while preserving the page-rounded result.
        let mut matched_pages = 0usize;
        let mut step_pages = 1usize;
        while matched_pages < max_pages {
            let hi_pages = matched_pages.saturating_add(step_pages).min(max_pages);
            let start = matched_pages * ps;
            let end = hi_pages * ps;
            if self.key[start..end] == key[start..end] {
                matched_pages = hi_pages;
                step_pages = step_pages.saturating_mul(2).max(1);
                continue;
            }

            let mut lo_pages = matched_pages;
            let mut bad_pages = hi_pages;
            while bad_pages - lo_pages > 1 {
                let mid_pages = lo_pages + (bad_pages - lo_pages) / 2;
                let start = lo_pages * ps;
                let end = mid_pages * ps;
                if self.key[start..end] == key[start..end] {
                    lo_pages = mid_pages;
                } else {
                    bad_pages = mid_pages;
                }
            }
            return lo_pages * ps;
        }

        matched_pages * ps
    }
}

/// Outcome of `TreeNodePool::match_child`, encoding the partial-vs-full
/// distinction in the type.
pub enum MatchChildResult {
    /// No child entry for `query_key`'s first page.
    NotFound,
    /// All of the matched child's stored key matched; caller may walk into
    /// `child_idx`. `node_key_len == child.key().len()`.
    FullMatch {
        child_idx: NodeIdx,
        node_key_len: usize,
    },
    /// Only a prefix of the child's key matched. Caller must call
    /// `split_node(node_split)` before continuing (or stop).
    PartialMatch(NodeSplit),
}

/// A validated node split: `child_idx` is a non-root node with a parent, and
/// `split_len` is page-aligned and in `[page_size, child.key().len() - page_size]`.
/// Only constructible via `match_child` returning `PartialMatch`, which validates
/// these invariants (test wrappers may bypass via struct literal).
pub struct NodeSplit {
    pub(crate) child_idx: NodeIdx,
    pub(crate) split_len: usize,
}

impl NodeSplit {
    pub fn split_len(&self) -> usize {
        self.split_len
    }
}

/// Arena-based pool that owns all tree nodes. Evicted slots are recycled via a
/// freelist; access a node by its `NodeIdx`.
///
/// Per-component `ComponentPoolState`s (sentinel pair + size aggregates) live
/// here so the high-level ops (`insert_leaf`, `split_node`, `evict_leaf`) can
/// splice nodes and keep size aggregates in sync. Sentinels are never freed and
/// excluded from `active_node_count()`.
pub struct TreeNodePool<K: ChildKeyType> {
    nodes: Vec<Option<TreeNode<K>>>,
    evicted_indices: Vec<NodeIdx>,
    page_size: PageSize,
    /// Device-tier per-component pool-level state.
    pub(crate) components: [ComponentPoolState; NUM_COMPONENT_TYPES],
    /// Sentinel slots allocated for this pool (one per LRU). Bumped by
    /// `alloc_sentinel`; excluded from `active_node_count`.
    sentinel_count: usize,
    /// Monotonic counter for SWA's lock-walk boundary uuids, minted via
    /// `acquire_next_swa_uuid_for_lock`. Starts at 1 so `0` stays distinguishable
    /// from default-zero memory. Plain `u64` is sufficient — the cache is
    /// single-threaded (PyO3 GIL serializes all entrypoints).
    next_swa_uuid_for_lock: u64,
    /// Whether SWA is configured for this cache.
    ///
    /// Gates the FULL-evict tombstone cascade: in FULL-only mode no node ever
    /// has an SWA value, so `iteratively_delete_tombstone_leaf`'s
    /// `has_value(SwaLRUSlot)` stop condition would treat every ancestor as a
    /// tombstone and over-delete. This short-circuits the cascade.
    has_swa_component: bool,
    /// Whether Mamba is configured for this cache.
    has_mamba_component: bool,
}

impl<K: ChildKeyType> TreeNodePool<K> {
    /// Construct from an already-validated `PageSize`. Initializes
    /// per-slot LRU sentinels via `LRUSlot::init`.
    pub fn new(
        page_size: PageSize,
        init_node_capacity: usize,
        has_swa_component: bool,
        has_mamba_component: bool,
    ) -> Self {
        let mut pool = Self {
            nodes: Vec::with_capacity(init_node_capacity),
            evicted_indices: Vec::new(),
            page_size,
            components: [ComponentPoolState::default(); NUM_COMPONENT_TYPES],
            sentinel_count: 0,
            // Start at 1 so 0 stays distinguishable from default-zero memory.
            next_swa_uuid_for_lock: 1,
            has_swa_component,
            has_mamba_component,
        };
        FullLRUSlot::init::<K>(&mut pool);
        SwaLRUSlot::init::<K>(&mut pool);
        MambaLRUSlot::init::<K>(&mut pool);
        pool
    }

    /// Allocate a sentinel TreeNode and bump `sentinel_count` so
    /// `active_node_count` continues to exclude it. Used exclusively by
    /// `LRUSlot::init`. Sentinels are never freed.
    pub(crate) fn alloc_sentinel(&mut self) -> NodeIdx {
        let idx = self.alloc(TreeNode::new_sentinel());
        self.sentinel_count += 1;
        idx
    }

    /// Get-or-mint the SWA lock-walk boundary uuid for `node_idx`. Returns the
    /// existing stamp if present (concurrent acquires hitting the same boundary
    /// share the marker); otherwise mints a fresh uuid, stamps it, and returns it.
    pub fn lazy_acquire_swa_uuid_for_lock(&mut self, node_idx: NodeIdx) -> u64 {
        if let Some(existing) = self.get(node_idx).swa_uuid_for_lock() {
            return existing;
        }
        let new_uuid = self.acquire_next_swa_uuid_for_lock();
        self.get_mut(node_idx).set_swa_uuid_for_lock(Some(new_uuid));
        new_uuid
    }

    /// Mint the next SWA lock-walk uuid, always advancing the counter (for
    /// stamp-if-absent semantics use `lazy_acquire_swa_uuid_for_lock`).
    ///
    /// `checked_add` panics on u64 overflow rather than silently wrapping and
    /// reissuing an in-use uuid, which would break the uniqueness guarantee
    /// SWA's release-walk relies on. The counter advances only after the check
    /// passes, so a panic leaves it unchanged.
    pub(crate) fn acquire_next_swa_uuid_for_lock(&mut self) -> u64 {
        let id = self.next_swa_uuid_for_lock;
        #[allow(
            clippy::expect_used,
            reason = "u64 overflow at 1.8e19 acquires; effectively impossible"
        )]
        let next = id.checked_add(1).expect(
            "acquire_next_swa_uuid_for_lock: u64 counter would overflow — \
             exhausted SWA uuid space (>1.8e19 acquires)",
        );
        self.next_swa_uuid_for_lock = next;
        id
    }

    /// Free a pool slot without parent cleanup. Use when the caller has already
    /// handled removing the node from its parent's children (e.g.,
    /// `replace_child`).
    pub(crate) fn free_slot(&mut self, idx: NodeIdx) {
        assert!(
            self.nodes[idx].is_some(),
            "free_slot: double-free at idx {idx} (pool size {})",
            self.nodes.len()
        );
        if FullLRUSlot::data(self.get(idx)).in_list {
            // Snapshot value_len + lock_ref before the remove so the counter
            // debit hits the right side of the evictable/protected split.
            let node = self.get(idx);
            let value_len = node.key().len();
            let lock_ref = node.lock_ref();
            FullLRUSlot::remove(self, idx);
            let state = FullLRUSlot::pool_state_mut(self);
            if lock_ref == 0 {
                state.unlocked_size -= value_len;
            } else {
                state.locked_size -= value_len;
            }
        }
        self.nodes[idx] = None;
        self.evicted_indices.push(idx);
    }

    /// Allocate a node in the pool. Returns its index.
    pub fn alloc(&mut self, node: TreeNode<K>) -> NodeIdx {
        if let Some(idx) = self.evicted_indices.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            self.nodes.push(Some(node));
            self.nodes.len() - 1
        }
    }

    /// Insert a leaf node and attach to `parent_idx` as a child.
    pub fn insert_leaf(
        &mut self,
        parent_idx: NodeIdx,
        node: TreeNode<K>,
    ) -> Result<NodeIdx, ChildKeyError> {
        let child_key_owned = node.child_key(self.page_size)?.to_owned();
        let child_idx = self.alloc(node);
        self.get_mut(child_idx).parent = Some(parent_idx);
        self.get_mut(parent_idx)
            .insert_child(child_key_owned, child_idx);
        let value_len = self.get(child_idx).key().len();
        FullLRUSlot::bump_mru(self, child_idx);
        FullLRUSlot::pool_state_mut(self).unlocked_size += value_len;
        if FullLRUSlot::has_value(self.get(child_idx)) {
            FullLRUSlot::postprocess_set_value(self, parent_idx);
        }
        Ok(child_idx)
    }

    /// Combined "find child + match key", returning a `MatchChildResult`.
    /// `&self` so callers can re-acquire the pool mutably for `split_node` in
    /// the `PartialMatch` arm.
    #[inline]
    pub fn match_child(&self, parent_idx: NodeIdx, query_key: &[K::Atom]) -> MatchChildResult {
        let Ok(child_key) = K::make_child_key(query_key, self.page_size) else {
            return MatchChildResult::NotFound;
        };
        let Some(child_idx) = self.get(parent_idx).get_child(child_key) else {
            return MatchChildResult::NotFound;
        };
        let child = self.get(child_idx);
        let prefix_len = child.match_key(query_key, self.page_size);
        if prefix_len < child.key().len() {
            MatchChildResult::PartialMatch(NodeSplit {
                child_idx,
                split_len: prefix_len,
            })
        } else {
            MatchChildResult::FullMatch {
                child_idx,
                node_key_len: prefix_len,
            }
        }
    }

    /// Split a node at `split_len`, inserting a new intermediate parent, and
    /// return the new intermediate's index. Infallible when `split` came from
    /// `match_child`. The defensive alignment assert catches test-constructed
    /// `NodeSplit`s with unaligned `split_len`, which would otherwise silently
    /// corrupt the tree (downstream `make_child_key` checks length, not
    /// alignment); out-of-range lengths still panic downstream after partial
    /// mutation.
    pub fn split_node(
        &mut self,
        components: &[Box<dyn crate::components::Component<K>>],
        split: NodeSplit,
    ) -> NodeIdx {
        let node_idx = split.child_idx;
        let split_len = split.split_len;
        let ps = self.page_size.get();
        #[allow(clippy::panic, reason = "callers must not pass root to split_node")]
        let parent_idx = self
            .get(node_idx)
            .parent()
            .unwrap_or_else(|| panic!("split_node: cannot split root node (idx {node_idx})"));
        assert!(
            split_len.is_multiple_of(ps),
            "split_node: split_len ({split_len}) must be aligned to page_size ({ps})",
        );

        // Pure structural split: key split + re-parent + child-map updates.
        // All per-component value/lock_ref/LRU/identity-marker fixups live in
        // each component's `redistribute_on_node_split` hook, dispatched below.
        let page_size = self.page_size;
        let mut prefix_key = std::mem::take(&mut self.get_mut(node_idx).key);
        let suffix_key = prefix_key.split_off(split_len);

        // `alloc` does not touch the LRU; the redistribute dispatch below does.
        let new_node_idx = self.alloc(TreeNode::new_child(Vec::new(), parent_idx, None));

        self.get_mut(node_idx).parent = Some(new_node_idx);

        // Replace in parent's children, borrowing from local prefix_key (zero
        // alloc). Test wrappers passing split_len < page_size panic mid-mutation.
        #[allow(
            clippy::expect_used,
            reason = "match_child guarantees split_len >= page_size"
        )]
        let original_child_key = K::make_child_key(&prefix_key, page_size)
            .expect("split_node: split_len < page_size — prefix has fewer than page_size tokens");
        self.get_mut(parent_idx)
            .replace_child(original_child_key, new_node_idx);

        // Add original as child of new node. `to_owned()` is unavoidable since
        // `suffix_key` is reused below as the original node's new key. Test
        // wrappers passing out-of-range split_len panic mid-mutation.
        #[allow(
            clippy::expect_used,
            reason = "match_child guarantees split_len <= key_len - page_size"
        )]
        let suffix_child_key = K::make_child_key(&suffix_key, page_size)
            .expect("split_node: split_len out of range — suffix has fewer than page_size tokens")
            .to_owned();
        self.get_mut(new_node_idx)
            .insert_child(suffix_child_key, node_idx);

        self.get_mut(new_node_idx).key = prefix_key;
        self.get_mut(node_idx).key = suffix_key;

        // Per-component redistribute dispatch. Each component fixes up its own
        // value / lock_ref / LRU / identity-marker across the split boundary.
        // Passing `&[]` (test-only) yields a structural-only split.
        for comp in components {
            comp.redistribute_on_node_split(self, new_node_idx, node_idx, split_len);
        }

        // New intermediate's full-device-child count: 1 iff its only child (the
        // original node) kept a FULL device value.
        self.get_mut(new_node_idx).num_children_with_device_full =
            u32::from(FullLRUSlot::has_value(self.get(node_idx)));

        new_node_idx
    }

    /// Evict a leaf and update all bookkeeping.
    pub fn evict_leaf(
        &mut self,
        idx: NodeIdx,
        result: &mut EvictResult,
    ) -> Result<(), ChildKeyError> {
        // TODO(Jialin): pass the configured components so this can iterate
        // generically instead of hard-coding SWA + Mamba snapshots + unlinks.
        let (
            parent_idx,
            swa_was_in_list,
            swa_has_value,
            mamba_was_in_list,
            mamba_has_value,
            swa_value_len,
            mamba_value_len,
        ) = {
            let node = self.get(idx);
            #[allow(clippy::panic, reason = "callers must not pass root to evict_leaf")]
            let parent_idx = node
                .parent()
                .unwrap_or_else(|| panic!("evict_leaf: cannot evict root node (idx {idx})"));
            // Ensure evict_leaf only applied on unlocked leaf node.
            assert!(
                node.is_leaf(),
                "evict_leaf: node at idx {idx} has {} children, expected 0",
                node.num_children(),
            );
            assert!(
                !node.is_locked(),
                "evict_leaf: node at idx {idx} has at least one component \
                 with lock_ref > 0 ({:?}); evicting a locked node would \
                 corrupt size aggregates and orphan that component's value",
                node.components
                    .iter()
                    .map(|c| c.lock_ref)
                    .collect::<Vec<_>>(),
            );
            (
                parent_idx,
                SwaLRUSlot::data(node).in_list,
                SwaLRUSlot::has_value(node),
                MambaLRUSlot::data(node).in_list,
                MambaLRUSlot::has_value(node),
                SwaLRUSlot::value_len(node),
                MambaLRUSlot::value_len(node),
            )
        };
        assert_eq!(
            swa_was_in_list, swa_has_value,
            "SWA invariant violated at evict_leaf entry for node_idx {idx}: \
             in_list ({swa_was_in_list}) != has_value ({swa_has_value})",
        );
        assert_eq!(
            mamba_was_in_list, mamba_has_value,
            "Mamba invariant violated at evict_leaf entry for node_idx {idx}: \
             in_list ({mamba_was_in_list}) != has_value ({mamba_has_value})",
        );

        // SWA/Mamba value drain. Must run before FULL, as SWA shallow-clones the
        // FULL value.
        {
            let node = self.get_mut(idx);
            MambaLRUSlot::take_value(node, result);
            SwaLRUSlot::take_value(node, result);
        }

        // FULL value drain.
        if let Some(full_value) = evict_full_value(self, idx, result) {
            result.freed[FullLRUSlot::COMPONENT as usize].push(full_value);
        }

        // SWA/Mamba LRU unlink + bookkeeping update.
        if swa_was_in_list {
            SwaLRUSlot::remove(self, idx);
            SwaLRUSlot::pool_state_mut(self).unlocked_size -= swa_value_len;
        }
        if mamba_was_in_list {
            MambaLRUSlot::remove(self, idx);
            MambaLRUSlot::pool_state_mut(self).unlocked_size -= mamba_value_len;
        }

        // Tree mutation: detach from parent, free the pool slot.
        #[allow(clippy::expect_used, reason = "validated as Some above")]
        let node = self.nodes[idx].take().expect("validated above");
        let child_key = node.child_key(self.page_size)?;
        self.get_mut(parent_idx).remove_child(child_key);

        self.evicted_indices.push(idx);
        Ok(())
    }

    /// Get a reference to a node.
    ///
    /// Panics if trying to access an evicted node.
    pub fn get(&self, idx: NodeIdx) -> &TreeNode<K> {
        let len = self.nodes.len();
        #[allow(clippy::panic, reason = "callers must not access evicted slots")]
        self.nodes[idx]
            .as_ref()
            .unwrap_or_else(|| panic!("accessing evicted node at idx {idx} (pool size {len})"))
    }

    /// Get a mutable reference to a node.
    ///
    /// Panics if trying to access an evicted node.
    pub fn get_mut(&mut self, idx: NodeIdx) -> &mut TreeNode<K> {
        let len = self.nodes.len();
        #[allow(clippy::panic, reason = "callers must not access evicted slots")]
        self.nodes[idx]
            .as_mut()
            .unwrap_or_else(|| panic!("accessing evicted node at idx {idx} (pool size {len})"))
    }

    /// The validated page size for this pool.
    pub fn page_size(&self) -> PageSize {
        self.page_size
    }

    /// Whether SWA was configured. Gates `FullLRUSlot::evict`'s tombstone
    /// cascade — see the field doc.
    pub fn has_swa_component(&self) -> bool {
        self.has_swa_component
    }

    /// Whether Mamba was configured. Gates `FullLRUSlot::evict`'s tombstone
    /// cascade — see the field doc.
    pub fn has_mamba_component(&self) -> bool {
        self.has_mamba_component
    }

    /// Total active (non-evicted) nodes, excluding LRU sentinel slots.
    pub fn active_node_count(&self) -> usize {
        self.nodes.len() - self.evicted_indices.len() - self.sentinel_count
    }
}

/// Production tree node: children keyed by token page (`Vec<i64>`).
/// Handles `page_size >= 1` (page_size=1 uses one-element page keys).
pub type PageTreeNode = TreeNode<Vec<i64>>;

/// Production tree node pool: children keyed by token page (`Vec<i64>`).
pub type PageTreeNodePool = TreeNodePool<Vec<i64>>;
