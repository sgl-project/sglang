use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use tch::Tensor;

use crate::component_type::{ComponentType, NUM_COMPONENT_TYPES};
use crate::components::{
    ComponentPoolState, EvictResult, FullSlot, LRUData, MambaSlot, Slot, SwaSlot, evict_full_value,
};
use crate::error::RadixCacheInitError;

/// Validated page size for the radix cache.
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

/// Child key type constructible from an atom slice.
pub trait ChildKeyType: Hash + Eq + Sized + Borrow<Self::Borrowed> {
    type Atom: Copy + Eq + Hash;
    type Borrowed: Hash + Eq + ToOwned<Owned = Self> + ?Sized;

    /// Extract a borrowed child key from an atom slice. Caller guarantees page alignment.
    fn make_child_key(key: &[Self::Atom], page_size: PageSize) -> &Self::Borrowed;
}

/// Page-aligned slice helper shared by all `ChildKeyType::make_child_key` impls.
fn page_aligned_slice<T>(key: &[T], page_size: PageSize) -> &[T] {
    let ps = page_size.get();
    debug_assert!(key.len() >= ps, "child key shorter than page_size");
    &key[..ps]
}

impl ChildKeyType for Vec<i64> {
    type Atom = i64;
    type Borrowed = [i64];

    fn make_child_key(key: &[i64], page_size: PageSize) -> &[i64] {
        page_aligned_slice(key, page_size)
    }
}

/// Bigram-keyed instantiation (EAGLE).
impl ChildKeyType for Vec<(i64, i64)> {
    type Atom = (i64, i64);
    type Borrowed = [(i64, i64)];

    fn make_child_key(key: &[(i64, i64)], page_size: PageSize) -> &[(i64, i64)] {
        page_aligned_slice(key, page_size)
    }
}

/// Index of a node inside a `TreeNodePool`.
///
/// TODO(Jialin): extend with a generation tag to avoid ABA on the recycling
/// freelist and improve debuggability.
pub type NodeIdx = usize;

/// Per-component per-node state. One entry per `ComponentType`.
#[derive(Default)]
pub struct ComponentNodeState {
    /// KV indices for this component on this node (None = no live value).
    pub value: Option<Tensor>,
    /// In-flight request count; while > 0 the slot is locked.
    pub lock_ref: u32,
    /// Intrusive LRU state for this component's slot.
    pub lru_data: LRUData,
}

/// Radix tree node generic over child key type. Owned by a `TreeNodePool`;
/// parent and children are `NodeIdx` indices, not references.
pub struct TreeNode<K: ChildKeyType> {
    /// Atoms stored at this node (the edge label from parent to this node).
    key: Vec<K::Atom>,
    /// Index of parent node in the pool. None for the root.
    parent: Option<NodeIdx>,
    /// Children keyed by child key, values are indices into the pool.
    children: HashMap<K, NodeIdx>,
    /// Device-tier per-component node-level state.
    ///
    /// TODO(Jialin): explore dynamic vs this static allocation, trading off
    /// perf against memory.
    pub(crate) components: [ComponentNodeState; NUM_COMPONENT_TYPES],
    /// SWA's lock-walk boundary marker.
    pub(crate) swa_uuid_for_lock: Option<u64>,
    /// Count of children holding a FULL device value.
    pub(crate) num_children_with_device_full: u32,
}

impl<K: ChildKeyType> TreeNode<K> {
    /// Construct a root node. FULL's `lock_ref` starts at 1 so the root is
    /// permanently protected.
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

    /// Construct a non-root node with a known parent. `value` goes into FULL's slot.
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

    /// Construct a sentinel node, never reachable through tree traversal.
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

    /// True iff this node is a namespace root (no parent).
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// FULL's KV cache slot indices for this segment. None for root / evicted.
    pub fn value(&self) -> Option<&Tensor> {
        self.components[ComponentType::Full as usize].value.as_ref()
    }

    /// Number of children.
    pub fn num_children(&self) -> usize {
        self.children.len()
    }

    /// True iff this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// FULL's lock-ref count: 0 means unlocked, > 0 means in use.
    pub fn lock_ref(&self) -> u32 {
        self.components[ComponentType::Full as usize].lock_ref
    }

    /// True iff any component's `lock_ref > 0`.
    pub fn is_locked(&self) -> bool {
        self.components.iter().any(|c| c.lock_ref > 0)
    }

    /// SWA's lock-walk boundary uuid for this node.
    pub fn swa_uuid_for_lock(&self) -> Option<u64> {
        self.swa_uuid_for_lock
    }

    /// Set this node's SWA lock-walk boundary uuid (`None` clears).
    pub fn set_swa_uuid_for_lock(&mut self, uuid: Option<u64>) {
        self.swa_uuid_for_lock = uuid;
    }

    /// Extract the child key from this node's key (the first page_size atoms).
    pub fn child_key(&self, page_size: PageSize) -> &K::Borrowed {
        K::make_child_key(&self.key, page_size)
    }

    /// Lookup a child index by borrowed key. Zero allocation.
    pub fn get_child(&self, child_key: &K::Borrowed) -> Option<NodeIdx> {
        self.children.get(child_key).copied()
    }

    /// Insert a child index.
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

    /// Replace an existing child index, returning the old one.
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

    /// Remove a child by key, returning its NodeIdx.
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
        let mut matched_pages = 0;
        while matched_pages < max_pages {
            let start = matched_pages * ps;
            let end = start + ps;
            if self.key[start..end] != key[start..end] {
                break;
            }
            matched_pages += 1;
        }
        matched_pages * ps
    }
}

/// Outcome of `TreeNodePool::match_child`.
pub enum MatchChildResult {
    /// No child entry for `query_key`'s first page.
    NotFound,
    /// All of the matched child's stored key matched.
    FullMatch {
        child_idx: NodeIdx,
        node_key_len: usize,
    },
    /// Only a prefix of the child's key matched; caller must `split_node` first.
    PartialMatch(NodeSplit),
}

/// A validated node split. Constructed via `match_child` returning `PartialMatch`.
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
pub struct TreeNodePool<K: ChildKeyType> {
    nodes: Vec<Option<TreeNode<K>>>,
    evicted_indices: Vec<NodeIdx>,
    page_size: PageSize,
    /// Device-tier per-component pool-level state.
    pub(crate) components: [ComponentPoolState; NUM_COMPONENT_TYPES],
    /// Sentinel slots allocated for this pool (one per LRU).
    sentinel_count: usize,
    /// Monotonic counter for SWA's lock-walk boundary uuids. Starts at 1.
    next_swa_uuid_for_lock: u64,
    /// Whether SWA is configured. Gates the FULL-evict tombstone cascade.
    has_swa_component: bool,
    /// Whether Mamba is configured for this cache.
    has_mamba_component: bool,
}

impl<K: ChildKeyType> TreeNodePool<K> {
    /// Construct from an already-validated `PageSize`.
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
            next_swa_uuid_for_lock: 1,
            has_swa_component,
            has_mamba_component,
        };
        FullSlot::init::<K>(&mut pool);
        SwaSlot::init::<K>(&mut pool);
        MambaSlot::init::<K>(&mut pool);
        pool
    }

    /// Allocate a sentinel TreeNode and bump `sentinel_count`.
    pub(crate) fn alloc_sentinel(&mut self) -> NodeIdx {
        let idx = self.alloc(TreeNode::new_sentinel());
        self.sentinel_count += 1;
        idx
    }

    /// Get-or-mint the SWA lock-walk boundary uuid for `node_idx`.
    pub fn lazy_acquire_swa_uuid_for_lock(&mut self, node_idx: NodeIdx) -> u64 {
        if let Some(existing) = self.get(node_idx).swa_uuid_for_lock() {
            return existing;
        }
        let new_uuid = self.acquire_next_swa_uuid_for_lock();
        self.get_mut(node_idx).set_swa_uuid_for_lock(Some(new_uuid));
        new_uuid
    }

    /// Mint the next SWA lock-walk uuid, always advancing the counter.
    pub(crate) fn acquire_next_swa_uuid_for_lock(&mut self) -> u64 {
        let id = self.next_swa_uuid_for_lock;
        self.next_swa_uuid_for_lock += 1;
        id
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
    pub fn insert_leaf(&mut self, parent_idx: NodeIdx, node: TreeNode<K>) -> NodeIdx {
        let child_key_owned = node.child_key(self.page_size).to_owned();
        let child_idx = self.alloc(node);
        self.get_mut(child_idx).parent = Some(parent_idx);
        self.get_mut(parent_idx)
            .insert_child(child_key_owned, child_idx);
        let value_len = self.get(child_idx).key().len();
        FullSlot::bump_mru(self, child_idx);
        FullSlot::pool_state_mut(self).unlocked_size += value_len;
        if FullSlot::has_value(self.get(child_idx)) {
            FullSlot::postprocess_set_value(self, parent_idx);
        }
        child_idx
    }

    /// Combined "find child + match key", returning a `MatchChildResult`.
    #[inline]
    pub fn match_child(&self, parent_idx: NodeIdx, query_key: &[K::Atom]) -> MatchChildResult {
        let child_key = K::make_child_key(query_key, self.page_size);
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
    /// return the new intermediate's index.
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
        let page_size = self.page_size;
        let mut prefix_key = std::mem::take(&mut self.get_mut(node_idx).key);
        let suffix_key = prefix_key.split_off(split_len);

        let new_node_idx = self.alloc(TreeNode::new_child(Vec::new(), parent_idx, None));

        self.get_mut(node_idx).parent = Some(new_node_idx);

        // Replace in parent's children, borrowing from local prefix_key.
        let original_child_key = K::make_child_key(&prefix_key, page_size);
        self.get_mut(parent_idx)
            .replace_child(original_child_key, new_node_idx);

        // Add original as child of new node.
        let suffix_child_key = K::make_child_key(&suffix_key, page_size).to_owned();
        self.get_mut(new_node_idx)
            .insert_child(suffix_child_key, node_idx);

        self.get_mut(new_node_idx).key = prefix_key;
        self.get_mut(node_idx).key = suffix_key;

        // Per-component redistribute dispatch across the split boundary.
        for comp in components {
            comp.redistribute_on_node_split(self, new_node_idx, node_idx, split_len);
        }

        // New intermediate's full-device-child count: 1 iff its only child kept
        // a FULL device value.
        self.get_mut(new_node_idx).num_children_with_device_full =
            u32::from(FullSlot::has_value(self.get(node_idx)));

        new_node_idx
    }

    /// Evict a leaf and update all bookkeeping.
    pub fn evict_leaf(&mut self, idx: NodeIdx, result: &mut EvictResult) {
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
                SwaSlot::data(node).in_list,
                SwaSlot::has_value(node),
                MambaSlot::data(node).in_list,
                MambaSlot::has_value(node),
                SwaSlot::value_len(node),
                MambaSlot::value_len(node),
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
            MambaSlot::take_value(node, result);
            SwaSlot::take_value(node, result);
        }

        // FULL value drain.
        if let Some(full_value) = evict_full_value(self, idx, result) {
            result.freed[FullSlot::COMPONENT as usize].push(full_value);
        }

        // SWA/Mamba LRU unlink + bookkeeping update.
        if swa_was_in_list {
            SwaSlot::lru_remove(self, idx);
            SwaSlot::pool_state_mut(self).unlocked_size -= swa_value_len;
        }
        if mamba_was_in_list {
            MambaSlot::lru_remove(self, idx);
            MambaSlot::pool_state_mut(self).unlocked_size -= mamba_value_len;
        }

        // Tree mutation: detach from parent, free the pool slot.
        #[allow(clippy::expect_used, reason = "validated as Some above")]
        let node = self.nodes[idx].take().expect("validated above");
        let child_key = node.child_key(self.page_size);
        self.get_mut(parent_idx).remove_child(child_key);

        self.evicted_indices.push(idx);
    }

    /// Get a reference to a node.
    pub fn get(&self, idx: NodeIdx) -> &TreeNode<K> {
        let len = self.nodes.len();
        #[allow(clippy::panic, reason = "callers must not access evicted slots")]
        self.nodes[idx]
            .as_ref()
            .unwrap_or_else(|| panic!("accessing evicted node at idx {idx} (pool size {len})"))
    }

    /// Get a mutable reference to a node.
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

    /// Whether SWA was configured.
    pub fn has_swa_component(&self) -> bool {
        self.has_swa_component
    }

    /// Whether Mamba was configured.
    pub fn has_mamba_component(&self) -> bool {
        self.has_mamba_component
    }

    /// Total active (non-evicted) nodes, excluding LRU sentinel slots.
    pub fn active_node_count(&self) -> usize {
        self.nodes.len() - self.evicted_indices.len() - self.sentinel_count
    }
}

/// Production tree node: children keyed by token page (`Vec<i64>`).
pub type PageTreeNode = TreeNode<Vec<i64>>;

/// Production tree node pool: children keyed by token page (`Vec<i64>`).
pub type PageTreeNodePool = TreeNodePool<Vec<i64>>;
