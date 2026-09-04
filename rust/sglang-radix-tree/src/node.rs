//! A radix-tree node, owned by the `NodeArena` and referenced by `NodeIdx_`.

use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::collections::hash_map::{DefaultHasher, RandomState};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use hashbrown::hash_map::Entry;
use hashbrown::{Equivalent, HashMap as HashBrownMap};
use sha2::{Digest, Sha256};
use tch::Tensor;

use crate::components::{ComponentType, FULL, NUM_COMPONENT_TYPES};

/// The two independent dimensions that partition a radix tree.
#[derive(Debug)]
struct KeyNamespaceData {
    extra_key: Option<Arc<str>>,
    cache_salt: Option<Arc<str>>,
    hash: u64,
}

/// Compact, shared namespace stored on nodes and child edges.
///
/// The default namespace is represented without an allocation. Non-default
/// namespaces share one immutable allocation down a radix path.
#[derive(Clone, Debug, Default)]
pub struct KeyNamespace(Option<Arc<KeyNamespaceData>>);

/// Borrowed namespace used by match/insert lookups without allocating.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyNamespaceRef<'a> {
    pub extra_key: Option<&'a str>,
    pub cache_salt: Option<&'a str>,
    hash: u64,
}

fn key_namespace_hash(extra_key: Option<&str>, cache_salt: Option<&str>) -> u64 {
    if extra_key.is_none() && cache_salt.is_none() {
        return 0;
    }
    let mut hasher = DefaultHasher::new();
    extra_key.hash(&mut hasher);
    cache_salt.hash(&mut hasher);
    hasher.finish()
}

impl<'a> KeyNamespaceRef<'a> {
    pub fn new(extra_key: Option<&'a str>, cache_salt: Option<&'a str>) -> Self {
        let cache_salt = cache_salt.filter(|salt| !salt.is_empty());
        Self {
            extra_key,
            cache_salt,
            hash: key_namespace_hash(extra_key, cache_salt),
        }
    }

    pub fn to_owned(self) -> KeyNamespace {
        if self.extra_key.is_none() && self.cache_salt.is_none() {
            return KeyNamespace::default();
        }
        KeyNamespace(Some(Arc::new(KeyNamespaceData {
            extra_key: self.extra_key.map(Into::into),
            cache_salt: self.cache_salt.map(Into::into),
            hash: self.hash,
        })))
    }
}

impl PartialEq for KeyNamespaceRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.extra_key == other.extra_key && self.cache_salt == other.cache_salt
    }
}

impl Eq for KeyNamespaceRef<'_> {}

impl Hash for KeyNamespaceRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl KeyNamespace {
    pub fn new(extra_key: Option<&str>, cache_salt: Option<&str>) -> Self {
        KeyNamespaceRef::new(extra_key, cache_salt).to_owned()
    }

    pub fn as_ref(&self) -> KeyNamespaceRef<'_> {
        match self.0.as_deref() {
            Some(namespace) => KeyNamespaceRef {
                extra_key: namespace.extra_key.as_deref(),
                cache_salt: namespace.cache_salt.as_deref(),
                hash: namespace.hash,
            },
            None => KeyNamespaceRef::default(),
        }
    }

    pub fn extra_key(&self) -> Option<&str> {
        self.as_ref().extra_key
    }

    pub fn cache_salt(&self) -> Option<&str> {
        self.as_ref().cache_salt
    }

    pub fn cache_salt_arc(&self) -> Option<Arc<str>> {
        self.0
            .as_ref()
            .and_then(|namespace| namespace.cache_salt.clone())
    }
}

impl PartialEq for KeyNamespace {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for KeyNamespace {}

impl Hash for KeyNamespace {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}

type ChildMap<K> = HashBrownMap<(KeyNamespace, K), NodeIdx_, RandomState>;

/// Borrowed view of a namespaced child edge used for allocation-free lookup.
struct ChildEdgeRef<'a, K: ChildKeyType> {
    namespace: KeyNamespaceRef<'a>,
    page: &'a [K::Atom],
}

impl<K: ChildKeyType> Hash for ChildEdgeRef<'_, K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.page.hash(state);
    }
}

impl<K: ChildKeyType> Equivalent<(KeyNamespace, K)> for ChildEdgeRef<'_, K> {
    fn equivalent(&self, edge: &(KeyNamespace, K)) -> bool {
        self.namespace == edge.0.as_ref() && self.page == edge.1.as_ref()
    }
}

/// A radix-tree node, generic over the child-key type `K` (single-token or bigram).
pub struct Node<K: ChildKeyType> {
    /// Parent handle; `None` for the root or a not-yet-attached child.
    pub parent: Option<NodeIdx_>,
    /// Own arena slot; stamped by the arena on allocation (hand-built nodes
    /// default it to the id value).
    pub(crate) idx: NodeIdx_,
    /// The namespace this node belongs to.
    pub namespace: KeyNamespace,
    /// Child edges keyed by (namespace, the child's page key); the namespace
    /// component mirrors the child's namespace at every level.
    pub children: ChildMap<K>,
    /// The page key labelling the edge from the parent (also this node's key in the
    /// parent's `children`); empty for the root.
    pub key: K,
    /// Per-(component × tier) value state, indexed by `ValueSlotIdx` (device
    /// slots first, host after); device states also sit at plain component
    /// indices.
    pub values: [ValueState; NUM_VALUE_SLOTS],
    /// SWA lock-window uuid; stamped where a device lock walk fills the window.
    pub swa_uuid: Option<i64>,
    /// SWA lock-window uuid for host locks; stamped where a host lock walk fills the window.
    pub swa_host_uuid: Option<i64>,
    /// Per-page hash chain; None when the node was never hashed.
    /// TODO: Store raw digests and hex-encode only at the Python or storage boundary.
    pub hash_value: Option<Vec<String>>,
    /// The in-flight write-through backup's ack id.
    pub write_through_pending_id: Option<usize>,
    /// Load-back anchor currently reading this node's host slots.
    pub load_back_pending_id: Option<NodeId>,
    /// Monotonic access tick for LRU ordering (exact; not wall-clock).
    pub last_access_counter: i64,
    /// Tick stamped at construction.
    pub creation_counter: i64,
    /// Match hits accumulated for write-through and LFU decisions.
    pub hit_count: i64,
    /// Eviction priority; the root uses `i64::MIN` and is never a leaf.
    pub priority: i64,
    /// This node's external handle; minted once, never recycled.
    pub id: NodeId,
}

impl<K: ChildKeyType> Node<K> {
    /// Whether this is the root (no parent).
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// The parent's id; panics on a root.
    #[track_caller]
    pub fn parent(&self) -> NodeIdx_ {
        self.parent
            .unwrap_or_else(|| panic!("node {} is a root and has no parent", self.id))
    }

    /// The parent's id, or `None` on a root.
    pub fn try_parent(&self) -> Option<NodeIdx_> {
        self.parent
    }

    /// Whether this node has no children (a tree leaf).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Tree-level: Full KV not on device (non-root with value=None).
    pub fn evicted(&self) -> bool {
        self.parent.is_some() && !self.has_device_value(FULL)
    }

    /// Tree-level: Full KV present on host.
    pub fn backuped(&self) -> bool {
        self.has_host_value(FULL)
    }

    /// The last page's hash value, or None when the node was never hashed.
    pub fn get_last_hash_value(&self) -> Option<&str> {
        self.hash_value
            .as_ref()
            .and_then(|h| h.last())
            .map(String::as_str)
    }

    /// The component's device value; panics if unset.
    pub fn device_value(&self, component_type: ComponentType) -> &Tensor {
        self.value_(ValueSlotIdx::device(component_type))
    }

    /// The component's device value, or None when unset.
    pub fn try_device_value(&self, component_type: ComponentType) -> Option<&Tensor> {
        self.try_value_(ValueSlotIdx::device(component_type))
    }

    /// Whether the component's device value is present.
    pub fn has_device_value(&self, component_type: ComponentType) -> bool {
        self.has_value_(ValueSlotIdx::device(component_type))
    }

    /// The component's device value length, or 0 when value-less.
    pub fn device_value_len(&self, component_type: ComponentType) -> usize {
        self.value_len_(ValueSlotIdx::device(component_type))
    }

    /// Set the component's device value; panics if already set.
    pub fn set_device_value(&mut self, component_type: ComponentType, value: Tensor) {
        self.set_value_(ValueSlotIdx::device(component_type), value);
    }

    /// Take the component's device value; panics if unset.
    pub fn take_device_value(&mut self, component_type: ComponentType) -> Tensor {
        self.take_value_(ValueSlotIdx::device(component_type))
    }

    /// The component's device lock refcount.
    pub fn device_lock_ref(&self, component_type: ComponentType) -> u32 {
        self.lock_ref_(ValueSlotIdx::device(component_type))
    }

    /// Bump the component's device lock refcount by one.
    pub fn inc_device_lock_ref(&mut self, component_type: ComponentType) {
        self.inc_lock_ref_(ValueSlotIdx::device(component_type));
    }

    /// Drop the component's device lock refcount by one; panics when unlocked.
    pub fn dec_device_lock_ref(&mut self, component_type: ComponentType) {
        self.dec_lock_ref_(ValueSlotIdx::device(component_type));
    }

    /// Copy the component's device lock refcount from `src_node`.
    pub fn copy_device_lock_ref(&mut self, component_type: ComponentType, src_node: &Node<K>) {
        let slot = ValueSlotIdx::device(component_type);
        self.set_lock_ref_(slot, src_node.lock_ref_(slot));
    }

    /// Split the component's device value between a new parent and the child.
    pub fn redistribute_child_device_value(
        parent_node: &mut Node<K>,
        child_node: &mut Node<K>,
        component_type: ComponentType,
        split_len: i64,
    ) {
        Node::redistribute_child_value_(
            parent_node,
            child_node,
            ValueSlotIdx::device(component_type),
            split_len,
        );
    }

    /// The component's host value; panics if unset.
    pub fn host_value(&self, component_type: ComponentType) -> &Tensor {
        self.value_(ValueSlotIdx::host(component_type))
    }

    /// The component's host value, or None when unset.
    pub fn try_host_value(&self, component_type: ComponentType) -> Option<&Tensor> {
        self.try_value_(ValueSlotIdx::host(component_type))
    }

    /// Whether the component's host value is present.
    pub fn has_host_value(&self, component_type: ComponentType) -> bool {
        self.has_value_(ValueSlotIdx::host(component_type))
    }

    /// The component's host value length, or 0 when value-less.
    pub fn host_value_len(&self, component_type: ComponentType) -> usize {
        self.value_len_(ValueSlotIdx::host(component_type))
    }

    /// Set the component's host value; panics if already set.
    pub fn set_host_value(&mut self, component_type: ComponentType, value: Tensor) {
        self.set_value_(ValueSlotIdx::host(component_type), value);
    }

    /// Take the component's host value; panics if unset.
    pub fn take_host_value(&mut self, component_type: ComponentType) -> Tensor {
        self.take_value_(ValueSlotIdx::host(component_type))
    }

    /// The component's host lock refcount (the host lock).
    pub fn host_lock_ref(&self, component_type: ComponentType) -> u32 {
        self.lock_ref_(ValueSlotIdx::host(component_type))
    }

    /// Bump the component's host lock refcount by one.
    pub fn inc_host_lock_ref(&mut self, component_type: ComponentType) {
        self.inc_lock_ref_(ValueSlotIdx::host(component_type));
    }

    /// Drop the component's host lock refcount by one; panics when unlocked.
    pub fn dec_host_lock_ref(&mut self, component_type: ComponentType) {
        self.dec_lock_ref_(ValueSlotIdx::host(component_type));
    }

    /// Split the component's host value between a new parent and the child.
    pub fn redistribute_child_host_value(
        parent_node: &mut Node<K>,
        child_node: &mut Node<K>,
        component_type: ComponentType,
        split_len: i64,
    ) {
        Node::redistribute_child_value_(
            parent_node,
            child_node,
            ValueSlotIdx::host(component_type),
            split_len,
        );
    }

    /// Whether any component holds a device lock on this node.
    pub fn is_device_locked(&self) -> bool {
        self.values[..NUM_COMPONENT_TYPES]
            .iter()
            .any(|state| state.lock_ref > 0)
    }

    /// Whether any component holds a host lock on this node.
    pub fn is_host_locked(&self) -> bool {
        self.values[NUM_COMPONENT_TYPES..]
            .iter()
            .any(|state| state.lock_ref > 0)
    }

    /// Whether an in-flight load-back currently pins this node.
    pub fn is_load_back_pending(&self) -> bool {
        self.load_back_pending_id.is_some()
    }

    // ==== Crate-internal tree wiring ====

    /// A fresh root: no parent, no value, lowest eviction priority.
    pub(crate) fn new_root(id: NodeId) -> Self {
        Node {
            parent: None,
            namespace: KeyNamespace::default(),
            children: ChildMap::with_hasher(RandomState::new()),
            key: K::default(),
            values: Default::default(),
            swa_uuid: None,
            swa_host_uuid: None,
            hash_value: Some(Vec::new()),
            write_through_pending_id: None,
            load_back_pending_id: None,
            last_access_counter: 0,
            creation_counter: 0,
            hit_count: 0,
            priority: i64::MIN,
            id,
            idx: NodeIdx_(id),
        }
    }

    /// A detached child reached by edge `key`; `attach_child` sets the parent link.
    pub(crate) fn new_child(id: NodeId, key: K, priority: i64) -> Self {
        Node {
            parent: None,
            namespace: KeyNamespace::default(),
            children: ChildMap::with_hasher(RandomState::new()),
            key,
            values: Default::default(),
            swa_uuid: None,
            swa_host_uuid: None,
            hash_value: None,
            write_through_pending_id: None,
            load_back_pending_id: None,
            last_access_counter: 0,
            creation_counter: 0,
            hit_count: 0,
            priority,
            id,
            idx: NodeIdx_(id),
        }
    }

    /// The namespaced edge key for this node's own edge from its parent.
    pub(crate) fn edge_key(&self, page_size: usize) -> (KeyNamespace, K) {
        (self.namespace.clone(), self.key.child_key(page_size))
    }

    /// Link `child` under this node, keyed by its namespaced page key; errors on
    /// a duplicate key. Panics if `child` is already attached (an internal invariant).
    /// The caller sets `child.namespace` first; the edge key mirrors it.
    pub(crate) fn attach_child(
        &mut self,
        child: &mut Node<K>,
        page_size: usize,
    ) -> Result<(), TreeCoreRuntimeError> {
        // Only fresh, detached nodes are ever attached.
        assert!(
            child.parent.is_none(),
            "attach_child: node {} is already attached to parent {:?}",
            child.id,
            child.parent
        );
        let parent_idx = self.idx;
        match self.children.entry(child.edge_key(page_size)) {
            Entry::Occupied(_) => Err(TreeCoreRuntimeError::DuplicateChildKey {
                parent: self.id,
                key: format!("{:?}", child.key),
            }),
            Entry::Vacant(slot) => {
                slot.insert(child.idx);
                child.parent = Some(parent_idx);
                Ok(())
            }
        }
    }

    /// Unlink this node from `parent`: drop it from `parent.children` and clear its own
    /// parent link; panics on a broken parent<->child link (an internal invariant).
    pub(crate) fn detach_from_parent(&mut self, parent: &mut Node<K>, page_size: usize) {
        // A live child is always registered under its namespaced page key.
        match parent.children.remove(&self.edge_key(page_size)) {
            Some(idx) if idx == self.idx => self.parent = None,
            found => panic!(
                "detach_from_parent: parent {} has no child {} under key {:?} (found {:?})",
                parent.id, self.id, self.key, found
            ),
        }
    }

    // ==== Internal slot-keyed lookups ====

    /// The slot's value state (internal slot-keyed lookup).
    pub fn state_(&self, slot: ValueSlotIdx) -> &ValueState {
        &self.values[slot.idx()]
    }

    /// The slot's mutable value state (internal slot-keyed lookup).
    pub fn state_mut_(&mut self, slot: ValueSlotIdx) -> &mut ValueState {
        &mut self.values[slot.idx()]
    }

    /// The slot's value, or None when unset (internal slot-keyed lookup).
    pub fn try_value_(&self, slot: ValueSlotIdx) -> Option<&Tensor> {
        self.state_(slot).value.as_ref()
    }

    /// The slot's value; panics if no value is set (internal slot-keyed lookup).
    pub fn value_(&self, slot: ValueSlotIdx) -> &Tensor {
        self.try_value_(slot).unwrap_or_else(|| {
            panic!(
                "value: {:?}/{} slot has no value on node {}",
                slot.component_type(),
                slot.tier(),
                self.id
            )
        })
    }

    /// Whether the slot's value is present (internal slot-keyed lookup).
    pub fn has_value_(&self, slot: ValueSlotIdx) -> bool {
        self.state_(slot).value.is_some()
    }

    /// The slot's value length, or 0 when value-less (internal slot-keyed lookup).
    pub fn value_len_(&self, slot: ValueSlotIdx) -> usize {
        self.state_(slot)
            .value
            .as_ref()
            .map_or(0, |v| v.size()[0] as usize)
    }

    /// Set the slot's value; panics if a value is already set (internal slot-keyed lookup).
    pub fn set_value_(&mut self, slot: ValueSlotIdx, value: Tensor) {
        if slot.component_type().single_value_per_node() {
            assert_eq!(
                value.size()[0],
                1,
                "set_value: {:?}/{} expects a single state slot on node {}",
                slot.component_type(),
                slot.tier(),
                self.id
            );
        } else {
            assert_eq!(
                value.size()[0] as usize,
                self.key.atom_len(),
                "set_value: {:?}/{} value length differs from the key on node {}",
                slot.component_type(),
                slot.tier(),
                self.id
            );
        }
        let node_id = self.id;
        let state = self.state_mut_(slot);
        assert!(
            state.value.is_none(),
            "set_value: {:?}/{} slot already set on node {node_id}",
            slot.component_type(),
            slot.tier()
        );
        state.value = Some(value);
    }

    /// Take the slot's value, leaving it value-less; panics if no value is set
    /// (internal slot-keyed lookup).
    pub fn take_value_(&mut self, slot: ValueSlotIdx) -> Tensor {
        let node_id = self.id;
        self.state_mut_(slot).value.take().unwrap_or_else(|| {
            panic!(
                "take_value: {:?}/{} slot has no value on node {node_id}",
                slot.component_type(),
                slot.tier()
            )
        })
    }

    /// Split the slot's value on `child_node` at `split_len`: the deep-copied head
    /// rows land on `parent_node`, the tail rows replace the value on `child_node`
    /// (internal slot-keyed lookup).
    pub fn redistribute_child_value_(
        parent_node: &mut Node<K>,
        child_node: &mut Node<K>,
        slot: ValueSlotIdx,
        split_len: i64,
    ) {
        let child_node_id = child_node.id;
        let value = child_node.take_value_(slot);
        let len = value.size()[0];
        // A boundary split would leave one side with a present-but-empty value.
        assert!(
            0 < split_len && split_len < len,
            "redistribute_child_value: split_len {split_len} out of range (0, {len}) on node {child_node_id}"
        );
        let head = value.narrow(0, 0, split_len).copy();
        let tail = value.narrow(0, split_len, len - split_len).copy();
        child_node.set_value_(slot, tail);
        parent_node.set_value_(slot, head);
    }

    /// The slot's lock refcount (internal slot-keyed lookup).
    pub fn lock_ref_(&self, slot: ValueSlotIdx) -> u32 {
        self.state_(slot).lock_ref
    }

    /// Set the slot's lock refcount (internal slot-keyed lookup).
    pub fn set_lock_ref_(&mut self, slot: ValueSlotIdx, lock_ref: u32) {
        self.state_mut_(slot).lock_ref = lock_ref;
    }

    /// Bump the slot's lock refcount by one (internal slot-keyed lookup).
    pub fn inc_lock_ref_(&mut self, slot: ValueSlotIdx) {
        self.state_mut_(slot).lock_ref += 1;
    }

    /// Drop the slot's lock refcount by one; panics on an unlocked node
    /// (internal slot-keyed lookup).
    pub fn dec_lock_ref_(&mut self, slot: ValueSlotIdx) {
        let node_id = self.id;
        let state = self.state_mut_(slot);
        state.lock_ref = state.lock_ref.checked_sub(1).unwrap_or_else(|| {
            panic!(
                "dec_lock_ref: {:?}/{} lock_ref underflow on node {node_id}",
                slot.component_type(),
                slot.tier()
            )
        });
    }
}

// Node handles and per-slot value state.

/// External node handle — the only node identity that crosses the FFI.
/// Minted monotonically and never recycled, so a freed node's id can never
/// alias a later allocation (the arena's id map is the ABA guard).
pub type NodeId = usize;

/// Internal arena slot index; recycled by the freelist and never crosses the FFI.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct NodeIdx_(pub(crate) usize);

impl std::fmt::Display for NodeIdx_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Number of (component × tier) value slots on a node: device tier first, host after.
pub const NUM_VALUE_SLOTS: usize = 2 * NUM_COMPONENT_TYPES;

/// Flat index of one (component × tier) value slot in a node's `values` array.
/// All slot arithmetic lives here; nothing else computes raw offsets.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ValueSlotIdx(usize);

impl ValueSlotIdx {
    /// The component's device-tier slot.
    pub const fn device(component_type: ComponentType) -> Self {
        Self(component_type.idx())
    }

    /// The component's host-tier slot.
    pub const fn host(component_type: ComponentType) -> Self {
        Self(NUM_COMPONENT_TYPES + component_type.idx())
    }

    /// Index into a node's `values` array.
    pub const fn idx(self) -> usize {
        self.0
    }

    /// The slot at a flat index; panics out of range.
    pub fn from_idx(idx: usize) -> Self {
        assert!(
            idx < NUM_VALUE_SLOTS,
            "from_idx: {idx} is not a value-slot index"
        );
        Self(idx)
    }

    /// Whether this is a host-tier slot.
    pub const fn is_host(self) -> bool {
        self.0 >= NUM_COMPONENT_TYPES
    }

    /// The component this slot belongs to.
    pub fn component_type(self) -> ComponentType {
        ComponentType::from_idx(self.0 % NUM_COMPONENT_TYPES)
    }

    /// The slot's tier name, for diagnostics.
    pub fn tier(self) -> &'static str {
        if self.is_host() { "host" } else { "device" }
    }
}

/// Per-(component × tier) node state: the KV-index `value` (held opaquely)
/// and the in-flight `lock_ref`.
#[derive(Default)]
pub struct ValueState {
    /// KV pool indices; `None` = value-less (root) or tombstone (evicted / out-of-window).
    pub value: Option<Tensor>,
    /// In-flight request refcount; a host slot's own `lock_ref` IS the host lock.
    pub lock_ref: u32,
}

// Tree-core runtime errors.

/// Errors surfaced from the tree-core runtime API when a caller violates a documented
/// contract (freeing an unallocated node, allocating under a freed parent).
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum TreeCoreRuntimeError {
    /// A public NodeId no longer names a live arena node.
    #[error("node {node_id} is not allocated")]
    NodeNotAllocated { node_id: NodeId },
    /// `begin_insert`/`insert` called while a resumable insert is suspended.
    #[error("concurrent insert walks")]
    ConcurrentInsertWalk,
    /// `resume_insert` called without a suspended insert.
    #[error("no in-flight insert")]
    NoInFlightInsert,
    /// A `NodeIdx_` beyond the arena's bounds — never allocated. `size` is the
    /// arena's current slot count, so valid ids are `0..size`.
    #[error("node access out of bounds: id {id} not in [0, {size})")]
    NodeAccessOutOfBound { id: NodeIdx_, size: usize },
    /// `free` called on an in-range slot that is already free — a double free.
    #[error("double free: node {id} is already free")]
    NodeDoubleFree { id: NodeIdx_ },
    /// `free` called on a root (no parent) — roots are protected.
    #[error("cannot free node {id}: it is a root (protected)")]
    RootNotFreeable { id: NodeIdx_ },
    /// `free` called on a non-leaf node — only leaves are freeable.
    #[error("cannot free non-leaf node {id}: it has {num_children} children")]
    FreeNonLeafNode { id: NodeIdx_, num_children: usize },
    /// `alloc_child` under an in-range parent slot that holds no live node.
    #[error("alloc_child: parent {id} is not allocated")]
    ParentNotAllocated { id: NodeIdx_ },
    /// `alloc_child` under a parent that already has a child at the same key.
    #[error("cannot add a child under parent {parent}: key {key} already has a child")]
    DuplicateChildKey { parent: NodeId, key: String },
    /// `demote` requires a device-resident Full value with a completed host backup.
    #[error(
        "cannot demote node {node_id}: expected device-resident Full value with host backup (evicted={evicted}, backuped={backuped})"
    )]
    InvalidDemoteState {
        node_id: NodeId,
        evicted: bool,
        backuped: bool,
    },
    /// SWA load-back cannot cross a node that has no value on either tier.
    #[error("SWA load-back traversal reached node {node_id} without a device or host value")]
    SwaLoadBackMissingValue { node_id: NodeId },
    /// A host insert below a non-root anchor must remain in that anchor's namespace.
    #[error("insert_host namespace does not match non-root anchor {node_id}")]
    InsertHostNamespaceMismatch { node_id: NodeId },
}

// Unigram and bigram child keys.

/// An owned child key (one radix page); `Atom` is the per-position token (`i64` single,
/// `(i64, i64)` bigram/EAGLE). `Borrow<[Atom]>` lets a borrowed page slice
/// query a `HashMap` keyed by owned keys.
pub trait ChildKeyType:
    Clone
    + Eq
    + Hash
    + Default
    + Debug
    + AsRef<[Self::Atom]>
    + Borrow<[Self::Atom]>
    + From<Vec<Self::Atom>>
{
    type Atom: Copy + Eq + Hash + Send + Sync;

    /// Whether this key represents overlapping token bigrams rather than
    /// individual tokens. This is type metadata, not per-node state.
    const IS_BIGRAM: bool;

    /// The key over the boundary's raw token ids; ownership passes straight
    /// through, so the unigram key never copies.
    fn key_from(token_ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self>;

    /// The atom's token ids as u32 storage-hash words.
    fn hash_words(atom: &Self::Atom) -> impl Iterator<Item = u32>;

    /// The raw token ids spanned by `atoms`; the unigram view borrows, bigram
    /// atoms (overlapping by one) materialize.
    fn raw_token_ids(atoms: &[Self::Atom]) -> Cow<'_, [i64]>;

    /// Number of atoms in this key; an atom is one radix position — a token normally,
    /// a token pair for bigram/EAGLE keys.
    fn atom_len(&self) -> usize {
        self.as_ref().len()
    }

    /// The first radix page (`page_size` atoms) as an owned key — a node's
    /// child-map key under its parent; panics on a key shorter than a page.
    fn child_key(&self, page_size: usize) -> Self {
        let atom_len = self.atom_len();
        assert!(
            atom_len >= page_size,
            "child_key: key of {atom_len} atoms is shorter than a page ({page_size})"
        );
        self.as_ref()[..page_size].to_vec().into()
    }

    /// The single page starting at `start`, zero-copy.
    fn page_at(&self, start: usize, page_size: usize) -> &[Self::Atom] {
        let end = start + page_size;
        let atom_len = self.atom_len();
        assert!(
            end <= atom_len,
            "page_at: page [{start}, {end}) reaches beyond the key length {atom_len}"
        );
        &self.as_ref()[start..end]
    }

    /// Page-quantized common-prefix length of the tail from `start` with `other`.
    fn match_len(&self, start: usize, other: &Self, page_size: usize) -> usize {
        let atom_len = self.atom_len();
        assert!(
            start <= atom_len,
            "match_len: start {start} beyond the key length {atom_len}"
        );
        let common = self.as_ref()[start..]
            .iter()
            .zip(other.as_ref())
            .take_while(|(a, b)| a == b)
            .count();
        common / page_size * page_size
    }

    /// The owned suffix from `start`; empty when `start` equals the length.
    fn suffix(&self, start: usize) -> Self {
        let atom_len = self.atom_len();
        assert!(
            start <= atom_len,
            "suffix: start {start} beyond the key length {atom_len}"
        );
        self.as_ref()[start..].to_vec().into()
    }

    /// The key truncated to a whole number of pages.
    fn page_aligned(&self, page_size: usize) -> Self {
        let aligned_len = self.atom_len() / page_size * page_size;
        self.as_ref()[..aligned_len].to_vec().into()
    }

    /// Split into (head, tail) owned keys at `split_idx`; panics on a boundary
    /// split, which would leave one side empty.
    fn split_at(&self, split_idx: usize) -> (Self, Self) {
        let atom_len = self.atom_len();
        assert!(
            0 < split_idx && split_idx < atom_len,
            "split_at: split_idx {split_idx} out of range (0, {atom_len})"
        );
        let (head, tail) = self.as_ref().split_at(split_idx);
        (head.to_vec().into(), tail.to_vec().into())
    }
}

/// A token id as a u32 storage-hash word; token ids beyond u32 are rejected.
fn hash_word(token_id: i64) -> u32 {
    u32::try_from(token_id).expect("token id does not fit in uint32")
}

impl ChildKeyType for Vec<i64> {
    type Atom = i64;
    const IS_BIGRAM: bool = false;

    fn key_from(token_ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
        token_ids
    }

    fn hash_words(atom: &i64) -> impl Iterator<Item = u32> {
        std::iter::once(hash_word(*atom))
    }

    fn raw_token_ids(atoms: &[i64]) -> Cow<'_, [i64]> {
        Cow::Borrowed(atoms)
    }
}

impl ChildKeyType for Vec<(i64, i64)> {
    type Atom = (i64, i64);
    const IS_BIGRAM: bool = true;

    /// N+1 raw token ids become N overlapping (t_i, t_{i+1}) bigram atoms.
    fn key_from(token_ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
        Cow::Owned(token_ids.windows(2).map(|w| (w[0], w[1])).collect())
    }

    fn hash_words(atom: &(i64, i64)) -> impl Iterator<Item = u32> {
        [hash_word(atom.0), hash_word(atom.1)].into_iter()
    }

    fn raw_token_ids(atoms: &[(i64, i64)]) -> Cow<'_, [i64]> {
        let Some(first) = atoms.first() else {
            return Cow::Owned(Vec::new());
        };
        Cow::Owned(
            std::iter::once(first.0)
                .chain(atoms.iter().map(|atom| atom.1))
                .collect(),
        )
    }
}

// Per-page hash chains.

pub(crate) const DIGEST_LEN: usize = 32;
pub(crate) type HashDigest = [u8; DIGEST_LEN];

/// SHA256(prior_digest || page atom words as little-endian u32 bytes).
pub(crate) fn hash_page<K: ChildKeyType>(
    page: &[K::Atom],
    prior: Option<&HashDigest>,
) -> HashDigest {
    let mut hasher = Sha256::new();
    if let Some(prior) = prior {
        hasher.update(prior);
    }
    for atom in page {
        for word in K::hash_words(atom) {
            hasher.update(word.to_le_bytes());
        }
    }
    hasher.finalize().into()
}

/// Lowercase-hex encoding of a digest.
fn digest_to_hex(digest: &HashDigest) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(DIGEST_LEN * 2);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Decode a chained-in hex hash back to its raw digest.
fn parse_prior_hash(prior_hash: &str) -> HashDigest {
    let bytes = prior_hash.as_bytes();
    assert_eq!(
        bytes.len(),
        DIGEST_LEN * 2,
        "prior hash must be a 64-char hex digest"
    );
    let nibble = |b: u8| -> u8 {
        match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => panic!("prior hash contains a non-hex character"),
        }
    };
    let mut digest = [0u8; DIGEST_LEN];
    for (i, out) in digest.iter_mut().enumerate() {
        *out = (nibble(bytes[i * 2]) << 4) | nibble(bytes[i * 2 + 1]);
    }
    digest
}

/// Per-page chained hashes over key atoms, seeded from an optional prior hex hash.
pub fn get_hash_str<K: ChildKeyType>(
    atoms: &[K::Atom],
    prior_hash: Option<&str>,
    page_size: usize,
) -> Vec<String> {
    assert!(page_size > 0, "page_size must be positive");
    // An empty prior chains nothing.
    let mut prior = prior_hash
        .filter(|prior| !prior.is_empty())
        .map(parse_prior_hash);
    let mut hash_values = Vec::with_capacity(atoms.len().div_ceil(page_size));
    for page in atoms.chunks(page_size) {
        let digest = hash_page::<K>(page, prior.as_ref());
        hash_values.push(digest_to_hex(&digest));
        prior = Some(digest);
    }
    hash_values
}

/// Per-page chained raw digests, seeded from an optional prior digest.
pub(crate) fn get_hash_digests<K: ChildKeyType>(
    atoms: &[K::Atom],
    prior: Option<&HashDigest>,
    page_size: usize,
) -> Vec<HashDigest> {
    assert!(page_size > 0, "page_size must be positive");
    let mut prior = prior.copied();
    let mut digests = Vec::with_capacity(atoms.len().div_ceil(page_size));
    for page in atoms.chunks(page_size) {
        let digest = hash_page::<K>(page, prior.as_ref());
        digests.push(digest);
        prior = Some(digest);
    }
    digests
}

/// The hash's first 16 hex chars as a signed 64-bit block id for events.
pub fn hash_str_to_int64(hash_str: &str) -> i64 {
    u64::from_str_radix(&hash_str[..16], 16).expect("hash must be a hex digest") as i64
}

/// The raw digest's first eight bytes as the event protocol's signed i64.
pub(crate) fn hash_digest_to_int64(digest: &HashDigest) -> i64 {
    i64::from_be_bytes(digest[..8].try_into().expect("digest has eight bytes"))
}

/// Split a node's hash list at a page boundary; None-safe when never hashed.
pub fn split_node_hash_value(
    hash_values: Option<Vec<String>>,
    split_idx: usize,
    page_size: usize,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let Some(mut new_node_hash) = hash_values else {
        return (None, None);
    };
    let child_hash = new_node_hash.split_off(split_idx / page_size);
    // Progressive splits must not retain the pre-split capacity on the head.
    new_node_hash.shrink_to_fit();
    (Some(new_node_hash), Some(child_hash))
}

// Node arena storage.

/// Owns every `Node`; parent/children/LRU hold `NodeIdx_`s into it, with a freelist
/// and a single root; child edges are keyed by (namespace, page key).
pub struct NodeArena<K: ChildKeyType> {
    /// Node store indexed by `NodeIdx_`; `None` marks a freed slot.
    nodes: Vec<Option<Node<K>>>,
    /// Freed slot ids available for reuse.
    free: Vec<NodeIdx_>,
    /// External handle -> live slot; a freed `NodeId` leaves the map, so a
    /// stale handle can never alias a recycled slot.
    id_map: HashMap<NodeId, NodeIdx_>,
    /// Next external handle; monotonic, never recycled (survives `reset`).
    next_id: NodeId,
    /// The tree's single root; every namespace hangs off it.
    root: NodeIdx_,
    /// Monotonic counter stamped into `Node::last_access_counter`.
    access_counter: i64,
    /// The component types this tree runs (every root is locked for each).
    component_types: Vec<ComponentType>,
    /// Atoms per radix page; children are keyed by their key's first page.
    page_size: usize,
}

impl<K: ChildKeyType> NodeArena<K> {
    /// Build an arena for the given component types and install a fresh root.
    pub fn new(component_types: Vec<ComponentType>, page_size: usize) -> Self {
        let mut arena = NodeArena {
            nodes: Vec::new(),
            free: Vec::new(),
            id_map: HashMap::new(),
            next_id: 0,
            root: NodeIdx_(0),
            access_counter: 0,
            component_types,
            page_size,
        };
        arena.reset();
        arena
    }

    /// Drop all nodes, then reinstall the root.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.free.clear();
        // next_id is NOT reset: pre-reset handles must miss, never alias.
        self.id_map.clear();
        self.access_counter = 0;
        self.root = self.alloc_root();
    }

    /// The live slot for an external handle; panics on a freed or unknown id.
    #[track_caller]
    pub fn resolve(&self, id: NodeId) -> NodeIdx_ {
        *self
            .id_map
            .get(&id)
            .unwrap_or_else(|| panic!("node {id} is not allocated"))
    }

    /// The live slot for an external handle, or None if freed/unknown.
    pub fn try_resolve(&self, id: NodeId) -> Option<NodeIdx_> {
        self.id_map.get(&id).copied()
    }

    /// Mint the next external handle for the slot and index it.
    fn mint_id_(&mut self, idx: NodeIdx_) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.id_map.insert(id, idx);
        id
    }

    /// Allocate a protected, value-less root: locked (`lock_ref = 1`) for each
    /// component type and never entering a leaf/LRU set.
    pub fn alloc_root(&mut self) -> NodeIdx_ {
        let idx = self.reserve();
        let id = self.mint_id_(idx);
        let tick = self.get_and_bump_access_counter();
        let node = self.nodes[idx.0].insert(Node::new_root(id));
        node.idx = idx;
        node.last_access_counter = tick;
        node.creation_counter = tick;
        for ct in &self.component_types {
            node.values[ct.idx()].lock_ref = 1;
        }
        idx
    }

    /// Every live node id, in slot order.
    pub fn live_ids(&self) -> impl Iterator<Item = NodeIdx_> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.as_ref().map(|_| NodeIdx_(idx)))
    }

    /// Per-page hash values for a node's key, chained from its parent's last hash.
    pub fn compute_node_hash_values(&self, node_id: NodeIdx_, page_size: usize) -> Vec<String> {
        let node = self.node(node_id);
        let parent_hash = node.parent.and_then(|parent_id| {
            let parent = self.node(parent_id);
            if parent.key.atom_len() > 0 {
                parent.get_last_hash_value()
            } else {
                None
            }
        });
        crate::node::get_hash_str::<K>(node.key.as_ref(), parent_hash, page_size)
    }

    /// The ancestor chain's hash values ending at `node_id`, in root-to-node
    /// order; the walk stops below the nearest never-hashed ancestor.
    pub fn prefix_hash_values(&self, node_id: Option<NodeIdx_>) -> Vec<String> {
        let mut chunks: Vec<&Vec<String>> = Vec::new();
        let mut cursor = node_id;
        while let Some(id) = cursor {
            let node = self.node(id);
            let Some(hash_value) = node.hash_value.as_ref() else {
                break;
            };
            chunks.push(hash_value);
            cursor = node.parent;
        }
        chunks
            .iter()
            .rev()
            .flat_map(|chunk| chunk.iter().cloned())
            .collect()
    }

    /// The node's caller-defined namespace key; None for the default.
    pub fn node_extra_key(&self, node_id: NodeIdx_) -> Option<&str> {
        self.node(node_id).namespace.extra_key()
    }

    /// The node's cache salt; None for an unsalted namespace.
    pub fn node_cache_salt(&self, node_id: NodeIdx_) -> Option<&str> {
        self.node(node_id).namespace.cache_salt()
    }

    /// The tree's single root.
    pub fn root(&self) -> NodeIdx_ {
        self.root
    }

    /// The node's child on the page within the namespace, if any.
    pub fn child_on_page(
        &self,
        id: NodeIdx_,
        extra_key: Option<&str>,
        page: &[K::Atom],
    ) -> Option<NodeIdx_> {
        self.child_on_page_in_namespace(
            id,
            KeyNamespaceRef::new(extra_key, /* cache_salt = */ None),
            page,
        )
    }

    /// The node's child on the page within the full radix namespace, if any.
    pub fn child_on_page_in_namespace(
        &self,
        id: NodeIdx_,
        namespace: KeyNamespaceRef<'_>,
        page: &[K::Atom],
    ) -> Option<NodeIdx_> {
        self.node(id)
            .children
            .get(&ChildEdgeRef::<K> { namespace, page })
            .copied()
    }

    /// The root's child on the key's first page within the namespace, if any.
    pub fn root_child(&self, extra_key: Option<&str>, page: &[K::Atom]) -> Option<NodeIdx_> {
        self.child_on_page(self.root, extra_key, page)
    }

    /// The root's child on the key's first page within the full radix namespace.
    pub fn root_child_in_namespace(
        &self,
        namespace: KeyNamespaceRef<'_>,
        page: &[K::Atom],
    ) -> Option<NodeIdx_> {
        self.child_on_page_in_namespace(self.root, namespace, page)
    }

    /// Whether any root edge files under the namespace.
    pub fn namespace_exists(&self, extra_key: Option<&str>) -> bool {
        self.full_namespace_exists(KeyNamespaceRef::new(
            extra_key, /* cache_salt = */ None,
        ))
    }

    /// Whether any root edge files under the full radix namespace.
    pub fn full_namespace_exists(&self, namespace: KeyNamespaceRef<'_>) -> bool {
        self.node(self.root)
            .children
            .keys()
            .any(|(stored, _)| stored.as_ref() == namespace)
    }

    /// Install `child` under `parent` on its namespaced `map_key`; returns the
    /// displaced child, if any. The key's namespace mirrors the child's.
    pub fn insert_child_edge(
        &mut self,
        parent: NodeIdx_,
        map_key: K,
        child: NodeIdx_,
    ) -> Option<NodeIdx_> {
        let namespace = self.node(child).namespace.clone();
        self.node_mut(parent)
            .children
            .insert((namespace, map_key), child)
    }

    /// Reserve a detached child and attach it under `parent`; `extra_key` names the
    /// namespace for root children (deeper nodes inherit the parent's).
    pub fn alloc_child(
        &mut self,
        parent: NodeIdx_,
        key: K,
        priority: i64,
        extra_key: Option<&str>,
    ) -> Result<NodeIdx_, TreeCoreRuntimeError> {
        self.alloc_child_in_namespace(
            parent,
            key,
            priority,
            KeyNamespaceRef::new(extra_key, /* cache_salt = */ None),
        )
    }

    /// Reserve and attach a child in the full radix namespace.
    pub fn alloc_child_in_namespace(
        &mut self,
        parent: NodeIdx_,
        key: K,
        priority: i64,
        namespace: KeyNamespaceRef<'_>,
    ) -> Result<NodeIdx_, TreeCoreRuntimeError> {
        // Validate the parent and attach the child before committing a slot, so a
        // rejected add reserves nothing.
        let size = self.nodes.len();
        match self.nodes.get(parent.0) {
            None => return Err(TreeCoreRuntimeError::NodeAccessOutOfBound { id: parent, size }),
            Some(None) => return Err(TreeCoreRuntimeError::ParentNotAllocated { id: parent }),
            Some(Some(_)) => {}
        }
        let idx = self
            .free
            .last()
            .copied()
            .unwrap_or(NodeIdx_(self.nodes.len()));
        let mut child_node = Node::new_child(self.next_id, key, priority);
        child_node.idx = idx;
        let tick = self.get_and_bump_access_counter();
        child_node.last_access_counter = tick;
        child_node.creation_counter = tick;
        let page_size = self.page_size;
        // Root children adopt the op namespace; deeper nodes inherit the parent's.
        child_node.namespace = if parent == self.root {
            namespace.to_owned()
        } else {
            self.node(parent).namespace.clone()
        };
        self.node_mut(parent)
            .attach_child(&mut child_node, page_size)?;
        match self.free.pop() {
            Some(popped) => {
                // The freed slot we peeked is still the freelist head and still free.
                assert_eq!(popped, idx, "freelist head changed between peek and pop");
                assert!(
                    self.nodes[idx.0].is_none(),
                    "freelist popped a live slot {idx} (freelist corruption)"
                );
                self.nodes[idx.0] = Some(child_node);
            }
            None => self.nodes.push(Some(child_node)),
        }
        self.mint_id_(idx);
        Ok(idx)
    }

    /// Allocate a detached node (empty key, no parent) for the tree to wire in.
    pub fn alloc_detached(&mut self, priority: i64) -> NodeIdx_ {
        let idx = self.reserve();
        let id = self.mint_id_(idx);
        let tick = self.get_and_bump_access_counter();
        let node = self.nodes[idx.0].insert(Node::new_child(id, K::default(), priority));
        node.idx = idx;
        node.last_access_counter = tick;
        node.creation_counter = tick;
        idx
    }

    /// Reserve an empty slot (reusing a freed one when available), returning its id.
    fn reserve(&mut self) -> NodeIdx_ {
        match self.free.pop() {
            Some(idx) => {
                assert!(
                    self.nodes[idx.0].is_none(),
                    "freelist popped a live slot {idx} (freelist corruption)"
                );
                idx
            }
            None => {
                self.nodes.push(None);
                NodeIdx_(self.nodes.len() - 1)
            }
        }
    }

    /// Detach a leaf from its parent and return its slot to the freelist.
    pub fn free_leaf(&mut self, id: NodeIdx_) -> Result<(), TreeCoreRuntimeError> {
        let size = self.nodes.len();
        // Validate and take the leaf out in one step.
        let mut child_node = match self.nodes.get_mut(id.0) {
            None => return Err(TreeCoreRuntimeError::NodeAccessOutOfBound { id, size }),
            Some(None) => return Err(TreeCoreRuntimeError::NodeDoubleFree { id }),
            Some(Some(node)) if node.is_root() => {
                return Err(TreeCoreRuntimeError::RootNotFreeable { id });
            }
            Some(Some(node)) if !node.is_leaf() => {
                return Err(TreeCoreRuntimeError::FreeNonLeafNode {
                    id,
                    num_children: node.children.len(),
                });
            }
            Some(slot) => slot.take().expect("validated non-root leaf"),
        };
        // A validated non-root leaf always has a parent to unlink from.
        let parent = child_node.parent();
        let page_size = self.page_size;
        let parent_node = self.node_mut(parent);
        child_node.detach_from_parent(parent_node, page_size);
        self.id_map.remove(&child_node.id);
        self.free.push(id);
        Ok(())
    }

    /// Number of live nodes.
    pub fn len(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    /// Shared access to a live node; panics on a dead or out-of-range id.
    #[track_caller]
    pub fn node(&self, id: NodeIdx_) -> &Node<K> {
        let size = self.nodes.len();
        self.nodes
            .get(id.0)
            .unwrap_or_else(|| panic!("node access out of bounds: id {id} not in [0, {size})"))
            .as_ref()
            .unwrap_or_else(|| panic!("node {id} is not allocated"))
    }

    /// Mutable access to a live node; panics on a dead or out-of-range id.
    #[track_caller]
    pub fn node_mut(&mut self, id: NodeIdx_) -> &mut Node<K> {
        let size = self.nodes.len();
        self.nodes
            .get_mut(id.0)
            .unwrap_or_else(|| panic!("node access out of bounds: id {id} not in [0, {size})"))
            .as_mut()
            .unwrap_or_else(|| panic!("node {id} is not allocated"))
    }

    /// The node's device value for the component; panics if unset.
    pub fn device_value(&self, id: NodeIdx_, component_type: ComponentType) -> &Tensor {
        self.node(id).device_value(component_type)
    }

    /// The node's device value for the component, or None when unset.
    pub fn try_device_value(&self, id: NodeIdx_, component_type: ComponentType) -> Option<&Tensor> {
        self.node(id).try_device_value(component_type)
    }

    /// Whether the node holds the component's device value.
    pub fn has_device_value(&self, id: NodeIdx_, component_type: ComponentType) -> bool {
        self.node(id).has_device_value(component_type)
    }

    /// The node's device value length for the component, or 0 when value-less.
    pub fn device_value_len(&self, id: NodeIdx_, component_type: ComponentType) -> usize {
        self.node(id).device_value_len(component_type)
    }

    /// Set the node's device value for the component; panics if already set.
    pub fn set_device_value(&mut self, id: NodeIdx_, component_type: ComponentType, value: Tensor) {
        self.node_mut(id).set_device_value(component_type, value);
    }

    /// Take the node's device value for the component; panics if unset.
    pub fn take_device_value(&mut self, id: NodeIdx_, component_type: ComponentType) -> Tensor {
        self.node_mut(id).take_device_value(component_type)
    }

    /// The node's device lock refcount for the component.
    pub fn device_lock_ref(&self, id: NodeIdx_, component_type: ComponentType) -> u32 {
        self.node(id).device_lock_ref(component_type)
    }

    /// The node's host value for the component; panics if unset.
    pub fn host_value(&self, id: NodeIdx_, component_type: ComponentType) -> &Tensor {
        self.node(id).host_value(component_type)
    }

    /// Whether the node holds the component's host value.
    pub fn has_host_value(&self, id: NodeIdx_, component_type: ComponentType) -> bool {
        self.node(id).has_host_value(component_type)
    }

    /// The node's host value length for the component, or 0 when value-less.
    pub fn host_value_len(&self, id: NodeIdx_, component_type: ComponentType) -> usize {
        self.node(id).host_value_len(component_type)
    }

    /// Set the node's host value for the component; panics if already set.
    pub fn set_host_value(&mut self, id: NodeIdx_, component_type: ComponentType, value: Tensor) {
        self.node_mut(id).set_host_value(component_type, value);
    }

    /// Take the node's host value for the component; panics if unset.
    pub fn take_host_value(&mut self, id: NodeIdx_, component_type: ComponentType) -> Tensor {
        self.node_mut(id).take_host_value(component_type)
    }

    /// The node's host lock refcount for the component.
    pub fn host_lock_ref(&self, id: NodeIdx_, component_type: ComponentType) -> u32 {
        self.node(id).host_lock_ref(component_type)
    }

    /// Bump the node's device lock refcount for the component.
    pub fn inc_device_lock_ref(&mut self, id: NodeIdx_, component_type: ComponentType) {
        self.node_mut(id).inc_device_lock_ref(component_type);
    }

    /// Drop the node's device lock refcount for the component; panics when unlocked.
    pub fn dec_device_lock_ref(&mut self, id: NodeIdx_, component_type: ComponentType) {
        self.node_mut(id).dec_device_lock_ref(component_type);
    }

    /// Bump the node's host lock refcount for the component.
    pub fn inc_host_lock_ref(&mut self, id: NodeIdx_, component_type: ComponentType) {
        self.node_mut(id).inc_host_lock_ref(component_type);
    }

    /// Mutable access to a live parent/child pair at once. Internal-only accessor:
    /// panics on a dead id or when `child_node_id` is not a child of `parent_node_id`.
    #[track_caller]
    pub fn node_pair_mut(
        &mut self,
        parent_node_id: NodeIdx_,
        child_node_id: NodeIdx_,
    ) -> (&mut Node<K>, &mut Node<K>) {
        assert_ne!(
            parent_node_id, child_node_id,
            "node_pair_mut: distinct nodes required, got {parent_node_id} twice"
        );
        let size = self.nodes.len();
        assert!(
            parent_node_id.0 < size && child_node_id.0 < size,
            "node_pair_mut: id out of bounds ({parent_node_id}, {child_node_id}) vs size {size}"
        );
        let [parent_slot, child_slot] = self
            .nodes
            .get_disjoint_mut([parent_node_id.0, child_node_id.0])
            .expect("distinct in-bounds indices");
        let parent_node = parent_slot.as_mut().expect("live node");
        let child_node = child_slot.as_mut().expect("live node");
        assert_eq!(
            child_node.try_parent(),
            Some(parent_node_id),
            "node_pair_mut: node {child_node_id} is not a child of {parent_node_id}"
        );
        (parent_node, child_node)
    }

    /// Advance the access counter by `delta` ticks and return the newest one,
    /// reserving the whole range for the caller to assign.
    pub fn get_and_batch_bump_access_counter(&mut self, delta: i64) -> i64 {
        assert!(
            delta > 0,
            "get_and_batch_bump_access_counter: delta {delta} must be positive"
        );
        self.access_counter += delta;
        self.access_counter
    }

    /// Bump the access counter and return the new tick (for stamping `last_access_counter`).
    pub fn get_and_bump_access_counter(&mut self) -> i64 {
        self.access_counter += 1;
        self.access_counter
    }
}

// Eviction-eligible node set.

/// Set of `NodeIdx_`s with O(1) membership ops and dense iteration.
#[derive(Default)]
pub struct EvictableNodeSet {
    /// Dense member list; order is unspecified (swap-remove).
    nodes: Vec<NodeIdx_>,
    /// Each member's position in `nodes`, indexed by `NodeIdx_`.
    slots: Vec<Option<usize>>,
}

impl EvictableNodeSet {
    pub fn new() -> Self {
        Default::default()
    }

    /// Whether `node_id` is a member.
    pub fn contains(&self, node_id: NodeIdx_) -> bool {
        self.slots.get(node_id.0).copied().flatten().is_some()
    }

    /// Insert `node_id`; no-op when already a member.
    pub fn add(&mut self, node_id: NodeIdx_) {
        if node_id.0 >= self.slots.len() {
            self.slots.resize(node_id.0 + 1, None);
        }
        if self.slots[node_id.0].is_some() {
            return;
        }
        self.slots[node_id.0] = Some(self.nodes.len());
        self.nodes.push(node_id);
    }

    /// Remove `node_id`; no-op when not a member.
    pub fn discard(&mut self, node_id: NodeIdx_) {
        let Some(slot) = self.slots.get(node_id.0).copied().flatten() else {
            return;
        };
        self.slots[node_id.0] = None;
        self.nodes.swap_remove(slot);
        // The swapped-in tail member (if any) now lives at `slot`.
        if let Some(&moved) = self.nodes.get(slot) {
            self.slots[moved.0] = Some(slot);
        }
    }

    /// The members, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = NodeIdx_> + '_ {
        self.nodes.iter().copied()
    }

    // Test-only conveniences: production callers use add/discard/contains/iter.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
#[cfg(test)]
#[path = "tests/node.rs"]
mod tests;
