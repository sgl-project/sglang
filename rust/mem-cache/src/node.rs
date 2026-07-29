//! A radix-tree node, owned by the `NodeArena` and referenced by `NodeIdx_`.

use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tch::Tensor;

use crate::components::{ComponentType, FULL, NUM_COMPONENT_TYPES};

/// A radix-tree node, generic over the child-key type `K` (single-token or bigram).
pub struct Node<K: ChildKeyType> {
    /// Parent handle; `None` for the root or a not-yet-attached child.
    pub parent: Option<NodeIdx_>,
    /// Own arena slot; stamped by the arena on allocation (hand-built nodes
    /// default it to the id value).
    pub(crate) idx: NodeIdx_,
    /// The namespace this node belongs to; `None` for the default namespace.
    pub extra_key: Option<Arc<str>>,
    /// Child edges keyed by (namespace, the child's page key); the namespace
    /// component mirrors the child's `extra_key` at every level.
    pub children: HashMap<(Option<Arc<str>>, K), NodeIdx_>,
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
    /// TODO(Jialin): store raw [u8; 32] digests and hex-encode only at the python/storage boundary.
    pub hash_value: Option<Vec<String>>,
    /// The in-flight write-through backup's ack id.
    pub write_through_pending_id: Option<usize>,
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
        // def is_root(self, node_id: NodeId) -> bool:
        //         """Whether the node is the tree root."""
        //         return self.node_by_id(node_id) is self.root_node
        todo!()
    }

    /// The parent's id; panics on a root.
    #[track_caller]
    pub fn parent(&self) -> NodeIdx_ {
        todo!()
    }

    /// The parent's id, or `None` on a root.
    pub fn try_parent(&self) -> Option<NodeIdx_> {
        todo!()
    }

    /// Whether this node has no children (a tree leaf).
    pub fn is_leaf(&self) -> bool {
        todo!()
    }

    /// Tree-level: Full KV not on device (non-root with value=None).
    pub fn evicted(&self) -> bool {
        // def evicted(self) -> bool:
        //         """Tree-level: Full KV not on device (non-root with value=None)."""
        //         return (
        //             self.parent is not None
        //             and self.component_data[ComponentType.FULL].value is None
        //         )
        todo!()
    }

    /// Tree-level: Full KV present on host.
    pub fn backuped(&self) -> bool {
        // def backuped(self) -> bool:
        //         """Tree-level: Full KV present on host."""
        //         return self.component_data[ComponentType.FULL].host_value is not None
        todo!()
    }

    /// The last page's hash value, or None when the node was never hashed.
    pub fn get_last_hash_value(&self) -> Option<&str> {
        // def get_last_hash_value(self) -> Optional[str]:
        //         if self.hash_value is None or len(self.hash_value) == 0:
        //             return None
        //         return self.hash_value[-1]
        todo!()
    }

    /// The component's device value; panics if unset.
    pub fn device_value(&self, component_type: ComponentType) -> &Tensor {
        todo!()
    }

    /// The component's device value, or None when unset.
    pub fn try_device_value(&self, component_type: ComponentType) -> Option<&Tensor> {
        todo!()
    }

    /// Whether the component's device value is present.
    pub fn has_device_value(&self, component_type: ComponentType) -> bool {
        todo!()
    }

    /// The component's device value length, or 0 when value-less.
    pub fn device_value_len(&self, component_type: ComponentType) -> usize {
        todo!()
    }

    /// Set the component's device value; panics if already set.
    pub fn set_device_value(&mut self, component_type: ComponentType, value: Tensor) {
        todo!()
    }

    /// Take the component's device value; panics if unset.
    pub fn take_device_value(&mut self, component_type: ComponentType) -> Tensor {
        todo!()
    }

    /// The component's device lock refcount.
    pub fn device_lock_ref(&self, component_type: ComponentType) -> u32 {
        todo!()
    }

    /// Bump the component's device lock refcount by one.
    pub fn inc_device_lock_ref(&mut self, component_type: ComponentType) {
        todo!()
    }

    /// Drop the component's device lock refcount by one; panics when unlocked.
    pub fn dec_device_lock_ref(&mut self, component_type: ComponentType) {
        todo!()
    }

    /// Copy the component's device lock refcount from `src_node`.
    pub fn copy_device_lock_ref(&mut self, component_type: ComponentType, src_node: &Node<K>) {
        todo!()
    }

    /// Split the component's device value between a new parent and the child.
    pub fn redistribute_child_device_value(
        parent_node: &mut Node<K>,
        child_node: &mut Node<K>,
        component_type: ComponentType,
        split_len: i64,
    ) {
        todo!()
    }

    /// The component's host value; panics if unset.
    pub fn host_value(&self, component_type: ComponentType) -> &Tensor {
        todo!()
    }

    /// The component's host value, or None when unset.
    pub fn try_host_value(&self, component_type: ComponentType) -> Option<&Tensor> {
        todo!()
    }

    /// Whether the component's host value is present.
    pub fn has_host_value(&self, component_type: ComponentType) -> bool {
        todo!()
    }

    /// The component's host value length, or 0 when value-less.
    pub fn host_value_len(&self, component_type: ComponentType) -> usize {
        todo!()
    }

    /// Set the component's host value; panics if already set.
    pub fn set_host_value(&mut self, component_type: ComponentType, value: Tensor) {
        todo!()
    }

    /// Take the component's host value; panics if unset.
    pub fn take_host_value(&mut self, component_type: ComponentType) -> Tensor {
        todo!()
    }

    /// The component's host lock refcount (the host lock).
    pub fn host_lock_ref(&self, component_type: ComponentType) -> u32 {
        todo!()
    }

    /// Bump the component's host lock refcount by one.
    pub fn inc_host_lock_ref(&mut self, component_type: ComponentType) {
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

    /// Drop the component's host lock refcount by one; panics when unlocked.
    pub fn dec_host_lock_ref(&mut self, component_type: ComponentType) {
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

    /// Split the component's host value between a new parent and the child.
    pub fn redistribute_child_host_value(
        parent_node: &mut Node<K>,
        child_node: &mut Node<K>,
        component_type: ComponentType,
        split_len: i64,
    ) {
        todo!()
    }

    /// Whether any component holds a device lock on this node.
    pub fn is_device_locked(&self) -> bool {
        todo!()
    }

    /// Whether any component holds a host lock on this node.
    pub fn is_host_locked(&self) -> bool {
        todo!()
    }

    // ==== Crate-internal tree wiring ====

    /// A fresh root: no parent, no value, lowest eviction priority.
    pub(crate) fn new_root(id: NodeId) -> Self {
        todo!()
    }

    /// A detached child reached by edge `key`; `attach_child` sets the parent link.
    pub(crate) fn new_child(id: NodeId, key: K, priority: i64) -> Self {
        todo!()
    }

    /// The namespaced edge key for this node's own edge from its parent.
    pub(crate) fn edge_key(&self, page_size: usize) -> (Option<Arc<str>>, K) {
        todo!()
    }

    /// Link `child` under this node, keyed by its namespaced page key; errors on
    /// a duplicate key. Panics if `child` is already attached (an internal invariant).
    /// The caller sets `child.extra_key` first; the edge key mirrors it.
    pub(crate) fn attach_child(
        &mut self,
        child: &mut Node<K>,
        page_size: usize,
    ) -> Result<(), TreeCoreRuntimeError> {
        todo!()
    }

    /// Unlink this node from `parent`: drop it from `parent.children` and clear its own
    /// parent link; panics on a broken parent<->child link (an internal invariant).
    pub(crate) fn detach_from_parent(&mut self, parent: &mut Node<K>, page_size: usize) {
        todo!()
    }

    // ==== Internal slot-keyed lookups ====

    /// The slot's value state (internal slot-keyed lookup).
    pub fn state_(&self, slot: ValueSlotIdx) -> &ValueState {
        todo!()
    }

    /// The slot's mutable value state (internal slot-keyed lookup).
    pub fn state_mut_(&mut self, slot: ValueSlotIdx) -> &mut ValueState {
        todo!()
    }

    /// The slot's value, or None when unset (internal slot-keyed lookup).
    pub fn try_value_(&self, slot: ValueSlotIdx) -> Option<&Tensor> {
        todo!()
    }

    /// The slot's value; panics if no value is set (internal slot-keyed lookup).
    pub fn value_(&self, slot: ValueSlotIdx) -> &Tensor {
        todo!()
    }

    /// Whether the slot's value is present (internal slot-keyed lookup).
    pub fn has_value_(&self, slot: ValueSlotIdx) -> bool {
        todo!()
    }

    /// The slot's value length, or 0 when value-less (internal slot-keyed lookup).
    pub fn value_len_(&self, slot: ValueSlotIdx) -> usize {
        todo!()
    }

    /// Set the slot's value; panics if a value is already set (internal slot-keyed lookup).
    pub fn set_value_(&mut self, slot: ValueSlotIdx, value: Tensor) {
        todo!()
    }

    /// Take the slot's value, leaving it value-less; panics if no value is set
    /// (internal slot-keyed lookup).
    pub fn take_value_(&mut self, slot: ValueSlotIdx) -> Tensor {
        todo!()
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
        todo!()
    }

    /// The slot's lock refcount (internal slot-keyed lookup).
    pub fn lock_ref_(&self, slot: ValueSlotIdx) -> u32 {
        todo!()
    }

    /// Set the slot's lock refcount (internal slot-keyed lookup).
    pub fn set_lock_ref_(&mut self, slot: ValueSlotIdx, lock_ref: u32) {
        todo!()
    }

    /// Bump the slot's lock refcount by one (internal slot-keyed lookup).
    pub fn inc_lock_ref_(&mut self, slot: ValueSlotIdx) {
        // def inc_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        //         node = self.node_by_id(node_id)
        //         result = IncLockRefResult()
        //         for component in self.components:
        //             result = component.acquire_component_lock(node=node, result=result)
        //         self._update_evictable_leaf_sets(node)
        //         return result
        todo!()
    }

    /// Drop the slot's lock refcount by one; panics on an unlocked node
    /// (internal slot-keyed lookup).
    pub fn dec_lock_ref_(&mut self, slot: ValueSlotIdx) {
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
}

// ==== Node handles and per-slot value state =============================

/// External node handle — the only node identity that crosses the FFI.
/// Minted monotonically and never recycled, so a freed node's id can never
/// alias a later allocation (the arena's id map is the ABA guard).
pub type NodeId = usize;

/// Internal arena slot index; recycled by the freelist and never crosses the FFI.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct NodeIdx_(pub(crate) usize);

impl std::fmt::Display for NodeIdx_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
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
        // def device(self):
        //         return self.tree_core.device
        todo!()
    }

    /// The component's host-tier slot.
    pub const fn host(component_type: ComponentType) -> Self {
        todo!()
    }

    /// Index into a node's `values` array.
    pub const fn idx(self) -> usize {
        todo!()
    }

    /// The slot at a flat index; panics out of range.
    pub fn from_idx(idx: usize) -> Self {
        todo!()
    }

    /// Whether this is a host-tier slot.
    pub const fn is_host(self) -> bool {
        todo!()
    }

    /// The component this slot belongs to.
    pub fn component_type(self) -> ComponentType {
        todo!()
    }

    /// The slot's tier name, for diagnostics.
    pub fn tier(self) -> &'static str {
        todo!()
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

// ==== Tree-core errors ==================================================

/// Errors surfaced from the tree-core runtime API when a caller violates a documented
/// contract (freeing an unallocated node, allocating under a freed parent).
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum TreeCoreRuntimeError {
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
}

// ==== Child keys (unigram and bigram pages) =============================

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
        todo!()
    }

    /// The first radix page (`page_size` atoms) as an owned key — a node's
    /// child-map key under its parent; panics on a key shorter than a page.
    fn child_key(&self, page_size: usize) -> Self {
        // def child_key(self, page_size: int = 1):
        //         """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        //         t = self.token_ids
        //         if self.is_bigram:
        //             if page_size == 1:
        //                 plain = (t[0], t[1])
        //             else:
        //                 plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        //         else:
        //             plain = t[0] if page_size == 1 else tuple(t[:page_size])
        //         return plain if self.extra_key is None else (self.extra_key, plain)
        todo!()
    }

    /// The single page starting at `start`, zero-copy.
    fn page_at(&self, start: usize, page_size: usize) -> &[Self::Atom] {
        todo!()
    }

    /// Page-quantized common-prefix length of the tail from `start` with `other`.
    fn match_len(&self, start: usize, other: &Self, page_size: usize) -> usize {
        todo!()
    }

    /// The owned suffix from `start`; empty when `start` equals the length.
    fn suffix(&self, start: usize) -> Self {
        todo!()
    }

    /// The key truncated to a whole number of pages.
    fn page_aligned(&self, page_size: usize) -> Self {
        // def page_aligned(self, page_size: int) -> RadixKey:
        //         if page_size == 1:
        //             return self
        //         aligned_len = len(self) // page_size * page_size
        //         return self[:aligned_len]
        todo!()
    }

    /// Split into (head, tail) owned keys at `split_idx`; panics on a boundary
    /// split, which would leave one side empty.
    fn split_at(&self, split_idx: usize) -> (Self, Self) {
        todo!()
    }
}

/// A token id as a u32 storage-hash word; token ids beyond u32 are rejected.
fn hash_word(token_id: i64) -> u32 {
    todo!()
}

impl ChildKeyType for Vec<i64> {
    type Atom = i64;

    fn key_from(token_ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
        todo!()
    }

    fn hash_words(atom: &i64) -> impl Iterator<Item = u32> {
        std::iter::empty()
    }

    fn raw_token_ids(atoms: &[i64]) -> Cow<'_, [i64]> {
        // def raw_token_ids(self) -> array:
        //         """token_ids honoring `limit` (copies only when capped)."""
        //         n = self._raw_len()
        //         t = self.token_ids
        //         return t if n == len(t) else t[:n]
        todo!()
    }
}

impl ChildKeyType for Vec<(i64, i64)> {
    type Atom = (i64, i64);

    /// N+1 raw token ids become N overlapping (t_i, t_{i+1}) bigram atoms.
    fn key_from(token_ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
        todo!()
    }

    fn hash_words(atom: &(i64, i64)) -> impl Iterator<Item = u32> {
        std::iter::empty()
    }

    fn raw_token_ids(atoms: &[(i64, i64)]) -> Cow<'_, [i64]> {
        // def raw_token_ids(self) -> array:
        //         """token_ids honoring `limit` (copies only when capped)."""
        //         n = self._raw_len()
        //         t = self.token_ids
        //         return t if n == len(t) else t[:n]
        todo!()
    }
}

// ==== Per-page hash chains ==============================================

const DIGEST_LEN: usize = 32;

/// SHA256(prior_digest || page atom words as little-endian u32 bytes).
fn hash_page<K: ChildKeyType>(
    page: &[K::Atom],
    prior: Option<&[u8; DIGEST_LEN]>,
) -> [u8; DIGEST_LEN] {
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
fn digest_to_hex(digest: &[u8; DIGEST_LEN]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(DIGEST_LEN * 2);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Decode a chained-in hex hash back to its raw digest.
fn parse_prior_hash(prior_hash: &str) -> [u8; DIGEST_LEN] {
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
    // def get_hash_str(
    //     token_ids: List[int],
    //     prior_hash: Optional[str] = None,
    //     page_size: Optional[int] = None,
    // ) -> str | List[str]:
    //     prior_digest = bytes.fromhex(prior_hash) if prior_hash else None
    //     return get_native_hash(token_ids, prior_digest, page_size)
    todo!()
}

/// The hash's first 16 hex chars as a signed 64-bit block id for events.
pub fn hash_str_to_int64(hash_str: &str) -> i64 {
    // def hash_str_to_int64(hash_str: str) -> int:
    //     """Convert SHA256 hex string to signed 64-bit integer for events.
    //
    //     Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    //     """
    //     uint64_val = int(hash_str[:16], 16)
    //     if uint64_val >= 2**63:
    //         return uint64_val - 2**64
    //     return uint64_val
    todo!()
}

/// Split a node's hash list at a page boundary; None-safe when never hashed.
pub fn split_node_hash_value(
    hash_values: Option<Vec<String>>,
    split_idx: usize,
    page_size: usize,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    // def split_node_hash_value(
    //     child_hash_value: Optional[List[str]], split_len: int, page_size: int
    // ) -> tuple[Optional[List[str]], Optional[List[str]]]:
    //     """Split hash_value between parent and child nodes during node splitting.
    //
    //     Args:
    //         child_hash_value: The hash_value list from the child node being split
    //         split_len: The length at which to split (in tokens)
    //         page_size: The page size for calculating number of pages
    //
    //     Returns:
    //         Tuple of (new_node_hash_value, updated_child_hash_value)
    //     """
    //     if child_hash_value is None:
    //         return None, None
    //
    //     if page_size == 1:
    //         split_pages = split_len
    //     else:
    //         split_pages = split_len // page_size
    //
    //     new_node_hash = child_hash_value[:split_pages]
    //     child_hash = child_hash_value[split_pages:]
    //
    //     return new_node_hash, child_hash
    todo!()
}

// ==== The node arena ====================================================

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
        todo!()
    }

    /// Drop all nodes, then reinstall the root.
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

    /// The live slot for an external handle; panics on a freed or unknown id.
    #[track_caller]
    pub fn resolve(&self, id: NodeId) -> NodeIdx_ {
        todo!()
    }

    /// The live slot for an external handle, or None if freed/unknown.
    pub fn try_resolve(&self, id: NodeId) -> Option<NodeIdx_> {
        todo!()
    }

    /// Mint the next external handle for the slot and index it.
    fn mint_id_(&mut self, idx: NodeIdx_) -> NodeId {
        todo!()
    }

    /// Allocate a protected, value-less root: locked (`lock_ref = 1`) for each
    /// component type and never entering a leaf/LRU set.
    pub fn alloc_root(&mut self) -> NodeIdx_ {
        todo!()
    }

    /// Every live node id, in slot order.
    pub fn live_ids(&self) -> impl Iterator<Item = NodeIdx_> + '_ {
        std::iter::empty()
    }

    /// Per-page hash values for a node's key, chained from its parent's last hash.
    pub fn compute_node_hash_values(&self, node_id: NodeIdx_, page_size: usize) -> Vec<String> {
        // def compute_node_hash_values(node: Any, page_size: int) -> List[str]:
        //     """Compute SHA256-based hash values for position-aware KV block IDs."""
        //     parent_hash = None
        //     if node.parent is not None and node.parent.hash_value is not None:
        //         if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
        //             parent_hash = node.parent.hash_value[-1]
        //
        //     hash_values = get_hash_str(node.key, parent_hash, page_size=page_size)
        //     assert isinstance(hash_values, list)
        //     return hash_values
        todo!()
    }

    /// The ancestor chain's hash values ending at `node_id`, in root-to-node
    /// order; the walk stops below the nearest never-hashed ancestor.
    pub fn prefix_hash_values(&self, node_id: Option<NodeIdx_>) -> Vec<String> {
        todo!()
    }

    /// The node's namespace; None for the default.
    pub fn node_extra_key(&self, node_id: NodeIdx_) -> Option<&str> {
        todo!()
    }

    /// The tree's single root.
    pub fn root(&self) -> NodeIdx_ {
        todo!()
    }

    /// The node's child on the page within the namespace, if any.
    pub fn child_on_page(
        &self,
        id: NodeIdx_,
        extra_key: Option<&str>,
        page: &[K::Atom],
    ) -> Option<NodeIdx_> {
        todo!()
    }

    /// The root's child on the key's first page within the namespace, if any.
    pub fn root_child(&self, extra_key: Option<&str>, page: &[K::Atom]) -> Option<NodeIdx_> {
        todo!()
    }

    /// Whether any root edge files under the namespace.
    pub fn namespace_exists(&self, extra_key: Option<&str>) -> bool {
        todo!()
    }

    /// Install `child` under `parent` on its namespaced `map_key`; returns the
    /// displaced child, if any. The key's namespace mirrors the child's.
    pub fn insert_child_edge(
        &mut self,
        parent: NodeIdx_,
        map_key: K,
        child: NodeIdx_,
    ) -> Option<NodeIdx_> {
        todo!()
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
        todo!()
    }

    /// Allocate a detached node (empty key, no parent) for the tree to wire in.
    pub fn alloc_detached(&mut self, priority: i64) -> NodeIdx_ {
        todo!()
    }

    /// Reserve an empty slot (reusing a freed one when available), returning its id.
    fn reserve(&mut self) -> NodeIdx_ {
        todo!()
    }

    /// Detach a leaf from its parent and return its slot to the freelist.
    pub fn free_leaf(&mut self, id: NodeIdx_) -> Result<(), TreeCoreRuntimeError> {
        todo!()
    }

    /// Number of live nodes.
    pub fn len(&self) -> usize {
        todo!()
    }

    /// Shared access to a live node; panics on a dead or out-of-range id.
    #[track_caller]
    pub fn node(&self, id: NodeIdx_) -> &Node<K> {
        todo!()
    }

    /// Mutable access to a live node; panics on a dead or out-of-range id.
    #[track_caller]
    pub fn node_mut(&mut self, id: NodeIdx_) -> &mut Node<K> {
        todo!()
    }

    /// The node's device value for the component; panics if unset.
    pub fn device_value(&self, id: NodeIdx_, component_type: ComponentType) -> &Tensor {
        todo!()
    }

    /// The node's device value for the component, or None when unset.
    pub fn try_device_value(&self, id: NodeIdx_, component_type: ComponentType) -> Option<&Tensor> {
        todo!()
    }

    /// Whether the node holds the component's device value.
    pub fn has_device_value(&self, id: NodeIdx_, component_type: ComponentType) -> bool {
        todo!()
    }

    /// The node's device value length for the component, or 0 when value-less.
    pub fn device_value_len(&self, id: NodeIdx_, component_type: ComponentType) -> usize {
        todo!()
    }

    /// Set the node's device value for the component; panics if already set.
    pub fn set_device_value(&mut self, id: NodeIdx_, component_type: ComponentType, value: Tensor) {
        todo!()
    }

    /// Take the node's device value for the component; panics if unset.
    pub fn take_device_value(&mut self, id: NodeIdx_, component_type: ComponentType) -> Tensor {
        todo!()
    }

    /// The node's device lock refcount for the component.
    pub fn device_lock_ref(&self, id: NodeIdx_, component_type: ComponentType) -> u32 {
        todo!()
    }

    /// The node's host value for the component; panics if unset.
    pub fn host_value(&self, id: NodeIdx_, component_type: ComponentType) -> &Tensor {
        todo!()
    }

    /// Whether the node holds the component's host value.
    pub fn has_host_value(&self, id: NodeIdx_, component_type: ComponentType) -> bool {
        todo!()
    }

    /// The node's host value length for the component, or 0 when value-less.
    pub fn host_value_len(&self, id: NodeIdx_, component_type: ComponentType) -> usize {
        todo!()
    }

    /// Set the node's host value for the component; panics if already set.
    pub fn set_host_value(&mut self, id: NodeIdx_, component_type: ComponentType, value: Tensor) {
        todo!()
    }

    /// Take the node's host value for the component; panics if unset.
    pub fn take_host_value(&mut self, id: NodeIdx_, component_type: ComponentType) -> Tensor {
        todo!()
    }

    /// The node's host lock refcount for the component.
    pub fn host_lock_ref(&self, id: NodeIdx_, component_type: ComponentType) -> u32 {
        todo!()
    }

    /// Bump the node's device lock refcount for the component.
    pub fn inc_device_lock_ref(&mut self, id: NodeIdx_, component_type: ComponentType) {
        todo!()
    }

    /// Drop the node's device lock refcount for the component; panics when unlocked.
    pub fn dec_device_lock_ref(&mut self, id: NodeIdx_, component_type: ComponentType) {
        todo!()
    }

    /// Bump the node's host lock refcount for the component.
    pub fn inc_host_lock_ref(&mut self, id: NodeIdx_, component_type: ComponentType) {
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

    /// Mutable access to a live parent/child pair at once. Internal-only accessor:
    /// panics on a dead id or when `child_node_id` is not a child of `parent_node_id`.
    #[track_caller]
    pub fn node_pair_mut(
        &mut self,
        parent_node_id: NodeIdx_,
        child_node_id: NodeIdx_,
    ) -> (&mut Node<K>, &mut Node<K>) {
        todo!()
    }

    /// Advance the access counter by `delta` ticks and return the newest one,
    /// reserving the whole range for the caller to assign.
    pub fn get_and_batch_bump_access_counter(&mut self, delta: i64) -> i64 {
        todo!()
    }

    /// Bump the access counter and return the new tick (for stamping `last_access_counter`).
    pub fn get_and_bump_access_counter(&mut self) -> i64 {
        todo!()
    }
}

// ==== Evictable leaf sets ===============================================

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
        todo!()
    }

    /// Whether `node_id` is a member.
    pub fn contains(&self, node_id: NodeIdx_) -> bool {
        todo!()
    }

    /// Insert `node_id`; no-op when already a member.
    pub fn add(&mut self, node_id: NodeIdx_) {
        todo!()
    }

    /// Remove `node_id`; no-op when not a member.
    pub fn discard(&mut self, node_id: NodeIdx_) {
        todo!()
    }

    /// The members, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = NodeIdx_> + '_ {
        std::iter::empty()
    }

    // Test-only conveniences: production callers use add/discard/contains/iter.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        todo!()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        todo!()
    }
}
