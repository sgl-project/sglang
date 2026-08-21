use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use tch::Tensor;

use crate::component_type::{ComponentType, NUM_COMPONENT_TYPES};
use crate::error::{ChildKeyError, RadixCacheInitError};
use crate::tree_node_lru::{
    evict_full_value, ComponentPoolState, EvictResult, FullLRUSlot, HostFullLRUSlot, LRUData,
    LRUSlot, MambaLRUSlot, SwaLRUSlot,
};

// TODO(Jialin): [Optimization][Major] TreeNode.children could be replaced by a more compact representation
// TODO(Jialin): [Optimization][Major] evict_leaf is the most costly operation due to TreeNode
// drop (HashMap deallocation). Reuse evicted nodes by clearing fields in place (key.clear(),
// children.clear()) instead of dropping — HashMap::clear() preserves the internal buffer,
// avoiding reallocation on the next insert into the recycled node.
// TODO(Jialin): [Optimization][Major] Add with_capacity(n) to preallocate children HashMap.
// Benchmarks show 2-3x insert speedup when resize/rehash is avoided.
// TODO(Jialin): [Optimization][Minor] Avoid child_key.to_owned if key exists in insert_child
// via hashbrown::entry_ref() which accepts borrowed keys natively.
// TODO(Jialin): [Optimization][Minor] Switch to hashbrown/ahash for faster non-cryptographic
// hashing. Rust's default SipHash is ~23ns/key at page=16 vs Python's near-zero cost hash.
// TODO(Jialin): [Cleanup] Convert remaining panics (insert_child duplicate key, evicted node
// access, split_node asserts) to proper Result types — see VacantChildSlot/SplitToken proof
// token TODOs on insert_leaf and split_node.
// TODO(Jialin): [Cleanup] Introduce compile-time invariants to prevent panics where possible.
// e.g., typed RootNodeIdx/ChildNodeIdx that can't be passed to evict_leaf/split_node, and
// would also restrict match_key to ChildNodeIdx (root has empty key — calling match_key on it
// trivially returns 0 and is meaningless).

/// Validated page size for the radix cache. Built once at construction
/// (`PageSize::new` rejects `0`); all consumers receive `PageSize` by
/// value (`Copy`) and treat it as infallible from there on.
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
/// is the borrowed counterpart of the owned key (`[Atom]` for every
/// K we ship today); kept as an associated type so external callers
/// can still spell `K::Borrowed` instead of `[K::Atom]`, and so
/// `<&K::Borrowed>::to_owned()` lands directly on `Self = K` without
/// a separate `From<Vec<Atom>>` constraint.
pub trait ChildKeyType: Hash + Eq + Sized + Borrow<Self::Borrowed> {
    type Atom: Copy + Eq + Hash;
    type Borrowed: Hash + Eq + ToOwned<Owned = Self> + ?Sized;

    /// Extract a borrowed child key from an atom slice. Each impl
    /// delegates to `page_aligned_slice` since the page-aligned-slice
    /// behavior is universal for the K instantiations we ship today;
    /// the per-impl method exists so the return type can be
    /// `&Self::Borrowed` (which equals `&[Self::Atom]` for both
    /// existing impls but is named distinctly per impl).
    fn make_child_key(
        key: &[Self::Atom],
        page_size: PageSize,
    ) -> Result<&Self::Borrowed, ChildKeyError>;
}

/// Page-aligned slice helper shared by all `ChildKeyType::make_child_key`
/// impls. Free function so we don't duplicate the body across impls.
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

/// Bigram-keyed instantiation. Each atom is a `(t[i], t[i+1])` pair;
/// callers (EAGLE today; potentially other overlap-pair workloads later)
/// build these from a raw `&[i64]` token slice via `windows(2).map(...)`.
/// Production wiring lands in a follow-up PR; this impl exists today so
/// PR-1 can characterize bench cost across both atom widths.
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
/// **Important**: `NodeIdx` should NOT be held outside of `TreeNodePool`. The pool
/// uses a freelist that recycles indices — an evicted index may be reallocated to a
/// new node (ABA problem). All state that references nodes (eviction queues, LRU
/// lists, etc.) should be owned by `TreeNodePool` or its parent `RadixCache`.
pub type NodeIdx = usize;

/// Per-component per-node state. One entry per `ComponentType`, stored in
/// `TreeNode.components: [ComponentNodeState; NUM_COMPONENT_TYPES]`.
///
/// Holds the per-component `value` (KV indices), `lock_ref` (in-flight
/// request count), and intrusive LRU links — each component carves its
/// own slice of per-node state without affecting other components.
///
/// `value: Option<Tensor>` semantics by component (per the migration
/// design's "use `Option<Tensor>` + invariants" decision):
///   * **FULL**: `Some` for all live non-root nodes; `None` only on root.
///     FULL never tombstones — when FULL evicts a leaf, the whole node
///     leaves the tree.
///   * **SWA** (when wired up): `Some` if the node sits within the
///     sliding window with live SWA KV; `None` otherwise — covers both
///     "never populated for SWA" (insert outside window from the start)
///     and "tombstoned" (window slid past). OSS treats both the same:
///     `is_tombstone = value is None`.
///   * **MAMBA** (when wired up): `Some` only on leaves with live Mamba
///     state; `None` on internal nodes (Mamba is per-leaf only,
///     `redistribute_on_node_split` nulls Mamba on prefix nodes) and on
///     root.
///
/// `lock_ref: u32` is the per-component in-flight request count.
/// `inc_lock_ref` / `dec_lock_ref` walk semantics differ per component
/// (FULL walks parents to root; SWA walks until sliding window is
/// filled; Mamba locks the leaf only).
///
/// `Default` defers to each field's default (None / 0 / all-zero
/// out-of-list link state). Neither `Copy` nor `Clone` is derived
/// (`tch::Tensor` is neither — copy/clone semantics for GPU resources
/// must be explicit). The array literal `[ComponentNodeState; N]` form
/// can't be used at construction (requires `Copy`); callers should
/// rely on the stdlib `Default` impl for `[T; N]` (`Default::default()`
/// in a struct-literal context where the field type pins the array
/// length).
#[derive(Default)]
pub struct ComponentNodeState {
    /// KV indices for this component on this node (None = no live value;
    /// for SWA that means "tombstone or never-populated", indistinguishable
    /// per OSS semantics; for FULL it's reserved for root).
    pub value: Option<Tensor>,
    /// Reference count from in-flight requests on this component's slot.
    /// While > 0 the slot is locked. Per-component locking lets each
    /// slot's `inc/dec_lock_ref` walk pattern be independent (FULL: walks
    /// parents to root; SWA: walks until window-filled; Mamba: leaf only).
    pub lock_ref: u32,
    /// Intrusive LRU state for this component's slot. Mutated only via the
    /// `LRUSlot` trait's algorithm methods (`bump_mru`, `remove`, etc.);
    /// never read or written directly outside the slot impl + the trait's
    /// default methods. Visibility is `pub` so the slot impls' default
    /// `data` / `data_mut` accessors (which live on the trait) can route
    /// through this field.
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
    ///
    /// TODO(Jialin): [Optimization][Major] Owning `Vec<K::Atom>` forces
    /// `radix_cache::insert_helper` to do `key[consumed..].to_vec()` for every
    /// new leaf — ~30 µs at n=100k. This is the dominant source of the
    /// "insert 0 match" bench regression vs Python (Python stores a
    /// `List[int]` reference without copying). The memcpy itself is
    /// memory-bandwidth bound and structural — it can't be eliminated within
    /// Rust-owned data. Eliminating it would require the cache to reference
    /// caller-owned buffers, which is a real engineering trade-off, not a
    /// free win. Deferred indefinitely pending production motivation.
    key: Vec<K::Atom>,
    /// Index of parent node in the pool. None for the root.
    parent: Option<NodeIdx>,
    /// Children keyed by child key, values are indices into the pool.
    children: HashMap<K, NodeIdx>,
    /// Device-tier per-component node-level state.
    pub(crate) components: [ComponentNodeState; NUM_COMPONENT_TYPES],
    /// Host-tier per-component node-level state.
    pub(crate) host_components: [ComponentNodeState; NUM_COMPONENT_TYPES],
    /// Monotonic hit count for a node, used by HiCache to decide if a host
    /// backup is needed.
    pub(crate) hit_count: u64,
    /// SWA's lock-walk boundary marker — `Some(uuid)` when this node
    /// is the boundary for one or more in-flight SWA acquires, `None`
    /// otherwise. Stored as a standalone field rather than per-
    /// component state because only the SWA acquire / release / split
    /// paths touch it; widening `ComponentNodeState` would pay the
    /// cost on FULL / Mamba slots that never use it.
    ///
    /// Storage today; mutation in a future PR. The field is allocated
    /// and round-trips via `swa_uuid_for_lock()` / `set_swa_uuid_for_lock`,
    /// but the production stamping (during SWA's acquire walk) and
    /// `_split_node` transfer-to-new-parent logic land with the SWA
    /// acquire/release PR. `split_node` does NOT yet move this field
    /// to the new intermediate parent.
    pub(crate) swa_uuid_for_lock: Option<u64>,
    /// Count of children holding a FULL device value. `is_evictable` treats a
    /// node with none as an (effective) leaf, regardless of whether HiCache is
    /// enabled.
    pub(crate) num_children_with_device_full: u32,
}

impl<K: ChildKeyType> TreeNode<K> {
    /// Construct a root node — no parent, empty key, no value. Used for the
    /// default namespace root and lazy-created `extra_key` namespace roots.
    /// FULL's `lock_ref` starts at 1 so the root is permanently protected
    /// (mirrors Python `RadixCache.reset()`'s `root_node.lock_ref = 1`).
    /// All other per-component fields default (value=None, lock_ref=0,
    /// LRU links not-in-list). Roots are never inserted into any LRU
    /// list (always locked, never evictable), so the link fields stay
    /// untouched after construction.
    pub fn new_root() -> Self {
        let mut components: [ComponentNodeState; NUM_COMPONENT_TYPES] = Default::default();
        components[ComponentType::Full as usize].lock_ref = 1;
        Self {
            key: Vec::new(),
            parent: None,
            children: HashMap::new(),
            components,
            host_components: Default::default(),
            hit_count: 0,
            swa_uuid_for_lock: None,
            num_children_with_device_full: 0,
        }
    }

    /// Construct a non-root node with a known parent. `parent_idx: NodeIdx`
    /// (not `Option`) encodes the invariant that children always have parents.
    /// `value` is `Option<Tensor>` because `split_node`'s transient intermediate
    /// and evicted children legitimately carry no value. The supplied `value`
    /// goes into FULL's slot (FULL is the canonical KV store); SWA/MAMBA
    /// values are populated separately by their components' `commit_insert_*`
    /// hooks if applicable. FULL's `lock_ref` starts at 0; the high-level
    /// pool ops (`insert_leaf`, `split_node`) splice the node into FULL's
    /// LRU via `FullLRUSlot::bump_mru` immediately after `alloc`, so the
    /// initial link values are never observed.
    pub fn new_child(key: Vec<K::Atom>, parent_idx: NodeIdx, value: Option<Tensor>) -> Self {
        let mut components: [ComponentNodeState; NUM_COMPONENT_TYPES] = Default::default();
        components[ComponentType::Full as usize].value = value;
        Self {
            key,
            parent: Some(parent_idx),
            children: HashMap::new(),
            components,
            host_components: Default::default(),
            hit_count: 0,
            swa_uuid_for_lock: None,
            num_children_with_device_full: 0,
        }
    }

    /// Construct a sentinel node — empty, never reachable through tree
    /// traversal. The sentinel slots per LRU are allocated at pool
    /// construction (and on `reset`) and never freed. The corresponding
    /// component's `lru_data` links are overwritten by `LRUSlot::init`
    /// immediately after `alloc`, cross-linking head <-> tail; the other
    /// components' slots stay in the default not-in-list state. All
    /// per-component value/lock_ref fields default (None / 0).
    pub fn new_sentinel() -> Self {
        Self {
            key: Vec::new(),
            parent: None,
            children: HashMap::new(),
            components: Default::default(),
            host_components: Default::default(),
            hit_count: 0,
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

    /// FULL's KV cache slot indices for this segment. None for root /
    /// evicted. (FULL-specific accessor for back-compat with callers that
    /// pre-date the per-component refactor; SWA/MAMBA value access goes
    /// through `node.components[ct].value` directly.)
    pub fn value(&self) -> Option<&Tensor> {
        self.components[ComponentType::Full as usize].value.as_ref()
    }

    /// Number of children.
    pub fn num_children(&self) -> usize {
        self.children.len()
    }

    /// True iff this node has no children — i.e., it's a leaf in the
    /// tree. Leaves are FULL-evictable (subject to the FULL lock_ref
    /// gate); internal nodes are SWA-tombstoneable but not
    /// FULL-evictable until they become leaves.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// FULL's lock-ref count: 0 means unlocked (eligible for eviction
    /// subject to leaf/value conditions), > 0 means in use by an in-flight
    /// request. SWA/MAMBA's lock_ref live in their own component slots and
    /// are accessed via `node.components[ct].lock_ref` (different walk
    /// patterns: SWA window-bounded, Mamba leaf-only).
    pub fn lock_ref(&self) -> u32 {
        self.components[ComponentType::Full as usize].lock_ref
    }

    /// True iff any component's `lock_ref > 0` on this node. Read-only
    /// helper used by `evict_leaf` as a defense-in-depth assertion at
    /// the actual delete site — the FULL/other invariant means
    /// `full_lock_ref == 0` already implies the others are 0, so this
    /// catches a future invariant break as a panic instead of a silent
    /// use-after-free of an orphaned non-FULL value.
    pub fn is_locked(&self) -> bool {
        self.components.iter().any(|c| c.lock_ref > 0)
    }

    /// SWA's lock-walk boundary uuid for this node — `Some` when this
    /// node is the boundary for one or more in-flight SWA acquires,
    /// `None` otherwise. See the field doc on `swa_uuid_for_lock` for
    /// the (planned) lifecycle; today this just round-trips with the
    /// setter, with no production write site wired up yet.
    pub fn swa_uuid_for_lock(&self) -> Option<u64> {
        self.swa_uuid_for_lock
    }

    /// Set this node's SWA lock-walk boundary uuid. `Some(uuid)` stamps;
    /// `None` clears (used during node-split to transfer the marker to
    /// the new parent). Production callers go through
    /// `TreeNodePool::lazy_acquire_swa_uuid_for_lock` for the get-or-mint-stamp
    /// flow; this raw setter is for split-node transfer (which moves an
    /// existing uuid wholesale) and tests.
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

    /// Insert a child index. Panics if the key already exists.
    /// Takes the key by value to avoid double-allocation: callers that already
    /// own a `K` (e.g., `insert_leaf` after `to_owned()`) move it straight in;
    /// callers with only a `&K::Borrowed` must `to_owned()` themselves.
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
    ///
    /// TODO(Jialin): [Cleanup] Restrict to ChildNodeIdx via the typed-index
    /// invariant tracked at the top of this file. Calling match_key on the root
    /// (empty key) trivially returns 0 — making it unrepresentable removes a
    /// degenerate code path and a category of nonsense calls from callers.
    pub fn match_key(&self, key: &[K::Atom], page_size: PageSize) -> usize {
        let ps = page_size.get();
        let max_pages = self.key.len().min(key.len()) / ps;
        if max_pages == 0 {
            return 0;
        }

        // Compare page-aligned chunks instead of counting token-by-token. Long
        // context cache hits are usually full-node matches; slice equality lets
        // LLVM/libcore use the optimized monomorphic comparison path for the
        // concrete atom type while preserving the same page-rounded result.
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

/// Outcome of `TreeNodePool::match_child` — encodes the partial-vs-full
/// distinction in the type so callers don't need to compare prefix_len
/// against child.key().len() themselves.
pub enum MatchChildResult {
    /// `parent_idx`'s children HashMap has no entry for `query_key`'s first page.
    NotFound,
    /// All of the matched child's stored key matched. Caller may walk into
    /// `child_idx`. `node_key_len == child.key().len()`.
    FullMatch {
        child_idx: NodeIdx,
        node_key_len: usize,
    },
    /// Only a prefix of the child's key matched. Caller must call
    /// `split_node(node_split)` before continuing (or stop).
    PartialMatch(NodeSplit),
}

/// Represents a valid node split: `child_idx` is a non-root node with a
/// parent, and `split_len` is in `[page_size, child.key().len() - page_size]`
/// and aligned to `page_size`. The only public path to construct one is
/// `match_child` returning `PartialMatch`, which validates these invariants.
/// Crate-internal test wrappers may construct via struct literal to
/// exercise `split_node`'s defensive asserts on intentionally invalid input.
pub struct NodeSplit {
    pub(crate) child_idx: NodeIdx,
    pub(crate) split_len: usize,
}

impl NodeSplit {
    pub fn split_len(&self) -> usize {
        self.split_len
    }
}

/// Arena-based pool that owns all tree nodes.
///
/// Evicted slots are recycled via a freelist. Access a node by its `NodeIdx`.
///
/// Per-component `ComponentPoolState`s (sentinel pair + size aggregates)
/// live here so the high-level pool ops (`insert_leaf`, `split_node`,
/// `evict_leaf`) can splice/unsplice nodes in lockstep with tree
/// mutations and keep the size aggregates in sync. Sentinel slots are allocated at
/// construction (and `reset`) and never freed; `active_node_count()`
/// excludes them so callers see the same node-count semantics as before.
pub struct TreeNodePool<K: ChildKeyType> {
    nodes: Vec<Option<TreeNode<K>>>,
    evicted_indices: Vec<NodeIdx>,
    page_size: PageSize,
    /// Device-tier per-component pool-level state.
    pub(crate) components: [ComponentPoolState; NUM_COMPONENT_TYPES],
    /// Host-tier per-component pool-level state.
    pub(crate) host_components: [ComponentPoolState; NUM_COMPONENT_TYPES],
    /// How many sentinel slots have been allocated for this pool.
    /// Bumped by `alloc_sentinel` (one call per sentinel during
    /// `LRUSlot::init`); excluded from `active_node_count` so
    /// callers see only user-visible nodes. Stored as a field rather than
    /// a const so adding new LRU instances (SWA, mamba) doesn't require
    /// touching this struct — each new slot's `init` just
    /// auto-bumps the counter as it allocates its pair.
    sentinel_count: usize,
    /// Monotonic counter for SWA's lock-walk boundary uuids. Each SWA
    /// acquire that newly-stamps a boundary node consumes one id via
    /// `acquire_next_swa_uuid_for_lock`; acquires that hit an already-stamped
    /// boundary reuse the existing one (orchestrated by
    /// `lazy_acquire_swa_uuid_for_lock`). Starts at 1 so `0` stays
    /// distinguishable from default-zero memory. Plain `u64` is
    /// sufficient — the radix cache is single-threaded (PyO3 GIL
    /// serializes all entrypoints).
    next_swa_uuid_for_lock: u64,
    /// Whether SWA is configured for this cache. Set at construction
    /// from the `sliding_window_size: Option<usize>` constructor arg
    /// — `Some(_)` means SWA is configured, `None` means FULL-only.
    ///
    /// Used as a discriminator for the FULL-evict tombstone cascade:
    /// in FULL-only mode, NO node ever has SWA value populated, so
    /// `iteratively_delete_tombstone_leaf`'s `has_value(SwaLRUSlot)`
    /// stop condition would treat every walked ancestor as a
    /// "tombstone" and over-delete. Gating the cascade on
    /// `has_swa_component` short-circuits it for FULL-only configs
    /// where tombstones can't exist by construction.
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
            host_components: [ComponentPoolState::default(); NUM_COMPONENT_TYPES],
            sentinel_count: 0,
            // Start at 1 so 0 is distinguishable from the default-zero
            // state of uninitialized memory.
            next_swa_uuid_for_lock: 1,
            has_swa_component,
            has_mamba_component,
        };
        // Initialize LRU slots.
        FullLRUSlot::init::<K>(&mut pool);
        HostFullLRUSlot::init::<K>(&mut pool);
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

    /// Lazily acquire the SWA lock-walk boundary uuid for `node_idx`
    /// — "lazy" in the sense that the counter advances only when
    /// needed: if the node already has a uuid stamped, return the
    /// existing one (concurrent-acquire reuse path — multiple
    /// in-flight acquires hitting the same boundary share the marker,
    /// mirrors OSS `swa_radix_cache.py`'s `inc_lock_ref` `if
    /// node.swa_uuid is None` branch). Only when no stamp exists does
    /// this mint a fresh uuid via `acquire_next_swa_uuid_for_lock`,
    /// stamp it on the node, and return it.
    ///
    /// Single entry point for the SWA acquire path — collapses the
    /// "read existing → mint → stamp" three-step into one call. The
    /// orchestration lives on the pool because both the counter
    /// (`next_swa_uuid_for_lock`) and the node storage are pool
    /// fields; the `Vec<Option<TreeNode>>` indirection on `nodes`
    /// blocks split-borrow via direct field access from callers.
    pub fn lazy_acquire_swa_uuid_for_lock(&mut self, node_idx: NodeIdx) -> u64 {
        if let Some(existing) = self.get(node_idx).swa_uuid_for_lock() {
            return existing;
        }
        let new_uuid = self.acquire_next_swa_uuid_for_lock();
        self.get_mut(node_idx).set_swa_uuid_for_lock(Some(new_uuid));
        new_uuid
    }

    /// Mint the next SWA lock-walk uuid from the monotonic counter.
    /// Always advances the counter — for the "stamp on a node only if
    /// not already stamped" semantic use `lazy_acquire_swa_uuid_for_lock`
    /// instead. Visible to the test wrapper so PR-21/n's
    /// counter-property tests can exercise monotonicity / start-at-1
    /// directly.
    ///
    /// Uses `checked_add` so a long-lived process that exhausts the
    /// `u64` counter (~1.8e19 distinct boundary stampings) panics
    /// loudly instead of silently wrapping and reissuing a uuid that
    /// was already in use — wrapping would break the boundary-uuid
    /// uniqueness guarantee that SWA's release-walk relies on.
    /// Compute → validate → commit ordering: counter advances ONLY
    /// after the overflow check passes; on panic the counter stays
    /// at its current value so the caller doesn't observe a stale id.
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
            // Snapshot value_len + lock_ref BEFORE the remove so the
            // counter debit can pick the right side of the
            // evictable/protected split (same rule as `lock` / `unlock`
            // use on the 0↔1 lock_ref transitions).
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
        // Bookkeeping updates after inserting the leaf.
        let value_len = self.get(child_idx).key().len();
        FullLRUSlot::bump_mru(self, child_idx);
        FullLRUSlot::pool_state_mut(self).unlocked_size += value_len;
        // After connecting the child, invoke postprocess_set_value so the
        // parent's bookkeeping stays accurate.
        if FullLRUSlot::has_value(self.get(child_idx)) {
            FullLRUSlot::postprocess_set_value(self, parent_idx);
        }
        Ok(child_idx)
    }

    /// Combined "find child + match key" — returns a `MatchChildResult` that
    /// callers can `match` exhaustively. Replaces the manual
    /// `parent.get_child` + `child.match_key` + `child.key().len()` dance.
    /// `&self` (read-only) so callers can re-acquire the pool mutably for
    /// `split_node` in the `PartialMatch` arm.
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

    /// Split a node at `split_len`, inserting a new intermediate parent.
    ///
    /// Before: `parent -[child_key]-> node(key=ABCD, children={...})`
    /// After:  `parent -[child_key]-> new_node(key=AB) -[suffix_key]-> node(key=CD, children={...})`
    ///
    /// Returns the index of the newly created intermediate node. Infallible
    /// when `split` was obtained from `match_child`. The single defensive
    /// alignment assert below catches `NodeSplit` instances constructed by
    /// in-crate test wrappers via struct literal with intentionally invalid
    /// input — without it, an unaligned `split_len` would silently corrupt
    /// the tree (downstream `make_child_key` only checks length, not
    /// alignment). Length-bound violations (split_len out of range or root
    /// node) still panic but via downstream `expect`s / `Vec::split_off`'s
    /// own bounds check, after partial mutation.
    ///
    /// TODO(Jialin): [Optimization][Minor] Collapse the two HashMap touches
    /// on `parent.children[child_key]` (the get_child during walk + the
    /// replace_child here) into a single op by holding an
    /// `OccupiedEntry`/bucket handle inside `NodeSplit`. Saves ~15-30ns per
    /// partial-match iteration. Held off because the lifetime cost (token
    /// grows `<'a>`, leaks through every API that accepts/returns it)
    /// outweighed the win at ship time. Worth revisiting alongside the
    /// `TreeNode.children: HashMap -> small-fanout Vec` TODO near the
    /// top of this file — that refactor changes the shape of "the entry
    /// handle" anyway (Vec index instead of HashMap entry).
    pub fn split_node(
        &mut self,
        components: &[Box<dyn crate::components::Component<K>>],
        split: NodeSplit,
    ) -> NodeIdx {
        let node_idx = split.child_idx;
        let split_len = split.split_len;
        let ps = self.page_size.get();
        // Extract parent_idx in a block so the immutable borrow on
        // self.get(node_idx) is dropped before we call self.get_mut() later.
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
        // Per-component concerns — value redistribution, lock_ref copy,
        // per-component LRU bump, identity-marker transfer (e.g. SWA's
        // `swa_uuid_for_lock`) — all live in each component's
        // `Component::redistribute_on_node_split` hook, dispatched at
        // the end of this method. Even FullComponent owns its own FULL
        // value/lock_ref/LRU work this way (see `FullComponent::
        // redistribute_on_node_split`); this keeps pool.split_node
        // decoupled from per-component value semantics.
        let page_size = self.page_size;
        let mut prefix_key = std::mem::take(&mut self.get_mut(node_idx).key);
        let suffix_key = prefix_key.split_off(split_len);

        // Allocate new intermediate. `alloc` does not touch the LRU; the
        // per-component LRU bump (via redistribute dispatch below)
        // handles each component's slot uniformly.
        let new_node_idx = self.alloc(TreeNode::new_child(Vec::new(), parent_idx, None));

        // Re-parent original node.
        self.get_mut(node_idx).parent = Some(new_node_idx);

        // Replace in parent's children — borrows from local prefix_key (zero alloc).
        // Infallible for NodeSplits from match_child (split_len >= page_size).
        // Test wrappers passing split_len < page_size panic here, mid-mutation.
        #[allow(
            clippy::expect_used,
            reason = "match_child guarantees split_len >= page_size"
        )]
        let original_child_key = K::make_child_key(&prefix_key, page_size)
            .expect("split_node: split_len < page_size — prefix has fewer than page_size tokens");
        self.get_mut(parent_idx)
            .replace_child(original_child_key, new_node_idx);

        // Add original as child of new node. `to_owned()` is unavoidable here
        // because `suffix_key` is needed below as the original node's new key.
        // Infallible for NodeSplits from match_child (split_len <= key_len - page_size).
        // Test wrappers passing out-of-range split_len panic here, mid-mutation.
        #[allow(
            clippy::expect_used,
            reason = "match_child guarantees split_len <= key_len - page_size"
        )]
        let suffix_child_key = K::make_child_key(&suffix_key, page_size)
            .expect("split_node: split_len out of range — suffix has fewer than page_size tokens")
            .to_owned();
        self.get_mut(new_node_idx)
            .insert_child(suffix_child_key, node_idx);

        // Set keys in their final destinations.
        self.get_mut(new_node_idx).key = prefix_key;
        self.get_mut(node_idx).key = suffix_key;

        // Per-component redistribute dispatch. Each configured component's
        // `redistribute_on_node_split` hook handles its own per-node
        // value / lock_ref / LRU / identity-marker fixups across the split
        // boundary. FullComponent's hook is required (no default) and
        // handles the FULL baseline — value slice, lock_ref copy, LRU
        // bump. SwaComponent (and future Mamba) handle their own analogs.
        //
        // Test wrappers that pass `&[]` here intentionally get
        // structural-only splits (no value/lock_ref/LRU work) — useful
        // for benching the pure structural primitive in isolation. Tests
        // that need FULL behavior post-split must pass FullComponent.
        for comp in components {
            comp.redistribute_on_node_split(self, new_node_idx, node_idx, split_len);
        }

        // Update the new intermediate's full-device-child count: 1 iff the
        // original node (now its only child) kept a FULL device value. Other
        // node counters are preserved.
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
        // TODO(Jialin): pass the configured components along so this
        // can iterate generically instead of hard-coding SWA + Mamba
        // snapshots + unlinks.
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

    /// Whether SWA was configured for this pool's parent
    /// `RadixCache`. Used by `FullLRUSlot::evict` to gate the
    /// tombstone-cascade cleanup — see the field doc.
    pub fn has_swa_component(&self) -> bool {
        self.has_swa_component
    }

    /// Whether Mamba was configured for this pool's parent
    /// `RadixCache`. Used by `FullLRUSlot::evict` to gate the
    /// Mamba tombstone-cascade cleanup — see the field doc.
    pub fn has_mamba_component(&self) -> bool {
        self.has_mamba_component
    }

    /// Total number of active (non-evicted) nodes in the pool, excluding
    /// LRU sentinel slots so callers see the same node-count semantics they
    /// did before sentinels existed.
    pub fn active_node_count(&self) -> usize {
        self.nodes.len() - self.evicted_indices.len() - self.sentinel_count
    }
}

/// Production tree node: children keyed by token page (`Vec<i64>`).
/// Handles `page_size >= 1` (page_size=1 uses one-element page keys).
pub type PageTreeNode = TreeNode<Vec<i64>>;

/// Production tree node pool: children keyed by token page (`Vec<i64>`).
pub type PageTreeNodePool = TreeNodePool<Vec<i64>>;
