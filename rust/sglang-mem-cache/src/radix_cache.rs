use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

use crate::components::{Component, FullComponent, IncLockRefResult, MambaComponent, SwaComponent};
use crate::deferred_action::DeferredAction;
use crate::error::{RadixCacheInitError, RadixCacheRuntimeError};
use crate::tree_node_lru::{
    EvictRequest, EvictResult, FullLRUSlot, LRUSlot, MambaLRUSlot, SwaLRUSlot,
};
use crate::tree_node_pool::{
    ChildKeyType, MatchChildResult, NodeIdx, PageSize, TreeNode, TreeNodePool,
};

/// Result of prefix match.
pub struct MatchResult {
    pub device_indices: Tensor,
    pub last_device_node_idx: NodeIdx,
    /// Chunk-aligned position past the cached prefix where prefill may
    /// snapshot SSM state to repair a tombstoned chunk boundary.
    pub mamba_branching_seqlen: Option<usize>,
    /// Cached Mamba state at `last_device_node_idx`; `None` on cache miss.
    pub mamba_value: Option<Tensor>,
}

pub struct InsertResult {
    /// The number of tokens in the insert key that matched existing nodes.
    pub prefix_len: usize,
    /// True if leaf creation was skipped (e.g. SWA vetoed because the
    /// suffix is outside the window).
    pub leaf_creation_skipped: bool,
    /// Indicates whether the cache has taken ownership of the
    /// `mamba_value` (`false`), or the caller should free it (`true`).
    pub mamba_value_exists: bool,
    /// KV cache allocator pool actions.
    pub deferred_actions: Vec<DeferredAction>,
}

/// Borrowed query key proven page-aligned and non-empty, so walk helpers
/// can skip alignment/empty checks. Atom-generic over single-token and
/// bigram instantiations.
struct PageAlignedQueryKey<'a, A> {
    key: &'a [A],
}

impl<'a, A> PageAlignedQueryKey<'a, A> {
    /// Returns `None` if the page-aligned length is 0 (empty input or
    /// `key.len() < page_size`). Caller should early-return on `None`.
    fn new(key: &'a [A], page_size: PageSize) -> Option<Self> {
        let ps = page_size.get();
        let aligned_len = key.len() / ps * ps;
        if aligned_len == 0 {
            None
        } else {
            Some(Self {
                key: &key[..aligned_len],
            })
        }
    }

    fn as_slice(&self) -> &[A] {
        self.key
    }
}

/// Build the tree's component list from the cache config. Currently
/// supported combinations: Full Attention; Full Attention + Sliding Window
/// Attention; Full Attention + Mamba (Linear Attention).
fn build_components<K: ChildKeyType>(
    sliding_window_size: Option<usize>,
    mamba_cache_chunk_size: Option<usize>,
    page_size: usize,
) -> Result<Vec<Box<dyn Component<K>>>, RadixCacheInitError> {
    if sliding_window_size.is_some() && mamba_cache_chunk_size.is_some() {
        return Err(RadixCacheInitError::SwaMambaComboNotSupported);
    }
    let mut components: Vec<Box<dyn Component<K>>> = Vec::new();
    components.push(Box::new(FullComponent::new()));
    if let Some(window) = sliding_window_size {
        components.push(Box::new(SwaComponent::new(window)?));
    }
    if let Some(chunk_size) = mamba_cache_chunk_size {
        components.push(Box::new(MambaComponent::new(chunk_size, page_size)?));
    }
    Ok(components)
}

/// Radix tree-based KV cache, generic over the child key type:
///   - `RadixCache<i64>`     for `page_size = 1` (one token per edge segment)
///   - `RadixCache<Vec<i64>>` for `page_size > 1` (token page per edge segment)
pub struct RadixCache<K: ChildKeyType> {
    /// Arena that owns all tree nodes; recycles slots via a freelist.
    tree_node_pool: TreeNodePool<K>,

    /// Root for queries with `extra_key = None`. Always present; re-allocated
    /// on `reset()`.
    default_root: NodeIdx,

    /// Root per `extra_key`, allocated lazily by `insert` on first use of a
    /// namespace.
    ///
    /// TODO(Jialin): drop entries when their subtree empties via eviction —
    /// otherwise transient extra_keys leak nodes over time.
    named_roots: HashMap<String, NodeIdx>,

    /// Validated once at construction; reused on every `reset()` rebuild and
    /// exposed via `page_size()`.
    page_size: PageSize,

    /// Preserved so `reset()` can rebuild the pool with the same capacity.
    init_node_capacity: usize,

    /// Empty Int64 tensor on the configured device, shallow-cloned on cache
    /// miss to avoid per-call allocation.
    empty_tensor: Tensor,

    /// Configured components (FULL always present; SWA/Mamba iff enabled).
    /// Iterating yields the validator chain that gates the match boundary and
    /// the per-component insert/evict/lock dispatch. Hot-path per-node LRU ops
    /// bypass this vec and stay statically dispatched on `*LRUSlot` markers.
    components: Vec<Box<dyn Component<K>>>,

    /// Whether SWA is configured. Stored separately from `components` to avoid
    /// iterating + downcasting, and so `reset()` can pass it to the rebuilt pool.
    has_swa_component: bool,

    /// Whether Mamba is configured. Parallel to `has_swa_component`.
    has_mamba_component: bool,

    /// Mamba chunk size; state checkpoints are saved only at multiples of it.
    mamba_cache_chunk_size: Option<usize>,
}

impl<K: ChildKeyType> RadixCache<K> {
    /// Construct a radix cache from per-cache config.
    ///
    /// - `init_node_capacity`: initial size of the tree node pool;
    /// - `sliding_window_size`: pass the per-token SWA window to enable
    ///   Sliding Window Attention;
    /// - `mamba_cache_chunk_size`: pass the SSM checkpoint chunk size to
    ///   enable Mamba (Linear Attention);
    pub fn new(
        device: Device,
        page_size: usize,
        init_node_capacity: usize,
        sliding_window_size: Option<usize>,
        mamba_cache_chunk_size: Option<usize>,
    ) -> Result<Self, RadixCacheInitError> {
        let page_size = PageSize::new(page_size)?;
        let has_swa_component = sliding_window_size.is_some();
        let has_mamba_component = mamba_cache_chunk_size.is_some();
        let components =
            build_components(sliding_window_size, mamba_cache_chunk_size, page_size.get())?;
        let mut tree_node_pool = TreeNodePool::<K>::new(
            page_size,
            init_node_capacity,
            has_swa_component,
            has_mamba_component,
        );
        let default_root = tree_node_pool.alloc(TreeNode::new_root());
        let empty_tensor = Tensor::empty([0], (Kind::Int64, device));
        Ok(Self {
            tree_node_pool,
            default_root,
            named_roots: HashMap::new(),
            page_size,
            init_node_capacity,
            empty_tensor,
            components,
            has_swa_component,
            has_mamba_component,
            mamba_cache_chunk_size,
        })
    }

    /// Recreate a new empty Radix Cache.
    pub fn reset(&mut self) {
        // Infallible: page_size was validated at construction.
        let mut new_tree_node_pool = TreeNodePool::<K>::new(
            self.page_size,
            self.init_node_capacity,
            self.has_swa_component,
            self.has_mamba_component,
        );
        let new_default_root = new_tree_node_pool.alloc(TreeNode::new_root());
        self.tree_node_pool = new_tree_node_pool;
        self.default_root = new_default_root;
        self.named_roots.clear();
    }

    fn empty_match_result(&self, last_device_node_idx: NodeIdx) -> MatchResult {
        MatchResult {
            device_indices: self.empty_tensor.shallow_clone(),
            last_device_node_idx,
            mamba_branching_seqlen: None,
            mamba_value: None,
        }
    }

    /// Resolve the namespace root for `extra_key`, lazily creating it on first
    /// use, so the returned node always lives in the query's namespace.
    fn get_or_create_root(&mut self, extra_key: Option<&str>) -> NodeIdx {
        match extra_key {
            None => self.default_root,
            Some(ek) => {
                if let Some(&idx) = self.named_roots.get(ek) {
                    idx
                } else {
                    let new_root = self.tree_node_pool.alloc(TreeNode::new_root());
                    self.named_roots.insert(ek.to_owned(), new_root);
                    new_root
                }
            }
        }
    }

    /// Find the longest cached prefix of `key` in the namespace selected by
    /// `extra_key` (default if `None`), splitting any node where the match ends
    /// mid-node and lazily creating the namespace root for an unseen `extra_key`.
    ///
    /// The validator-approved path is bumped to MRU on every component
    /// afterward. The returned boundary only advances past nodes that ALL
    /// component validators approve:
    /// - FULL: always approves.
    /// - SWA: gates on the contiguous-present run reaching `sliding_window_size`.
    /// With no approved boundary the result is empty and the boundary is root.
    pub fn match_prefix(
        &mut self,
        key: &[K::Atom],
        extra_key: Option<&str>,
    ) -> Result<MatchResult, RadixCacheRuntimeError> {
        let root = self.get_or_create_root(extra_key);
        let aligned_key = match PageAlignedQueryKey::new(key, self.page_size) {
            Some(k) => k,
            None => return Ok(self.empty_match_result(root)),
        };
        self.match_prefix_helper(root, aligned_key)
    }

    /// Prefix match from `root` along `key`, splitting any node where the
    /// match ends mid-key.
    fn match_prefix_helper(
        &mut self,
        root: NodeIdx,
        key: PageAlignedQueryKey<K::Atom>,
    ) -> Result<MatchResult, RadixCacheRuntimeError> {
        let key = key.as_slice();
        let key_len = key.len();
        let mut node_idx = root;
        let mut consumed = 0usize;
        let mut values: Vec<Tensor> = Vec::new();

        // Stateful per-node match validators. Mainly SWA: after walking through
        // tombstones, the matched length must reach the sliding window size.
        let mut validators: Vec<Box<dyn crate::components::MatchValidator<K>>> = self
            .components
            .iter()
            .filter_map(|c| c.create_match_validator())
            .collect();

        let mut last_matched_node_idx = root;
        let mut last_device_node_idx = root;
        let mut last_device_value_len: usize = 0;

        while consumed < key_len {
            let remaining_key = &key[consumed..];
            // One walk step: no match, full match, or split on partial match.
            let (matched_node_idx, terminated) =
                match self.tree_node_pool.match_child(node_idx, remaining_key) {
                    MatchChildResult::NotFound => break,
                    MatchChildResult::FullMatch {
                        child_idx,
                        node_key_len,
                    } => {
                        consumed += node_key_len;
                        (child_idx, false)
                    }
                    MatchChildResult::PartialMatch(node_split) => (
                        self.tree_node_pool.split_node(&self.components, node_split),
                        true,
                    ),
                };

            // A device-absent node ends the device-present prefix.
            let matched_node = self.tree_node_pool.get(matched_node_idx);
            let Some(device_value) = matched_node.value() else {
                break;
            };
            values.push(device_value.shallow_clone());
            node_idx = matched_node_idx;

            let mut all_valid = true;
            // Validators are stateful: run all of them, no short-circuit.
            for v in validators.iter_mut() {
                all_valid &= v.validate(matched_node);
            }
            if all_valid {
                last_matched_node_idx = matched_node_idx;
                last_device_value_len = values.len();
                last_device_node_idx = matched_node_idx;
            }
            if terminated {
                break;
            }
        }

        self.bump_mru_walk(last_matched_node_idx);

        // Device values up to the validator-approved boundary.
        let device_indices = if last_device_value_len == 0 {
            self.empty_tensor.shallow_clone()
        } else {
            Tensor::cat(&values[..last_device_value_len], 0)
        };

        let (mamba_branching_seqlen, mamba_value) = match self.mamba_cache_chunk_size {
            Some(chunk_size) => {
                // Populated only on a partial match (walk extended past the
                // validator-approved boundary).
                let branching_seqlen = if last_device_value_len < values.len() {
                    let total: usize = values.iter().map(|v| v.size()[0] as usize).sum();
                    let aligned = total / chunk_size * chunk_size;
                    (aligned > 0).then_some(aligned)
                } else {
                    None
                };
                let mv = MambaLRUSlot::value(self.tree_node_pool.get(last_matched_node_idx))
                    .map(|t| t.shallow_clone());
                (branching_seqlen, mv)
            }
            None => (None, None),
        };

        Ok(MatchResult {
            device_indices,
            last_device_node_idx,
            mamba_branching_seqlen,
            mamba_value,
        })
    }

    /// Insert `(key, value)` into the namespace selected by `extra_key`
    /// (default if `None`). `value` is a 1-D `Int64` tensor of KV slot indices,
    /// at least page-aligned-key-length long (excess is truncated). Inputs are
    /// validated before any tree mutation, so a bad call never half-modifies the
    /// tree. The stored slice is deep-copied; callers may mutate/drop `value`
    /// after this returns.
    ///
    /// Returns the prefix length already cached before this insert; the caller
    /// frees `value[:prefix_len]` as redundant duplicates.
    pub fn insert(
        &mut self,
        key: &[K::Atom],
        value: &Tensor,
        extra_key: Option<&str>,
        prev_prefix_len: usize,
        swa_evicted_seqlen: usize,
        mamba_value: Option<Tensor>,
    ) -> Result<InsertResult, RadixCacheRuntimeError> {
        self.validate_insert_value(key.len(), value, mamba_value.as_ref())?;
        let aligned_key = match PageAlignedQueryKey::new(key, self.page_size) {
            Some(k) => k,
            None => {
                return Ok(InsertResult {
                    prefix_len: 0,
                    leaf_creation_skipped: false,
                    // Empty key: ownership NOT taken; caller should free mamba_value.
                    mamba_value_exists: true,
                    deferred_actions: Vec::new(),
                });
            }
        };
        let root = self.get_or_create_root(extra_key);
        self.insert_helper(
            root,
            aligned_key,
            value,
            prev_prefix_len,
            swa_evicted_seqlen,
            mamba_value,
        )
    }

    /// Validate insert inputs.
    fn validate_insert_value(
        &self,
        key_len: usize,
        value: &Tensor,
        mamba_value: Option<&Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        if value.kind() != Kind::Int64 {
            return Err(RadixCacheRuntimeError::InsertValueWrongDtype { got: value.kind() });
        }
        let shape = value.size();
        if shape.len() != 1 {
            return Err(RadixCacheRuntimeError::InsertValueWrongShape { got: shape });
        }
        let value_len = shape[0] as usize;
        let ps = self.page_size.get();
        let aligned_key_len = key_len / ps * ps;
        if value_len < aligned_key_len {
            return Err(RadixCacheRuntimeError::InsertValueLengthMismatch {
                aligned_key_len,
                value_len,
            });
        }
        let cache_device = self.empty_tensor.device();
        let value_device = value.device();
        if value_device != cache_device {
            return Err(RadixCacheRuntimeError::InsertValueWrongDevice {
                expected: cache_device,
                got: value_device,
            });
        }
        if let Some(m) = mamba_value {
            if self.mamba_cache_chunk_size.is_none() {
                return Err(RadixCacheRuntimeError::InsertMambaValueWithoutMambaConfigured);
            }
            if m.kind() != Kind::Int64 {
                return Err(RadixCacheRuntimeError::InsertMambaValueWrongDtype { got: m.kind() });
            }
            let mshape = m.size();
            if mshape.len() != 1 || mshape[0] != 1 {
                return Err(RadixCacheRuntimeError::InsertMambaValueWrongShape { got: mshape });
            }
            let mamba_device = m.device();
            if mamba_device != cache_device {
                return Err(RadixCacheRuntimeError::InsertMambaValueWrongDevice {
                    expected: cache_device,
                    got: mamba_device,
                });
            }
        }
        Ok(())
    }

    /// Bring `node_idx` and its ancestors to MRU in every component's LRU.
    /// SWA tombstones are skipped at the slot level via `in_list` gating.
    fn bump_mru_walk(&mut self, node_idx: NodeIdx) {
        for comp in self.components.iter() {
            comp.bump_mru_walk(&mut self.tree_node_pool, node_idx);
        }
    }

    /// Take `min(consumed_from)` across components for one overlap node — any
    /// component claiming a slot vetoes its freeing as a duplicate. Default is
    /// `node_key_len` (claim nothing); SWA overrides for tombstone recovery.
    #[allow(clippy::too_many_arguments)]
    fn consume_value(
        &mut self,
        child_idx: NodeIdx,
        node_key_len: usize,
        total_prefix_len: usize,
        prev_prefix_len: usize,
        value_slice: &Tensor,
        swa_evicted_seqlen: usize,
        deferred: &mut Vec<DeferredAction>,
    ) -> Result<usize, RadixCacheRuntimeError> {
        let mut consumed_from = node_key_len;
        for comp in self.components.iter() {
            let comp_consumed = comp.consume_value(
                &mut self.tree_node_pool,
                &self.components,
                child_idx,
                node_key_len,
                total_prefix_len,
                prev_prefix_len,
                value_slice,
                swa_evicted_seqlen,
                deferred,
            )?;
            consumed_from = consumed_from.min(comp_consumed);
        }
        Ok(consumed_from)
    }

    /// True if ANY component vetoes leaf creation. Default is no veto; SWA
    /// vetoes when the entire suffix is outside the SWA window.
    fn should_skip_leaf_creation(
        &self,
        total_prefix_len: usize,
        key_len: usize,
        swa_evicted_seqlen: usize,
    ) -> bool {
        self.components
            .iter()
            .any(|c| c.should_skip_leaf_creation(total_prefix_len, key_len, swa_evicted_seqlen))
    }

    /// Let each component inspect/split the new leaf and emit deferred actions.
    /// Default is a no-op; SWA splits at the SWA boundary (if straddling) and
    /// emits `SwaStamp` for the in-window portion.
    fn commit_insert_data_on_new_leaf(
        &mut self,
        leaf_idx: NodeIdx,
        consumed: usize,
        swa_evicted_seqlen: usize,
        deferred: &mut Vec<DeferredAction>,
    ) {
        for comp in self.components.iter() {
            comp.commit_insert_data_on_new_leaf(
                &mut self.tree_node_pool,
                &self.components,
                leaf_idx,
                consumed,
                swa_evicted_seqlen,
                deferred,
            );
        }
    }

    /// Walk from `root` along `key`, splitting nodes on partial matches, then
    /// append a new leaf for the unmatched suffix. Runs component hooks at three
    /// points: per overlap node (`consume_value`), before leaf creation
    /// (`should_skip_leaf_creation`), and after leaf creation
    /// (`commit_insert_data_on_new_leaf`).
    fn insert_helper(
        &mut self,
        root: NodeIdx,
        key: PageAlignedQueryKey<K::Atom>,
        value: &Tensor,
        prev_prefix_len: usize,
        swa_evicted_seqlen: usize,
        mamba_value: Option<Tensor>,
    ) -> Result<InsertResult, RadixCacheRuntimeError> {
        let key = key.as_slice();
        let key_len = key.len();
        let mut node_idx = root;
        let mut consumed = 0usize;
        let mut deferred: Vec<DeferredAction> = Vec::new();

        // ---- Overlap walk ----
        // Match the longest page-aligned prefix, one edge per step.
        while consumed < key_len {
            let remaining_key = &key[consumed..];
            let (current_node_idx, step_len, last_step) =
                match self.tree_node_pool.match_child(node_idx, remaining_key) {
                    MatchChildResult::NotFound => break,
                    MatchChildResult::FullMatch {
                        child_idx,
                        node_key_len,
                    } => (child_idx, node_key_len, false),
                    MatchChildResult::PartialMatch(node_split) => {
                        let split_len = node_split.split_len();
                        let new_idx = self.tree_node_pool.split_node(&self.components, node_split);
                        (new_idx, split_len, true)
                    }
                };

            let value_slice = value.narrow(0, consumed as i64, step_len as i64);

            let consumed_from = self.consume_value(
                current_node_idx,
                step_len,
                consumed,
                prev_prefix_len,
                &value_slice,
                swa_evicted_seqlen,
                &mut deferred,
            )?;

            // Free the duplicate band: value indices past the caller's locked
            // prefix but not claimed by any component. The locked prefix (caller
            // frees) and component-claimed slices (e.g. SWA recovery) are kept.
            let dup_start = prev_prefix_len.saturating_sub(consumed);
            if dup_start < consumed_from {
                deferred.push(DeferredAction::FullDupFreed {
                    freed_indices: value_slice.narrow(
                        0,
                        dup_start as i64,
                        (consumed_from - dup_start) as i64,
                    ),
                });
            }

            consumed += step_len;
            node_idx = current_node_idx;
            if last_step {
                break;
            }
        }

        // ---- Leaf creation ----
        let remaining_len = key_len - consumed;
        let mut leaf_creation_skipped = false;
        let mut new_leaf_created = false;

        if remaining_len > 0 {
            let skip = self.should_skip_leaf_creation(consumed, remaining_len, swa_evicted_seqlen);

            if skip {
                deferred.push(DeferredAction::FullDupFreed {
                    freed_indices: value.narrow(0, consumed as i64, remaining_len as i64),
                });
                leaf_creation_skipped = true;
            } else {
                let remaining_key = key[consumed..].to_vec();
                let remaining_value = value
                    .narrow(0, consumed as i64, remaining_len as i64)
                    .copy();
                let leaf = TreeNode::new_child(remaining_key, node_idx, Some(remaining_value));
                #[allow(clippy::expect_used, reason = "child key just confirmed absent above")]
                let leaf_idx = self.tree_node_pool.insert_leaf(node_idx, leaf).expect(
                    "first-page child key was just confirmed absent \
                         at this parent",
                );

                self.commit_insert_data_on_new_leaf(
                    leaf_idx,
                    consumed,
                    swa_evicted_seqlen,
                    &mut deferred,
                );

                node_idx = leaf_idx;
                new_leaf_created = true;
            }
        }

        // ---- Populate mamba value ----
        let mamba_value_exists = if let Some(mv) = mamba_value {
            assert!(
                !leaf_creation_skipped,
                "leaf_creation_skipped is unreachable for Mamba",
            );
            if new_leaf_created || !MambaLRUSlot::has_value(self.tree_node_pool.get(node_idx)) {
                MambaLRUSlot::set_value(&mut self.tree_node_pool, node_idx, mv)?;
                let delta = MambaLRUSlot::value_len(self.tree_node_pool.get(node_idx));
                MambaLRUSlot::bump_mru(&mut self.tree_node_pool, node_idx);
                MambaLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += delta;
                false
            } else {
                true
            }
        } else {
            false
        };

        self.bump_mru_walk(node_idx);

        Ok(InsertResult {
            prefix_len: consumed,
            leaf_creation_skipped,
            mamba_value_exists,
            deferred_actions: deferred,
        })
    }

    /// Configured page size (1 for token, >1 for page).
    pub fn page_size(&self) -> usize {
        self.page_size.get()
    }

    /// Number of live nodes in the underlying tree_node_pool (always >= 1 — root).
    pub fn active_tree_node_count(&self) -> usize {
        self.tree_node_pool.active_node_count()
    }

    // TODO: prefix the FULL accessors `full_*` to match the `swa_*` / `mamba_*`
    // accessors; touches the PyO3 surface, stubs, and test call sites.

    /// Sum of `key.len()` across FULL device-value unreferenced nodes.
    pub fn evictable_token_size(&self) -> usize {
        FullLRUSlot::unlocked_size(&self.tree_node_pool)
    }

    /// Sum of `key.len()` across FULL device-value referenced nodes.
    pub fn protected_token_size(&self) -> usize {
        FullLRUSlot::locked_size(&self.tree_node_pool)
    }

    /// Total tokens (evictable + protected) across FULL and SWA components.
    pub fn total_token_size(&self) -> usize {
        let mut total = FullLRUSlot::total_size(&self.tree_node_pool);
        if self.has_swa_component {
            total += SwaLRUSlot::total_size(&self.tree_node_pool);
        }
        total
    }

    /// Total Mamba slots (evictable + protected); separate from `total_token_size` because Mamba's unit is slots, not tokens.
    pub fn mamba_total_size(&self) -> usize {
        if self.has_mamba_component {
            MambaLRUSlot::total_size(&self.tree_node_pool)
        } else {
            0
        }
    }

    /// Sum of `key.len()` across SWA device-value unreferenced nodes.
    pub fn swa_evictable_token_size(&self) -> usize {
        SwaLRUSlot::unlocked_size(&self.tree_node_pool)
    }

    /// Sum of `key.len()` across SWA device-value referenced nodes.
    pub fn swa_protected_token_size(&self) -> usize {
        SwaLRUSlot::locked_size(&self.tree_node_pool)
    }

    /// Count of unlocked nodes with a Mamba value populated.
    pub fn mamba_evictable_token_size(&self) -> usize {
        MambaLRUSlot::unlocked_size(&self.tree_node_pool)
    }

    /// Count of locked nodes with a Mamba value populated.
    pub fn mamba_protected_token_size(&self) -> usize {
        MambaLRUSlot::locked_size(&self.tree_node_pool)
    }

    /// Acquire: dispatch to each component's `inc_lock_ref` (FULL first, then
    /// SWA) and aggregate. `delta` sums per-component contributions;
    /// `swa_uuid_for_lock` comes from the at-most-one component that produces it.
    /// Forward order is required so FULL's `lock_ref` is bumped before SWA's
    /// per-slot mutator-assert checks `swa_lock_ref <= full_lock_ref`. Caller
    /// must pass `swa_uuid_for_lock` back to `dec_lock_ref`.
    // TODO(perf): collapse the per-component walks into one coordinated
    // leaf-to-root walk; today the leaf-to-boundary segment is visited twice for
    // a SWA-configured cache.
    pub fn inc_lock_ref(&mut self, node_idx: NodeIdx) -> IncLockRefResult {
        let mut delta: i64 = 0;
        let mut swa_uuid_for_lock: Option<u64> = None;
        for c in self.components.iter() {
            if let Some(r) = c.inc_lock_ref(&mut self.tree_node_pool, node_idx) {
                delta += r.delta;
                // At-most-one component produces this; `.or()` keeps the first
                // `Some` so a later `None` can't clobber it.
                swa_uuid_for_lock = swa_uuid_for_lock.or(r.swa_uuid_for_lock);
            }
        }
        IncLockRefResult {
            delta,
            swa_uuid_for_lock,
        }
    }

    /// Release: dispatch to each component's `dec_lock_ref` in REVERSE (SWA
    /// first, then FULL) and sum the deltas. Pass back the `swa_uuid_for_lock`
    /// from the matching `inc_lock_ref` so SWA stops at the right boundary.
    ///
    /// Reverse order is load-bearing: `FullLRUSlot::dec_lock_ref` asserts
    /// `swa_lock_ref <= new full_lock_ref`, so FULL must be dec'd last to keep
    /// the gap valid through every step. Panics on lock_ref underflow — callers
    /// must match dec calls to inc calls exactly.
    // TODO(perf): same single-walk opportunity as `inc_lock_ref`.
    pub fn dec_lock_ref(&mut self, node_idx: NodeIdx, swa_uuid_for_lock: Option<u64>) -> i64 {
        let mut delta: i64 = 0;
        for c in self.components.iter().rev() {
            if let Some(d) = c.dec_lock_ref(&mut self.tree_node_pool, node_idx, swa_uuid_for_lock) {
                delta += d;
            }
        }
        delta
    }

    /// Best-effort to evict at least `num_tokens` per component.
    pub fn evict(&mut self, request: EvictRequest) -> EvictResult {
        let mut result = EvictResult::default();
        // FULL eviction runs first: it can evict SWA values too, shrinking
        // SWA's residual budget.
        for c in self.components.iter() {
            c.evict(&mut self.tree_node_pool, &request, &mut result);
        }
        result
    }

    /// Write per-node SWA values back into the tree.
    pub fn apply_swa_writes(
        &mut self,
        node_indices: Vec<NodeIdx>,
        swa_values: Vec<tch::Tensor>,
    ) -> Result<(), RadixCacheRuntimeError> {
        if node_indices.len() != swa_values.len() {
            return Err(RadixCacheRuntimeError::ApplySwaWritesMismatch {
                indices: node_indices.len(),
                values: swa_values.len(),
            });
        }
        // Accumulate credit across all nodes, commit to pool state once after
        // the loop (saves N-1 pool_state_mut indexes).
        let mut evictable_size_credit: usize = 0;
        for (idx, value) in node_indices.into_iter().zip(swa_values) {
            // Snapshot pre-mutation state (immutable borrow released before the
            // stamp's mutable borrow); the stamp doesn't touch these.
            let node = self.tree_node_pool.get(idx);
            let in_list_at_entry = SwaLRUSlot::data(node).in_list;
            let value_present_at_entry = SwaLRUSlot::has_value(node);
            let key_len = node.key().len();
            // SWA value existence must match SWA LRU membership.
            assert_eq!(
                in_list_at_entry, value_present_at_entry,
                "SWA invariant violated at apply_swa_writes entry for node_idx {idx}: \
                 in_list ({in_list_at_entry}) != value.is_some() ({value_present_at_entry})",
            );

            SwaLRUSlot::replace_value(&mut self.tree_node_pool, idx, value);

            SwaLRUSlot::bump_mru(&mut self.tree_node_pool, idx);
            if !in_list_at_entry {
                evictable_size_credit += key_len;
            }
        }
        SwaLRUSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += evictable_size_credit;
        Ok(())
    }
}

/// Production radix cache: children keyed by token page (`Vec<i64>`).
/// Handles `page_size >= 1` — `page_size=1` uses one-element page keys.
pub type PageRadixCache = RadixCache<Vec<i64>>;

/// Bigram-keyed radix cache: children keyed by `(t[i], t[i+1])` pairs, for
/// callers (e.g. EAGLE) that want overlap-pair keys.
pub type BigramRadixCache = RadixCache<Vec<(i64, i64)>>;
