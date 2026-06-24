use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

use crate::component_type::ComponentType;
use crate::components::{Component, FullComponent, IncLockRefResult, MambaComponent, SwaComponent};
use crate::components::{EvictRequest, EvictResult, Slot, SwaSlot};
use crate::deferred_action::DeferredAction;
use crate::error::{RadixCacheInitError, RadixCacheRuntimeError};
use crate::tree_node_pool::{
    ChildKeyType, MatchChildResult, NodeIdx, PageSize, TreeNode, TreeNodePool,
};

/// Result of prefix match.
pub struct MatchResult {
    pub device_indices: Tensor,
    pub last_device_node_idx: NodeIdx,
    /// Chunk-aligned position past the cached prefix for SSM state repair.
    pub mamba_branching_seqlen: Option<usize>,
    /// Cached Mamba state at `last_device_node_idx`.
    pub mamba_value: Option<Tensor>,
}

/// Result of insert.
pub struct InsertResult {
    /// Tokens in the insert key that matched existing nodes.
    pub prefix_len: usize,
    /// True if leaf creation was skipped.
    pub leaf_creation_skipped: bool,
    /// True if the caller still owns `mamba_value` and must free it.
    pub mamba_value_exists: bool,
    /// KV cache allocator pool actions.
    pub deferred_actions: Vec<DeferredAction>,
}

/// Borrowed query key proven page-aligned and non-empty.
struct PageAlignedQueryKey<'a, A> {
    key: &'a [A],
}

impl<'a, A> PageAlignedQueryKey<'a, A> {
    /// `None` when the page-aligned length is 0.
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

/// Build the tree's component list from the cache config.
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

/// Radix tree-based KV cache, generic over the child key type.
pub struct RadixCache<K: ChildKeyType> {
    /// Arena that owns all tree nodes; recycles slots via a freelist.
    tree_node_pool: TreeNodePool<K>,

    /// Root for queries with `extra_key = None`.
    default_root: NodeIdx,

    /// Root per `extra_key`, allocated lazily on first use.
    ///
    /// TODO(Jialin): drop entries when their subtree empties via eviction.
    named_roots: HashMap<String, NodeIdx>,

    page_size: PageSize,

    init_node_capacity: usize,

    /// Empty Int64 tensor, shallow-cloned on cache miss.
    empty_tensor: Tensor,

    /// Configured components (FULL always present; SWA/Mamba iff enabled).
    components: Vec<Box<dyn Component<K>>>,

    has_swa_component: bool,

    has_mamba_component: bool,

    /// Mamba chunk size; state checkpoints saved only at multiples of it.
    mamba_cache_chunk_size: Option<usize>,
}

impl<K: ChildKeyType> RadixCache<K> {
    /// Construct a radix cache from per-cache config.
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

    /// Resolve the namespace root for `extra_key`, lazily creating it.
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

    /// Find the longest cached prefix of `key` in the `extra_key` namespace,
    /// splitting any node where the match ends mid-node. The boundary advances
    /// only past nodes all component validators approve.
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

    /// Prefix match from `root` along `key`, splitting on a mid-key match.
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

        // Stateful per-node match validators.
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
            // Validators are stateful: run all, no short-circuit.
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

        let device_indices = if last_device_value_len == 0 {
            self.empty_tensor.shallow_clone()
        } else {
            Tensor::cat(&values[..last_device_value_len], 0)
        };

        // Each component fills its own MatchResult fields; the driver stays
        // component-agnostic (mirrors Python's _match_post_processor loop).
        let mut result = MatchResult {
            device_indices,
            last_device_node_idx,
            mamba_branching_seqlen: None,
            mamba_value: None,
        };
        for component in &self.components {
            component.finalize_match_result(
                &self.tree_node_pool,
                last_matched_node_idx,
                &values,
                last_device_value_len,
                &mut result,
            );
        }
        Ok(result)
    }

    /// Insert `(key, value)` into the `extra_key` namespace, deep-copying the
    /// stored slice. Returns the prefix length already cached before this insert.
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
    fn bump_mru_walk(&mut self, node_idx: NodeIdx) {
        for comp in self.components.iter() {
            comp.bump_mru_walk(&mut self.tree_node_pool, node_idx);
        }
    }

    /// Take `min(consumed_from)` across components for one overlap node.
    #[allow(clippy::too_many_arguments)]
    fn update_component_on_insert_overlap(
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
            let comp_consumed = comp.update_component_on_insert_overlap(
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

    /// True if any component vetoes leaf creation.
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
    /// append a new leaf for the unmatched suffix.
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

            let consumed_from = self.update_component_on_insert_overlap(
                current_node_idx,
                step_len,
                consumed,
                prev_prefix_len,
                &value_slice,
                swa_evicted_seqlen,
                &mut deferred,
            )?;

            // Free indices past the caller's locked prefix not claimed by any
            // component.
            let dup_start = prev_prefix_len.saturating_sub(consumed);
            if dup_start < consumed_from {
                deferred.push(DeferredAction::FullFree {
                    full_to_free: value_slice.narrow(
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
                deferred.push(DeferredAction::FullFree {
                    full_to_free: value.narrow(0, consumed as i64, remaining_len as i64),
                });
                leaf_creation_skipped = true;
            } else {
                let remaining_key = key[consumed..].to_vec();
                let remaining_value = value
                    .narrow(0, consumed as i64, remaining_len as i64)
                    .copy();
                let leaf = TreeNode::new_child(remaining_key, node_idx, Some(remaining_value));
                let leaf_idx = self.tree_node_pool.insert_leaf(node_idx, leaf);

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

        // ---- Per-component insert-value commit (e.g. Mamba aux SSM state) ----
        let mut result = InsertResult {
            prefix_len: consumed,
            leaf_creation_skipped,
            mamba_value_exists: false,
            deferred_actions: deferred,
        };
        for c in &self.components {
            c.commit_insert_value(
                &mut self.tree_node_pool,
                node_idx,
                new_leaf_created,
                mamba_value.as_ref(),
                &mut result,
            )?;
        }

        self.bump_mru_walk(node_idx);
        Ok(result)
    }

    /// Configured page size.
    pub fn page_size(&self) -> usize {
        self.page_size.get()
    }

    /// Number of live nodes in the underlying tree_node_pool.
    pub fn active_tree_node_count(&self) -> usize {
        self.tree_node_pool.active_node_count()
    }

    /// Unreferenced (evictable) token size for one component.
    pub fn component_evictable_size(&self, ct: ComponentType) -> usize {
        self.tree_node_pool.components[ct as usize].unlocked_size
    }

    /// Referenced (protected) token size for one component.
    pub fn component_protected_size(&self, ct: ComponentType) -> usize {
        self.tree_node_pool.components[ct as usize].locked_size
    }

    /// Total (evictable + protected) token size for one component.
    pub fn component_total_size(&self, ct: ComponentType) -> usize {
        let s = &self.tree_node_pool.components[ct as usize];
        s.unlocked_size + s.locked_size
    }

    /// `(FULL tokens, auxiliary tokens)`, aux = SWA + Mamba. Mirrors
    /// `UnifiedRadixCache.total_size()`; absent components contribute 0.
    pub fn total_size(&self) -> (usize, usize) {
        let full = self.component_total_size(ComponentType::Full);
        let aux = self.component_total_size(ComponentType::Swa)
            + self.component_total_size(ComponentType::Mamba);
        (full, aux)
    }

    /// Acquire: dispatch to each component's `inc_lock_ref` (FULL first, then
    /// SWA) and aggregate. Forward order keeps `swa_lock_ref <= full_lock_ref`.
    pub fn inc_lock_ref(&mut self, node_idx: NodeIdx) -> IncLockRefResult {
        let mut delta: i64 = 0;
        let mut swa_uuid_for_lock: Option<u64> = None;
        for c in self.components.iter() {
            if let Some(r) = c.inc_lock_ref(&mut self.tree_node_pool, node_idx) {
                delta += r.delta;
                swa_uuid_for_lock = swa_uuid_for_lock.or(r.swa_uuid_for_lock);
            }
        }
        IncLockRefResult {
            delta,
            swa_uuid_for_lock,
        }
    }

    /// Release: dispatch to each component's `dec_lock_ref` in REVERSE (SWA
    /// first, then FULL) and sum the deltas. Reverse order keeps
    /// `swa_lock_ref <= full_lock_ref` valid at every step.
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
        // FULL runs first: it can evict SWA values, shrinking SWA's budget.
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
        // Accumulate credit, commit to pool state once after the loop.
        let mut evictable_size_credit: usize = 0;
        for (idx, value) in node_indices.into_iter().zip(swa_values) {
            // Snapshot pre-mutation state.
            let node = self.tree_node_pool.get(idx);
            let in_list_at_entry = SwaSlot::data(node).in_list;
            let value_present_at_entry = SwaSlot::has_value(node);
            let key_len = node.key().len();
            // SWA value existence must match SWA LRU membership.
            assert_eq!(
                in_list_at_entry, value_present_at_entry,
                "SWA invariant violated at apply_swa_writes entry for node_idx {idx}: \
                 in_list ({in_list_at_entry}) != value.is_some() ({value_present_at_entry})",
            );

            SwaSlot::replace_value(&mut self.tree_node_pool, idx, value);

            SwaSlot::bump_mru(&mut self.tree_node_pool, idx);
            if !in_list_at_entry {
                evictable_size_credit += key_len;
            }
        }
        SwaSlot::pool_state_mut(&mut self.tree_node_pool).unlocked_size += evictable_size_credit;
        Ok(())
    }
}

/// Production radix cache: children keyed by token page (`Vec<i64>`).
pub type PageRadixCache = RadixCache<Vec<i64>>;

/// Bigram-keyed radix cache: children keyed by `(t[i], t[i+1])` pairs.
pub type BigramRadixCache = RadixCache<Vec<(i64, i64)>>;
