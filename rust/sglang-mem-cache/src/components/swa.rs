//! SWA component: sliding-window match gating and lock-walk.

use tch::Tensor;

use super::{Component, IncLockRefResult, MatchValidator};
use super::{
    EvictRequest, EvictResult, FullSlot, Slot, dec_lock_ref_non_full, evict_non_full,
    inc_lock_ref_non_full,
};
use crate::component_type::ComponentType;
use crate::deferred_action::DeferredAction;
use crate::error::{RadixCacheInitError, RadixCacheRuntimeError};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, NodeSplit, TreeNode, TreeNodePool};

pub struct SwaComponent {
    /// Sliding window size in tokens.
    sliding_window_size: usize,
}

impl SwaComponent {
    /// Construct a SwaComponent with `sliding_window_size` tokens.
    pub fn new(sliding_window_size: usize) -> Result<Self, RadixCacheInitError> {
        if sliding_window_size == 0 {
            return Err(RadixCacheInitError::InvalidSlidingWindowSize {
                got: sliding_window_size,
            });
        }
        Ok(Self {
            sliding_window_size,
        })
    }
}

impl<K: ChildKeyType> Component<K> for SwaComponent {
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>> {
        Some(Box::new(SwaMatchValidator::new(self.sliding_window_size)))
    }

    /// SWA lock-walk acquire: bump `lock_ref` leaf -> up until accumulated
    /// key length reaches `sliding_window_size`, stamping a boundary uuid.
    /// Must run after `FullComponent::inc_lock_ref`.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult> {
        let mut delta: i64 = 0;
        let mut accumulated_token_count: usize = 0;
        let mut swa_uuid_for_lock: Option<u64> = None;
        let mut current = node_idx;
        while let Some(parent) = pool.get(current).parent() {
            let key_len = pool.get(current).key().len();
            delta += SwaSlot::inc_lock_ref(pool, current);
            accumulated_token_count += key_len;
            if accumulated_token_count >= self.sliding_window_size {
                swa_uuid_for_lock = Some(pool.lazy_acquire_swa_uuid_for_lock(current));
                break;
            }
            current = parent;
        }
        Some(IncLockRefResult {
            delta,
            swa_uuid_for_lock,
        })
    }

    /// SWA lock-walk release, inverse of `inc_lock_ref`: decrement
    /// `lock_ref` leaf -> up, stopping at the node matching the request's
    /// `swa_uuid_for_lock` (or root if `None`). Must run before FULL.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64> {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = pool.get(current).parent() {
            let this_uuid = pool.get(current).swa_uuid_for_lock();
            delta += SwaSlot::dec_lock_ref(pool, current);
            // `is_some()` rules out the (None, None) match (walk to root).
            if this_uuid.is_some() && this_uuid == swa_uuid_for_lock {
                break;
            }
            current = parent;
        }
        Some(delta)
    }

    /// Evict SWA values and remove unreferenced leaves.
    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult) {
        let ct = ComponentType::Swa as usize;
        let target = request.num_tokens[ct];
        let already = result.evicted[ct];
        if already < target {
            evict_non_full::<K, SwaSlot>(pool, target - already, result);
        }
    }

    /// SWA tombstone-recovery hook. Called per-overlap-node during
    /// insert. If the node is an SWA tombstone (SWA value evicted, FULL
    /// value remains), recovers the SWA value by replacing the old FULL
    /// value with the incoming insert value and emitting a `SwaRecover`
    /// action; the Python orchestrator translates the new full indices
    /// to SWA indices via `apply_swa_writes`.
    ///
    /// Three branches based on where `swa_evicted_seqlen` falls relative
    /// to the node's position in the token sequence:
    ///
    /// ```text
    ///   0              total_prefix_len                    total_prefix_len + node_key_len
    ///   |              |                                   |
    ///   v              v                                   v
    ///   [== prefix ==][============ node key ==============]
    ///
    ///   Branch 1: seqlen <= total_prefix_len
    ///        seqlen
    ///          v
    ///   [== prefix ==][========= in SWA window ============]  → full recover
    ///
    ///   Branch 2: total_prefix_len < seqlen < total_prefix_len + node_key_len
    ///                       seqlen
    ///                         v
    ///   [== prefix ==][= out =][====== in SWA window ======]  → split, partial recover
    ///                    ^
    ///                    split here (start_idx = seqlen - total_prefix_len)
    ///
    ///   Branch 3: seqlen >= total_prefix_len + node_key_len
    ///                                                        seqlen
    ///                                                          v
    ///   [== prefix ==][========= outside SWA window ========]  → no-op
    /// ```
    ///
    /// Returns `consumed_from`: how far into `value_slice` SWA claimed
    /// ownership.
    /// - Branch 1: returns 0 (claimed entire slice)
    /// - Branch 2: returns start_idx (claimed from start_idx onward)
    /// - Branch 3: returns node_key_len (claimed nothing)
    /// - Not a tombstone / no SWA frontier: returns node_key_len
    #[allow(clippy::too_many_arguments)]
    fn update_component_on_insert_overlap(
        &self,
        pool: &mut TreeNodePool<K>,
        components: &[Box<dyn Component<K>>],
        child_idx: NodeIdx,
        node_key_len: usize,
        total_prefix_len: usize,
        prev_prefix_len: usize,
        value_slice: &Tensor,
        swa_evicted_seqlen: usize,
        deferred: &mut Vec<DeferredAction>,
    ) -> Result<usize, RadixCacheRuntimeError> {
        // Whole node inside the caller-protected prefix: skip recovery.
        if prev_prefix_len >= total_prefix_len + node_key_len {
            return Ok(node_key_len);
        }

        // Not a tombstone: nothing to recover.
        if SwaSlot::has_value(pool.get(child_idx)) {
            return Ok(node_key_len);
        }

        assert_eq!(
            SwaSlot::lock_ref(pool.get(child_idx)),
            0,
            "SWA tombstone at node_idx={child_idx} has non-zero lock_ref"
        );

        if swa_evicted_seqlen <= total_prefix_len {
            // Entire node in window: full recover.
            #[allow(
                clippy::expect_used,
                reason = "tombstone invariant: FULL value retained"
            )]
            let old_full = FullSlot::value(pool.get(child_idx))
                .expect("tombstone node must have FULL value")
                .shallow_clone();
            let new_full = value_slice.copy();
            FullSlot::replace_value(pool, child_idx, new_full.shallow_clone());
            deferred.push(DeferredAction::SwaRecover {
                node_idx: child_idx,
                old_full_to_free: old_full,
                new_full_value: new_full,
            });
            Ok(0)
        } else if swa_evicted_seqlen < total_prefix_len + node_key_len {
            // Node straddles boundary: split, then recover in-window suffix.
            let start_idx = swa_evicted_seqlen - total_prefix_len;
            let split = NodeSplit {
                child_idx,
                split_len: start_idx,
            };
            let _new_parent = pool.split_node(components, split);
            #[allow(
                clippy::expect_used,
                reason = "split_node narrows but preserves FULL value"
            )]
            let old_full = FullSlot::value(pool.get(child_idx))
                .expect("split suffix must have FULL value")
                .shallow_clone();
            let new_full = value_slice
                .narrow(0, start_idx as i64, (node_key_len - start_idx) as i64)
                .copy();
            FullSlot::replace_value(pool, child_idx, new_full.shallow_clone());
            deferred.push(DeferredAction::SwaRecover {
                node_idx: child_idx,
                old_full_to_free: old_full,
                new_full_value: new_full,
            });
            Ok(start_idx)
        } else {
            // Entire node outside window: no recovery.
            Ok(node_key_len)
        }
    }

    /// SWA veto: skip leaf creation when the new suffix lies exactly at the
    /// SWA eviction boundary (a born-tombstone, useless for SWA matches).
    fn should_skip_leaf_creation(
        &self,
        total_prefix_len: usize,
        key_len: usize,
        swa_evicted_seqlen: usize,
    ) -> bool {
        assert!(
            swa_evicted_seqlen <= total_prefix_len + key_len,
            "SwaComponent::should_skip_leaf_creation: swa_evicted_seqlen ({swa_evicted_seqlen}) \
             > total_prefix_len ({total_prefix_len}) + key_len ({key_len}) — caller's SWA \
             eviction watermark exceeds the request's data range",
        );
        swa_evicted_seqlen == total_prefix_len + key_len
    }

    /// SWA boundary-split + stamp hook: split a boundary-straddling leaf
    /// into a tombstone parent and in-window child, then emit `SwaStamp`.
    fn commit_insert_data_on_new_leaf(
        &self,
        pool: &mut TreeNodePool<K>,
        components: &[Box<dyn Component<K>>],
        leaf_idx: NodeIdx,
        consumed: usize,
        swa_evicted_seqlen: usize,
        deferred: &mut Vec<DeferredAction>,
    ) {
        let key_len = pool.get(leaf_idx).key().len();
        let split_pos = swa_evicted_seqlen as isize - consumed as isize;

        // Leaf must have at least one in-window token.
        assert!(
            split_pos < key_len as isize,
            "SwaComponent::commit_insert_data_on_new_leaf: split_pos ({split_pos}) >= key_len ({key_len}) \
             — entire leaf is outside SWA window, should_skip_leaf_creation should have \
             vetoed leaf creation (consumed={consumed}, swa_evicted_seqlen={swa_evicted_seqlen})",
        );

        // Straddles boundary: split into tombstone parent + in-window child.
        if split_pos > 0 {
            let split = NodeSplit {
                child_idx: leaf_idx,
                split_len: split_pos as usize,
            };
            let _tombstone_parent = pool.split_node(components, split);
        }

        #[allow(
            clippy::expect_used,
            reason = "in-window leaf invariant: FULL value populated"
        )]
        let full_value = FullSlot::value(pool.get(leaf_idx))
            .expect("in-window leaf must have FULL value")
            .shallow_clone();
        deferred.push(DeferredAction::SwaStamp {
            node_idx: leaf_idx,
            full_value,
        });
    }

    /// Bump SWA LRU recency from `node_idx` up through ancestors.
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        SwaSlot::bump_mru_walk(pool, node_idx);
    }

    /// SWA-side redistribution after a node split: lock_ref copy,
    /// swa_uuid_for_lock transfer, value slice, and LRU bump. Upholds the
    /// SWA invariant `lock_ref > 0 implies has_value`.
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    ) {
        // 1. Copy SWA lock_ref so an in-flight release walk can pass through.
        let child_swa_lock_ref = SwaSlot::lock_ref(pool.get(child_idx));
        SwaSlot::set_lock_ref(pool.get_mut(new_parent_idx), child_swa_lock_ref);

        // 2. Transfer the SWA uuid (boundary marker) to the parent slice.
        let transferred_uuid = pool.get_mut(child_idx).swa_uuid_for_lock.take();
        pool.get_mut(new_parent_idx)
            .set_swa_uuid_for_lock(transferred_uuid);

        // 3. Slice SWA value across the boundary if present (else tombstone).
        let Some(child_swa) = SwaSlot::value(pool.get(child_idx)) else {
            return;
        };
        let child_swa = child_swa.shallow_clone();
        let total_len = child_swa.size()[0];
        let parent_part = child_swa.narrow(0, 0, split_len as i64);
        let child_part = child_swa.narrow(0, split_len as i64, total_len - split_len as i64);

        SwaSlot::replace_value(pool, new_parent_idx, parent_part);
        SwaSlot::replace_value(pool, child_idx, child_part);

        // 4. Insert both halves into SWA's LRU, child more recent than parent.
        SwaSlot::bump_mru_split(pool, new_parent_idx, child_idx);
    }
}

/// SWA's per-walk match validator. Approves an SWA-present node if no
/// tombstone has been seen, or the contiguous run since the last tombstone
/// reaches `sliding_window_size`.
pub struct SwaMatchValidator {
    sliding_window_size: usize,
    seen_tombstone: bool,
    current_match_len: usize,
}

impl SwaMatchValidator {
    pub fn new(sliding_window_size: usize) -> Self {
        Self {
            sliding_window_size,
            seen_tombstone: false,
            current_match_len: 0,
        }
    }
}

impl<K: ChildKeyType> MatchValidator<K> for SwaMatchValidator {
    fn validate(&mut self, n: &TreeNode<K>) -> bool {
        if !SwaSlot::has_value(n) {
            self.seen_tombstone = true;
            self.current_match_len = 0;
            return false;
        }
        self.current_match_len += n.key().len();
        !self.seen_tombstone || self.current_match_len >= self.sliding_window_size
    }
}

/// SWA component slot.
pub struct SwaSlot;

impl Slot for SwaSlot {
    const COMPONENT: ComponentType = ComponentType::Swa;
    const NAME: &'static str = "Swa";

    /// SWA's freed handle is a clone of FULL's value, not SWA's own value.
    fn take_value<K: ChildKeyType>(node: &mut TreeNode<K>, result: &mut EvictResult) {
        let ct = Self::COMPONENT as usize;
        if node.components[ct].value.take().is_some() {
            if let Some(full) = node.components[ComponentType::Full as usize].value.as_ref() {
                let cloned = full.shallow_clone();
                result.evicted[ct] += cloned.size()[0] as usize;
                result.freed[ct].push(cloned);
            }
        }
    }

    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        inc_lock_ref_non_full::<K, Self>(pool, node_idx, /* enforce_full_cap */ true)
    }

    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        dec_lock_ref_non_full::<K, Self>(pool, node_idx)
    }
}
