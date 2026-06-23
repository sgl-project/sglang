//! SWA component: sliding-window match gating and a window-bounded
//! lock-walk that stamps a boundary uuid once the accumulated key
//! length reaches `sliding_window_size`.

use tch::Tensor;

use super::{Component, IncLockRefResult, MatchValidator};
use crate::component_type::ComponentType;
use crate::deferred_action::DeferredAction;
use crate::error::{RadixCacheInitError, RadixCacheRuntimeError};
use crate::tree_node_lru::{
    EvictRequest, EvictResult, FullLRUSlot, LRUSlot, SwaLRUSlot, evict_non_full,
};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, NodeSplit, TreeNode, TreeNodePool};

pub struct SwaComponent {
    /// Sliding window size in tokens.
    sliding_window_size: usize,
}

impl SwaComponent {
    /// Construct a SwaComponent with `sliding_window_size` tokens.
    /// Rejects `0`: a zero window passes the validator vacuously and
    /// defeats SWA gating.
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

    /// SWA lock-walk acquire. Walk leaf -> up bumping SWA's `lock_ref`
    /// and accumulating `node.key().len()` until it reaches
    /// `sliding_window_size`; the boundary node IS bumped (inclusive
    /// check). At the boundary, `lazy_acquire_swa_uuid_for_lock` reuses
    /// the existing stamp (concurrent-acquire reuse) or mints a fresh one.
    ///
    /// The uuid is a node identity-marker persisting across acquire /
    /// release cycles, NOT a per-acquire token: concurrent acquires at
    /// the same boundary share it so each release stops at the same node.
    /// Clearing it on release would corrupt other in-flight release walks.
    ///
    /// Dispatch order: `FullComponent::inc_lock_ref` must run first so
    /// FULL's `lock_ref >= prospective new SWA lock_ref` (enforced by the
    /// per-slot `SwaLRUSlot::inc_lock_ref` assert at every bump).
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
            // On the 0->1 transition for SWA-populated nodes, returns the
            // negative delta (key.len() shifted evictable -> protected);
            // 0 otherwise.
            delta += SwaLRUSlot::inc_lock_ref(pool, current);
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

    /// SWA lock-walk release, inverse of `inc_lock_ref`. Walk leaf -> up
    /// decrementing SWA's `lock_ref`, stopping AFTER decrementing the node
    /// whose uuid matches the request's `swa_uuid_for_lock` (boundary node
    /// IS decremented). Does NOT clear the node's uuid — see `inc_lock_ref`.
    /// When the request's uuid is `None` (inc walk didn't fill the window),
    /// walks all the way to root, symmetric with the inc walk.
    ///
    /// Dispatch order: SWA decrements before FULL so FULL's per-slot
    /// `dec_lock_ref` assert (`swa_lock_ref <= prospective new
    /// full_lock_ref`) doesn't fire transiently while they're equal.
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
            delta += SwaLRUSlot::dec_lock_ref(pool, current);
            // `is_some()` rules out the (None, None) match: a `None`
            // request uuid (window didn't fill) keeps walking to root.
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
            evict_non_full::<K, SwaLRUSlot>(pool, target - already, result);
        }
    }

    /// SWA tombstone-recovery hook, called per-overlap-node during insert.
    /// If the node is an SWA tombstone (SWA value evicted, FULL value
    /// remains), recover the SWA value by replacing the old FULL value
    /// with the incoming slice and emitting `SwaRecover`.
    ///
    /// Returns how far into `value_slice` SWA claimed ownership, by branch
    /// on where `swa_evicted_seqlen` falls within the node:
    /// - Branch 1 (in window): full recover, returns 0
    /// - Branch 2 (straddles boundary): split + partial recover, returns start_idx
    /// - Branch 3 (outside window) / not a tombstone: no-op, returns node_key_len
    #[allow(clippy::too_many_arguments)]
    fn consume_value(
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
        // `prev_prefix_len` is the caller-protected prefix (absolute
        // coords). If the whole overlap node lies within it, recovering
        // would free FULL slots still referenced by an in-flight request,
        // so skip recovery and claim nothing.
        if prev_prefix_len >= total_prefix_len + node_key_len {
            return Ok(node_key_len);
        }

        // Not a tombstone: SWA already has a value, nothing to recover.
        // Real tombstones always have lock_ref==0 (synthetic post-split
        // tombstones can't exist: redistribute_on_node_split slices the
        // SWA value across splits instead).
        if SwaLRUSlot::has_value(pool.get(child_idx)) {
            return Ok(node_key_len);
        }

        assert_eq!(
            SwaLRUSlot::lock_ref(pool.get(child_idx)),
            0,
            "SWA tombstone at node_idx={child_idx} has non-zero lock_ref"
        );

        if swa_evicted_seqlen <= total_prefix_len {
            // Branch 1: entire node within SWA window — full recover.
            #[allow(
                clippy::expect_used,
                reason = "tombstone invariant: FULL value retained"
            )]
            let old_full = FullLRUSlot::value(pool.get(child_idx))
                .expect("tombstone node must have FULL value")
                .shallow_clone();
            let new_full = value_slice.copy();
            FullLRUSlot::replace_value(pool, child_idx, new_full.shallow_clone());
            deferred.push(DeferredAction::SwaRecover {
                node_idx: child_idx,
                freed_full: old_full,
                source_value: new_full,
            });
            Ok(0)
        } else if swa_evicted_seqlen < total_prefix_len + node_key_len {
            // Branch 2: node straddles the boundary — split at start_idx,
            // then recover the in-window suffix (`child_idx` after split).
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
            let old_full = FullLRUSlot::value(pool.get(child_idx))
                .expect("split suffix must have FULL value")
                .shallow_clone();
            let new_full = value_slice
                .narrow(0, start_idx as i64, (node_key_len - start_idx) as i64)
                .copy();
            FullLRUSlot::replace_value(pool, child_idx, new_full.shallow_clone());
            deferred.push(DeferredAction::SwaRecover {
                node_idx: child_idx,
                freed_full: old_full,
                source_value: new_full,
            });
            Ok(start_idx)
        } else {
            // Branch 3: entire node is outside SWA window — no recovery.
            Ok(node_key_len)
        }
    }

    /// SWA veto: skip leaf creation if the new suffix lies exactly at the
    /// SWA eviction boundary (a born-tombstone with no SWA value is useless
    /// for SWA matches). The leading assert surfaces a too-large `seqlen`
    /// (evicted past the request's data) here rather than deferring it to
    /// `commit_insert_data_on_new_leaf`'s post-leaf assert.
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

    /// SWA boundary-split + stamp hook. If the new leaf straddles the SWA
    /// eviction boundary (Case B), split it into a tombstone parent
    /// (outside window) and an in-window child, then emit `SwaStamp` on the
    /// child; otherwise (Case A, entire leaf in window) emit `SwaStamp` for
    /// the whole leaf. Case C (entire leaf outside) is asserted unreachable
    /// — `should_skip_leaf_creation` must have vetoed it.
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

        // Precondition: leaf must have at least one in-window token (Case A
        // or B). Case C should have been vetoed by should_skip_leaf_creation.
        assert!(
            split_pos < key_len as isize,
            "SwaComponent::commit_insert_data_on_new_leaf: split_pos ({split_pos}) >= key_len ({key_len}) \
             — entire leaf is outside SWA window, should_skip_leaf_creation should have \
             vetoed leaf creation (consumed={consumed}, swa_evicted_seqlen={swa_evicted_seqlen})",
        );

        // Case B: leaf straddles the boundary — split into tombstone parent
        // + in-window child (leaf_idx becomes the suffix). Case A: no split.
        if split_pos > 0 {
            let split = NodeSplit {
                child_idx: leaf_idx,
                split_len: split_pos as usize,
            };
            let _tombstone_parent = pool.split_node(components, split);
        }

        // Emit SwaStamp so the orchestrator translates full -> swa and
        // stamps the SWA value + LRU insertion on leaf_idx.
        #[allow(
            clippy::expect_used,
            reason = "in-window leaf invariant: FULL value populated"
        )]
        let full_value = FullLRUSlot::value(pool.get(leaf_idx))
            .expect("in-window leaf must have FULL value")
            .shallow_clone();
        deferred.push(DeferredAction::SwaStamp {
            node_idx: leaf_idx,
            source_value: full_value,
        });
    }

    /// Bump SWA LRU recency from `node_idx` up through ancestors. Gates on
    /// the per-node `in_list` flag, so SWA tombstones are skipped (an
    /// access bump doesn't revive them).
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        SwaLRUSlot::bump_mru_walk(pool, node_idx);
    }

    /// SWA-side redistribution after a node split: lock_ref copy,
    /// swa_uuid_for_lock transfer, value slice, and LRU bump.
    ///
    /// Copying SWA lock_ref and slicing the value to the new parent
    /// upholds the SWA invariant `lock_ref > 0 implies has_value`. Without
    /// them, splitting a SWA-populated, SWA-locked node would leave the new
    /// parent a synthetic tombstone (no value, lock_ref > 0) that breaks
    /// any future walk landing on it.
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    ) {
        // 1. Copy SWA lock_ref. Unconditional: an in-flight acquire that
        // bumped the original child needs the new intermediate to absorb
        // its later release-walk past the boundary.
        let child_swa_lock_ref = SwaLRUSlot::lock_ref(pool.get(child_idx));
        SwaLRUSlot::set_lock_ref(pool.get_mut(new_parent_idx), child_swa_lock_ref);

        // 2. Transfer the SWA uuid (boundary marker): it belongs to the
        // parent slice after the split, so an in-flight acquire's release
        // walk still finds and stops at the right node.
        let transferred_uuid = pool.get_mut(child_idx).swa_uuid_for_lock.take();
        pool.get_mut(new_parent_idx)
            .set_swa_uuid_for_lock(transferred_uuid);

        // 3. Slice SWA value across the boundary if present. Tombstones
        // leave both halves tombstoned, with bump_mru_split skipped (step 4)
        // to preserve the `in_list <=> value.is_some()` invariant.
        let Some(child_swa) = SwaLRUSlot::value(pool.get(child_idx)) else {
            return;
        };
        let child_swa = child_swa.shallow_clone();
        let total_len = child_swa.size()[0];
        let parent_part = child_swa.narrow(0, 0, split_len as i64);
        let child_part = child_swa.narrow(0, split_len as i64, total_len - split_len as i64);

        SwaLRUSlot::replace_value(pool, new_parent_idx, parent_part);
        SwaLRUSlot::replace_value(pool, child_idx, child_part);

        // 4. Both halves now have SWA value; ensure both are in SWA's LRU
        // with the child more recent than the parent (leaf-MRU ordering).
        SwaLRUSlot::bump_mru_split(pool, new_parent_idx, child_idx);
    }
}

/// SWA's per-walk match validator. Tracks contiguous-SWA-present run
/// length (in tokens) and approves a node iff its SWA value is present
/// (not a tombstone) AND either no tombstone has been seen yet
/// (pre-tombstone shortcut) or the run since the last tombstone reaches
/// `sliding_window_size`.
///
/// The pre-tombstone shortcut treats any path that hasn't hit a tombstone
/// as already-window-filled: freshly-inserted, never-evicted entries are
/// guaranteed valid SWA, so window-fill gating only kicks in after a
/// tombstone splits the path.
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
        if !SwaLRUSlot::has_value(n) {
            self.seen_tombstone = true;
            self.current_match_len = 0;
            return false;
        }
        self.current_match_len += n.key().len();
        // Pre-tombstone shortcut: approve any SWA-present node before
        // the first tombstone. After the first tombstone, require the
        // contiguous run to fill the window.
        !self.seen_tombstone || self.current_match_len >= self.sliding_window_size
    }
}
