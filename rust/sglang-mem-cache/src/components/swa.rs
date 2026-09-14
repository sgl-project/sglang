//! SWA component. Match validation: contiguous-SWA-present run-length
//! gating with tombstone-reset and pre-tombstone shortcut. Lock-walk:
//! window-bounded leaf → up traversal that bumps SWA's `lock_ref` per
//! node, accumulates `node.key().len()`, and stamps (or reuses) a
//! boundary uuid once the accumulated count reaches
//! `sliding_window_size`.
//!
//! `SwaComponent` is the orchestrator-facing shell — it captures the
//! `sliding_window_size` config at construction (cache-instance
//! constant per OSS `swa_radix_cache.py`'s `__init__`-time capture)
//! and produces a fresh `SwaMatchValidator` per `match_prefix` walk.
//!
//! Future SWA PR additions to this file: cascade evict orchestration,
//! insert hooks. Each new method on `Component` for those will land
//! here, delegating to `tree_node_lru.rs::SwaLRUSlot`'s static methods
//! for the hot-path per-node mutations.

use tch::Tensor;

use super::{Component, IncLockRefResult, MatchValidator};
use crate::component_type::ComponentType;
use crate::deferred_action::DeferredAction;
use crate::error::{RadixCacheInitError, RadixCacheRuntimeError};
use crate::tree_node_lru::{
    evict_non_full, EvictRequest, EvictResult, FullLRUSlot, LRUSlot, SwaLRUSlot,
};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, NodeSplit, TreeNode, TreeNodePool};

pub struct SwaComponent {
    /// Sliding window size in tokens. Captured at `RadixCache::new`
    /// from the `sliding_window_size: Option<usize>` constructor arg
    /// (this shell is only constructed when the user passed a
    /// `Some(W)` — `None` callers skip SWA entirely).
    sliding_window_size: usize,
}

impl SwaComponent {
    /// Construct a SwaComponent with `sliding_window_size` tokens.
    /// Rejects `0` as degenerate (the validator would always pass
    /// vacuously, defeating SWA gating). Validation lives here so
    /// SWA owns its own config invariants — `RadixCache::new` just
    /// propagates the error.
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

    /// SWA lock-walk acquire. Walk leaf → up bumping SWA's `lock_ref`
    /// (via `SwaLRUSlot::inc_lock_ref` — also asserts the FULL/other
    /// invariant) and accumulating `node.key().len()` until the
    /// accumulated count reaches `sliding_window_size`. The boundary
    /// node IS bumped (window-fill check is inclusive — matches OSS
    /// `swa_radix_cache.py`'s `inc_lock_ref` window-fill loop). At
    /// the boundary, calls `pool.lazy_acquire_swa_uuid_for_lock(node_idx)`
    /// which returns the existing uuid if one is stamped (concurrent-
    /// acquire reuse) or mints + stamps a fresh one otherwise (matches
    /// OSS `swa_radix_cache.py`'s `inc_lock_ref` `if node.swa_uuid is
    /// None` branch).
    ///
    /// Multiple concurrent acquires hitting the same boundary share
    /// the uuid, so each release stops at the same node — the uuid is
    /// a node identity-marker persisting across acquire/release
    /// cycles, NOT a per-acquire token. Clearing on release would
    /// corrupt other in-flight requests' release walks (they'd walk
    /// past the boundary and underflow `swa_lock_ref` on parents that
    /// were never bumped).
    ///
    /// Returns `IncLockRefResult { delta: 0, swa_uuid_for_lock: ... }`.
    /// `delta` is `0` because SWA's pool-state aggregates aren't wired
    /// up yet — see the TODO at the size-update site below for the
    /// intended OSS-matching behavior.
    ///
    /// Dispatch-order assumption: `FullComponent::inc_lock_ref` must
    /// have already run for this node so FULL's `lock_ref >=
    /// prospective new SWA lock_ref`. `RadixCache::inc_lock_ref`'s
    /// forward iteration (FULL first, then SWA) provides this; the
    /// per-slot `SwaLRUSlot::inc_lock_ref` mutator-assert verifies it
    /// at every per-node bump.
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
            // Bump SWA's lock_ref via the slot mutator (asserts the
            // FULL/other invariant on the prospective new value, plus
            // u32 overflow). On the 0→1 transition for SWA-populated
            // nodes, returns the negative delta (key.len() shifted
            // from evictable to protected); 0 otherwise (already-locked
            // or unpopulated nodes don't shift).
            delta += SwaLRUSlot::inc_lock_ref(pool, current);
            accumulated_token_count += key_len;
            // Boundary check: if window now full, get-or-mint the
            // boundary uuid and stop walking. Check is inclusive — the
            // boundary node IS counted. `lazy_acquire_swa_uuid_for_lock`
            // returns the existing stamp (concurrent-acquire reuse) or
            // mints + stamps a fresh one in a single call.
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

    /// SWA lock-walk release. Inverse of `inc_lock_ref` — walk
    /// leaf → up decrementing SWA's `lock_ref` (via
    /// `SwaLRUSlot::dec_lock_ref` — also asserts no underflow).
    /// Stops AFTER decrementing the node whose `swa_uuid_for_lock`
    /// matches the request's `swa_uuid_for_lock` (matches OSS
    /// `swa_radix_cache.py`'s `dec_lock_ref` `dec_lock_swa = False` shape:
    /// the boundary node IS decremented, then SWA's walk stops).
    ///
    /// **Does NOT clear the node's uuid** — the uuid persists across
    /// acquire / release cycles for concurrent-acquire safety. See
    /// the uuid-reuse comment on `inc_lock_ref` for the trace.
    ///
    /// When `swa_uuid_for_lock` is `None` (the inc walk didn't fill
    /// the window because `sliding_window_size > total_path_len` —
    /// matches OSS `swa_radix_cache.py`'s `dec_lock_ref` "unlocks to
    /// the root, exclusive" comment), walks all the way to root
    /// decrementing every node — symmetric with the inc walk's same
    /// path coverage in that case.
    ///
    /// Dispatch-order assumption: `FullComponent::dec_lock_ref` must
    /// NOT have run yet for this call. SWA decrements first so
    /// FULL's per-slot `dec_lock_ref` mutator-assert (which checks
    /// `swa_lock_ref <= prospective new full_lock_ref`) doesn't fire
    /// transiently when `full_lock_ref == swa_lock_ref` before the
    /// dec. `RadixCache::dec_lock_ref`'s reverse iteration (SWA
    /// first, then FULL) provides this. OSS
    /// `unified_radix_cache.py`'s `dec_lock_ref` iterates forward; we deviate to
    /// keep our per-slot assert load-bearing.
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
            // Decrement via slot mutator (asserts no underflow). On
            // the 1→0 transition for SWA-populated nodes, returns the
            // positive delta (key.len() shifted from protected back to
            // evictable); 0 otherwise.
            delta += SwaLRUSlot::dec_lock_ref(pool, current);
            // Boundary check: stop after decrementing the node whose
            // uuid matches the request's `swa_uuid_for_lock`. The
            // `is_some()` guard rules out the (None, None) match —
            // when the request's uuid is `None` (window didn't fill),
            // keep walking to root.
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
        // `prev_prefix_len` is the caller-protected prefix in absolute
        // coords. If the entire overlap node lies within it, recovery
        // would replace the FULL value AND emit `SwaRecover` to free
        // the old FULL slots — but those slots are still referenced by
        // an in-flight request. Skip recovery, claim nothing; the
        // outer FULL dup-free path also gates on `prev_prefix_len` so
        // the caller's slots stay untouched. Mirrors baseline
        // `unified_cache_components/swa_component.py::update_component_on_insert_overlap`'s
        // `if params.prev_prefix_len >= total_prefix_len + prefix_len: return prefix_len`.
        if prev_prefix_len >= total_prefix_len + node_key_len {
            return Ok(node_key_len);
        }

        // Not a tombstone → SWA already has a value; nothing to recover.
        // Real tombstones (created via SWA evict) always have lock_ref==0;
        // synthetic post-split tombstones can't exist anymore because
        // SwaComponent::redistribute_on_node_split slices the SWA value
        // across the split, so the new parent inherits its share rather
        // than becoming a tombstone with lock_ref>0.
        if SwaLRUSlot::has_value(pool.get(child_idx)) {
            return Ok(node_key_len);
        }

        assert_eq!(
            SwaLRUSlot::lock_ref(pool.get(child_idx)),
            0,
            "SWA tombstone at node_idx={child_idx} has non-zero lock_ref"
        );

        if swa_evicted_seqlen <= total_prefix_len {
            // Branch 1: entire node is within SWA window — full recover.
            // Replace FULL's value with the incoming slice; the old FULL
            // value is freed by Python via the SwaRecover action.
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
            // then recover the in-window child (which is `child_idx` after
            // split_node re-parents it as the suffix).
            let start_idx = swa_evicted_seqlen - total_prefix_len;
            let split = NodeSplit {
                child_idx,
                split_len: start_idx,
            };
            let _new_parent = pool.split_node(components, split);
            // After split: child_idx is now the suffix (in-window portion).
            // Its FULL value has been narrowed to value[start_idx:] by split_node.
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

    /// SWA veto: skip leaf creation if the new suffix lies exactly at
    /// the SWA eviction boundary (would produce a born-tombstone with
    /// no SWA value, useless for SWA matches). Matches OSS
    /// `swa_radix_cache.py`'s defensive `==` check — the
    /// `-page_size` fix in `_evict_swa` keeps `seqlen` strictly less
    /// than `total_prefix_len + key_len` in normal operation, so this
    /// check guards against unexpected eviction states from other
    /// code paths.
    ///
    /// Strict `==` (not `>=`) plus a leading assert: a too-large
    /// `seqlen` (caller bug — evicted past the request's data) is
    /// surfaced HERE at the veto check rather than silently absorbed,
    /// or deferred to `commit_insert_data_on_new_leaf`'s post-leaf
    /// `split_pos < key_len` assert. Earlier panic = clearer
    /// diagnostic (no half-created leaf or in-flight deferred
    /// actions to reason about).
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

    /// SWA boundary-split + stamp hook. If the new leaf straddles the
    /// SWA eviction boundary, splits it into a tombstone parent
    /// (outside the SWA window, no SWA value) and a non-tombstone child
    /// (inside the window, gets SWA value via `SwaStamp` →
    /// `apply_swa_writes`). Otherwise, just emits `SwaStamp` for the
    /// entire leaf.
    ///
    /// ```text
    ///   0              consumed (= node_start)            consumed + key_len
    ///   |              |                                  |
    ///   v              v                                  v
    ///   [== prefix ==][============ new leaf key =========]
    ///
    ///   split_pos = swa_evicted_seqlen - consumed
    ///
    ///   Case A: split_pos <= 0 → entire leaf in window
    ///        seqlen
    ///          v
    ///   [== prefix ==][======== in SWA window ============]  → SwaStamp
    ///
    ///   Case B: 0 < split_pos < key_len → straddles boundary
    ///                       seqlen
    ///                         v
    ///   [== prefix ==][= out =][====== in SWA window =====]  → split + SwaStamp
    ///                    ^
    ///                    split here
    ///
    ///   Case C: split_pos >= key_len → entire leaf outside
    ///   (asserted unreachable — should_skip_leaf_creation must veto)
    /// ```
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
        // or Case B). Case C (split_pos >= key_len, entire leaf outside the
        // SWA window) must be vetoed by should_skip_leaf_creation before
        // commit_insert_data_on_new_leaf runs — reaching here is a caller bug.
        assert!(
            split_pos < key_len as isize,
            "SwaComponent::commit_insert_data_on_new_leaf: split_pos ({split_pos}) >= key_len ({key_len}) \
             — entire leaf is outside SWA window, should_skip_leaf_creation should have \
             vetoed leaf creation (consumed={consumed}, swa_evicted_seqlen={swa_evicted_seqlen})",
        );

        // Case B (split_pos > 0): leaf straddles the boundary — split
        // into tombstone parent + in-window child first. After split,
        // leaf_idx is the suffix (in-window child) with its FULL value
        // narrowed by split_node.
        // Case A (split_pos <= 0): entire leaf is within the SWA window
        // — no split needed.
        if split_pos > 0 {
            let split = NodeSplit {
                child_idx: leaf_idx,
                split_len: split_pos as usize,
            };
            let _tombstone_parent = pool.split_node(components, split);
        }

        // Common path: emit SwaStamp so Python translates full→swa and
        // apply_swa_writes stamps the SWA value + LRU insertion on
        // leaf_idx (the entire leaf in Case A, the in-window suffix in
        // Case B).
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

    /// Bump SWA LRU recency from `node_idx` upward through ancestors.
    /// `SwaLRUSlot::bump_mru_walk` gates on the per-node `in_list`
    /// flag, so SWA tombstones (no SWA value, not in SWA's LRU) are
    /// skipped — they're not revived by an access bump.
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        SwaLRUSlot::bump_mru_walk(pool, node_idx);
    }

    /// SWA-specific redistribution after a node split. Owns ALL SWA-side
    /// concerns at split time: lock_ref copy, swa_uuid_for_lock transfer,
    /// value slice, and LRU bump.
    ///
    /// `pool.split_node` only handles structural changes (key split,
    /// re-parent, child-map updates) plus FULL-component baseline (FULL
    /// value redistribution, FULL lock_ref copy, FULL LRU bump). FULL
    /// must stay in pool because EVERY split site needs it — including
    /// SwaComponent's own internal splits (`consume_value` Branch 2,
    /// `commit_insert_data_on_new_leaf` Case B), which can't dispatch
    /// per-component redistribute from inside a component method.
    ///
    /// SWA can opt-in via this hook because SwaComponent's internal
    /// split sites operate on SWA-empty nodes (tombstones with SWA
    /// lock_ref==0 by the inc_lock_ref invariant; fresh leaves with
    /// lock_ref==0 just-created). Skipping SWA handling on those
    /// internal splits is correct.
    ///
    /// **Why this matters for outer split sites**: without SWA value
    /// and lock_ref copy, splitting a SWA-populated SWA-locked node
    /// would leave the new parent as a "synthetic tombstone" with no
    /// SWA value but SWA lock_ref==0, while the child keeps both SWA
    /// value and lock_ref. The SWA inc/dec_lock_ref invariant
    /// `lock_ref > 0 implies has_value` would break on the parent if
    /// a future walk landed there. Copying SWA lock_ref to the parent
    /// keeps the invariant intact, and slicing the value gives the
    /// parent its in-list contribution.
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    ) {
        // 1. Copy SWA lock_ref from child to new parent. Unconditional —
        // an in-flight SWA acquire that bumped lock_ref on the original
        // child needs the new intermediate to absorb its later release-
        // walk past the boundary (mirrors FULL's lock_ref copy in pool).
        let child_swa_lock_ref = SwaLRUSlot::lock_ref(pool.get(child_idx));
        SwaLRUSlot::set_lock_ref(pool.get_mut(new_parent_idx), child_swa_lock_ref);

        // 2. Transfer SWA uuid (boundary marker) from child to new parent.
        // The boundary semantically belongs to the parent slice after the
        // split — an in-flight SWA acquire that stamped the original
        // node sees the same uuid on the new parent, so its release
        // walk still finds and stops at the right node. Mirrors OSS
        // `swa_radix_cache.py::_split_node`'s `new_node.swa_uuid =
        // child.swa_uuid; child.swa_uuid = None`. `take()` clears the
        // child's stamp; `None` (no in-flight acquire) is a no-op.
        let transferred_uuid = pool.get_mut(child_idx).swa_uuid_for_lock.take();
        pool.get_mut(new_parent_idx)
            .set_swa_uuid_for_lock(transferred_uuid);

        // 3. Slice SWA value across the boundary if present. Tombstones
        // (no SWA value) leave both halves tombstoned — and bump_mru_split
        // is correctly SKIPPED in that case to preserve the
        // `in_list ⟺ value.is_some()` invariant that apply_swa_writes
        // asserts.
        let Some(child_swa) = SwaLRUSlot::value(pool.get(child_idx)) else {
            return;
        };
        let child_swa = child_swa.shallow_clone();
        let total_len = child_swa.size()[0];
        let parent_part = child_swa.narrow(0, 0, split_len as i64);
        let child_part = child_swa.narrow(0, split_len as i64, total_len - split_len as i64);

        SwaLRUSlot::replace_value(pool, new_parent_idx, parent_part);
        SwaLRUSlot::replace_value(pool, child_idx, child_part);

        // 4. Both halves now have SWA value; ensure both are in SWA's LRU.
        // Original child was already in SWA's LRU (had SWA value
        // pre-split); new parent is being added for the first time.
        // bump_mru_split handles both via bump_mru's in-list /
        // not-in-list uniformity, and encodes the leaf-MRU ordering
        // (child more recent than parent — matches FULL's policy in
        // pool.split_node).
        SwaLRUSlot::bump_mru_split(pool, new_parent_idx, child_idx);
    }
}

/// SWA's per-walk match validator. Mirrors OSS `swa_component.py`'s
/// `create_match_validator`. Tracks contiguous-SWA-present run length
/// (in tokens) and approves a node iff:
///
///   * the node's SWA component value is present (not a tombstone), AND
///   * either no tombstone has been seen yet (pre-tombstone shortcut),
///     OR the run since the last tombstone has reached
///     `sliding_window_size` tokens.
///
/// The pre-tombstone shortcut mirrors OSS's `state["len"] =
/// float("inf")` initialization (and the equivalent
/// `match_len_since_tombstone = float("inf")` in `swa_radix_cache.py`):
/// any path from root that hasn't yet hit a tombstone is treated as
/// already-window-filled. The window-fill gating only applies after a
/// tombstone splits the path. Rationale per the OSS comment: freshly-
/// inserted, never-evicted cache entries are guaranteed valid SWA —
/// no point requiring them to re-fill the window.
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
