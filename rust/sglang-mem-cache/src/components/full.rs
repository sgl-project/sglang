//! Tree node component which maintains device and host FULL values.

use tch::Tensor;

use super::{Component, IncLockRefResult, MatchValidator};
use crate::component_type::ComponentType;
use crate::deferred_action::DeferredAction;
use crate::error::RadixCacheRuntimeError;
use crate::tree_node_lru::{
    evict_full, EvictRequest, EvictResult, FullLRUSlot, HostFullLRUSlot, LRUSlot,
};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNodePool};

pub struct FullComponent {
    /// Whether HiCache enabled.
    enable_hicache: bool,
    /// Whether the HiCache write-back policy is active
    hicache_write_back: bool,
}

impl FullComponent {
    pub fn new(enable_hicache: bool, hicache_write_back: bool) -> Self {
        Self {
            enable_hicache,
            hicache_write_back,
        }
    }
}

impl<K: ChildKeyType> Component<K> for FullComponent {
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>> {
        // FULL has no boundary-gating logic — return `None` instead
        // of a wrapper that always says `true`. The orchestrator skips
        // the per-node `validate()` call (and the allocation) for any
        // component that returns `None`. FULL-only deployments pay
        // zero per-call allocation and zero per-node vtable hits.
        None
    }

    /// Walk from `node_idx` up to root (but not including) to bump each node's
    /// lock_ref counter.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult> {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        // Walk up via parent links, stopping at the namespace root
        // (which has `parent == None`).
        while let Some(parent) = pool.get(current).parent() {
            delta += FullLRUSlot::inc_lock_ref(pool, current);
            current = parent;
        }
        Some(IncLockRefResult {
            delta,
            swa_uuid_for_lock: None,
        })
    }

    /// Walk from `node_idx` up to root (but not including) to decrease each
    /// node's lock_ref counter.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        _swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64> {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = pool.get(current).parent() {
            delta += FullLRUSlot::dec_lock_ref(pool, current);
            current = parent;
        }
        Some(delta)
    }

    /// Evict FULL values and remove unreferenced leaves.
    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult) {
        let ct = ComponentType::Full as usize;
        let target = request.num_tokens[ct];
        let already = result.evicted[ct];
        if already < target {
            evict_full(pool, target - already, self.hicache_write_back, result);
        }
    }

    /// Bump device and host LRU accordingly; a node-level bump is skipped when
    /// its tier value is absent (not-in-list).
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        FullLRUSlot::bump_mru_walk(pool, node_idx);
        if self.enable_hicache {
            HostFullLRUSlot::bump_mru_walk(pool, node_idx);
        }
    }

    /// Redistribute node values from `child_idx` to `new_parent_idx`
    /// (`child_idx` was just structurally split into
    /// `new_parent_idx -> child_idx` at `split_len`).
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    ) {
        // Redistribute full value, and ensure value present if HiCache is off.
        let require_device_value = !self.enable_hicache;
        FullLRUSlot::redistribute_on_split(
            pool,
            new_parent_idx,
            child_idx,
            split_len,
            require_device_value,
        );

        // Redistribute hit count and host full value if there's any.
        let child_hit_count = pool.get(child_idx).hit_count;
        pool.get_mut(new_parent_idx).hit_count = child_hit_count;
        HostFullLRUSlot::redistribute_on_split(pool, new_parent_idx, child_idx, split_len, false);
    }

    /// When the device value is not set (backed up by HiCache), claim the overlap
    /// slice to repopulate it (un-evict); otherwise it is a duplicate.
    #[allow(clippy::too_many_arguments)]
    fn consume_value(
        &self,
        pool: &mut TreeNodePool<K>,
        _components: &[Box<dyn Component<K>>],
        child_idx: NodeIdx,
        node_key_len: usize,
        _total_prefix_len: usize,
        _prev_prefix_len: usize,
        value_slice: &Tensor,
        _swa_evicted_seqlen: usize,
        _deferred: &mut Vec<DeferredAction>,
    ) -> Result<usize, RadixCacheRuntimeError> {
        if FullLRUSlot::has_value(pool.get(child_idx)) {
            return Ok(node_key_len);
        }
        // A device-absent node is guaranteed to be unreferenced by any request
        // (lock_ref == 0); a non-zero lock_ref is an invariant violation.
        if FullLRUSlot::lock_ref(pool.get(child_idx)) != 0 {
            return Err(RadixCacheRuntimeError::UnevictLockedNode {
                node_idx: child_idx,
            });
        }
        // TODO(Jialin): fold set_value + bump_mru + unlocked_size credit into a
        // shared set_value(idx, value, update_lru) helper.
        FullLRUSlot::set_value(pool, child_idx, value_slice.shallow_clone())?;
        FullLRUSlot::bump_mru(pool, child_idx);
        FullLRUSlot::pool_state_mut(pool).unlocked_size += node_key_len;
        Ok(0)
    }
}
