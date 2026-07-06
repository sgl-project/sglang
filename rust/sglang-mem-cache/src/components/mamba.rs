//! Mamba component for the radix tree.

use tch::Tensor;

use super::{Component, IncLockRefResult, MatchValidator};
use super::{
    EvictRequest, EvictResult, Slot, dec_lock_ref_non_full, evict_non_full, inc_lock_ref_non_full,
};
use crate::component_type::ComponentType;
use crate::error::{RadixCacheInitError, RadixCacheRuntimeError};
use crate::radix_cache::{InsertResult, MatchResult};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNode, TreeNodePool};

/// Mamba radix-tree component.
pub struct MambaComponent {
    mamba_cache_chunk_size: usize,
}

/// Approves nodes that have a Mamba value.
pub struct MambaMatchValidator;

impl<K: ChildKeyType> MatchValidator<K> for MambaMatchValidator {
    fn validate(&mut self, n: &TreeNode<K>) -> bool {
        MambaSlot::has_value(n)
    }
}

impl MambaComponent {
    pub fn new(
        mamba_cache_chunk_size: usize,
        page_size: usize,
    ) -> Result<Self, RadixCacheInitError> {
        if mamba_cache_chunk_size < page_size {
            return Err(RadixCacheInitError::MambaCacheChunkSizeBelowPageSize {
                chunk_size: mamba_cache_chunk_size,
                page_size,
            });
        }
        Ok(Self {
            mamba_cache_chunk_size,
        })
    }
}

impl<K: ChildKeyType> Component<K> for MambaComponent {
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>> {
        Some(Box::new(MambaMatchValidator))
    }

    /// Fill the Mamba branching seqlen + cached state on the match result.
    fn finalize_match_result(
        &self,
        pool: &TreeNodePool<K>,
        last_matched_node_idx: NodeIdx,
        values: &[Tensor],
        last_device_value_len: usize,
        result: &mut MatchResult,
    ) {
        // Branching seqlen only when the walk extended past the device boundary.
        if last_device_value_len < values.len() {
            let total: usize = values.iter().map(|v| v.size()[0] as usize).sum();
            let aligned = total / self.mamba_cache_chunk_size * self.mamba_cache_chunk_size;
            result.mamba_branching_seqlen = (aligned > 0).then_some(aligned);
        }
        result.mamba_value =
            MambaSlot::value(pool.get(last_matched_node_idx)).map(|t| t.shallow_clone());
    }

    /// Populate the Mamba SSM value on the inserted node; sets
    /// `result.mamba_value_exists` when an existing node already held one (caller
    /// keeps its slot).
    fn commit_insert_value(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        new_leaf: bool,
        value: Option<&Tensor>,
        result: &mut InsertResult,
    ) -> Result<(), RadixCacheRuntimeError> {
        let Some(mv) = value else {
            return Ok(());
        };
        if new_leaf || !MambaSlot::has_value(pool.get(node_idx)) {
            MambaSlot::set_value(pool, node_idx, mv.shallow_clone())?;
            let delta = MambaSlot::value_len(pool.get(node_idx));
            MambaSlot::bump_mru(pool, node_idx);
            MambaSlot::pool_state_mut(pool).unlocked_size += delta;
        } else {
            result.mamba_value_exists = true;
        }
        Ok(())
    }

    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult> {
        let delta = if MambaSlot::has_value(pool.get(node_idx)) {
            MambaSlot::inc_lock_ref(pool, node_idx)
        } else {
            0
        };
        Some(IncLockRefResult {
            delta,
            swa_uuid_for_lock: None,
        })
    }

    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        _swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64> {
        let delta = if MambaSlot::has_value(pool.get(node_idx)) {
            MambaSlot::dec_lock_ref(pool, node_idx)
        } else {
            0
        };
        Some(delta)
    }

    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult) {
        let ct = ComponentType::Mamba as usize;
        let target = request.num_tokens[ct];
        let already = result.evicted[ct];
        if already < target {
            evict_non_full::<K, MambaSlot>(pool, target - already, result);
        }
    }

    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        MambaSlot::bump_mru_walk(pool, node_idx);
    }

    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        _new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        _split_len: usize,
    ) {
        // Mamba state stays at the child; the new parent has none.
        if MambaSlot::has_value(pool.get(child_idx)) {
            MambaSlot::bump_mru(pool, child_idx);
        }
    }
}

/// Mamba component slot.
pub struct MambaSlot;

impl Slot for MambaSlot {
    const COMPONENT: ComponentType = ComponentType::Mamba;
    const NAME: &'static str = "Mamba";

    /// 1 SSM state per TreeNode.
    fn value_len<K: ChildKeyType>(_node: &TreeNode<K>) -> usize {
        1
    }

    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        inc_lock_ref_non_full::<K, Self>(pool, node_idx, /* enforce_full_cap */ true)
    }

    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        dec_lock_ref_non_full::<K, Self>(pool, node_idx)
    }
}
