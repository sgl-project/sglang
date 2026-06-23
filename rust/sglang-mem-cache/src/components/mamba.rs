//! Mamba component for the radix tree.

use super::{Component, IncLockRefResult, MatchValidator};
use crate::component_type::ComponentType;
use crate::error::RadixCacheInitError;
use crate::tree_node_lru::{EvictRequest, EvictResult, LRUSlot, MambaLRUSlot, evict_non_full};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNode, TreeNodePool};

/// Mamba radix-tree component.
pub struct MambaComponent {
    #[allow(dead_code)]
    mamba_cache_chunk_size: usize,
}

/// Approves nodes that have a Mamba value.
pub struct MambaMatchValidator;

impl<K: ChildKeyType> MatchValidator<K> for MambaMatchValidator {
    fn validate(&mut self, n: &TreeNode<K>) -> bool {
        MambaLRUSlot::has_value(n)
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

    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult> {
        let delta = if MambaLRUSlot::has_value(pool.get(node_idx)) {
            MambaLRUSlot::inc_lock_ref(pool, node_idx)
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
        let delta = if MambaLRUSlot::has_value(pool.get(node_idx)) {
            MambaLRUSlot::dec_lock_ref(pool, node_idx)
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
            evict_non_full::<K, MambaLRUSlot>(pool, target - already, result);
        }
    }

    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        MambaLRUSlot::bump_mru_walk(pool, node_idx);
    }

    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        _new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        _split_len: usize,
    ) {
        // Mamba state stays at the child; the new parent has none.
        if MambaLRUSlot::has_value(pool.get(child_idx)) {
            MambaLRUSlot::bump_mru(pool, child_idx);
        }
    }
}
