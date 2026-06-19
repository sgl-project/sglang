//! Mamba component for the radix tree.

use super::{Component, IncLockRefResult, MatchValidator};
use crate::component_type::ComponentType;
use crate::error::RadixCacheInitError;
use crate::tree_node_lru::{evict_non_full, EvictRequest, EvictResult, LRUSlot, MambaLRUSlot};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNode, TreeNodePool};

/// Per-component shell that hosts Mamba-specific radix-tree logic.
pub struct MambaComponent {
    /// Chunk granularity for Mamba SSM checkpoints — only positions at
    /// multiples of this size carry SSM states. Guaranteed >= `page_size`.
    #[allow(dead_code)]
    mamba_cache_chunk_size: usize,
}

/// Per-walk Mamba validator: approve iff the node has a Mamba value.
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

    /// Single-node update: bump Mamba's `lock_ref` only if the node
    /// has a Mamba value populated.
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

    /// Single-node update: decrement Mamba's `lock_ref` only if the
    /// node has a Mamba value populated. `swa_uuid_for_lock` is ignored.
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

    /// Evict Mamba values and remove unreferenced leaves.
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

    /// Redistribute node values on split.
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        _new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        _split_len: usize,
    ) {
        // a) Mamba lock_ref is NOT copied to the new parent (Mamba
        //    acquire targets the leaf only, never intermediates).
        // b) Mamba value stays at the child (split point has no SSM
        //    state; new parent is born a tombstone).
        // c) Only bump the child's Mamba LRU position; the new parent
        //    has no Mamba value and is not in Mamba's LRU.
        if MambaLRUSlot::has_value(pool.get(child_idx)) {
            MambaLRUSlot::bump_mru(pool, child_idx);
        }
    }
}
