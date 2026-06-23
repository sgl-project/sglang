use tch::Tensor;

use crate::tree_node_pool::NodeIdx;

/// Actions that coordinate the Python-owned KV-pool allocator with the Rust
/// radix tree. The Python orchestrator applies each one after the radix-tree
/// call returns.
#[derive(Debug)]
pub enum DeferredAction {
    /// Free full-attention KV indices duplicated by an overlapping prefix.
    FullDupFreed { freed_indices: Tensor },

    /// Recover an SWA tombstone node, replacing its FULL value with a new one.
    SwaRecover {
        node_idx: NodeIdx,
        freed_full: Tensor,
        source_value: Tensor,
    },

    /// Stamp SWA value on a newly created leaf.
    SwaStamp {
        node_idx: NodeIdx,
        source_value: Tensor,
    },
}
