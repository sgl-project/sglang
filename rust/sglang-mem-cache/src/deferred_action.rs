use tch::Tensor;

use crate::tree_node_pool::NodeIdx;

/// Actions that coordinate Python-owned components (e.g. KV-pool allocator,
/// HiCache controller) with the Rust radix tree. The Python orchestrator
/// applies each one after the radix-tree call returns.
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

    // ---- HiCache actions ----
    /// Back up the device value to host. Emitted on the insert path when
    /// reuse reaches the write-through threshold; the orchestrator copies
    /// `value` device→host and calls back `set_host_full_values` to stamp.
    FullWriteThroughBackup { node_idx: NodeIdx, value: Tensor },

    /// Evict the device value of a backed-up node; the node stays as host value exists.
    FullDeviceEvictOnBackedUp {
        node_idx: NodeIdx,
        device_value: Tensor,
    },

    /// Evict the host value (the node is deleted if unreferenced).
    FullHostEvict {
        node_idx: NodeIdx,
        host_value: Tensor,
    },

    /// Write-back a device-only victim to host on evict
    FullWriteBackOnEvict { node_idx: NodeIdx, value: Tensor },
}
