use tch::Tensor;

use crate::component_type::ComponentType;
use crate::tree_node_pool::NodeIdx;

/// An action emitted by the Rust radix tree for the Python orchestrator to apply.
#[derive(Debug)]
pub enum DeferredAction {
    /// Free full-attention KV indices duplicated by an overlapping prefix.
    FullFree { full_to_free: Tensor },

    /// Recover an SWA tombstone node, replacing its FULL value.
    SwaRecover {
        node_idx: NodeIdx,
        old_full_to_free: Tensor,
        /// Full KV mapped (full->swa) into the node's SWA value.
        new_full_value: Tensor,
    },

    /// Stamp an SWA value on a newly created leaf.
    SwaStamp {
        node_idx: NodeIdx,
        /// Full KV mapped (full->swa) into the node's SWA value.
        full_value: Tensor,
    },
}

impl DeferredAction {
    /// The component that owns this action; the Python consumer routes by it.
    pub fn component_type(&self) -> ComponentType {
        match self {
            DeferredAction::FullFree { .. } => ComponentType::Full,
            DeferredAction::SwaRecover { .. } | DeferredAction::SwaStamp { .. } => {
                ComponentType::Swa
            }
        }
    }
}
