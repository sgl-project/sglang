//! Tree operation definitions for HA synchronization
//!
//! Defines serializable tree operations that can be synchronized across HA cluster nodes

use serde::{Deserialize, Serialize};

/// Tree insert operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TreeInsertOp {
    pub text: String,
    pub tenant: String, // worker URL
}

/// Tree remove operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TreeRemoveOp {
    pub tenant: String, // worker URL
}

/// Tree operation type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TreeOperation {
    Insert(TreeInsertOp),
    Remove(TreeRemoveOp),
}

/// Tree state for a specific model
/// Contains a sequence of operations that can be applied to reconstruct the tree
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct TreeState {
    pub model_id: String,
    pub operations: Vec<TreeOperation>,
    pub version: u64,
}

impl TreeState {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            operations: Vec::new(),
            version: 0,
        }
    }

    pub fn add_operation(&mut self, operation: TreeOperation) {
        self.operations.push(operation);
        self.version += 1;
    }
}
