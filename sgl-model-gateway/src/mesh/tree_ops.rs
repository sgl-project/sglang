//! Tree operation definitions for mesh synchronization
//!
//! Defines serializable tree operations that can be synchronized across mesh cluster nodes

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_insert_op_creation() {
        let op = TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        assert_eq!(op.text, "test_text");
        assert_eq!(op.tenant, "http://worker1:8000");
    }

    #[test]
    fn test_tree_remove_op_creation() {
        let op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        assert_eq!(op.tenant, "http://worker1:8000");
    }

    #[test]
    fn test_tree_operation_insert() {
        let insert_op = TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Insert(insert_op.clone());

        match &operation {
            TreeOperation::Insert(op) => {
                assert_eq!(op.text, "test_text");
                assert_eq!(op.tenant, "http://worker1:8000");
            }
            TreeOperation::Remove(_) => panic!("Expected Insert operation"),
        }
    }

    #[test]
    fn test_tree_operation_remove() {
        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Remove(remove_op.clone());

        match &operation {
            TreeOperation::Insert(_) => panic!("Expected Remove operation"),
            TreeOperation::Remove(op) => {
                assert_eq!(op.tenant, "http://worker1:8000");
            }
        }
    }

    #[test]
    fn test_tree_operation_serialization() {
        let insert_op = TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Insert(insert_op);

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: TreeOperation = serde_json::from_str(&serialized).unwrap();

        match (&operation, &deserialized) {
            (TreeOperation::Insert(a), TreeOperation::Insert(b)) => {
                assert_eq!(a.text, b.text);
                assert_eq!(a.tenant, b.tenant);
            }
            _ => panic!("Operations should match"),
        }
    }

    #[test]
    fn test_tree_operation_remove_serialization() {
        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        let operation = TreeOperation::Remove(remove_op);

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: TreeOperation = serde_json::from_str(&serialized).unwrap();

        match (&operation, &deserialized) {
            (TreeOperation::Remove(a), TreeOperation::Remove(b)) => {
                assert_eq!(a.tenant, b.tenant);
            }
            _ => panic!("Operations should match"),
        }
    }

    #[test]
    fn test_tree_state_new() {
        let state = TreeState::new("model1".to_string());
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.operations.len(), 0);
        assert_eq!(state.version, 0);
    }

    #[test]
    fn test_tree_state_default() {
        let state = TreeState::default();
        assert_eq!(state.model_id, "");
        assert_eq!(state.operations.len(), 0);
        assert_eq!(state.version, 0);
    }

    #[test]
    fn test_tree_state_add_operation() {
        let mut state = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            text: "text1".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Insert(insert_op));

        assert_eq!(state.operations.len(), 1);
        assert_eq!(state.version, 1);

        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Remove(remove_op));

        assert_eq!(state.operations.len(), 2);
        assert_eq!(state.version, 2);
    }

    #[test]
    fn test_tree_state_add_multiple_operations() {
        let mut state = TreeState::new("model1".to_string());

        for i in 0..5 {
            let insert_op = TreeInsertOp {
                text: format!("text_{}", i),
                tenant: format!("http://worker{}:8000", i),
            };
            state.add_operation(TreeOperation::Insert(insert_op));
        }

        assert_eq!(state.operations.len(), 5);
        assert_eq!(state.version, 5);
    }

    #[test]
    fn test_tree_state_serialization() {
        let mut state = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Insert(insert_op));

        let remove_op = TreeRemoveOp {
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Remove(remove_op));

        let serialized = serde_json::to_string(&state).unwrap();
        let deserialized: TreeState = serde_json::from_str(&serialized).unwrap();

        assert_eq!(state.model_id, deserialized.model_id);
        assert_eq!(state.operations.len(), deserialized.operations.len());
        assert_eq!(state.version, deserialized.version);
    }

    #[test]
    fn test_tree_state_clone() {
        let mut state = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        state.add_operation(TreeOperation::Insert(insert_op));

        let cloned = state.clone();
        assert_eq!(state.model_id, cloned.model_id);
        assert_eq!(state.operations.len(), cloned.operations.len());
        assert_eq!(state.version, cloned.version);
    }

    #[test]
    fn test_tree_state_equality() {
        let mut state1 = TreeState::new("model1".to_string());
        let mut state2 = TreeState::new("model1".to_string());

        let insert_op = TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        state1.add_operation(TreeOperation::Insert(insert_op.clone()));
        state2.add_operation(TreeOperation::Insert(insert_op));

        assert_eq!(state1, state2);
    }

    #[test]
    fn test_tree_operation_hash() {
        use std::collections::HashSet;

        let insert_op1 = TreeInsertOp {
            text: "text1".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };
        let insert_op2 = TreeInsertOp {
            text: "text1".to_string(),
            tenant: "http://worker1:8000".to_string(),
        };

        let op1 = TreeOperation::Insert(insert_op1);
        let op2 = TreeOperation::Insert(insert_op2);

        let mut set = HashSet::new();
        set.insert(op1.clone());
        set.insert(op2.clone());

        // Same operations should be considered equal
        assert_eq!(set.len(), 1);
    }
}
