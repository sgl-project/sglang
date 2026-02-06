//! Test utilities for mesh module

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use parking_lot::RwLock;

use super::{
    service::{gossip::NodeState, ClusterState},
    stores::{MembershipState, StateStores},
    sync::MeshSyncManager,
};

/// Create test StateStores with a given node name
pub fn create_test_stores(self_name: String) -> Arc<StateStores> {
    Arc::new(StateStores::with_self_name(self_name))
}

/// Create test MeshSyncManager
pub fn create_test_sync_manager(self_name: String) -> Arc<MeshSyncManager> {
    let stores = create_test_stores(self_name.clone());
    Arc::new(MeshSyncManager::new(stores, self_name))
}

/// Create test cluster state with given nodes
pub fn create_test_cluster_state(
    nodes: Vec<(String, String, i32)>, // (name, address, status)
) -> ClusterState {
    let mut state = BTreeMap::new();
    for (name, address, status) in nodes {
        state.insert(
            name.clone(),
            NodeState {
                name: name.clone(),
                address,
                status,
                version: 1,
                metadata: HashMap::new(),
            },
        );
    }
    Arc::new(RwLock::new(state))
}

/// Create test membership state
#[allow(dead_code)]
pub fn create_test_membership_state(name: String, address: String, status: i32) -> MembershipState {
    MembershipState {
        name,
        address,
        status,
        version: 1,
        metadata: BTreeMap::new(),
    }
}

#[cfg(test)]
mod test_utils_tests {
    use super::*;

    #[test]
    fn test_create_test_stores() {
        let stores = create_test_stores("test_node".to_string());
        assert!(!stores.rate_limit.is_owner("key1"));
    }

    #[test]
    fn test_create_test_sync_manager() {
        let manager = create_test_sync_manager("test_node".to_string());
        assert_eq!(manager.self_name(), "test_node");
    }

    #[test]
    fn test_create_test_cluster_state() {
        let state = create_test_cluster_state(vec![
            ("node1".to_string(), "127.0.0.1:8000".to_string(), 1),
            ("node2".to_string(), "127.0.0.1:8001".to_string(), 1),
        ]);
        let read_state = state.read();
        assert_eq!(read_state.len(), 2);
        assert!(read_state.contains_key("node1"));
        assert!(read_state.contains_key("node2"));
    }
}
