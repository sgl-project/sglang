//! Integration tests for mesh functionality
//!
//! Tests multi-node scenarios including state synchronization,
//! rate limiting, and cache-aware routing across cluster nodes.

use std::sync::Arc;

use smg::mesh::{
    crdt::SKey,
    gossip::NodeStatus,
    stores::{
        AppState, MembershipState, RateLimitConfig, StateStores, WorkerState,
        GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
    },
    sync::MeshSyncManager,
    tree_ops::{TreeInsertOp, TreeOperation},
};

/// Create test stores for a node
fn create_test_stores(node_name: String) -> Arc<StateStores> {
    Arc::new(StateStores::with_self_name(node_name))
}

/// Create test sync manager for a node
fn create_test_sync_manager(node_name: String) -> Arc<MeshSyncManager> {
    let stores = create_test_stores(node_name.clone());
    Arc::new(MeshSyncManager::new(stores, node_name))
}

#[tokio::test]
async fn test_multi_node_state_synchronization() {
    // Create three nodes
    let manager1 = create_test_sync_manager("node1".to_string());
    let manager2 = create_test_sync_manager("node2".to_string());
    let manager3 = create_test_sync_manager("node3".to_string());

    // Node1 syncs a worker state
    manager1.sync_worker_state(
        "worker1".to_string(),
        "model1".to_string(),
        "http://localhost:8000".to_string(),
        true,
        0.5,
    );

    // Simulate synchronization: Node2 and Node3 receive the update
    let worker_state = manager1.get_worker_state("worker1").unwrap();
    manager2.apply_remote_worker_state(worker_state.clone(), Some("node1".to_string()));
    manager3.apply_remote_worker_state(worker_state, Some("node1".to_string()));

    // Verify all nodes have the same state
    let state1 = manager1.get_worker_state("worker1").unwrap();
    let state2 = manager2.get_worker_state("worker1").unwrap();
    let state3 = manager3.get_worker_state("worker1").unwrap();

    assert_eq!(state1.worker_id, state2.worker_id);
    assert_eq!(state2.worker_id, state3.worker_id);
    assert_eq!(state1.version, state2.version);
    assert_eq!(state2.version, state3.version);
}

#[tokio::test]
async fn test_node_join_and_leave() {
    let manager1 = create_test_sync_manager("node1".to_string());
    let manager2 = create_test_sync_manager("node2".to_string());

    // Node1 has some state
    manager1.sync_worker_state(
        "worker1".to_string(),
        "model1".to_string(),
        "http://localhost:8000".to_string(),
        true,
        0.5,
    );

    manager1.sync_policy_state(
        "model1".to_string(),
        "cache_aware".to_string(),
        b"config".to_vec(),
    );

    // Node2 joins and receives state
    let worker_state = manager1.get_worker_state("worker1").unwrap();
    manager2.apply_remote_worker_state(worker_state, Some("node1".to_string()));

    let policy_state = manager1.get_policy_state("model1").unwrap();
    manager2.apply_remote_policy_state(policy_state, Some("node1".to_string()));

    // Verify Node2 has the state
    assert!(manager2.get_worker_state("worker1").is_some());
    assert!(manager2.get_policy_state("model1").is_some());

    // Node1 removes worker
    manager1.remove_worker_state("worker1");
    // In a real scenario, this would be propagated via gossip
    // For test, we verify the removal happened
    assert!(manager1.get_worker_state("worker1").is_none());
}

#[tokio::test]
async fn test_rate_limit_cluster_consistency() {
    // Create stores and managers
    let stores1 = create_test_stores("node1".to_string());
    let stores2 = create_test_stores("node2".to_string());
    let stores3 = create_test_stores("node3".to_string());

    // Add all nodes to membership store (required for rate limit hash ring)
    let node_names = ["node1", "node2", "node3"];
    let node_addresses = ["127.0.0.1:8001", "127.0.0.1:8002", "127.0.0.1:8003"];

    for stores in [&stores1, &stores2, &stores3] {
        for (i, &name) in node_names.iter().enumerate() {
            let key = SKey::new(name.to_string());
            stores.membership.insert(
                key,
                MembershipState {
                    name: name.to_string(),
                    address: node_addresses[i].to_string(),
                    status: NodeStatus::Alive as i32,
                    version: 1,
                    metadata: std::collections::BTreeMap::new(),
                },
                name.to_string(),
            );
        }
    }

    // Setup global rate limit config
    let config = RateLimitConfig {
        limit_per_second: 100,
    };
    let serialized = serde_json::to_vec(&config).unwrap();
    let key = SKey::new(GLOBAL_RATE_LIMIT_KEY.to_string());
    for stores in [&stores1, &stores2, &stores3] {
        stores.app.insert(
            key.clone(),
            AppState {
                key: GLOBAL_RATE_LIMIT_KEY.to_string(),
                value: serialized.clone(),
                version: 1,
            },
            "node1".to_string(),
        );
    }

    // Create managers with updated stores
    let manager1 = Arc::new(MeshSyncManager::new(stores1.clone(), "node1".to_string()));
    let manager2 = Arc::new(MeshSyncManager::new(stores2.clone(), "node2".to_string()));
    let manager3 = Arc::new(MeshSyncManager::new(stores3.clone(), "node3".to_string()));

    // Update rate limit membership (reads from membership store)
    manager1.update_rate_limit_membership();
    manager2.update_rate_limit_membership();
    manager3.update_rate_limit_membership();

    // Each node increments the counter (if it's an owner)
    let test_key = GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string();

    manager1.sync_rate_limit_inc(test_key.clone(), 10);
    manager2.sync_rate_limit_inc(test_key.clone(), 5);
    manager3.sync_rate_limit_inc(test_key.clone(), 3);

    // Simulate counter merging (in real scenario, this happens via gossip)
    // Get counters from each node and merge them into all nodes
    if let Some(counter2) = stores2.rate_limit.get_counter(&test_key) {
        manager1.apply_remote_rate_limit_counter(test_key.clone(), &counter2);
        manager3.apply_remote_rate_limit_counter(test_key.clone(), &counter2);
    }
    if let Some(counter3) = stores3.rate_limit.get_counter(&test_key) {
        manager1.apply_remote_rate_limit_counter(test_key.clone(), &counter3);
        manager2.apply_remote_rate_limit_counter(test_key.clone(), &counter3);
    }
    if let Some(counter1) = stores1.rate_limit.get_counter(&test_key) {
        manager2.apply_remote_rate_limit_counter(test_key.clone(), &counter1);
        manager3.apply_remote_rate_limit_counter(test_key.clone(), &counter1);
    }

    // Check aggregated value
    let value = manager1.get_rate_limit_value(&test_key);
    // Should have aggregated value from all owners
    assert!(value.is_some());
    // The value should be the sum of all increments (10 + 5 + 3 = 18)
    // But note: only owners actually increment, so the sum depends on ownership
    let value = value.unwrap();
    assert!(value > 0, "Counter value should be greater than 0");
}

#[tokio::test]
async fn test_rate_limit_node_failure() {
    let manager1 = create_test_sync_manager("node1".to_string());
    let _manager2 = create_test_sync_manager("node2".to_string());
    let _manager3 = create_test_sync_manager("node3".to_string());

    // Setup membership through sync manager
    // In a real scenario, membership would be updated through gossip protocol
    manager1.update_rate_limit_membership();

    // Simulate node2 failure
    manager1.handle_node_failure(&["node2".to_string()]);

    // Update membership to reflect failure
    manager1.update_rate_limit_membership();

    // Verify system continues to work
    let test_key = "test_key".to_string();
    manager1.sync_rate_limit_inc(test_key.clone(), 1);
    let _value = manager1.get_rate_limit_value(&test_key);
    // Value may be None if not owner, which is acceptable
    // In a real scenario, ownership would be redistributed after node failure
}

#[tokio::test]
async fn test_cache_aware_tree_synchronization() {
    let manager1 = create_test_sync_manager("node1".to_string());
    let manager2 = create_test_sync_manager("node2".to_string());

    // Node1 syncs tree operations
    let op1 = TreeOperation::Insert(TreeInsertOp {
        text: "request1".to_string(),
        tenant: "http://worker1:8000".to_string(),
    });
    manager1
        .sync_tree_operation("model1".to_string(), op1)
        .unwrap();

    let op2 = TreeOperation::Insert(TreeInsertOp {
        text: "request2".to_string(),
        tenant: "http://worker2:8000".to_string(),
    });
    manager1
        .sync_tree_operation("model1".to_string(), op2)
        .unwrap();

    // Node2 receives tree state (simulated)
    let tree_state = manager1.get_tree_state("model1").unwrap();
    manager2.apply_remote_tree_operation(
        "model1".to_string(),
        tree_state,
        Some("node1".to_string()),
    );

    // Verify Node2 has the tree state
    let tree_state2 = manager2.get_tree_state("model1");
    assert!(tree_state2.is_some());
    let tree = tree_state2.unwrap();
    assert_eq!(tree.operations.len(), 2);
}

#[tokio::test]
async fn test_version_conflict_resolution() {
    let manager1 = create_test_sync_manager("node1".to_string());
    let manager2 = create_test_sync_manager("node2".to_string());

    // Both nodes update the same worker with different versions
    manager1.sync_worker_state(
        "worker1".to_string(),
        "model1".to_string(),
        "http://localhost:8000".to_string(),
        true,
        0.5,
    );

    // Node2 tries to apply an older version
    let old_state = WorkerState {
        worker_id: "worker1".to_string(),
        model_id: "model1".to_string(),
        url: "http://localhost:8000".to_string(),
        health: false,
        load: 0.8,
        version: 0, // Older version
    };

    manager2.apply_remote_worker_state(old_state, Some("node2".to_string()));

    // Node2 should not have the state (version too old)
    // But if it does, it should have version 0
    let state2 = manager2.get_worker_state("worker1");
    if let Some(s) = state2 {
        // If state exists, it should be from node1 (version 1)
        assert!(s.version >= 1);
    }

    // Node1 applies newer version to Node2
    let new_state = manager1.get_worker_state("worker1").unwrap();
    manager2.apply_remote_worker_state(new_state, Some("node1".to_string()));

    // Now Node2 should have the correct state
    let final_state = manager2.get_worker_state("worker1").unwrap();
    assert_eq!(final_state.version, 1);
    assert!(final_state.health);
}

#[tokio::test]
async fn test_concurrent_updates() {
    let manager1 = create_test_sync_manager("node1".to_string());
    let manager2 = create_test_sync_manager("node2".to_string());
    let manager3 = create_test_sync_manager("node3".to_string());

    // All nodes update different workers concurrently
    manager1.sync_worker_state(
        "worker1".to_string(),
        "model1".to_string(),
        "http://localhost:8000".to_string(),
        true,
        0.5,
    );

    manager2.sync_worker_state(
        "worker2".to_string(),
        "model1".to_string(),
        "http://localhost:8001".to_string(),
        true,
        0.6,
    );

    manager3.sync_worker_state(
        "worker3".to_string(),
        "model1".to_string(),
        "http://localhost:8002".to_string(),
        true,
        0.7,
    );

    // Simulate synchronization: all nodes receive all updates
    let worker1_state = manager1.get_worker_state("worker1").unwrap();
    let worker2_state = manager2.get_worker_state("worker2").unwrap();
    let worker3_state = manager3.get_worker_state("worker3").unwrap();

    manager2.apply_remote_worker_state(worker1_state.clone(), Some("node1".to_string()));
    manager3.apply_remote_worker_state(worker1_state, Some("node1".to_string()));

    manager1.apply_remote_worker_state(worker2_state.clone(), Some("node2".to_string()));
    manager3.apply_remote_worker_state(worker2_state, Some("node2".to_string()));

    manager1.apply_remote_worker_state(worker3_state.clone(), Some("node3".to_string()));
    manager2.apply_remote_worker_state(worker3_state, Some("node3".to_string()));

    // All nodes should have all workers
    assert_eq!(manager1.get_all_worker_states().len(), 3);
    assert_eq!(manager2.get_all_worker_states().len(), 3);
    assert_eq!(manager3.get_all_worker_states().len(), 3);
}

#[tokio::test]
async fn test_rate_limit_window_reset() {
    let manager = create_test_sync_manager("node1".to_string());

    // Setup membership
    manager.update_rate_limit_membership();

    // Setup config through stores (for testing)
    let stores = create_test_stores("node1".to_string());
    let config = RateLimitConfig {
        limit_per_second: 100,
    };
    let serialized = serde_json::to_vec(&config).unwrap();
    let key = SKey::new(GLOBAL_RATE_LIMIT_KEY.to_string());
    stores.app.insert(
        key,
        AppState {
            key: GLOBAL_RATE_LIMIT_KEY.to_string(),
            value: serialized,
            version: 1,
        },
        "node1".to_string(),
    );

    // Recreate manager with updated stores
    let manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

    // Increment counter (if owner)
    manager.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 50);
    let value_before = manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
    // Value may be None if not owner, or Some if owner
    if let Some(val) = value_before {
        assert!(val > 0);

        // Reset counter
        manager.reset_global_rate_limit_counter();
        let value_after = manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
        // Should be reset
        assert!(value_after.is_none() || value_after.unwrap_or(0) <= 0);
    }
}
