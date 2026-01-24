//! Mesh state synchronization module
//!
//! Handles synchronization of worker and policy states across mesh cluster nodes

use std::sync::Arc;

use tracing::debug;

use super::{
    crdt::SKey,
    gossip::NodeStatus,
    stores::{
        tree_state_key, PolicyState, RateLimitConfig, StateStores, WorkerState,
        GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
    },
    tree_ops::{TreeOperation, TreeState},
};

/// Mesh sync manager for coordinating state synchronization
#[derive(Clone, Debug)]
pub struct MeshSyncManager {
    pub(crate) stores: Arc<StateStores>,
    self_name: String,
}

impl MeshSyncManager {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self { stores, self_name }
    }

    /// Get the node name (actor) for this sync manager
    pub fn self_name(&self) -> &str {
        &self.self_name
    }

    /// Sync worker state to mesh stores
    pub fn sync_worker_state(
        &self,
        worker_id: String,
        model_id: String,
        url: String,
        health: bool,
        load: f64,
    ) {
        let key = SKey::new(worker_id.clone());

        // Get current version if exists, otherwise start at 1
        let current_version = self
            .stores
            .worker
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);
        let new_version = current_version + 1;

        let state = WorkerState {
            worker_id: worker_id.clone(),
            model_id,
            url,
            health,
            load,
            version: new_version,
        };

        // Use self node name as actor
        let actor = self.self_name.clone();
        self.stores.worker.insert(key, state, actor);
        debug!(
            "Synced worker state to mesh {} (version: {})",
            worker_id, new_version
        );
    }

    /// Remove worker state from mesh stores
    pub fn remove_worker_state(&self, worker_id: &str) {
        let key = SKey::new(worker_id.to_string());
        self.stores.worker.remove(&key);
        debug!("Removed worker state from mesh {}", worker_id);
    }

    /// Sync policy state to mesh stores
    pub fn sync_policy_state(&self, model_id: String, policy_type: String, config: Vec<u8>) {
        let key = SKey::new(format!("policy:{}", model_id));

        // Get current version if exists, otherwise start at 1
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);
        let new_version = current_version + 1;

        let state = PolicyState {
            model_id: model_id.clone(),
            policy_type,
            config,
            version: new_version,
        };

        // Use self node name as actor
        let actor = self.self_name.clone();
        self.stores.policy.insert(key, state, actor);
        debug!(
            "Synced policy state to mesh model={} (version: {})",
            model_id, new_version
        );
    }

    /// Remove policy state from mesh stores
    pub fn remove_policy_state(&self, model_id: &str) {
        let key = SKey::new(format!("policy:{}", model_id));
        self.stores.policy.remove(&key);
        debug!("Removed policy state from mesh model={}", model_id);
    }

    /// Get worker state from mesh stores
    pub fn get_worker_state(&self, worker_id: &str) -> Option<WorkerState> {
        let key = SKey::new(worker_id.to_string());
        self.stores.worker.get(&key)
    }

    /// Get all worker states from mesh stores
    pub fn get_all_worker_states(&self) -> Vec<WorkerState> {
        self.stores.worker.all().into_values().collect()
    }

    /// Get policy state from mesh stores
    pub fn get_policy_state(&self, model_id: &str) -> Option<PolicyState> {
        let key = SKey::new(format!("policy:{}", model_id));
        self.stores.policy.get(&key)
    }

    /// Get all policy states from mesh stores
    pub fn get_all_policy_states(&self) -> Vec<PolicyState> {
        self.stores.policy.all().into_values().collect()
    }

    /// Apply worker state update from remote node
    /// The actor should be extracted from the state update context (e.g., from StateUpdate message)
    pub fn apply_remote_worker_state(&self, state: WorkerState, actor: Option<String>) {
        let key = SKey::new(state.worker_id.clone());
        // Use provided actor, or fallback to a default if not available
        // In practice, actor should come from the StateUpdate message
        let actor = actor.unwrap_or_else(|| "remote".to_string());

        // Check if we should update based on version
        let current_version = self
            .stores
            .worker
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);

        if state.version > current_version {
            self.stores.worker.insert(key, state.clone(), actor.clone());
            debug!(
                "Applied remote worker state update: {} (version: {} -> {})",
                state.worker_id, current_version, state.version
            );
        } else {
            debug!(
                "Skipped remote worker state update: {} (version {} <= current {})",
                state.worker_id, state.version, current_version
            );
        }
    }

    /// Apply policy state update from remote node
    /// The actor should be extracted from the state update context (e.g., from StateUpdate message)
    pub fn apply_remote_policy_state(&self, state: PolicyState, actor: Option<String>) {
        let key = SKey::new(format!("policy:{}", state.model_id));
        // Use provided actor, or fallback to a default if not available
        let actor = actor.unwrap_or_else(|| "remote".to_string());

        // Check if we should update based on version
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);

        if state.version > current_version {
            self.stores.policy.insert(key, state.clone(), actor.clone());
            debug!(
                "Applied remote policy state update: {} (version: {} -> {})",
                state.model_id, current_version, state.version
            );
        } else {
            debug!(
                "Skipped remote policy state update: {} (version {} <= current {})",
                state.model_id, state.version, current_version
            );
        }
    }

    /// Update rate-limit hash ring with current membership
    pub fn update_rate_limit_membership(&self) {
        // Get all alive nodes from membership store
        let all_members = self.stores.membership.all();
        let alive_nodes: Vec<String> = all_members
            .values()
            .filter(|m| m.status == NodeStatus::Alive as i32)
            .map(|m| m.name.clone())
            .collect();

        self.stores.rate_limit.update_membership(&alive_nodes);
        debug!(
            "Updated rate-limit hash ring with {} alive nodes",
            alive_nodes.len()
        );
    }

    /// Handle node failure and transfer rate-limit ownership
    pub fn handle_node_failure(&self, failed_nodes: &[String]) {
        if failed_nodes.is_empty() {
            return;
        }

        debug!("Handling node failure for rate-limit: {:?}", failed_nodes);

        // Check which keys need ownership transfer
        let affected_keys = self
            .stores
            .rate_limit
            .check_ownership_transfer(failed_nodes);

        if !affected_keys.is_empty() {
            debug!(
                "Ownership transfer needed for {} rate-limit keys",
                affected_keys.len()
            );

            // Update membership to reflect node failures
            self.update_rate_limit_membership();

            // For each affected key, we may need to initialize counters if we're now an owner
            for key in &affected_keys {
                if self.stores.rate_limit.is_owner(key) {
                    debug!("This node is now owner of rate-limit key: {}", key);
                    // Counter will be created on first inc() call
                }
            }
        }
    }

    /// Sync rate-limit counter increment (only if this node is an owner)
    pub fn sync_rate_limit_inc(&self, key: String, delta: i64) {
        if !self.stores.rate_limit.is_owner(&key) {
            // Not an owner, skip
            return;
        }

        self.stores
            .rate_limit
            .inc(key.clone(), self.self_name.clone(), delta);
        debug!("Synced rate-limit increment: key={}, delta={}", key, delta);
    }

    /// Apply remote rate-limit counter update (merge CRDT)
    pub fn apply_remote_rate_limit_counter(
        &self,
        key: String,
        counter: &super::crdt::SyncPNCounter,
    ) {
        // Merge counter regardless of ownership (for CRDT consistency)
        self.stores.rate_limit.merge_counter(key.clone(), counter);
        debug!("Applied remote rate-limit counter update: key={}", key);
    }

    /// Get rate-limit value (aggregate from all owners)
    pub fn get_rate_limit_value(&self, key: &str) -> Option<i64> {
        self.stores.rate_limit.value(key)
    }

    /// Get global rate limit configuration from AppStore
    pub fn get_global_rate_limit_config(&self) -> Option<RateLimitConfig> {
        let key = SKey::new(GLOBAL_RATE_LIMIT_KEY.to_string());
        self.stores
            .app
            .get(&key)
            .and_then(|app_state| serde_json::from_slice::<RateLimitConfig>(&app_state.value).ok())
    }

    /// Check if global rate limit is exceeded
    /// Returns (is_exceeded, current_count, limit)
    pub fn check_global_rate_limit(&self) -> (bool, i64, u64) {
        let config = self.get_global_rate_limit_config().unwrap_or_default();

        if config.limit_per_second == 0 {
            // Rate limit disabled
            return (false, 0, 0);
        }

        // Increment counter if this node is an owner
        self.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 1);

        // Get aggregated counter value from all owners
        let current_count = self
            .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
            .unwrap_or(0);

        let is_exceeded = current_count > config.limit_per_second as i64;
        (is_exceeded, current_count, config.limit_per_second)
    }

    /// Reset global rate limit counter (called periodically for time window reset)
    pub fn reset_global_rate_limit_counter(&self) {
        // Reset by decrementing the current value
        // Since we use PNCounter, we can't directly reset, but we can track the window
        // For simplicity, we'll use a time-based approach where counters are reset periodically
        // The actual reset logic will be handled by the window manager
        let current_count = self
            .get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY)
            .unwrap_or(0);

        if current_count > 0 {
            // Decrement by current count to effectively reset
            // Note: This is a workaround since PNCounter doesn't support direct reset
            // In production, you might want to use a different approach like timestamped counters
            self.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), -current_count);
        }
    }

    /// Sync tree operation to mesh stores
    /// This adds a tree operation (insert or remove) to the tree state for a specific model
    pub fn sync_tree_operation(
        &self,
        model_id: String,
        operation: TreeOperation,
    ) -> Result<(), String> {
        let key = SKey::new(tree_state_key(&model_id));

        // Get current tree state or create new one
        let mut tree_state = if let Some(policy_state) = self.stores.policy.get(&key) {
            // Deserialize existing tree state
            serde_json::from_slice::<TreeState>(&policy_state.config)
                .unwrap_or_else(|_| TreeState::new(model_id.clone()))
        } else {
            TreeState::new(model_id.clone())
        };

        // Add the new operation
        tree_state.add_operation(operation);

        // Serialize and store back
        let serialized = serde_json::to_vec(&tree_state)
            .map_err(|e| format!("Failed to serialize tree state: {}", e))?;

        // Get current version if exists
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);
        let new_version = current_version + 1;

        let state = PolicyState {
            model_id: model_id.clone(),
            policy_type: "tree_state".to_string(),
            config: serialized,
            version: new_version,
        };

        let actor = self.self_name.clone();
        self.stores.policy.insert(key, state, actor);
        debug!(
            "Synced tree operation to mesh: model={} (version: {})",
            model_id, new_version
        );

        Ok(())
    }

    /// Get tree state for a model from mesh stores
    pub fn get_tree_state(&self, model_id: &str) -> Option<TreeState> {
        let key = SKey::new(tree_state_key(model_id));
        self.stores
            .policy
            .get(&key)
            .and_then(|policy_state| serde_json::from_slice::<TreeState>(&policy_state.config).ok())
    }

    /// Apply remote tree operation to local policy
    /// This is called when receiving tree state updates from other nodes
    pub fn apply_remote_tree_operation(
        &self,
        model_id: String,
        tree_state: TreeState,
        actor: Option<String>,
    ) {
        let key = SKey::new(tree_state_key(&model_id));
        let actor = actor.unwrap_or_else(|| "remote".to_string());

        // Check if we should update based on version
        let current_version = self
            .stores
            .policy
            .get_metadata(&key)
            .map(|(v, _)| v)
            .unwrap_or(0);

        if tree_state.version > current_version {
            // Serialize tree state
            if let Ok(serialized) = serde_json::to_vec(&tree_state) {
                let state = PolicyState {
                    model_id: model_id.clone(),
                    policy_type: "tree_state".to_string(),
                    config: serialized,
                    version: tree_state.version,
                };

                self.stores.policy.insert(key, state, actor.clone());
                debug!(
                    "Applied remote tree state update: model={} (version: {} -> {})",
                    model_id, current_version, tree_state.version
                );
            } else {
                debug!(
                    "Failed to serialize remote tree state for model={}",
                    model_id
                );
            }
        } else {
            debug!(
                "Skipped remote tree state update: model={} (version {} <= current {})",
                model_id, tree_state.version, current_version
            );
        }
    }
}

/// Optional mesh sync manager (can be None if mesh is not enabled)
pub type OptionalMeshSyncManager = Option<Arc<MeshSyncManager>>;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::mesh::stores::{
        AppState, MembershipState, RateLimitConfig, StateStores, GLOBAL_RATE_LIMIT_COUNTER_KEY,
        GLOBAL_RATE_LIMIT_KEY,
    };

    fn create_test_sync_manager() -> MeshSyncManager {
        let stores = Arc::new(StateStores::new());
        MeshSyncManager::new(stores, "test_node".to_string())
    }

    fn create_test_manager(self_name: String) -> MeshSyncManager {
        let stores = Arc::new(StateStores::with_self_name(self_name.clone()));
        MeshSyncManager::new(stores, self_name)
    }

    #[test]
    fn test_sync_manager_new() {
        let manager = create_test_sync_manager();
        // Should create without panicking
        assert_eq!(manager.get_all_worker_states().len(), 0);
        assert_eq!(manager.get_all_policy_states().len(), 0);
    }

    #[test]
    fn test_sync_worker_state() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        let state = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state.worker_id, "worker1");
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.url, "http://localhost:8000");
        assert!(state.health);
        assert_eq!(state.load, 0.5);
        assert_eq!(state.version, 1);
    }

    #[test]
    fn test_sync_multiple_worker_states() {
        let manager = create_test_sync_manager();

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        manager.sync_worker_state(
            "worker2".to_string(),
            "model1".to_string(),
            "http://localhost:8001".to_string(),
            false,
            0.8,
        );

        manager.sync_worker_state(
            "worker3".to_string(),
            "model2".to_string(),
            "http://localhost:8002".to_string(),
            true,
            0.3,
        );

        let all_states = manager.get_all_worker_states();
        assert_eq!(all_states.len(), 3);

        let worker1 = manager.get_worker_state("worker1").unwrap();
        assert_eq!(worker1.worker_id, "worker1");
        assert!(worker1.health);

        let worker2 = manager.get_worker_state("worker2").unwrap();
        assert_eq!(worker2.worker_id, "worker2");
        assert!(!worker2.health);

        let worker3 = manager.get_worker_state("worker3").unwrap();
        assert_eq!(worker3.worker_id, "worker3");
        assert_eq!(worker3.model_id, "model2");
    }

    #[test]
    fn test_sync_worker_state_version_increment() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        let state1 = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state1.version, 1);

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            false,
            0.8,
        );

        let state2 = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state2.version, 2);
        assert!(!state2.health);
        assert_eq!(state2.load, 0.8);
    }

    #[test]
    fn test_remove_worker_state() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        assert!(manager.get_worker_state("worker1").is_some());

        manager.remove_worker_state("worker1");

        assert!(manager.get_worker_state("worker1").is_none());
        assert_eq!(manager.get_all_worker_states().len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_worker_state() {
        let manager = create_test_sync_manager();

        // Should not panic
        manager.remove_worker_state("nonexistent");
        assert!(manager.get_worker_state("nonexistent").is_none());
    }

    #[test]
    fn test_sync_policy_state() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_policy_state(
            "model1".to_string(),
            "cache_aware".to_string(),
            b"config_data".to_vec(),
        );

        let state = manager.get_policy_state("model1").unwrap();
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.policy_type, "cache_aware");
        assert_eq!(state.config, b"config_data");
        assert_eq!(state.version, 1);
    }

    #[test]
    fn test_sync_multiple_policy_states() {
        let manager = create_test_sync_manager();

        manager.sync_policy_state(
            "model1".to_string(),
            "round_robin".to_string(),
            b"config1".to_vec(),
        );

        manager.sync_policy_state(
            "model2".to_string(),
            "random".to_string(),
            b"config2".to_vec(),
        );

        manager.sync_policy_state(
            "model3".to_string(),
            "consistent_hash".to_string(),
            b"config3".to_vec(),
        );

        let all_states = manager.get_all_policy_states();
        assert_eq!(all_states.len(), 3);

        let policy1 = manager.get_policy_state("model1").unwrap();
        assert_eq!(policy1.model_id, "model1");
        assert_eq!(policy1.policy_type, "round_robin");

        let policy2 = manager.get_policy_state("model2").unwrap();
        assert_eq!(policy2.model_id, "model2");
        assert_eq!(policy2.policy_type, "random");
    }

    #[test]
    fn test_remove_policy_state() {
        let manager = create_test_sync_manager();

        manager.sync_policy_state(
            "model1".to_string(),
            "round_robin".to_string(),
            b"config".to_vec(),
        );

        assert!(manager.get_policy_state("model1").is_some());

        manager.remove_policy_state("model1");

        assert!(manager.get_policy_state("model1").is_none());
        assert_eq!(manager.get_all_policy_states().len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_policy_state() {
        let manager = create_test_sync_manager();

        // Should not panic
        manager.remove_policy_state("nonexistent");
        assert!(manager.get_policy_state("nonexistent").is_none());
    }

    #[test]
    fn test_apply_remote_worker_state() {
        let manager = create_test_manager("node1".to_string());

        // Apply remote state with higher version
        let remote_state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.5,
            version: 5,
        };

        manager.apply_remote_worker_state(remote_state.clone(), Some("node2".to_string()));

        let state = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state.version, 5);
    }

    #[test]
    fn test_apply_remote_worker_state_basic() {
        let manager = create_test_sync_manager();

        let remote_state = WorkerState {
            worker_id: "remote_worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.6,
            version: 1,
        };

        manager.apply_remote_worker_state(remote_state.clone(), None);

        let state = manager.get_worker_state("remote_worker1");
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.worker_id, "remote_worker1");
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.url, "http://localhost:8000");
        assert!(state.health);
        assert_eq!(state.load, 0.6);
    }

    #[test]
    fn test_apply_remote_worker_state_version_check() {
        let manager = create_test_manager("node1".to_string());

        // First insert local state
        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        // Try to apply older version - should be skipped
        let old_state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: false,
            load: 0.8,
            version: 0, // Older version
        };

        manager.apply_remote_worker_state(old_state, Some("node2".to_string()));

        // Should still have version 1
        let state = manager.get_worker_state("worker1").unwrap();
        assert_eq!(state.version, 1);
        assert!(state.health); // Not updated
    }

    #[test]
    fn test_apply_remote_policy_state() {
        let manager = create_test_sync_manager();

        let remote_state = PolicyState {
            model_id: "model1".to_string(),
            policy_type: "remote_policy".to_string(),
            config: b"remote_config".to_vec(),
            version: 1,
        };

        manager.apply_remote_policy_state(remote_state.clone(), None);

        let state = manager.get_policy_state("model1");
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.policy_type, "remote_policy");
        assert_eq!(state.config, b"remote_config");
    }

    #[test]
    fn test_mixed_local_and_remote_states() {
        let manager = create_test_sync_manager();

        // Add local worker
        manager.sync_worker_state(
            "local_worker".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        // Add remote worker
        let remote_state = WorkerState {
            worker_id: "remote_worker".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8001".to_string(),
            health: true,
            load: 0.7,
            version: 1,
        };
        manager.apply_remote_worker_state(remote_state, None);

        let all_states = manager.get_all_worker_states();
        assert_eq!(all_states.len(), 2);

        assert!(manager.get_worker_state("local_worker").is_some());
        assert!(manager.get_worker_state("remote_worker").is_some());
    }

    #[test]
    fn test_update_worker_state() {
        let manager = create_test_sync_manager();

        // Initial state
        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        // Update state
        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            false,
            0.9,
        );

        let state = manager.get_worker_state("worker1").unwrap();
        assert!(!state.health);
        assert_eq!(state.load, 0.9);
        assert_eq!(manager.get_all_worker_states().len(), 1);
    }

    #[test]
    fn test_update_policy_state() {
        let manager = create_test_sync_manager();

        // Initial state
        manager.sync_policy_state(
            "model1".to_string(),
            "round_robin".to_string(),
            b"config1".to_vec(),
        );

        // Update state
        manager.sync_policy_state(
            "model1".to_string(),
            "random".to_string(),
            b"config2".to_vec(),
        );

        let state = manager.get_policy_state("model1").unwrap();
        assert_eq!(state.policy_type, "random");
        assert_eq!(state.config, b"config2");
        assert_eq!(manager.get_all_policy_states().len(), 1);
    }

    #[test]
    fn test_get_all_worker_states_empty() {
        let manager = create_test_sync_manager();
        let states = manager.get_all_worker_states();
        assert!(states.is_empty());
    }

    #[test]
    fn test_get_all_policy_states_empty() {
        let manager = create_test_sync_manager();
        let states = manager.get_all_policy_states();
        assert!(states.is_empty());
    }

    #[test]
    fn test_update_rate_limit_membership() {
        let manager = create_test_manager("node1".to_string());

        // Add membership nodes
        let key1 = SKey::new("node1".to_string());
        manager.stores.membership.insert(
            key1,
            MembershipState {
                name: "node1".to_string(),
                address: "127.0.0.1:8000".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
            "node1".to_string(),
        );

        let key2 = SKey::new("node2".to_string());
        manager.stores.membership.insert(
            key2,
            MembershipState {
                name: "node2".to_string(),
                address: "127.0.0.1:8001".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
            "node1".to_string(),
        );

        manager.update_rate_limit_membership();

        // Check that hash ring was updated
        let owners = manager.stores.rate_limit.get_owners("test_key");
        assert!(!owners.is_empty());
    }

    #[test]
    fn test_handle_node_failure() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership
        let key1 = SKey::new("node1".to_string());
        manager.stores.membership.insert(
            key1,
            MembershipState {
                name: "node1".to_string(),
                address: "127.0.0.1:8000".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
            "node1".to_string(),
        );

        let key2 = SKey::new("node2".to_string());
        manager.stores.membership.insert(
            key2,
            MembershipState {
                name: "node2".to_string(),
                address: "127.0.0.1:8001".to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: BTreeMap::new(),
            },
            "node1".to_string(),
        );

        manager.update_rate_limit_membership();

        // Handle node failure
        manager.handle_node_failure(&["node2".to_string()]);

        // Membership should be updated
        manager.update_rate_limit_membership();
    }

    #[test]
    fn test_sync_rate_limit_inc() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership to make node1 an owner
        manager
            .stores
            .rate_limit
            .update_membership(&["node1".to_string()]);

        let test_key = "test_key".to_string();
        if manager.stores.rate_limit.is_owner(&test_key) {
            manager.sync_rate_limit_inc(test_key.clone(), 5);

            let value = manager.get_rate_limit_value(&test_key);
            assert_eq!(value, Some(5));
        }
    }

    #[test]
    fn test_sync_rate_limit_inc_non_owner() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership without node1
        manager
            .stores
            .rate_limit
            .update_membership(&["node2".to_string(), "node3".to_string()]);

        let test_key = "test_key".to_string();
        if !manager.stores.rate_limit.is_owner(&test_key) {
            manager.sync_rate_limit_inc(test_key.clone(), 5);

            // Should not increment if not owner
            let value = manager.get_rate_limit_value(&test_key);
            assert_eq!(value, None);
        }
    }

    #[test]
    fn test_get_global_rate_limit_config() {
        let manager = create_test_manager("node1".to_string());

        // Initially should be None
        assert!(manager.get_global_rate_limit_config().is_none());

        // Set config
        let key = SKey::new(GLOBAL_RATE_LIMIT_KEY.to_string());
        let config = RateLimitConfig {
            limit_per_second: 100,
        };
        let serialized = serde_json::to_vec(&config).unwrap();
        manager.stores.app.insert(
            key,
            AppState {
                key: GLOBAL_RATE_LIMIT_KEY.to_string(),
                value: serialized,
                version: 1,
            },
            "node1".to_string(),
        );

        let retrieved = manager.get_global_rate_limit_config().unwrap();
        assert_eq!(retrieved.limit_per_second, 100);
    }

    #[test]
    fn test_check_global_rate_limit() {
        let manager = create_test_manager("node1".to_string());

        // Setup config
        let key = SKey::new(GLOBAL_RATE_LIMIT_KEY.to_string());
        let config = RateLimitConfig {
            limit_per_second: 10,
        };
        let serialized = serde_json::to_vec(&config).unwrap();
        manager.stores.app.insert(
            key,
            AppState {
                key: GLOBAL_RATE_LIMIT_KEY.to_string(),
                value: serialized,
                version: 1,
            },
            "node1".to_string(),
        );

        // Setup membership
        manager
            .stores
            .rate_limit
            .update_membership(&["node1".to_string()]);

        // Check rate limit
        let (is_exceeded, _current_count, limit) = manager.check_global_rate_limit();
        assert!(!is_exceeded); // First check should not exceed
        assert_eq!(limit, 10);

        // Increment multiple times
        for _ in 0..15 {
            manager.check_global_rate_limit();
        }

        let (is_exceeded2, current_count2, _) = manager.check_global_rate_limit();
        // Should exceed after many increments
        assert!(is_exceeded2 || current_count2 > 10);
    }

    #[test]
    fn test_reset_global_rate_limit_counter() {
        let manager = create_test_manager("node1".to_string());

        // Setup membership
        manager
            .stores
            .rate_limit
            .update_membership(&["node1".to_string()]);

        // Increment counter
        if manager
            .stores
            .rate_limit
            .is_owner(GLOBAL_RATE_LIMIT_COUNTER_KEY)
        {
            manager.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 10);
            let value = manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            assert!(value.is_some() && value.unwrap() > 0);

            // Reset
            manager.reset_global_rate_limit_counter();
            let value_after = manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            // Should be reset (0 or negative)
            assert!(value_after.is_none() || value_after.unwrap() <= 0);
        }
    }

    #[test]
    fn test_sync_tree_operation() {
        let manager = create_test_manager("node1".to_string());

        use crate::mesh::tree_ops::{TreeInsertOp, TreeOperation};

        let op = TreeOperation::Insert(TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://localhost:8000".to_string(),
        });

        let result = manager.sync_tree_operation("model1".to_string(), op);
        assert!(result.is_ok());

        // Verify tree state was stored
        let tree_state = manager.get_tree_state("model1");
        assert!(tree_state.is_some());
        let tree = tree_state.unwrap();
        assert_eq!(tree.model_id, "model1");
        assert_eq!(tree.operations.len(), 1);
    }

    #[test]
    fn test_get_tree_state() {
        let manager = create_test_manager("node1".to_string());

        // Initially should be None
        assert!(manager.get_tree_state("model1").is_none());

        // Sync an operation
        use crate::mesh::tree_ops::{TreeInsertOp, TreeOperation};
        let op = TreeOperation::Insert(TreeInsertOp {
            text: "test_text".to_string(),
            tenant: "http://localhost:8000".to_string(),
        });
        manager
            .sync_tree_operation("model1".to_string(), op)
            .unwrap();

        let tree_state = manager.get_tree_state("model1");
        assert!(tree_state.is_some());
    }

    #[test]
    fn test_apply_remote_tree_operation() {
        let manager = create_test_manager("node1".to_string());

        use crate::mesh::tree_ops::{TreeInsertOp, TreeOperation, TreeState};

        let mut tree_state = TreeState::new("model1".to_string());
        tree_state.version = 5;
        tree_state.add_operation(TreeOperation::Insert(TreeInsertOp {
            text: "remote_text".to_string(),
            tenant: "http://localhost:8001".to_string(),
        }));
        // add_operation increments version, so version is now 6

        manager.apply_remote_tree_operation(
            "model1".to_string(),
            tree_state,
            Some("node2".to_string()),
        );

        let retrieved = manager.get_tree_state("model1").unwrap();
        assert_eq!(retrieved.version, 6); // add_operation increments version from 5 to 6
        assert_eq!(retrieved.operations.len(), 1);
    }

    #[test]
    fn test_get_all_worker_states() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://localhost:8000".to_string(),
            true,
            0.5,
        );

        manager.sync_worker_state(
            "worker2".to_string(),
            "model2".to_string(),
            "http://localhost:8001".to_string(),
            false,
            0.8,
        );

        let all_states = manager.get_all_worker_states();
        assert_eq!(all_states.len(), 2);
    }

    #[test]
    fn test_get_all_policy_states() {
        let manager = create_test_manager("node1".to_string());

        manager.sync_policy_state("model1".to_string(), "cache_aware".to_string(), vec![]);

        manager.sync_policy_state("model2".to_string(), "round_robin".to_string(), vec![]);

        let all_states = manager.get_all_policy_states();
        assert_eq!(all_states.len(), 2);
    }
}
