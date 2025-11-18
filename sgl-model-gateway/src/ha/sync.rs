//! HA state synchronization module
//!
//! Handles synchronization of worker and policy states across HA cluster nodes

use std::sync::Arc;

use tracing::debug;

use super::{
    crdt::SKey,
    gossip::NodeStatus,
    stores::{MembershipState, PolicyState, StateStores, WorkerState},
};

/// HA sync manager for coordinating state synchronization
#[derive(Clone, Debug)]
pub struct HASyncManager {
    stores: Arc<StateStores>,
    self_name: String,
}

impl HASyncManager {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self { stores, self_name }
    }

    /// Get the node name (actor) for this sync manager
    pub fn self_name(&self) -> &str {
        &self.self_name
    }

    /// Sync worker state to HA stores
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
            "Synced worker state to HA: {} (version: {})",
            worker_id, new_version
        );
    }

    /// Remove worker state from HA stores
    pub fn remove_worker_state(&self, worker_id: &str) {
        let key = SKey::new(worker_id.to_string());
        self.stores.worker.remove(&key);
        debug!("Removed worker state from HA: {}", worker_id);
    }

    /// Sync policy state to HA stores
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
            "Synced policy state to HA: model={} (version: {})",
            model_id, new_version
        );
    }

    /// Remove policy state from HA stores
    pub fn remove_policy_state(&self, model_id: &str) {
        let key = SKey::new(format!("policy:{}", model_id));
        self.stores.policy.remove(&key);
        debug!("Removed policy state from HA: model={}", model_id);
    }

    /// Get worker state from HA stores
    pub fn get_worker_state(&self, worker_id: &str) -> Option<WorkerState> {
        let key = SKey::new(worker_id.to_string());
        self.stores.worker.get(&key)
    }

    /// Get all worker states from HA stores
    pub fn get_all_worker_states(&self) -> Vec<WorkerState> {
        self.stores.worker.all().into_values().collect()
    }

    /// Get policy state from HA stores
    pub fn get_policy_state(&self, model_id: &str) -> Option<PolicyState> {
        let key = SKey::new(format!("policy:{}", model_id));
        self.stores.policy.get(&key)
    }

    /// Get all policy states from HA stores
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
}

/// Optional HA sync manager (can be None if HA is not enabled)
pub type OptionalHASyncManager = Option<Arc<HASyncManager>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ha::stores::StateStores;

    fn create_test_sync_manager() -> HASyncManager {
        let stores = Arc::new(StateStores::new());
        HASyncManager::new(stores, "test_node".to_string())
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
        let manager = create_test_sync_manager();

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://worker1:8000".to_string(),
            true,
            0.5,
        );

        let state = manager.get_worker_state("worker1");
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.worker_id, "worker1");
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.url, "http://worker1:8000");
        assert!(state.health);
        assert_eq!(state.load, 0.5);
    }

    #[test]
    fn test_sync_multiple_worker_states() {
        let manager = create_test_sync_manager();

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://worker1:8000".to_string(),
            true,
            0.5,
        );

        manager.sync_worker_state(
            "worker2".to_string(),
            "model1".to_string(),
            "http://worker2:8000".to_string(),
            false,
            0.8,
        );

        manager.sync_worker_state(
            "worker3".to_string(),
            "model2".to_string(),
            "http://worker3:8000".to_string(),
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
    fn test_remove_worker_state() {
        let manager = create_test_sync_manager();

        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://worker1:8000".to_string(),
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
        let manager = create_test_sync_manager();

        let config = b"policy_config_data".to_vec();
        manager.sync_policy_state(
            "model1".to_string(),
            "round_robin".to_string(),
            config.clone(),
        );

        let state = manager.get_policy_state("model1");
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.model_id, "model1");
        assert_eq!(state.policy_type, "round_robin");
        assert_eq!(state.config, config);
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
        let manager = create_test_sync_manager();

        let remote_state = WorkerState {
            worker_id: "remote_worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://remote:8000".to_string(),
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
        assert_eq!(state.url, "http://remote:8000");
        assert!(state.health);
        assert_eq!(state.load, 0.6);
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
            "http://local:8000".to_string(),
            true,
            0.5,
        );

        // Add remote worker
        let remote_state = WorkerState {
            worker_id: "remote_worker".to_string(),
            model_id: "model1".to_string(),
            url: "http://remote:8000".to_string(),
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
            "http://worker1:8000".to_string(),
            true,
            0.5,
        );

        // Update state
        manager.sync_worker_state(
            "worker1".to_string(),
            "model1".to_string(),
            "http://worker1:8000".to_string(),
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
}
