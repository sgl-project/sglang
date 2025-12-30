//! HA state synchronization module
//!
//! Handles synchronization of worker and policy states across HA cluster nodes

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::{
    crdt::SKey,
    stores::{StateStores, WorkerState, PolicyState, StoreType},
};

/// HA sync manager for coordinating state synchronization
#[derive(Clone, Debug)]
pub struct HASyncManager {
    stores: Arc<StateStores>,
}

impl HASyncManager {
    pub fn new(stores: Arc<StateStores>) -> Self {
        Self { stores }
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
        let state = WorkerState {
            worker_id: worker_id.clone(),
            model_id,
            url,
            health,
            load,
            version: 1, // TODO: Track version properly
        };

        // Use self node name as actor (need to get from context)
        let actor = "local".to_string(); // TODO: Get actual node name
        self.stores.worker.insert(key, state, actor);
        debug!("Synced worker state to HA: {}", worker_id);
    }

    /// Remove worker state from HA stores
    pub fn remove_worker_state(&self, worker_id: &str) {
        let key = SKey::new(worker_id.to_string());
        self.stores.worker.remove(&key);
        debug!("Removed worker state from HA: {}", worker_id);
    }

    /// Sync policy state to HA stores
    pub fn sync_policy_state(
        &self,
        model_id: String,
        policy_type: String,
        config: Vec<u8>,
    ) {
        let key = SKey::new(format!("policy:{}", model_id));
        let state = PolicyState {
            model_id: model_id.clone(),
            policy_type,
            config,
            version: 1, // TODO: Track version properly
        };

        let actor = "local".to_string(); // TODO: Get actual node name
        self.stores.policy.insert(key, state, actor);
        debug!("Synced policy state to HA: model={}", model_id);
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
    pub fn apply_remote_worker_state(&self, state: WorkerState) {
        let key = SKey::new(state.worker_id.clone());
        let actor = "remote".to_string(); // TODO: Get actual remote node name
        self.stores.worker.insert(key, state, actor);
        debug!("Applied remote worker state update");
    }

    /// Apply policy state update from remote node
    pub fn apply_remote_policy_state(&self, state: PolicyState) {
        let key = SKey::new(format!("policy:{}", state.model_id));
        let actor = "remote".to_string(); // TODO: Get actual remote node name
        self.stores.policy.insert(key, state, actor);
        debug!("Applied remote policy state update");
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
        HASyncManager::new(stores)
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
        
        manager.apply_remote_worker_state(remote_state.clone());
        
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
        
        manager.apply_remote_policy_state(remote_state.clone());
        
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
        manager.apply_remote_worker_state(remote_state);
        
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

