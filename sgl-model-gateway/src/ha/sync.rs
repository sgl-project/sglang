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

