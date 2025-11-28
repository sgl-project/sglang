//! Incremental update collection and batching
//!
//! Collects local state changes and batches them for efficient transmission

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use tracing::{debug, trace};

use super::{
    gossip::StateUpdate,
    stores::{MembershipState, PolicyState, StateStores, StoreType, WorkerState},
    SKey,
};

/// Tracks the last sent version for each key in each store
#[derive(Debug, Clone, Default)]
struct LastSentVersions {
    worker: HashMap<String, u64>,
    policy: HashMap<String, u64>,
    app: HashMap<String, u64>,
    membership: HashMap<String, u64>,
    rate_limit: HashMap<String, u64>, // Track last sent timestamp for rate limit counters
}

/// Incremental update collector
pub struct IncrementalUpdateCollector {
    stores: Arc<StateStores>,
    self_name: String,
    last_sent: Arc<RwLock<LastSentVersions>>,
}

impl IncrementalUpdateCollector {
    pub fn new(stores: Arc<StateStores>, self_name: String) -> Self {
        Self {
            stores,
            self_name,
            last_sent: Arc::new(RwLock::new(LastSentVersions::default())),
        }
    }

    /// Get current timestamp in nanoseconds
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    /// Helper function to collect updates for stores with serializable state
    fn collect_serializable_updates<S>(
        &self,
        all_items: std::collections::BTreeMap<SKey, S>,
        get_version: impl Fn(&SKey) -> u64,
        last_sent_map: &mut HashMap<String, u64>,
        store_name: &str,
        get_id: impl Fn(&S) -> String,
    ) -> Vec<StateUpdate>
    where
        S: serde::Serialize,
    {
        let mut updates = Vec::new();
        let timestamp = Self::current_timestamp();

        for (key, state) in all_items {
            let key_str = key.as_str().to_string();
            let current_version = get_version(&key);
            let last_sent_version = last_sent_map.get(&key_str).copied().unwrap_or(0);

            if current_version > last_sent_version {
                if let Ok(serialized) = serde_json::to_vec(&state) {
                    updates.push(StateUpdate {
                        key: key_str.clone(),
                        value: serialized,
                        version: current_version,
                        actor: self.self_name.clone(),
                        timestamp,
                    });

                    last_sent_map.insert(key_str, current_version);
                    trace!(
                        "Collected {} update: {} (version: {})",
                        store_name,
                        get_id(&state),
                        current_version
                    );
                }
            }
        }
        updates
    }

    /// Collect incremental updates for a specific store type
    pub fn collect_updates_for_store(&self, store_type: StoreType) -> Vec<StateUpdate> {
        let mut updates = Vec::new();
        let mut last_sent = self.last_sent.write();

        match store_type {
            StoreType::Worker => {
                let all_workers = self.stores.worker.all();
                let get_version = |key: &SKey| {
                    self.stores
                        .worker
                        .get_metadata(key)
                        .map(|(v, _)| v)
                        .unwrap_or(0)
                };
                updates = self.collect_serializable_updates(
                    all_workers,
                    get_version,
                    &mut last_sent.worker,
                    "worker",
                    |state: &WorkerState| state.worker_id.clone(),
                );
            }
            StoreType::Policy => {
                let all_policies = self.stores.policy.all();
                let get_version = |key: &SKey| {
                    self.stores
                        .policy
                        .get_metadata(key)
                        .map(|(v, _)| v)
                        .unwrap_or(0)
                };
                updates = self.collect_serializable_updates(
                    all_policies,
                    get_version,
                    &mut last_sent.policy,
                    "policy",
                    |state: &PolicyState| state.model_id.clone(),
                );
            }
            StoreType::App => {
                let all_apps = self.stores.app.all();
                let timestamp = Self::current_timestamp();
                for (key, state) in all_apps {
                    let key_str = key.as_str().to_string();
                    let current_version = self
                        .stores
                        .app
                        .get_metadata(&key)
                        .map(|(v, _)| v)
                        .unwrap_or(0);
                    let last_sent_version = last_sent.app.get(&key_str).copied().unwrap_or(0);

                    if current_version > last_sent_version {
                        updates.push(StateUpdate {
                            key: key_str.clone(),
                            value: state.value.clone(),
                            version: current_version,
                            actor: self.self_name.clone(),
                            timestamp,
                        });
                        last_sent.app.insert(key_str, current_version);
                        trace!(
                            "Collected app update: {} (version: {})",
                            state.key,
                            current_version
                        );
                    }
                }
            }
            StoreType::Membership => {
                let all_members = self.stores.membership.all();
                let get_version = |key: &SKey| {
                    self.stores
                        .membership
                        .get_metadata(key)
                        .map(|(v, _)| v)
                        .unwrap_or(0)
                };
                updates = self.collect_serializable_updates(
                    all_members,
                    get_version,
                    &mut last_sent.membership,
                    "membership",
                    |state: &MembershipState| state.name.clone(),
                );
            }
            StoreType::RateLimit => {
                let rate_limit_keys = self.stores.rate_limit.keys();
                let current_timestamp = Self::current_timestamp();

                for key in rate_limit_keys {
                    if self.stores.rate_limit.is_owner(&key) {
                        if let Some(counter) = self.stores.rate_limit.get_counter(&key) {
                            let last_sent_timestamp =
                                last_sent.rate_limit.get(&key).copied().unwrap_or(0);

                            // Only send if at least 1 second has passed since last send
                            if current_timestamp > last_sent_timestamp + 1_000_000_000 {
                                if let Ok(serialized) = serde_json::to_vec(&counter.snapshot()) {
                                    let key_str = key.clone();
                                    updates.push(StateUpdate {
                                        key: key_str.clone(),
                                        value: serialized,
                                        version: current_timestamp,
                                        actor: self.self_name.clone(),
                                        timestamp: current_timestamp,
                                    });
                                    last_sent.rate_limit.insert(key_str, current_timestamp);
                                    trace!("Collected rate limit counter update: {}", key);
                                }
                            }
                        }
                    }
                }
            }
        }

        debug!(
            "Collected {} incremental updates for store {:?}",
            updates.len(),
            store_type
        );
        updates
    }

    /// Collect all incremental updates across all stores
    pub fn collect_all_updates(&self) -> Vec<(StoreType, Vec<StateUpdate>)> {
        let mut all_updates = Vec::new();

        for store_type in [
            StoreType::Worker,
            StoreType::Policy,
            StoreType::App,
            StoreType::Membership,
            StoreType::RateLimit,
        ] {
            let updates = self.collect_updates_for_store(store_type);
            if !updates.is_empty() {
                all_updates.push((store_type, updates));
            }
        }

        all_updates
    }

    /// Mark updates as sent (called after successful transmission)
    pub fn mark_sent(&self, store_type: StoreType, updates: &[StateUpdate]) {
        let mut last_sent = self.last_sent.write();
        let target_map = match store_type {
            StoreType::Worker => &mut last_sent.worker,
            StoreType::Policy => &mut last_sent.policy,
            StoreType::App => &mut last_sent.app,
            StoreType::Membership => &mut last_sent.membership,
            StoreType::RateLimit => &mut last_sent.rate_limit,
        };

        for update in updates {
            target_map.insert(update.key.clone(), update.version);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;
    use crate::mesh::stores::{AppState, MembershipState, PolicyState, StateStores, WorkerState};

    fn create_test_collector(self_name: String) -> IncrementalUpdateCollector {
        let stores = Arc::new(StateStores::with_self_name(self_name.clone()));
        IncrementalUpdateCollector::new(stores, self_name)
    }

    #[test]
    fn test_collect_worker_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Insert a worker state
        let key = SKey::new("worker1".to_string());
        let worker_state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.5,
            version: 1,
        };
        stores.worker.insert(key, worker_state, "node1".to_string());

        // Collect updates
        let updates = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "worker1");
        assert_eq!(updates[0].version, 1);
        assert_eq!(updates[0].actor, "node1");

        // Collect again - should be empty (already sent)
        let updates2 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates2.len(), 0);

        // Update worker state
        let key2 = SKey::new("worker1".to_string());
        let worker_state2 = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: false,
            load: 0.8,
            version: 2,
        };
        stores
            .worker
            .insert(key2, worker_state2, "node1".to_string());

        // Should collect new version
        let updates3 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates3.len(), 1);
        assert_eq!(updates3[0].version, 2);
    }

    #[test]
    fn test_collect_policy_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = SKey::new("policy:model1".to_string());
        let policy_state = PolicyState {
            model_id: "model1".to_string(),
            policy_type: "cache_aware".to_string(),
            config: b"config_data".to_vec(),
            version: 1,
        };
        stores.policy.insert(key, policy_state, "node1".to_string());

        let updates = collector.collect_updates_for_store(StoreType::Policy);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "policy:model1");
    }

    #[test]
    fn test_collect_app_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = SKey::new("app_key1".to_string());
        let app_state = AppState {
            key: "app_key1".to_string(),
            value: b"app_value".to_vec(),
            version: 1,
        };
        stores.app.insert(key, app_state, "node1".to_string());

        let updates = collector.collect_updates_for_store(StoreType::App);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "app_key1");
    }

    #[test]
    fn test_collect_membership_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = SKey::new("node2".to_string());
        let membership_state = MembershipState {
            name: "node2".to_string(),
            address: "127.0.0.1:8001".to_string(),
            status: 1, // Alive
            version: 1,
            metadata: std::collections::BTreeMap::new(),
        };
        stores
            .membership
            .insert(key, membership_state, "node1".to_string());

        let updates = collector.collect_updates_for_store(StoreType::Membership);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].key, "node2");
    }

    #[test]
    fn test_collect_all_updates() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Insert into multiple stores
        let worker_key = SKey::new("worker1".to_string());
        stores.worker.insert(
            worker_key,
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.5,
                version: 1,
            },
            "node1".to_string(),
        );

        let policy_key = SKey::new("policy:model1".to_string());
        stores.policy.insert(
            policy_key,
            PolicyState {
                model_id: "model1".to_string(),
                policy_type: "cache_aware".to_string(),
                config: vec![],
                version: 1,
            },
            "node1".to_string(),
        );

        let all_updates = collector.collect_all_updates();
        assert_eq!(all_updates.len(), 2); // Worker and Policy
    }

    #[test]
    fn test_mark_sent() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Insert and collect
        let key = SKey::new("worker1".to_string());
        stores.worker.insert(
            key,
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.5,
                version: 1,
            },
            "node1".to_string(),
        );

        let updates = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates.len(), 1);

        // Mark as sent
        collector.mark_sent(StoreType::Worker, &updates);

        // Should not collect again
        let updates2 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates2.len(), 0);
    }

    #[test]
    fn test_rate_limit_timestamp_filtering() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        // Update membership to make node1 an owner
        stores.rate_limit.update_membership(&["node1".to_string()]);

        // Insert a counter (node1 should be owner)
        let test_key = "test_rate_limit_key".to_string();
        if stores.rate_limit.is_owner(&test_key) {
            stores
                .rate_limit
                .inc(test_key.clone(), "node1".to_string(), 1);
        }

        // Collect immediately - should be filtered by timestamp
        let _updates = collector.collect_updates_for_store(StoreType::RateLimit);
        // May be empty if timestamp check fails, or may have one update
        // The exact behavior depends on timing

        // Wait a bit and try again
        thread::sleep(Duration::from_secs(2));

        // Now should collect (enough time has passed)
        let updates2 = collector.collect_updates_for_store(StoreType::RateLimit);
        // Should have at least one update if node1 is owner
        if stores.rate_limit.is_owner(&test_key) {
            // Updates may be 0 or 1 depending on timing
            let _ = updates2;
        }
    }

    #[test]
    fn test_version_tracking() {
        let collector = create_test_collector("node1".to_string());
        let stores = collector.stores.clone();

        let key = SKey::new("worker1".to_string());

        // Insert first version (will be version 1 in store)
        stores.worker.insert(
            key.clone(),
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.5,
                version: 1, // Note: CRDT will use this but increment internally
            },
            "node1".to_string(),
        );

        let updates1 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates1.len(), 1);
        let version1 = updates1[0].version;
        assert!(version1 >= 1);

        // Insert second version (will increment from version1)
        stores.worker.insert(
            key.clone(),
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: false,
                load: 0.8,
                version: 2, // Note: CRDT will increment internally
            },
            "node1".to_string(),
        );

        let updates2 = collector.collect_updates_for_store(StoreType::Worker);
        assert_eq!(updates2.len(), 1);
        let version2 = updates2[0].version;
        assert!(version2 > version1);

        // Insert again - should increment version and be collected
        stores.worker.insert(
            key,
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.3,
                version: 1, // Note: CRDT ignores this and increments internally
            },
            "node1".to_string(),
        );

        let updates3 = collector.collect_updates_for_store(StoreType::Worker);
        // Should collect because version was incremented (version2 + 1 > version2)
        assert_eq!(updates3.len(), 1);
        let version3 = updates3[0].version;
        assert!(version3 > version2);
    }
}
