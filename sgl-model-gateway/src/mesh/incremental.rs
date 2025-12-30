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
