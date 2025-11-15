//! Incremental update collection and batching
//!
//! Collects local state changes and batches them for efficient transmission

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, trace};

use super::{
    crdt::SKey,
    gossip::StateUpdate,
    stores::{AppState, PolicyState, StateStores, StoreType, WorkerState},
};

/// Tracks the last sent version for each key in each store
#[derive(Debug, Clone, Default)]
struct LastSentVersions {
    worker: HashMap<String, u64>,
    policy: HashMap<String, u64>,
    app: HashMap<String, u64>,
    membership: HashMap<String, u64>,
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

    /// Collect incremental updates for a specific store type
    pub fn collect_updates_for_store(
        &self,
        store_type: StoreType,
    ) -> Vec<StateUpdate> {
        let mut updates = Vec::new();
        let mut last_sent = self.last_sent.write();

        match store_type {
            StoreType::Worker => {
                let all_workers = self.stores.worker.all();
                for (key, state) in all_workers {
                    let key_str = key.as_str().to_string();
                    let current_version = self.stores.worker.get_metadata(&key)
                        .map(|(v, _)| v)
                        .unwrap_or(0);
                    
                    let last_sent_version = last_sent.worker.get(&key_str).copied().unwrap_or(0);
                    
                    // Only include if version has changed
                    if current_version > last_sent_version {
                        if let Ok(serialized) = serde_json::to_vec(&state) {
                            updates.push(StateUpdate {
                                key: key_str.clone(),
                                value: serialized,
                                version: current_version,
                                actor: self.self_name.clone(),
                                timestamp: SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_nanos() as u64,
                            });
                            
                            // Update last sent version
                            last_sent.worker.insert(key_str, current_version);
                            trace!("Collected worker update: {} (version: {})", state.worker_id, current_version);
                        }
                    }
                }
            }
            StoreType::Policy => {
                let all_policies = self.stores.policy.all();
                for (key, state) in all_policies {
                    let key_str = key.as_str().to_string();
                    let current_version = self.stores.policy.get_metadata(&key)
                        .map(|(v, _)| v)
                        .unwrap_or(0);
                    
                    let last_sent_version = last_sent.policy.get(&key_str).copied().unwrap_or(0);
                    
                    // Only include if version has changed
                    if current_version > last_sent_version {
                        if let Ok(serialized) = serde_json::to_vec(&state) {
                            updates.push(StateUpdate {
                                key: key_str.clone(),
                                value: serialized,
                                version: current_version,
                                actor: self.self_name.clone(),
                                timestamp: SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_nanos() as u64,
                            });
                            
                            // Update last sent version
                            last_sent.policy.insert(key_str, current_version);
                            trace!("Collected policy update: {} (version: {})", state.model_id, current_version);
                        }
                    }
                }
            }
            StoreType::App => {
                let all_apps = self.stores.app.all();
                for (key, state) in all_apps {
                    let key_str = key.as_str().to_string();
                    let current_version = self.stores.app.get_metadata(&key)
                        .map(|(v, _)| v)
                        .unwrap_or(0);
                    
                    let last_sent_version = last_sent.app.get(&key_str).copied().unwrap_or(0);
                    
                    // Only include if version has changed
                    if current_version > last_sent_version {
                        updates.push(StateUpdate {
                            key: key_str.clone(),
                            value: state.value.clone(),
                            version: current_version,
                            actor: self.self_name.clone(),
                            timestamp: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64,
                        });
                        
                        // Update last sent version
                        last_sent.app.insert(key_str, current_version);
                        trace!("Collected app update: {} (version: {})", state.key, current_version);
                    }
                }
            }
            StoreType::Membership => {
                let all_members = self.stores.membership.all();
                for (key, state) in all_members {
                    let key_str = key.as_str().to_string();
                    let current_version = self.stores.membership.get_metadata(&key)
                        .map(|(v, _)| v)
                        .unwrap_or(0);
                    
                    let last_sent_version = last_sent.membership.get(&key_str).copied().unwrap_or(0);
                    
                    // Only include if version has changed
                    if current_version > last_sent_version {
                        if let Ok(serialized) = serde_json::to_vec(&state) {
                            updates.push(StateUpdate {
                                key: key_str.clone(),
                                value: serialized,
                                version: current_version,
                                actor: self.self_name.clone(),
                                timestamp: SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_nanos() as u64,
                            });
                            
                            // Update last sent version
                            last_sent.membership.insert(key_str, current_version);
                            trace!("Collected membership update: {} (version: {})", state.name, current_version);
                        }
                    }
                }
            }
        }

        debug!("Collected {} incremental updates for store {:?}", updates.len(), store_type);
        updates
    }

    /// Collect all incremental updates across all stores
    pub fn collect_all_updates(&self) -> Vec<(StoreType, Vec<StateUpdate>)> {
        let mut all_updates = Vec::new();
        
        for store_type in [StoreType::Worker, StoreType::Policy, StoreType::App, StoreType::Membership] {
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
        match store_type {
            StoreType::Worker => {
                for update in updates {
                    last_sent.worker.insert(update.key.clone(), update.version);
                }
            }
            StoreType::Policy => {
                for update in updates {
                    last_sent.policy.insert(update.key.clone(), update.version);
                }
            }
            StoreType::App => {
                for update in updates {
                    last_sent.app.insert(update.key.clone(), update.version);
                }
            }
            StoreType::Membership => {
                for update in updates {
                    last_sent.membership.insert(update.key.clone(), update.version);
                }
            }
        }
    }
}

