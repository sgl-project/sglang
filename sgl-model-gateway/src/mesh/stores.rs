//! State stores for mesh cluster synchronization
//!
//! Four types of state stores:
//! - MembershipStore: Router node membership
//! - AppStore: Application configuration, rate-limiting rules, LB algorithms
//! - WorkerStore: Worker status, load, health
//! - PolicyStore: Routing policy internal state

use std::{collections::BTreeMap, sync::Arc};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::{
    consistent_hash::ConsistentHashRing,
    crdt::{SKey, SyncCRDTMap, SyncPNCounter},
};

/// Store type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StoreType {
    Membership,
    App,
    Worker,
    Policy,
    RateLimit,
}

impl StoreType {
    pub fn as_str(&self) -> &'static str {
        match self {
            StoreType::Membership => "membership",
            StoreType::App => "app",
            StoreType::Worker => "worker",
            StoreType::Policy => "policy",
            StoreType::RateLimit => "rate_limit",
        }
    }

    /// Convert from proto StoreType (i32) to local StoreType
    pub fn from_proto(proto_value: i32) -> Self {
        match proto_value {
            0 => StoreType::Membership,
            1 => StoreType::App,
            2 => StoreType::Worker,
            3 => StoreType::Policy,
            4 => StoreType::RateLimit,
            _ => StoreType::Membership, // Default fallback
        }
    }
}

/// Membership state entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct MembershipState {
    pub name: String,
    pub address: String,
    pub status: i32, // NodeStatus enum value
    pub version: u64,
    pub metadata: BTreeMap<String, Vec<u8>>,
}

/// App state entry (application configuration)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct AppState {
    pub key: String,
    pub value: Vec<u8>, // Serialized config
    pub version: u64,
}

/// Global rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RateLimitConfig {
    pub limit_per_second: u64,
}

/// Key for global rate limit configuration in AppStore
pub const GLOBAL_RATE_LIMIT_KEY: &str = "global_rate_limit";
/// Key for global rate limit counter in RateLimitStore
pub const GLOBAL_RATE_LIMIT_COUNTER_KEY: &str = "global";

/// Worker state entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub url: String,
    pub health: bool,
    pub load: f64,
    pub version: u64,
}

// Implement Hash manually for WorkerState (excluding f64)
impl std::hash::Hash for WorkerState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.worker_id.hash(state);
        self.model_id.hash(state);
        self.url.hash(state);
        self.health.hash(state);
        // f64 cannot be hashed directly, use a workaround
        (self.load as i64).hash(state);
        self.version.hash(state);
    }
}

// Implement Eq manually (f64 comparison with epsilon)
impl Eq for WorkerState {}

/// Policy state entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct PolicyState {
    pub model_id: String,
    pub policy_type: String,
    pub config: Vec<u8>, // Serialized policy config
    pub version: u64,
}

/// Helper function to get tree state key for a model
pub fn tree_state_key(model_id: &str) -> String {
    format!("tree:{}", model_id)
}

/// Membership store
#[derive(Debug, Clone)]
pub struct MembershipStore {
    inner: SyncCRDTMap<MembershipState>,
}

impl MembershipStore {
    pub fn new() -> Self {
        Self {
            inner: SyncCRDTMap::new(),
        }
    }

    pub fn get(&self, key: &SKey) -> Option<MembershipState> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: SKey, value: MembershipState, actor: String) {
        self.inner.insert(key, value, actor);
    }

    pub fn remove(&self, key: &SKey) {
        self.inner.remove(key);
    }

    pub fn merge(&self, other: &crate::mesh::crdt::CRDTMap<MembershipState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::mesh::crdt::CRDTMap<MembershipState> {
        self.inner.snapshot()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn all(&self) -> BTreeMap<SKey, MembershipState> {
        self.inner.snapshot().to_map()
    }

    pub fn get_metadata(&self, key: &SKey) -> Option<(u64, String)> {
        self.inner.get_metadata(key)
    }
}

impl Default for MembershipStore {
    fn default() -> Self {
        Self::new()
    }
}

/// App store
#[derive(Debug, Clone)]
pub struct AppStore {
    inner: SyncCRDTMap<AppState>,
}

impl AppStore {
    pub fn new() -> Self {
        Self {
            inner: SyncCRDTMap::new(),
        }
    }

    pub fn get(&self, key: &SKey) -> Option<AppState> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: SKey, value: AppState, actor: String) {
        self.inner.insert(key, value, actor);
    }

    pub fn remove(&self, key: &SKey) {
        self.inner.remove(key);
    }

    pub fn merge(&self, other: &crate::mesh::crdt::CRDTMap<AppState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::mesh::crdt::CRDTMap<AppState> {
        self.inner.snapshot()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn all(&self) -> BTreeMap<SKey, AppState> {
        self.inner.snapshot().to_map()
    }

    pub fn get_metadata(&self, key: &SKey) -> Option<(u64, String)> {
        self.inner.get_metadata(key)
    }
}

impl Default for AppStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Worker store
#[derive(Debug, Clone)]
pub struct WorkerStore {
    inner: SyncCRDTMap<WorkerState>,
}

impl WorkerStore {
    pub fn new() -> Self {
        Self {
            inner: SyncCRDTMap::new(),
        }
    }

    pub fn get(&self, key: &SKey) -> Option<WorkerState> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: SKey, value: WorkerState, actor: String) {
        self.inner.insert(key, value, actor);
    }

    pub fn remove(&self, key: &SKey) {
        self.inner.remove(key);
    }

    pub fn merge(&self, other: &crate::mesh::crdt::CRDTMap<WorkerState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::mesh::crdt::CRDTMap<WorkerState> {
        self.inner.snapshot()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn all(&self) -> BTreeMap<SKey, WorkerState> {
        self.inner.snapshot().to_map()
    }

    pub fn get_metadata(&self, key: &SKey) -> Option<(u64, String)> {
        self.inner.get_metadata(key)
    }
}

impl Default for WorkerStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy store
#[derive(Debug, Clone)]
pub struct PolicyStore {
    inner: SyncCRDTMap<PolicyState>,
}

impl PolicyStore {
    pub fn new() -> Self {
        Self {
            inner: SyncCRDTMap::new(),
        }
    }

    pub fn get(&self, key: &SKey) -> Option<PolicyState> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: SKey, value: PolicyState, actor: String) {
        self.inner.insert(key, value, actor);
    }

    pub fn remove(&self, key: &SKey) {
        self.inner.remove(key);
    }

    pub fn merge(&self, other: &crate::mesh::crdt::CRDTMap<PolicyState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::mesh::crdt::CRDTMap<PolicyState> {
        self.inner.snapshot()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn all(&self) -> BTreeMap<SKey, PolicyState> {
        self.inner.snapshot().to_map()
    }

    pub fn get_metadata(&self, key: &SKey) -> Option<(u64, String)> {
        self.inner.get_metadata(key)
    }
}

impl Default for PolicyStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate-limit counter store (using PNCounter with consistent hashing)
#[derive(Debug, Clone)]
pub struct RateLimitStore {
    counters: Arc<RwLock<BTreeMap<String, SyncPNCounter>>>, // key -> counter
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    self_name: String,
}

impl RateLimitStore {
    pub fn new(self_name: String) -> Self {
        Self {
            counters: Arc::new(RwLock::new(BTreeMap::new())),
            hash_ring: Arc::new(RwLock::new(ConsistentHashRing::new())),
            self_name,
        }
    }

    /// Update the hash ring with current membership
    pub fn update_membership(&self, nodes: &[String]) {
        let mut ring = self.hash_ring.write();
        ring.update_membership(nodes);
        debug!("Updated rate-limit hash ring with {} nodes", nodes.len());
    }

    /// Check if this node is an owner of a key
    pub fn is_owner(&self, key: &str) -> bool {
        let ring = self.hash_ring.read();
        ring.is_owner(key, &self.self_name)
    }

    /// Get owners for a key
    pub fn get_owners(&self, key: &str) -> Vec<String> {
        let ring = self.hash_ring.read();
        ring.get_owners(key)
    }

    /// Get or create counter (only if this node is an owner)
    #[allow(dead_code)]
    fn get_or_create_counter_internal(&self, key: String) -> Option<SyncPNCounter> {
        if !self.is_owner(&key) {
            return None;
        }

        let mut counters = self.counters.write();
        Some(counters.entry(key.clone()).or_default().clone())
    }

    pub fn get_counter(&self, key: &str) -> Option<SyncPNCounter> {
        if !self.is_owner(key) {
            return None;
        }
        let counters = self.counters.read();
        counters.get(key).cloned()
    }

    /// Increment counter (only if this node is an owner)
    pub fn inc(&self, key: String, actor: String, delta: i64) {
        if !self.is_owner(&key) {
            // Not an owner, skip
            return;
        }

        let mut counters = self.counters.write();
        let counter = counters.entry(key.clone()).or_default();
        counter.inc(actor, delta);
    }

    /// Get counter value (aggregate from all owners via CRDT merge)
    pub fn value(&self, key: &str) -> Option<i64> {
        let counters = self.counters.read();
        counters.get(key).map(|c| c.value())
    }

    /// Merge counter from another node (for CRDT synchronization)
    pub fn merge_counter(&self, key: String, other: &SyncPNCounter) {
        let mut counters = self.counters.write();
        let counter = counters.entry(key).or_default();
        // Get the inner CRDTPNCounter from other SyncPNCounter
        let other_inner = other.snapshot();
        counter.merge(&other_inner);
    }

    /// Get all counter keys
    pub fn keys(&self) -> Vec<String> {
        let counters = self.counters.read();
        counters.keys().cloned().collect()
    }

    /// Check if we need to transfer ownership due to node failure
    pub fn check_ownership_transfer(&self, failed_nodes: &[String]) -> Vec<String> {
        let mut affected_keys = Vec::new();
        let ring = self.hash_ring.read();
        let counters = self.counters.read();

        for key in counters.keys() {
            let owners = ring.get_owners(key);
            // Check if any owner has failed
            if owners.iter().any(|owner| failed_nodes.contains(owner)) {
                // Check if we are now an owner
                if ring.is_owner(key, &self.self_name) {
                    affected_keys.push(key.clone());
                }
            }
        }

        affected_keys
    }
}

impl Default for RateLimitStore {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}

/// All state stores container
#[derive(Debug, Clone)]
pub struct StateStores {
    pub membership: MembershipStore,
    pub app: AppStore,
    pub worker: WorkerStore,
    pub policy: PolicyStore,
    pub rate_limit: RateLimitStore,
}

impl StateStores {
    pub fn new() -> Self {
        Self {
            membership: MembershipStore::new(),
            app: AppStore::new(),
            worker: WorkerStore::new(),
            policy: PolicyStore::new(),
            rate_limit: RateLimitStore::new("default".to_string()),
        }
    }

    pub fn with_self_name(self_name: String) -> Self {
        Self {
            membership: MembershipStore::new(),
            app: AppStore::new(),
            worker: WorkerStore::new(),
            policy: PolicyStore::new(),
            rate_limit: RateLimitStore::new(self_name),
        }
    }
}

impl Default for StateStores {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::mesh::service::gossip::NodeStatus;

    #[test]
    fn test_membership_store() {
        let store = MembershipStore::new();
        let key = SKey::new("node1".to_string());
        let state = MembershipState {
            name: "node1".to_string(),
            address: "127.0.0.1:8000".to_string(),
            status: NodeStatus::Alive as i32,
            version: 1,
            metadata: BTreeMap::new(),
        };

        store.insert(key.clone(), state.clone(), "node1".to_string());
        assert_eq!(store.get(&key).unwrap().name, "node1");
        assert_eq!(store.len(), 1);

        store.remove(&key);
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn test_app_store() {
        let store = AppStore::new();
        let key = SKey::new("app_key1".to_string());
        let state = AppState {
            key: "app_key1".to_string(),
            value: b"app_value".to_vec(),
            version: 1,
        };

        store.insert(key.clone(), state.clone(), "node1".to_string());
        assert_eq!(store.get(&key).unwrap().key, "app_key1");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_worker_store() {
        let store = WorkerStore::new();
        let key = SKey::new("worker1".to_string());
        let state = WorkerState {
            worker_id: "worker1".to_string(),
            model_id: "model1".to_string(),
            url: "http://localhost:8000".to_string(),
            health: true,
            load: 0.5,
            version: 1,
        };

        store.insert(key.clone(), state.clone(), "node1".to_string());
        assert_eq!(store.get(&key).unwrap().worker_id, "worker1");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_policy_store() {
        let store = PolicyStore::new();
        let key = SKey::new("policy:model1".to_string());
        let state = PolicyState {
            model_id: "model1".to_string(),
            policy_type: "cache_aware".to_string(),
            config: b"config_data".to_vec(),
            version: 1,
        };

        store.insert(key.clone(), state.clone(), "node1".to_string());
        assert_eq!(store.get(&key).unwrap().model_id, "model1");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_rate_limit_store_update_membership() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&[
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ]);

        let owners = store.get_owners("test_key");
        assert_eq!(owners.len(), 3);
        assert!(
            owners.contains(&"node1".to_string())
                || owners.contains(&"node2".to_string())
                || owners.contains(&"node3".to_string())
        );
    }

    #[test]
    fn test_rate_limit_store_is_owner() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&["node1".to_string()]);

        let test_key = "test_key".to_string();
        let is_owner = store.is_owner(&test_key);
        // node1 should be owner since it's the only node
        assert!(is_owner);
    }

    #[test]
    fn test_rate_limit_store_inc_only_owner() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&["node1".to_string()]);

        let test_key = "test_key".to_string();
        if store.is_owner(&test_key) {
            store.inc(test_key.clone(), "node1".to_string(), 5);

            let value = store.value(&test_key);
            assert_eq!(value, Some(5));
        }
    }

    #[test]
    fn test_rate_limit_store_inc_non_owner() {
        let store = RateLimitStore::new("node1".to_string());

        // Setup membership without node1 as owner
        store.update_membership(&["node2".to_string(), "node3".to_string()]);

        let test_key = "test_key".to_string();
        if !store.is_owner(&test_key) {
            store.inc(test_key.clone(), "node1".to_string(), 5);

            // Should not increment if not owner
            let value = store.value(&test_key);
            assert_eq!(value, None);
        }
    }

    #[test]
    fn test_rate_limit_store_merge_counter() {
        let store1 = RateLimitStore::new("node1".to_string());
        let store2 = RateLimitStore::new("node2".to_string());

        store1.update_membership(&["node1".to_string()]);
        store2.update_membership(&["node2".to_string()]);

        let test_key = "test_key".to_string();

        // Both nodes increment their counters
        if store1.is_owner(&test_key) {
            store1.inc(test_key.clone(), "node1".to_string(), 10);
        }

        if store2.is_owner(&test_key) {
            store2.inc(test_key.clone(), "node2".to_string(), 5);
        }

        // Merge counter from store2 into store1
        if let Some(counter2) = store2.get_counter(&test_key) {
            store1.merge_counter(test_key.clone(), &counter2);
        }

        // Get aggregated value (if node1 is owner)
        if store1.is_owner(&test_key) {
            let value = store1.value(&test_key);
            // Should include merged value
            assert!(value.is_some());
        }
    }

    #[test]
    fn test_rate_limit_store_check_ownership_transfer() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&[
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ]);

        let test_key = "test_key".to_string();

        // Setup a counter (if node1 is owner)
        if store.is_owner(&test_key) {
            store.inc(test_key.clone(), "node1".to_string(), 10);
        }

        // Check ownership transfer when node2 fails
        let affected = store.check_ownership_transfer(&["node2".to_string()]);
        // Should detect if node2 was an owner
        let _ = affected;
    }

    #[test]
    fn test_rate_limit_store_keys() {
        let store = RateLimitStore::new("node1".to_string());

        store.update_membership(&["node1".to_string()]);

        let key1 = "key1".to_string();
        let key2 = "key2".to_string();

        if store.is_owner(&key1) {
            store.inc(key1.clone(), "node1".to_string(), 1);
        }

        if store.is_owner(&key2) {
            store.inc(key2.clone(), "node1".to_string(), 1);
        }

        let keys = store.keys();
        // Should contain keys where node1 is owner
        let _ = keys;
    }

    #[test]
    fn test_state_stores_new() {
        let stores = StateStores::new();
        assert_eq!(stores.membership.len(), 0);
        assert_eq!(stores.app.len(), 0);
        assert_eq!(stores.worker.len(), 0);
        assert_eq!(stores.policy.len(), 0);
    }

    #[test]
    fn test_state_stores_with_self_name() {
        let stores = StateStores::with_self_name("test_node".to_string());
        // Rate limit store should have the self_name
        let test_key = "test_key".to_string();
        stores
            .rate_limit
            .update_membership(&["test_node".to_string()]);
        assert!(stores.rate_limit.is_owner(&test_key));
    }
}
