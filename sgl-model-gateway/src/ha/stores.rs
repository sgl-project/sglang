//! State stores for HA cluster synchronization
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RateLimitConfig {
    pub limit_per_second: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            limit_per_second: 0, // 0 means disabled
        }
    }
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

    pub fn merge(&self, other: &crate::ha::crdt::CRDTMap<MembershipState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::ha::crdt::CRDTMap<MembershipState> {
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

    pub fn merge(&self, other: &crate::ha::crdt::CRDTMap<AppState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::ha::crdt::CRDTMap<AppState> {
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

    pub fn merge(&self, other: &crate::ha::crdt::CRDTMap<WorkerState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::ha::crdt::CRDTMap<WorkerState> {
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

    pub fn merge(&self, other: &crate::ha::crdt::CRDTMap<PolicyState>) {
        self.inner.merge(other);
    }

    pub fn snapshot(&self) -> crate::ha::crdt::CRDTMap<PolicyState> {
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
        Some(
            counters
                .entry(key.clone())
                .or_default()
                .clone(),
        )
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
        let counter = counters
            .entry(key.clone())
            .or_default();
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
