//! State stores for HA cluster synchronization
//!
//! Four types of state stores:
//! - MembershipStore: Router node membership
//! - AppStore: Application configuration, rate-limiting rules, LB algorithms
//! - WorkerStore: Worker status, load, health
//! - PolicyStore: Routing policy internal state

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::crdt::{SKey, SyncCRDTMap, SyncPNCounter};

/// Store type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StoreType {
    Membership,
    App,
    Worker,
    Policy,
}

impl StoreType {
    pub fn as_str(&self) -> &'static str {
        match self {
            StoreType::Membership => "membership",
            StoreType::App => "app",
            StoreType::Worker => "worker",
            StoreType::Policy => "policy",
        }
    }

    /// Convert from proto StoreType (i32) to local StoreType
    pub fn from_proto(proto_value: i32) -> Self {
        match proto_value {
            0 => StoreType::Membership,
            1 => StoreType::App,
            2 => StoreType::Worker,
            3 => StoreType::Policy,
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

/// Rate-limit counter store (using PNCounter)
#[derive(Debug, Clone)]
pub struct RateLimitStore {
    counters: BTreeMap<String, SyncPNCounter>, // key -> counter
}

impl RateLimitStore {
    pub fn new() -> Self {
        Self {
            counters: BTreeMap::new(),
        }
    }

    pub fn get_or_create_counter(&mut self, key: String) -> &mut SyncPNCounter {
        self.counters
            .entry(key.clone())
            .or_insert_with(SyncPNCounter::new)
    }

    pub fn get_counter(&self, key: &str) -> Option<&SyncPNCounter> {
        self.counters.get(key)
    }

    pub fn inc(&mut self, key: String, actor: String, delta: i64) {
        let counter = self.get_or_create_counter(key);
        counter.inc(actor, delta);
    }

    pub fn value(&self, key: &str) -> Option<i64> {
        self.counters.get(key).map(|c| c.value())
    }
}

impl Default for RateLimitStore {
    fn default() -> Self {
        Self::new()
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
            rate_limit: RateLimitStore::new(),
        }
    }
}

impl Default for StateStores {
    fn default() -> Self {
        Self::new()
    }
}
