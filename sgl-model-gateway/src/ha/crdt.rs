//! CRDT (Conflict-free Replicated Data Types) wrapper for HA state synchronization
//!
//! This module provides CRDT data structures for eventual consistency:
//! - Map<SKey, LWWReg> for Last-Write-Wins Register maps
//! - PNCounter for rate-limit and load balance aggregates

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;

use crdts::{CvRDT, PNCounter};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// State key for CRDT maps
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SKey(pub String);

impl SKey {
    pub fn new(key: String) -> Self {
        Self(key)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for SKey {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SKey {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Last-Write-Wins Register wrapper
/// Simplified implementation using timestamp and version
#[derive(Debug, Clone)]
#[derive(serde::Serialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct LWWRegister<T: Clone + Serialize + DeserializeOwned> {
    value: T,
    timestamp: u64,
    version: u64,
    actor: String,
}

impl<T: Clone + Serialize + DeserializeOwned> LWWRegister<T> {
    pub fn new(value: T, actor: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self {
            value,
            timestamp,
            version: 1,
            actor,
        }
    }

    pub fn read(&self) -> &T {
        &self.value
    }

    pub fn write(&mut self, value: T, actor: String) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.value = value;
        self.timestamp = timestamp;
        self.version += 1;
        self.actor = actor;
    }

    pub fn merge(&mut self, other: &Self) {
        // Last-Write-Wins: choose the one with higher timestamp, or higher version if equal
        if other.timestamp > self.timestamp
            || (other.timestamp == self.timestamp && other.version > self.version)
        {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.version = other.version;
            self.actor = other.actor.clone();
        }
    }
}

/// CRDT Map wrapper using LWWRegister for values
/// Simplified implementation using BTreeMap with LWWRegister values
#[derive(Debug, Clone)]
#[derive(serde::Serialize)]
#[serde(bound(serialize = "T: Serialize + DeserializeOwned"))]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct CRDTMap<T: Clone + Serialize + DeserializeOwned> {
    inner: BTreeMap<SKey, LWWRegister<T>>,
}

impl<T: Clone + Serialize + DeserializeOwned> Default for CRDTMap<T> {
    fn default() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }
}

impl<T: Clone + Serialize + DeserializeOwned> CRDTMap<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &SKey) -> Option<&T> {
        self.inner.get(key).map(|reg| reg.read())
    }

    pub fn insert(&mut self, key: SKey, value: T, actor: String) {
        let reg = LWWRegister::new(value, actor);
        self.inner.insert(key, reg);
    }

    pub fn remove(&mut self, key: &SKey) {
        self.inner.remove(key);
    }

    pub fn contains_key(&self, key: &SKey) -> bool {
        self.inner.contains_key(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SKey, &T)> {
        self.inner.iter().map(|(k, v)| (k, v.read()))
    }

    pub fn keys(&self) -> impl Iterator<Item = &SKey> {
        self.inner.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.inner.values().map(|v| v.read())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn merge(&mut self, other: &Self) {
        for (key, other_reg) in &other.inner {
            match self.inner.get_mut(key) {
                Some(self_reg) => {
                    self_reg.merge(other_reg);
                }
                None => {
                    self.inner.insert(key.clone(), other_reg.clone());
                }
            }
        }
    }

    pub fn to_map(&self) -> BTreeMap<SKey, T> {
        self.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

/// Positive-Negative Counter for rate-limit and load balance aggregates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRDTPNCounter {
    inner: PNCounter<String>,
}

impl Default for CRDTPNCounter {
    fn default() -> Self {
        Self {
            inner: PNCounter::new(),
        }
    }
}

impl CRDTPNCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inc(&mut self, actor: String, delta: i64) {
        // PNCounter API: inc(actor) increments by 1, dec(actor) decrements by 1
        if delta > 0 {
            for _ in 0..delta as u64 {
                self.inner.inc(actor.clone());
            }
        } else if delta < 0 {
            for _ in 0..(-delta) as u64 {
                self.inner.dec(actor.clone());
            }
        }
    }

    pub fn value(&self) -> i64 {
        // PNCounter returns BigInt, convert to i64
        let big_val = self.inner.read();
        // Convert BigInt to i64
        // Use try_into or manual conversion
        use num_traits::ToPrimitive;
        big_val.to_i64().unwrap_or(0)
    }

    pub fn merge(&mut self, other: &Self) {
        // Merge PNCounter using CvRDT trait
        // CvRDT::merge takes &mut self and other by value, but we need to clone
        let other_clone = other.inner.clone();
        <PNCounter<String> as CvRDT>::merge(&mut self.inner, other_clone);
    }
}

/// Thread-safe wrapper for CRDT Map
#[derive(Debug, Clone)]
pub struct SyncCRDTMap<T: Clone + Serialize + DeserializeOwned> {
    inner: Arc<RwLock<CRDTMap<T>>>,
}

impl<T: Clone + Serialize + DeserializeOwned> Default for SyncCRDTMap<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(CRDTMap::new())),
        }
    }
}

impl<T: Clone + Serialize + DeserializeOwned> SyncCRDTMap<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &SKey) -> Option<T> {
        self.inner.read().get(key).cloned()
    }

    pub fn insert(&self, key: SKey, value: T, actor: String) {
        self.inner.write().insert(key, value, actor);
    }

    pub fn remove(&self, key: &SKey) {
        self.inner.write().remove(key);
    }

    pub fn contains_key(&self, key: &SKey) -> bool {
        self.inner.read().contains_key(key)
    }

    pub fn merge(&self, other: &CRDTMap<T>) {
        self.inner.write().merge(other);
    }

    pub fn snapshot(&self) -> CRDTMap<T> {
        self.inner.read().clone()
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

/// Thread-safe wrapper for PNCounter
#[derive(Debug, Clone)]
pub struct SyncPNCounter {
    inner: Arc<RwLock<CRDTPNCounter>>,
}

impl Default for SyncPNCounter {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(CRDTPNCounter::new())),
        }
    }
}

impl SyncPNCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inc(&self, actor: String, delta: i64) {
        self.inner.write().inc(actor, delta);
    }

    pub fn value(&self) -> i64 {
        self.inner.read().value()
    }

    pub fn merge(&self, other: &CRDTPNCounter) {
        let mut inner = self.inner.write();
        inner.merge(other);
    }

    pub fn snapshot(&self) -> CRDTPNCounter {
        self.inner.read().clone()
    }
}


