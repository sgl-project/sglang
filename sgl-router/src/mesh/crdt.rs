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

use crdts::{CmRDT, CvRDT, PNCounter};
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tracing::warn;

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
#[derive(Debug, Clone, serde::Serialize)]
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
#[derive(Debug, Clone, serde::Serialize)]
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
        // Check if key already exists to preserve version
        if let Some(existing_reg) = self.inner.get_mut(&key) {
            // Update existing register, which will increment version
            existing_reg.write(value, actor);
        } else {
            // New entry, start with version 1
            let reg = LWWRegister::new(value, actor);
            self.inner.insert(key, reg);
        }
    }

    /// Get the version and actor for a key
    pub fn get_metadata(&self, key: &SKey) -> Option<(u64, String)> {
        self.inner
            .get(key)
            .map(|reg| (reg.version, reg.actor.clone()))
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
        // PNCounter API: inc(actor) and dec(actor) return operations that need to be applied
        if delta > 0 {
            for _ in 0..delta as u64 {
                let op = self.inner.inc(actor.clone());
                <PNCounter<String> as CmRDT>::apply(&mut self.inner, op);
            }
        } else if delta < 0 {
            for _ in 0..(-delta) as u64 {
                let op = self.inner.dec(actor.clone());
                <PNCounter<String> as CmRDT>::apply(&mut self.inner, op);
            }
        }
    }

    pub fn value(&self) -> i64 {
        // PNCounter returns BigInt, convert to i64
        let big_val = self.inner.read();
        // Convert BigInt to i64
        use num_traits::ToPrimitive;
        // Try direct conversion first
        match big_val.to_i64() {
            Some(val) => val,
            None => {
                // If direct conversion fails, try via string
                let val_str = big_val.to_string();
                match val_str.parse::<i64>() {
                    Ok(val) => val,
                    Err(_) => {
                        warn!(
                            "PNCounter value {} exceeds i64 range, truncating to 0. This may indicate rate limit counters growing too large.",
                            big_val
                        );
                        0
                    }
                }
            }
        }
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

    /// Get the version and actor for a key
    pub fn get_metadata(&self, key: &SKey) -> Option<(u64, String)> {
        self.inner.read().get_metadata(key)
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

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;

    #[test]
    fn test_lww_register_create_and_read() {
        let reg = LWWRegister::new("value1".to_string(), "actor1".to_string());
        assert_eq!(reg.read(), "value1");
        assert_eq!(reg.version, 1);
        assert_eq!(reg.actor, "actor1");
    }

    #[test]
    fn test_lww_register_version_increment() {
        let mut reg = LWWRegister::new("value1".to_string(), "actor1".to_string());
        let initial_version = reg.version;
        reg.write("value2".to_string(), "actor2".to_string());
        assert_eq!(reg.version, initial_version + 1);
        assert_eq!(reg.read(), "value2");
        assert_eq!(reg.actor, "actor2");
    }

    #[test]
    fn test_lww_register_merge_timestamp_priority() {
        let mut reg1 = LWWRegister::new("value1".to_string(), "actor1".to_string());
        thread::sleep(Duration::from_millis(10)); // Ensure different timestamp
        let reg2 = LWWRegister::new("value2".to_string(), "actor2".to_string());

        // reg2 has newer timestamp, should win
        reg1.merge(&reg2);
        assert_eq!(reg1.read(), "value2");
        assert_eq!(reg1.actor, "actor2");
    }

    #[test]
    fn test_lww_register_merge_version_priority() {
        let mut reg1 = LWWRegister::new("value1".to_string(), "actor1".to_string());
        let mut reg2 = LWWRegister::new("value2".to_string(), "actor2".to_string());

        // Set same timestamp but different versions
        reg2.timestamp = reg1.timestamp;
        reg2.version = reg1.version + 1;

        reg1.merge(&reg2);
        assert_eq!(reg1.read(), "value2");
        assert_eq!(reg1.version, reg2.version);
    }

    #[test]
    fn test_lww_register_concurrent_merge() {
        let mut reg1 = LWWRegister::new("value1".to_string(), "actor1".to_string());
        thread::sleep(Duration::from_millis(10));
        let reg2 = LWWRegister::new("value2".to_string(), "actor2".to_string());
        thread::sleep(Duration::from_millis(10));
        let reg3 = LWWRegister::new("value3".to_string(), "actor3".to_string());

        // Merge in different orders should give same result (latest wins)
        reg1.merge(&reg2);
        reg1.merge(&reg3);
        assert_eq!(reg1.read(), "value3");

        let mut reg4 = LWWRegister::new("value1".to_string(), "actor1".to_string());
        thread::sleep(Duration::from_millis(10));
        let reg5 = LWWRegister::new("value2".to_string(), "actor2".to_string());
        thread::sleep(Duration::from_millis(10));
        let reg6 = LWWRegister::new("value3".to_string(), "actor3".to_string());

        reg4.merge(&reg6);
        reg4.merge(&reg5);
        // reg6 should win (latest timestamp)
        assert_eq!(reg4.read(), "value3");
    }

    #[test]
    fn test_crdt_map_insert_get_remove() {
        let mut map = CRDTMap::new();
        let key = SKey::new("key1".to_string());

        map.insert(key.clone(), "value1".to_string(), "actor1".to_string());
        assert_eq!(map.get(&key), Some(&"value1".to_string()));
        assert_eq!(map.len(), 1);

        map.remove(&key);
        assert_eq!(map.get(&key), None);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_crdt_map_version_management() {
        let mut map = CRDTMap::new();
        let key = SKey::new("key1".to_string());

        map.insert(key.clone(), "value1".to_string(), "actor1".to_string());
        let (version1, actor1) = map.get_metadata(&key).unwrap();
        assert_eq!(version1, 1);
        assert_eq!(actor1, "actor1");

        map.insert(key.clone(), "value2".to_string(), "actor2".to_string());
        let (version2, actor2) = map.get_metadata(&key).unwrap();
        assert_eq!(version2, 2);
        assert_eq!(actor2, "actor2");
    }

    #[test]
    fn test_crdt_map_merge() {
        let mut map1 = CRDTMap::new();
        let mut map2 = CRDTMap::new();

        let key1 = SKey::new("key1".to_string());
        let key2 = SKey::new("key2".to_string());

        map1.insert(key1.clone(), "value1".to_string(), "actor1".to_string());
        map2.insert(key2.clone(), "value2".to_string(), "actor2".to_string());

        map1.merge(&map2);
        assert_eq!(map1.get(&key1), Some(&"value1".to_string()));
        assert_eq!(map1.get(&key2), Some(&"value2".to_string()));
        assert_eq!(map1.len(), 2);
    }

    #[test]
    fn test_crdt_map_merge_conflict_resolution() {
        let mut map1 = CRDTMap::new();
        let mut map2 = CRDTMap::new();

        let key = SKey::new("key1".to_string());

        map1.insert(key.clone(), "value1".to_string(), "actor1".to_string());
        thread::sleep(Duration::from_millis(10));
        map2.insert(key.clone(), "value2".to_string(), "actor2".to_string());

        // map2 has newer timestamp, should win
        map1.merge(&map2);
        assert_eq!(map1.get(&key), Some(&"value2".to_string()));
    }

    #[test]
    fn test_pn_counter_inc_dec() {
        let mut counter = CRDTPNCounter::new();
        assert_eq!(counter.value(), 0);

        counter.inc("actor1".to_string(), 5);
        assert_eq!(counter.value(), 5);

        counter.inc("actor2".to_string(), 3);
        assert_eq!(counter.value(), 8);

        counter.inc("actor1".to_string(), -2);
        assert_eq!(counter.value(), 6);
    }

    #[test]
    fn test_pn_counter_merge() {
        let mut counter1 = CRDTPNCounter::new();
        let mut counter2 = CRDTPNCounter::new();

        counter1.inc("actor1".to_string(), 10);
        counter2.inc("actor2".to_string(), 5);

        counter1.merge(&counter2);
        assert_eq!(counter1.value(), 15);
    }

    #[test]
    fn test_pn_counter_merge_idempotent() {
        let mut counter1 = CRDTPNCounter::new();
        let mut counter2 = CRDTPNCounter::new();

        counter1.inc("actor1".to_string(), 10);
        counter2.inc("actor1".to_string(), 10);

        counter1.merge(&counter2);
        // Merging same operations should not double count
        assert_eq!(counter1.value(), 10);
    }

    #[test]
    fn test_sync_crdt_map() {
        let map = SyncCRDTMap::new();
        let key = SKey::new("key1".to_string());

        map.insert(key.clone(), "value1".to_string(), "actor1".to_string());
        assert_eq!(map.get(&key), Some("value1".to_string()));

        let (version, actor) = map.get_metadata(&key).unwrap();
        assert_eq!(version, 1);
        assert_eq!(actor, "actor1");
    }

    #[test]
    fn test_sync_pn_counter() {
        let counter = SyncPNCounter::new();
        assert_eq!(counter.value(), 0);

        counter.inc("actor1".to_string(), 10);
        assert_eq!(counter.value(), 10);

        let snapshot = counter.snapshot();
        let counter2 = SyncPNCounter::new();
        counter2.merge(&snapshot);
        assert_eq!(counter2.value(), 10);
    }

    #[test]
    fn test_sync_pn_counter_concurrent() {
        let counter = Arc::new(SyncPNCounter::new());
        let mut handles = vec![];

        for i in 0..10 {
            let counter_clone = counter.clone();
            let handle = thread::spawn(move || {
                counter_clone.inc(format!("actor{}", i), 1);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.value(), 10);
    }
}
