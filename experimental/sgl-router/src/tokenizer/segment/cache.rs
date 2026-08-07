// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Storage backend for the segment cache: segment bytes → token ids.
//!
//! [`SegmentStore`] is the one storage abstraction [`super::SegmentCache`]
//! depends on. Today there is a single in-process backend ([`InProcStore`]); a
//! shared/cross-process tier (e.g. an shm-backed store, or a tiered store
//! combining the two) can be added later as another `SegmentStore` impl without
//! changing the cache.

use moka::sync::Cache;
use std::sync::Arc;

/// A store of segment → token-ids. Implement to add a backend (in-proc today;
/// shm / tiered / NUMA-sharded later). Must be thread-safe: one instance is
/// shared across all requests.
///
/// **Correctness contract:** a `get` hit MUST return the exact ids previously
/// `insert`ed for THAT segment — i.e. the store is collision-free. How it
/// achieves that is the backend's business ([`InProcStore`] keys by the bytes;
/// a hash-keyed shm store must byte-verify on hit).
pub trait SegmentStore: Send + Sync {
    /// Cached ids for `seg`, if present.
    fn get(&self, seg: &str) -> Option<Arc<[u32]>>;

    /// Store `ids` for `seg`.
    fn insert(&self, seg: &str, ids: Arc<[u32]>);

    /// Approximate number of cached segments (observability / tests).
    fn entry_count(&self) -> u64;

    /// Reserved hint for backends with expensive lookups (a future shm store):
    /// warm the slot for `seg` ahead of a `get`. Default no-op — in-proc stores
    /// have nothing to prefetch.
    fn prefetch(&self, _seg: &str) {}
}

/// In-process [`SegmentStore`]: a bounded `moka` map (LRU/TinyLFU eviction)
/// keyed by the segment string itself, so lookups are **collision-free** with no
/// manual hashing — a hit is always the exact segment's ids. Values are
/// `Arc<[u32]>` so a hit clones a pointer, not the token vector. Thread-safe:
/// `moka::sync::Cache` supports concurrent get/insert.
pub struct InProcStore {
    inner: Cache<Box<str>, Arc<[u32]>>,
}

impl InProcStore {
    /// `capacity` is the max number of cached segments.
    pub fn new(capacity: u64) -> Self {
        Self {
            inner: Cache::builder().max_capacity(capacity).build(),
        }
    }
}

impl std::fmt::Debug for InProcStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProcStore")
            .field("entry_count", &self.inner.entry_count())
            .finish()
    }
}

impl SegmentStore for InProcStore {
    fn get(&self, seg: &str) -> Option<Arc<[u32]>> {
        self.inner.get(seg)
    }

    fn insert(&self, seg: &str, ids: Arc<[u32]>) {
        self.inner.insert(seg.into(), ids);
    }

    fn entry_count(&self) -> u64 {
        self.inner.run_pending_tasks();
        self.inner.entry_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_by_exact_segment() {
        let c = InProcStore::new(16);
        assert!(c.get("<|im_start|>user\nhi").is_none());
        c.insert("<|im_start|>user\nhi", Arc::from(vec![1u32, 2, 3]));
        assert_eq!(
            c.get("<|im_start|>user\nhi").as_deref(),
            Some(&[1u32, 2, 3][..])
        );
        // a different segment is a distinct key (no hash-collision aliasing).
        assert!(c.get("<|im_start|>user\nbye").is_none());
        assert_eq!(c.entry_count(), 1);
    }
}
