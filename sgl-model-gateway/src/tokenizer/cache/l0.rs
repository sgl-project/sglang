//! L0 Cache: Whole-string exact match cache
//!
//! This is the simplest and most effective cache layer.
//! Key: input string → Value: full encoding result (Arc-wrapped for zero-copy cache hits)
//!
//! Expected hit rate: 60-90% for workloads with repeated system prompts

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use dashmap::DashMap;

use super::super::traits::Encoding;

/// L0 cache implementation using DashMap for lock-free reads
/// Uses Arc<Encoding> internally to provide zero-copy cache hits
pub struct L0Cache {
    /// The cache map: input string → Arc-wrapped encoding for cheap cloning
    map: Arc<DashMap<String, Arc<Encoding>>>,
    /// Maximum number of entries before eviction
    max_entries: usize,
    /// Cache hit counter
    hits: AtomicU64,
    /// Cache miss counter
    misses: AtomicU64,
}

impl L0Cache {
    /// Create a new L0 cache with the specified capacity
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: Arc::new(DashMap::with_capacity(max_entries.min(1024))),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get an encoding from the cache (returns Arc for zero-copy access)
    #[inline]
    pub fn get(&self, key: &str) -> Option<Arc<Encoding>> {
        match self.map.get(key) {
            Some(entry) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                // Arc::clone is cheap (just increment reference count)
                Some(Arc::clone(entry.value()))
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert an encoding into the cache
    pub fn insert(&self, key: String, value: Encoding) {
        // Simple eviction: if we're at capacity, remove a random entry
        // DashMap doesn't support LRU directly, so we use a simple strategy
        if self.map.len() >= self.max_entries {
            // Get the key to remove in a separate scope to ensure iterator is dropped
            let key_to_remove = { self.map.iter().next().map(|entry| entry.key().clone()) }; // Iterator fully dropped here, all locks released

            // Now remove it
            if let Some(k) = key_to_remove {
                self.map.remove(&k);
            }
        }

        self.map.insert(key, Arc::new(value));
    }

    /// Insert a pre-wrapped Arc encoding into the cache (avoids double-wrapping)
    pub fn insert_arc(&self, key: String, value: Arc<Encoding>) {
        if self.map.len() >= self.max_entries {
            let key_to_remove = { self.map.iter().next().map(|entry| entry.key().clone()) };
            if let Some(k) = key_to_remove {
                self.map.remove(&k);
            }
        }
        self.map.insert(key, value);
    }

    /// Get the current number of entries in the cache
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        CacheStats {
            hits,
            misses,
            entries: self.len(),
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.map.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Rough estimate:
        // - Each entry: key (string) + value (encoding ~250 tokens * 4 bytes) + overhead
        // - Average: ~2.2KB per entry
        self.len() * 2200
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::traits::Encoding;

    fn mock_encoding(tokens: Vec<u32>) -> Encoding {
        Encoding::Sp(tokens)
    }

    #[test]
    fn test_basic_get_set() {
        let cache = L0Cache::new(10);

        // Miss
        assert!(cache.get("hello").is_none());

        // Insert
        cache.insert("hello".to_string(), mock_encoding(vec![1, 2, 3]));

        // Hit - now returns Arc<Encoding>
        let result = cache.get("hello");
        assert!(result.is_some());
        assert_eq!(result.unwrap().token_ids(), &[1, 2, 3]);
    }

    #[test]
    fn test_eviction() {
        let cache = L0Cache::new(2);

        cache.insert("a".to_string(), mock_encoding(vec![1]));
        cache.insert("b".to_string(), mock_encoding(vec![2]));

        // Should evict when adding third
        cache.insert("c".to_string(), mock_encoding(vec![3]));

        // Cache should have exactly 2 entries
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_stats() {
        let cache = L0Cache::new(10);

        cache.insert("test".to_string(), mock_encoding(vec![1, 2, 3]));

        // 1 miss (initial get that returned None)
        let _ = cache.get("missing");

        // 1 hit
        let _ = cache.get("test");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_clear() {
        let cache = L0Cache::new(10);

        cache.insert("test".to_string(), mock_encoding(vec![1, 2, 3]));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get("test").is_none());
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let cache = Arc::new(L0Cache::new(1000));
        let mut handles = vec![];

        // Spawn 10 threads
        for i in 0..10 {
            let cache_clone = cache.clone();
            handles.push(thread::spawn(move || {
                // Each thread inserts and reads
                let key = format!("key_{}", i);
                cache_clone.insert(key.clone(), mock_encoding(vec![i as u32]));

                // Read it back
                let result = cache_clone.get(&key);
                assert!(result.is_some());
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 10 entries
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_arc_reuse() {
        // Test that multiple gets return the same Arc (reference counting)
        let cache = L0Cache::new(10);
        cache.insert("test".to_string(), mock_encoding(vec![1, 2, 3]));

        let arc1 = cache.get("test").unwrap();
        let arc2 = cache.get("test").unwrap();

        // Both should point to the same allocation
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }
}
