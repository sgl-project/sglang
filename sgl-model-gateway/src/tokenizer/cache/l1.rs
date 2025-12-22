//! L1 Cache: Special-token boundary prefix cache
//!
//! Caches tokenization results at ALL special token boundaries.
//! Special tokens (like `<|im_start|>`, `<|im_end|>`) are atomic in BPE tokenizers (special: true, normalized: false),
//! making them the ONLY safe split points that guarantee correctness.
//!
//! **Design**: Cache at every special token boundary (not at fixed granularity intervals)
//! - Simple: No granularity parameter, no search windows
//! - Efficient: Fewer cache entries (10 instead of 64 for typical 8KB prompt)
//! - Natural: Aligns with actual chat template structure
//!
//! Example:
//!
//! Template: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\n{query}<|im_end|>"
//!
//! Request 1: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|>"
//! Request 2: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nHello!<|im_end|>"
//!
//! Cache points: After each "<|im_end|>" (atomic tokens, guaranteed safe)
//! Result: tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)

use std::{
    mem::size_of,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use blake3;
use dashmap::DashMap;

use super::super::traits::TokenIdType;

/// Hash type for cache keys
type Blake3Hash = [u8; 32];

/// Number of shards for concurrent access
const NUM_SHARDS: usize = 16;

/// Find ALL special token boundaries in the text
///
/// **ONLY uses special tokens** - these are atomic (special: true, normalized: false) in BPE,
/// guaranteeing: tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)
///
/// No fallback to whitespace/punctuation - better to not cache than risk corruption.
///
/// Common special tokens:
/// - ChatML: `<|im_start|>`, `<|im_end|>`
/// - Llama 3: `<|begin_of_text|>`, `<|end_of_text|>`, `<|eot_id|>`
/// - GPT: `<|endoftext|>`
/// - Custom: `<|reserved_special_token_N|>`
///
/// Returns positions immediately after each special token (where prefixes can be cached).
fn find_special_token_boundaries(text: &str, special_tokens: &[&str]) -> Vec<usize> {
    if special_tokens.is_empty() {
        return Vec::new();
    }

    let mut boundaries = Vec::new();

    // Find all special token end positions
    for &token in special_tokens {
        let mut start = 0;
        while let Some(pos) = text[start..].find(token) {
            let boundary = start + pos + token.len();
            // Only cache boundaries that leave some suffix to tokenize
            if boundary < text.len() {
                boundaries.push(boundary);
            }
            start = boundary;
        }
    }

    // Sort and deduplicate (in case multiple special tokens end at same position)
    boundaries.sort_unstable();
    boundaries.dedup();

    boundaries
}

/// A cached prefix entry
/// Uses Arc<[TokenIdType]> for zero-copy access to tokens
#[derive(Debug, Clone)]
struct CachedPrefix {
    /// The pre-computed token IDs for this prefix (Arc for zero-copy cloning)
    tokens: Arc<[TokenIdType]>,
    /// Last access timestamp (for LRU eviction)
    last_accessed: Arc<AtomicU64>,
    /// Size in bytes (for memory tracking during eviction)
    size_bytes: usize,
}

/// L1 cache implementation with special-token-boundary prefix matching
pub struct L1Cache {
    /// Sharded maps for concurrent access
    /// Key: Blake3 hash of bytes[0..boundary]
    /// Value: Cached token IDs for that prefix
    shards: Vec<Arc<DashMap<Blake3Hash, CachedPrefix>>>,
    /// Maximum memory in bytes
    max_memory: usize,
    /// Current memory usage estimate
    current_memory: AtomicU64,
    /// Cache hit counter
    hits: AtomicU64,
    /// Cache miss counter
    misses: AtomicU64,
    /// Monotonic counter for LRU timestamps
    access_counter: AtomicU64,
}

impl L1Cache {
    /// Create a new L1 cache with the specified memory limit
    pub fn new(max_memory: usize) -> Self {
        let shards = (0..NUM_SHARDS).map(|_| Arc::new(DashMap::new())).collect();

        Self {
            shards,
            max_memory,
            current_memory: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
        }
    }

    /// Try to find the longest prefix match at special token boundaries
    /// Returns (cached_tokens, byte_offset) if found
    ///
    /// Uses pre-computed tokens cached during insertion.
    /// Returns Vec<TokenIdType> as the caller needs to extend it with suffix tokens.
    pub fn longest_prefix_match(
        &self,
        input: &str,
        special_tokens: &[&str],
    ) -> Option<(Vec<TokenIdType>, usize)> {
        let boundaries = find_special_token_boundaries(input, special_tokens);

        if boundaries.is_empty() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Search backwards from the longest boundary to find the best match
        for &boundary_pos in boundaries.iter().rev() {
            let prefix = &input[0..boundary_pos];
            let prefix_bytes = prefix.as_bytes();
            let hash = blake3::hash(prefix_bytes);
            let hash_bytes: Blake3Hash = *hash.as_bytes();

            let shard_idx = hash_bytes[0] as usize % NUM_SHARDS;

            if let Some(entry) = self.shards[shard_idx].get(&hash_bytes) {
                // Update last accessed timestamp for LRU
                let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);
                entry.last_accessed.store(timestamp, Ordering::Relaxed);

                self.hits.fetch_add(1, Ordering::Relaxed);
                // Convert Arc<[T]> to Vec<T> - caller will extend with suffix tokens
                return Some((entry.tokens.to_vec(), boundary_pos));
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert prefix entries at ALL special token boundaries
    ///
    /// Re-tokenizes each prefix to ensure correctness (BPE tokenization is not prefix-stable).
    /// This is more expensive on cache misses but provides correct tokens for cache hits.
    ///
    /// Optimized for workloads with high prefix reuse (e.g., chat templates with repeated system prompts).
    pub fn insert_at_boundaries<E: super::super::traits::Encoder + ?Sized>(
        &self,
        input: &str,
        tokenizer: &E,
        special_tokens: &[&str],
    ) -> anyhow::Result<()> {
        let boundaries = find_special_token_boundaries(input, special_tokens);

        if boundaries.is_empty() {
            return Ok(());
        }

        // Calculate how much memory we need and tokenize each prefix
        let mut entries_to_insert = Vec::with_capacity(boundaries.len());
        for &boundary_pos in &boundaries {
            // Extract prefix up to this special token boundary
            let prefix = &input[0..boundary_pos];
            let prefix_bytes = prefix.as_bytes();
            let hash = blake3::hash(prefix_bytes);
            let hash_bytes: Blake3Hash = *hash.as_bytes();

            // Re-tokenize the prefix for guaranteed correctness
            // This is the only way to know the exact token boundaries
            let prefix_encoding = tokenizer.encode(prefix)?;
            // Convert to Arc<[TokenIdType]> for zero-copy sharing
            let prefix_tokens: Arc<[TokenIdType]> = prefix_encoding.token_ids().into();

            // Size = text bytes + token storage
            let size_bytes = boundary_pos + prefix_tokens.len() * size_of::<TokenIdType>();

            entries_to_insert.push((hash_bytes, prefix_tokens, size_bytes));
        }

        if entries_to_insert.is_empty() {
            return Ok(());
        }

        let total_size_needed: usize = entries_to_insert.iter().map(|(_, _, size)| size).sum();

        // Evict if necessary
        let current = self.current_memory.load(Ordering::Relaxed) as usize;
        if current + total_size_needed > self.max_memory {
            self.evict_lru(total_size_needed);
        }

        // Insert all entries
        let current_timestamp = self.access_counter.load(Ordering::Relaxed);
        for (hash_bytes, prefix_tokens, size_bytes) in entries_to_insert {
            let shard_idx = hash_bytes[0] as usize % NUM_SHARDS;

            let cached = CachedPrefix {
                tokens: prefix_tokens, // Already Arc<[TokenIdType]>
                last_accessed: Arc::new(AtomicU64::new(current_timestamp)),
                size_bytes,
            };

            self.shards[shard_idx].insert(hash_bytes, cached);
            self.current_memory
                .fetch_add(size_bytes as u64, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Evict least recently used entries using approximate LRU via random sampling
    ///
    /// This uses an approximate LRU strategy that's much faster than true LRU:
    /// - Samples K random entries from the cache (K=32)
    /// - Evicts the oldest entry among the samples
    /// - Repeats until enough space is freed
    ///
    /// This provides O(samples) complexity instead of O(total_entries * log(total_entries)),
    /// avoiding latency spikes when eviction is triggered on large caches.
    ///
    /// The approximation is excellent in practice - sampling 32 entries from a large cache
    /// gives high probability of finding very old entries.
    fn evict_lru(&self, space_needed: usize) {
        const SAMPLE_SIZE: usize = 32; // Number of entries to sample per eviction round
        let mut freed = 0usize;
        let mut iteration = 0usize;

        // Keep evicting until we have enough space
        while freed < space_needed {
            // Collect samples from shards
            let mut samples: Vec<(usize, Blake3Hash, u64, usize)> = Vec::with_capacity(SAMPLE_SIZE);

            // Sample entries across different shards
            for i in 0..SAMPLE_SIZE {
                // Distribute samples across shards using iteration and index for variety
                let shard_idx = (iteration * SAMPLE_SIZE + i) % NUM_SHARDS;

                // Get first entry from that shard (DashMap iteration order is arbitrary)
                if let Some(entry) = self.shards[shard_idx].iter().next() {
                    let hash = *entry.key();
                    let timestamp = entry.value().last_accessed.load(Ordering::Relaxed);
                    let size = entry.value().size_bytes;
                    samples.push((shard_idx, hash, timestamp, size));
                }
            }

            if samples.is_empty() {
                // Cache is empty, nothing to evict
                break;
            }

            // Find the oldest entry among samples
            if let Some((shard_idx, hash, _, _)) =
                samples.iter().min_by_key(|(_, _, ts, _)| ts).copied()
            {
                // Remove it
                if let Some((_, removed)) = self.shards[shard_idx].remove(&hash) {
                    freed += removed.size_bytes;
                    self.current_memory
                        .fetch_sub(removed.size_bytes as u64, Ordering::Relaxed);
                }
            }

            iteration += 1;
        }
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Get cache statistics
    pub fn stats(&self) -> L1CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        L1CacheStats {
            hits,
            misses,
            entries: self.len(),
            memory_bytes: self.current_memory.load(Ordering::Relaxed) as usize,
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
        self.current_memory.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct L1CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub memory_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    #[test]
    fn test_basic_prefix_match() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = MockTokenizer::new();

        // Realistic ChatML template with special tokens
        let input1 = "<|im_start|>system\nYou are a helpful assistant that provides clear and detailed responses.<|im_end|><|im_start|>user\nHello there! How are you doing today?<|im_end|>";

        // Insert at special token boundaries (re-tokenizes prefixes)
        cache
            .insert_at_boundaries(input1, &tokenizer, special_tokens)
            .unwrap();

        // Should have cached at special token boundaries
        assert!(!cache.is_empty());

        // Search with same prefix but different user query
        let input2 = "<|im_start|>system\nYou are a helpful assistant that provides clear and detailed responses.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|>";
        let result = cache.longest_prefix_match(input2, special_tokens);

        // Should find a match at the special token boundary (after system message)
        assert!(result.is_some());
        let (tokens, offset) = result.unwrap();
        assert!(offset > 0);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_short_input_with_boundaries() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = MockTokenizer::new();

        // Short input with special tokens
        let input = "<|im_start|>user\nHi<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens)
            .unwrap();

        // Should cache at <|im_start|> boundary (has suffix left)
        assert!(!cache.is_empty());

        // Should find a match
        let result = cache.longest_prefix_match(input, special_tokens);
        assert!(result.is_some());
    }

    #[test]
    fn test_longest_match() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = MockTokenizer::new();

        // Create multi-turn conversation with multiple special token boundaries (~400 bytes)
        let input = "<|im_start|>system\nYou are a helpful AI assistant that provides detailed and accurate responses.<|im_end|><|im_start|>user\nHello there! How are you today? Can you help me understand how tokenization works in language models?<|im_end|><|im_start|>assistant\nI'm doing well, thank you! I'd be happy to explain tokenization. Tokenization is the process of breaking text into smaller units called tokens.<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens)
            .unwrap();

        // Should have multiple entries at special token boundaries
        assert!(cache.len() >= 2); // At least 2 boundaries

        // Search with partial conversation - should match at a special token boundary
        let partial_input = "<|im_start|>system\nYou are a helpful AI assistant that provides detailed and accurate responses.<|im_end|><|im_start|>user\nHello there! How are you today? Can you help me understand how tokenization works in language models?<|im_end|>";
        let result = cache.longest_prefix_match(partial_input, special_tokens);

        // Should find a match at a special token boundary
        assert!(result.is_some());
        let (_, offset) = result.unwrap();
        assert!(offset > 0);
        assert!(offset <= partial_input.len());
    }

    #[test]
    fn test_stats() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = MockTokenizer::new();

        // ChatML input with special tokens
        let input = "<|im_start|>system\nYou are a helpful assistant that provides detailed answers.<|im_end|><|im_start|>user\nHello there! How are you today?<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens)
            .unwrap();

        // Try to find match
        let _ = cache.longest_prefix_match(input, special_tokens);

        let stats = cache.stats();
        // Should have at least one hit (the longest special token boundary should match)
        assert!(stats.hits >= 1);
        assert_eq!(stats.hit_rate, 1.0);
    }

    #[test]
    fn test_clear() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = MockTokenizer::new();

        // ChatML input with special tokens
        let input = "<|im_start|>system\nYou are a helpful assistant that provides clear and detailed responses.<|im_end|><|im_start|>user\nHello there!<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens)
            .unwrap();
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_lru_eviction() {
        // Create a small cache (5KB) to trigger eviction
        let cache = L1Cache::new(5 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>", "<|eot_id|>"];
        let tokenizer = MockTokenizer::new();

        // Insert first conversation
        let input1 = "<|im_start|>system\nYou are a helpful assistant specialized in mathematics.<|im_end|><|im_start|>user\nCan you explain calculus to me?<|im_end|><|im_start|>assistant\nCertainly! Calculus is a branch of mathematics that studies continuous change.<|im_end|><|eot_id|>";
        cache
            .insert_at_boundaries(input1, &tokenizer, special_tokens)
            .unwrap();

        // Access the first entry to update its timestamp
        let result = cache.longest_prefix_match(input1, special_tokens);
        assert!(result.is_some());

        // Insert second conversation
        let input2 = "<|im_start|>system\nYou are a helpful assistant specialized in physics.<|im_end|><|im_start|>user\nWhat is quantum mechanics?<|im_end|><|im_start|>assistant\nQuantum mechanics is the fundamental theory describing nature at atomic and subatomic scales.<|im_end|><|eot_id|>";
        cache
            .insert_at_boundaries(input2, &tokenizer, special_tokens)
            .unwrap();

        // Access the second entry to make it more recent
        let result = cache.longest_prefix_match(input2, special_tokens);
        assert!(result.is_some());

        // Insert third conversation (should trigger eviction of oldest)
        let input3 = "<|im_start|>system\nYou are a helpful assistant specialized in chemistry.<|im_end|><|im_start|>user\nExplain the periodic table to me please.<|im_end|><|im_start|>assistant\nThe periodic table is a tabular arrangement of chemical elements organized by atomic number and electron configuration.<|im_end|><|eot_id|>";
        cache
            .insert_at_boundaries(input3, &tokenizer, special_tokens)
            .unwrap();

        // Verify cache didn't exceed max memory
        let stats = cache.stats();
        assert!(stats.memory_bytes <= 5 * 1024);

        // The most recently accessed entries should still be present
        let result = cache.longest_prefix_match(input3, special_tokens);
        assert!(result.is_some());
    }
}
