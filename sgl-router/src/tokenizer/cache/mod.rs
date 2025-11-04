//! Tokenizer Caching Layer
//!
//! Provides a caching wrapper around any tokenizer implementation to speed up
//! repeated tokenization of the same strings (e.g., system prompts).
//!
//! # Architecture
//! - **L0 Cache**: Whole-string exact match (90% of wins)
//! - **L1 Cache**: Prefix matching at fixed boundaries (future work)
//!
//! # Usage
//! ```ignore
//! let tokenizer = Arc::new(HuggingFaceTokenizer::from_file("tokenizer.json")?);
//! let cached = Arc::new(CachedTokenizer::new(tokenizer, CacheConfig::default()));
//! let encoding = cached.encode("Hello world")?;
//! ```

mod fingerprint;
mod l0;
mod l1;

use std::sync::Arc;

use anyhow::Result;
pub use fingerprint::TokenizerFingerprint;
pub use l0::{CacheStats, L0Cache};
pub use l1::{L1Cache, L1CacheStats};
use rayon::prelude::*;

use super::traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer};

/// Configuration for the tokenizer cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable L0 (whole-string) cache
    pub enable_l0: bool,
    /// Maximum number of entries in L0 cache
    pub l0_max_entries: usize,
    /// Enable L1 (prefix) cache
    pub enable_l1: bool,
    /// Maximum memory for L1 cache in bytes
    pub l1_max_memory: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000, // ~22MB memory for typical prompts
            enable_l1: false,       // Opt-in for now
            l1_max_memory: 50 * 1024 * 1024, // 50MB
        }
    }
}

/// A caching wrapper around any tokenizer
pub struct CachedTokenizer {
    /// The underlying tokenizer
    inner: Arc<dyn Tokenizer>,
    /// L0 cache (whole-string exact match)
    l0: Option<L0Cache>,
    /// L1 cache (prefix matching at fixed boundaries)
    l1: Option<L1Cache>,
    /// Configuration
    #[allow(dead_code)]
    config: CacheConfig,
    /// Fingerprint for cache invalidation
    fingerprint: TokenizerFingerprint,
    /// Cached special token strings (extracted once at construction)
    special_token_strings: Vec<String>,
}

impl CachedTokenizer {
    /// Create a new cached tokenizer
    pub fn new(inner: Arc<dyn Tokenizer>, config: CacheConfig) -> Self {
        let fingerprint = TokenizerFingerprint::from_tokenizer(inner.as_ref());

        let l0 = if config.enable_l0 {
            Some(L0Cache::new(config.l0_max_entries))
        } else {
            None
        };

        let l1 = if config.enable_l1 {
            Some(L1Cache::new(config.l1_max_memory))
        } else {
            None
        };

        // Extract special tokens once at construction time
        let special_token_strings = Self::extract_special_token_strings(&inner);

        Self {
            inner,
            l0,
            l1,
            config,
            fingerprint,
            special_token_strings,
        }
    }

    /// Extract all special token strings from the tokenizer (called once at construction)
    fn extract_special_token_strings(tokenizer: &Arc<dyn Tokenizer>) -> Vec<String> {
        let special_tokens = tokenizer.get_special_tokens();
        let mut tokens = Vec::new();

        if let Some(ref token) = special_tokens.bos_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.eos_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.unk_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.sep_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.pad_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.cls_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.mask_token {
            tokens.push(token.clone());
        }

        tokens.extend(special_tokens.additional_special_tokens.iter().cloned());
        tokens
    }

    /// Get L0 cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.l0.as_ref().map(|cache| cache.stats())
    }

    /// Get L1 cache statistics
    pub fn l1_cache_stats(&self) -> Option<L1CacheStats> {
        self.l1.as_ref().map(|cache| cache.stats())
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Some(l0) = &self.l0 {
            l0.clear();
        }
        if let Some(l1) = &self.l1 {
            l1.clear();
        }
    }

    /// Get the fingerprint of the underlying tokenizer
    pub fn fingerprint(&self) -> &TokenizerFingerprint {
        &self.fingerprint
    }

    /// Get a reference to the inner (wrapped) tokenizer
    pub fn inner(&self) -> &Arc<dyn Tokenizer> {
        &self.inner
    }
}

impl Encoder for CachedTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        // Collect special tokens once if L1 is enabled (avoid redundant allocation)
        let special_tokens: Option<Vec<&str>> = self.l1.as_ref().map(|_| {
            self.special_token_strings
                .iter()
                .map(|s| s.as_str())
                .collect()
        });

        // L0 cache lookup (exact match)
        if let Some(l0) = &self.l0 {
            if let Some(cached) = l0.get(input) {
                return Ok(cached);
            }
        }

        // L1 cache lookup (prefix match at special token boundaries)
        if let Some(l1) = &self.l1 {
            let tokens = special_tokens.as_ref().unwrap();

            if let Some((prefix_tokens, prefix_len)) = l1.longest_prefix_match(input, tokens) {
                // We have a prefix match - tokenize the suffix
                let suffix = &input[prefix_len..];
                if !suffix.is_empty() {
                    let suffix_encoding = self.inner.encode(suffix)?;

                    // Merge prefix tokens + suffix tokens
                    // Safe because we're splitting at special token boundaries
                    let mut merged_tokens = prefix_tokens;
                    merged_tokens.extend_from_slice(suffix_encoding.token_ids());

                    let merged_encoding = Encoding::Sp(merged_tokens);

                    // Cache the full result in L0
                    if let Some(l0) = &self.l0 {
                        l0.insert(input.to_string(), merged_encoding.clone());
                    }

                    return Ok(merged_encoding);
                }
            }
        }

        // Full tokenization (both L0 and L1 miss)
        let encoding = self.inner.encode(input)?;

        // Cache in L0
        if let Some(l0) = &self.l0 {
            l0.insert(input.to_string(), encoding.clone());
        }

        // Cache in L1 at special token boundaries
        // Re-tokenizes prefixes for correctness (optimized for high prefix reuse)
        if let Some(l1) = &self.l1 {
            let tokens = special_tokens.as_ref().unwrap();
            let _ = l1.insert_at_boundaries(input, self.inner.as_ref(), tokens);
            // Ignore errors in cache insertion - cache is best-effort
        }

        Ok(encoding)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        // Process each input in parallel, leveraging thread-safe caches
        // This maintains the parallelism from the underlying HuggingFaceTokenizer
        inputs.par_iter().map(|&input| self.encode(input)).collect()
    }
}

impl Decoder for CachedTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        // Decoding is not cached (it's fast enough and rarely repeated)
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CachedTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    #[test]
    fn test_cache_hit() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let input = "Hello world";

        // First call - miss
        let result1 = cached.encode(input).unwrap();

        // Second call - hit
        let result2 = cached.encode(input).unwrap();

        // Results should be identical
        assert_eq!(result1.token_ids(), result2.token_ids());

        // Check cache stats
        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_disabled() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = CacheConfig {
            enable_l0: false,
            l0_max_entries: 0,
            enable_l1: false,
            l1_max_memory: 0,
        };
        let cached = CachedTokenizer::new(tokenizer, config);

        let input = "Hello world";

        // Both calls should work even without cache
        let result1 = cached.encode(input).unwrap();
        let result2 = cached.encode(input).unwrap();

        assert_eq!(result1.token_ids(), result2.token_ids());

        // No cache stats available
        assert!(cached.cache_stats().is_none());
    }

    #[test]
    fn test_encode_batch() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let inputs = vec!["Hello", "world", "Hello"]; // "Hello" repeated

        let results = cached.encode_batch(&inputs).unwrap();

        assert_eq!(results.len(), 3);

        // With parallel execution, duplicate inputs may be processed simultaneously
        // and both see cache misses. Verify results are correct instead.
        assert_eq!(results[0].token_ids(), results[2].token_ids()); // Both "Hello" should match

        // After batch processing, cache should be populated
        // Subsequent calls should hit the cache
        let _ = cached.encode("Hello").unwrap();
        let stats = cached.cache_stats().unwrap();

        // Should have at least 1 hit from the call above (cache was populated by batch)
        assert!(
            stats.hits >= 1,
            "Expected at least 1 cache hit after batch processing"
        );
    }

    #[test]
    fn test_decoder_passthrough() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let tokens = vec![1, 2, 3];
        let decoded = cached.decode(&tokens, false).unwrap();

        // Should just pass through to inner tokenizer
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_tokenizer_trait_methods() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer.clone(), CacheConfig::default());

        // Should pass through to inner tokenizer
        assert_eq!(cached.vocab_size(), tokenizer.vocab_size());
        assert!(cached.token_to_id("Hello").is_some());
        assert!(cached.id_to_token(1).is_some());
    }
}
