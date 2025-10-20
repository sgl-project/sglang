//! Tokenizer Fingerprinting for Cache Invalidation
//!
//! Creates a unique fingerprint of a tokenizer's configuration to detect
//! when the tokenizer has changed and the cache needs to be cleared.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use super::super::traits::Tokenizer;

/// A fingerprint of a tokenizer's configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenizerFingerprint {
    /// Size of the vocabulary
    pub vocab_size: usize,
    /// Hash of a sample of vocabulary tokens (for speed)
    pub vocab_hash: u64,
    /// Hash of special tokens
    pub special_tokens_hash: u64,
}

impl TokenizerFingerprint {
    /// Create a fingerprint from a tokenizer
    pub fn from_tokenizer(tokenizer: &dyn Tokenizer) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let vocab_hash = Self::compute_vocab_hash(tokenizer);
        let special_tokens_hash = Self::compute_special_tokens_hash(tokenizer);

        Self {
            vocab_size,
            vocab_hash,
            special_tokens_hash,
        }
    }

    /// Compute a hash of the vocabulary by sampling tokens
    fn compute_vocab_hash(tokenizer: &dyn Tokenizer) -> u64 {
        let mut hasher = DefaultHasher::new();
        let vocab_size = tokenizer.vocab_size();

        // Sample up to 1000 tokens for speed
        let sample_size = vocab_size.min(1000);
        let step = if sample_size > 0 {
            vocab_size / sample_size
        } else {
            1
        };

        for i in (0..vocab_size).step_by(step.max(1)) {
            if let Some(token) = tokenizer.id_to_token(i as u32) {
                token.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Compute a hash of special tokens
    fn compute_special_tokens_hash(tokenizer: &dyn Tokenizer) -> u64 {
        let mut hasher = DefaultHasher::new();
        let special_tokens = tokenizer.get_special_tokens();

        special_tokens.bos_token.hash(&mut hasher);
        special_tokens.eos_token.hash(&mut hasher);
        special_tokens.unk_token.hash(&mut hasher);
        special_tokens.sep_token.hash(&mut hasher);
        special_tokens.pad_token.hash(&mut hasher);
        special_tokens.cls_token.hash(&mut hasher);
        special_tokens.mask_token.hash(&mut hasher);
        special_tokens.additional_special_tokens.hash(&mut hasher);

        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    #[test]
    fn test_fingerprint_equality() {
        let tokenizer1 = MockTokenizer::new();
        let tokenizer2 = MockTokenizer::new();

        let fp1 = TokenizerFingerprint::from_tokenizer(&tokenizer1);
        let fp2 = TokenizerFingerprint::from_tokenizer(&tokenizer2);

        // Same tokenizer config should produce same fingerprint
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_consistency() {
        let tokenizer = MockTokenizer::new();

        let fp1 = TokenizerFingerprint::from_tokenizer(&tokenizer);
        let fp2 = TokenizerFingerprint::from_tokenizer(&tokenizer);

        // Fingerprint should be consistent
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.vocab_size, tokenizer.vocab_size());
    }
}
