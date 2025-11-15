use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use anyhow::Result;

/// Type alias for token IDs
pub type TokenIdType = u32;

/// Core encoding trait - separate from decoding for modularity
pub trait Encoder: Send + Sync {
    fn encode(&self, input: &str) -> Result<Encoding>;
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>>;
}

/// Core decoding trait - can be implemented independently
pub trait Decoder: Send + Sync {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String>;
}

/// Combined tokenizer trait
pub trait Tokenizer: Encoder + Decoder {
    fn vocab_size(&self) -> usize;
    fn get_special_tokens(&self) -> &SpecialTokens;
    fn token_to_id(&self, token: &str) -> Option<TokenIdType>;
    fn id_to_token(&self, id: TokenIdType) -> Option<String>;

    /// Enable downcasting to concrete types
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Contains the results of tokenizing text: token IDs, string tokens, and their spans
#[derive(Debug, Clone)]
pub enum Encoding {
    /// Hugging Face
    Hf(Box<tokenizers::tokenizer::Encoding>),
    /// Sentence Piece
    Sp(Vec<TokenIdType>),
    /// Tiktoken (for GPT models) - now uses u32 in tiktoken-rs 0.7.0
    Tiktoken(Vec<TokenIdType>),
}

impl Encoding {
    /// Returns a reference to token IDs - zero-copy operation
    pub fn token_ids(&self) -> &[TokenIdType] {
        match self {
            Encoding::Hf(inner) => inner.get_ids(),
            Encoding::Sp(inner) => inner,
            Encoding::Tiktoken(inner) => inner,
        }
    }

    /// Deprecated: Use token_ids() instead (kept for compatibility)
    #[deprecated(since = "0.1.0", note = "Use token_ids() instead")]
    pub fn token_ids_ref(&self) -> &[TokenIdType] {
        self.token_ids()
    }

    /// Get a hash of the token IDs for caching purposes
    pub fn get_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Hash implementation for Encoding
impl Hash for Encoding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Encoding::Hf(inner) => inner.get_ids().hash(state),
            Encoding::Sp(inner) => inner.hash(state),
            Encoding::Tiktoken(inner) => inner.hash(state),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Vec<String>,
}
