use anyhow::Result;
use std::ops::Deref;
use std::sync::Arc;

pub mod mock;
pub mod stream;
pub mod traits;

#[cfg(test)]
mod tests;

pub use stream::DecodeStream;
pub use traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer as TokenizerTrait};

/// Main tokenizer wrapper that provides a unified interface for different tokenizer implementations
#[derive(Clone)]
pub struct Tokenizer(Arc<dyn traits::Tokenizer>);

impl Tokenizer {
    /// Create a tokenizer from a file path
    /// Will be implemented in Phase 3 with factory pattern
    pub fn from_file(_file_path: &str) -> Result<Tokenizer> {
        // TODO: Implement factory pattern in Phase 3
        unimplemented!("Factory pattern will be implemented in Phase 3")
    }

    /// Create a tokenizer from an Arc<dyn Tokenizer>
    pub fn from_arc(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        Tokenizer(tokenizer)
    }

    /// Create a stateful sequence object for decoding token_ids into text
    pub fn decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> DecodeStream {
        DecodeStream::new(self.0.clone(), prompt_token_ids, skip_special_tokens)
    }

    /// Direct encode method
    pub fn encode(&self, input: &str) -> Result<Encoding> {
        self.0.encode(input)
    }

    /// Direct batch encode method
    pub fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.0.encode_batch(inputs)
    }

    /// Direct decode method
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.0.decode(token_ids, skip_special_tokens)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    /// Get special tokens
    pub fn get_special_tokens(&self) -> &SpecialTokens {
        self.0.get_special_tokens()
    }

    /// Convert token string to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    /// Convert ID to token string
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
}

impl Deref for Tokenizer {
    type Target = Arc<dyn traits::Tokenizer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Arc<dyn traits::Tokenizer>> for Tokenizer {
    fn from(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        Tokenizer(tokenizer)
    }
}
